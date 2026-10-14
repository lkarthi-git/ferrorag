use crate::idempotency::{IdempotencyStore, IdempotencyStatus};
use crate::node::Node;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, trace, warn,error};


/// A trait for types that contain a unique, deterministic identifier 
/// used for idempotency checks.
pub trait Idempotent {
    fn get_id(&self) -> String;
}

/// A middleware node that prevents duplicate processing of the same item.
///
/// It uses an external `IdempotencyStore` (e.g., Redis, DynamoDB) to coordinate
/// state across distributed workers or concurrent tasks. If an item has already 
/// been processed, or is currently being processed by another task, it is skipped.
pub struct IdempotencyNode<N, S> {
    pub inner_node: N,
    pub store: S, 
}

impl<N, S> IdempotencyNode<N, S> {
    /// Creates a new `IdempotencyNode`.
    pub fn new(inner_node: N, store: S) -> Self {
        Self { inner_node, store }
    }
}

impl<N, S, I, O, E> Node for IdempotencyNode<N, S> 
where 
    N: Node<Input = I, Output = Vec<O>, Error = E> + Send + Sync,
    S: IdempotencyStore + Send + Sync,
    I: Idempotent + Send + Sync,
    O: Default + Send,
    E: From<S::Error> + Send + From<std::io::Error>, 
{
    type Input = I;
    type Output = Vec<O>;
    type Error = E;

    fn name(&self) -> &'static str {
        "IdempotencyNode"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let node_name = self.name();
        let id = input.get_id();
        trace!(item_id = %id, "Checking idempotency status");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System clock was set backwards")
            .as_secs();
        // 1. Check if we should process this item
        match self.store.check_and_lock(&id).await? {
            IdempotencyStatus::Completed => {
                metrics::counter!("pipeline_node_idempotency_skipped_total", "node" => node_name, "reason" => "completed").increment(1);
                debug!(
                    item_id = %id, 
                    "Item already processed. Skipping execution."
                );
                return Ok(Vec::new()); 
            }
            IdempotencyStatus::InProgress { expires_at } => {
                // NOTE: This drops the item if another task is working on it.
                // If strict consistency is required, you might want to implement a 
                // retry loop/delay here to wait for the other task to finish.
                if now < expires_at {
                        // ACTIVE LOCK: Someone else is currently working on this.
                        metrics::counter!("pipeline_node_idempotency_skipped_total", "node" => node_name, "reason" => "in_progress").increment(1);
                        warn!(
                            item_id = %id, 
                            "Item currently in progress elsewhere. Dropping to prevent duplication."
                        );
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::WouldBlock, 
                            "Item is currently in progress by another worker"
                        ).into());
                    } else {
                        // STALE LOCK: The previous worker died. 
                        metrics::counter!("pipeline_node_idempotency_passed_total", "node" => node_name, "reason" => "stale_lock_stolen").increment(1);
                        trace!(
                            item_id = %id, 
                            "Stale lock detected. Stealing idempotency lock."
                        );
                        
                        // CRITICAL: You must update the database to claim the lock so other threads don't also steal it!
                        // You need a method on your store like `steal_lock(&id)` to reset the expiration.
                        self.store.steal_lock(&id).await?; 
                        
                        // After stealing, we naturally fall out of the match statement 
                        // and proceed down to self.inner_node.execute(input).await
                    }
            }
            IdempotencyStatus::New => {
                metrics::counter!("pipeline_node_idempotency_passed_total", "node" => node_name).increment(1);
                trace!(
                    item_id = %id, 
                    "Item is new. Idempotency lock acquired."
                );
            } 
        }

        // 2. Execute the inner node
        match self.inner_node.execute(input).await {
            Ok(output) => {
                trace!(item_id = %id, "Execution successful. Marking as completed in store.");
                self.store.mark_success(&id).await?;
                Ok(output)
            }
            Err(error) => {
                warn!(
                    item_id = %id, 
                    // Using debug formatting for the error if it doesn't implement Display,
                    // but assuming standard error traits are met.
                    "Execution failed. Releasing lock so item can be retried."
                );
                
                if let Err(store_err) = self.store.release_lock(&id).await {
                    error!(
                        item_id = %id, 
                        error = %store_err, 
                        "Failed to release idempotency lock. Item may be stuck as InProgress."
                    );
                }

                Err(error)
            }
        }
    }

    fn flush(&self) -> Result<Self::Output, Self::Error> {
        self.inner_node.flush()
    }
}

#[cfg(test)]

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;
    use std::io;
    use std::time::{SystemTime, UNIX_EPOCH};

    // --- Mock Setup ---

    struct TestInput {
        id: String,
        should_fail: bool,
    }

    impl Idempotent for TestInput {
        fn get_id(&self) -> String {
            self.id.clone()
        }
    }

    struct MockNode;
    impl Node for MockNode {
        type Input = TestInput;
        type Output = Vec<String>;
        type Error = io::Error;

        async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
            if input.should_fail {
                Err(io::Error::new(io::ErrorKind::Other, "Mock Node Execution Failure"))
            } else {
                Ok(vec![format!("Processed {}", input.id)])
            }
        }
        
        fn name(&self) -> &'static str {
            "MockNode"
        }
    }

    /// An in-memory mock store to verify the node interacts with the store correctly.
    #[derive(Clone)]
    struct MockStore {
        state: Arc<Mutex<HashMap<String, IdempotencyStatus>>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                state: Arc::new(Mutex::new(HashMap::new())),
            }
        }
        
        fn seed(&self, id: &str, status: IdempotencyStatus) {
            self.state.lock().unwrap().insert(id.to_string(), status);
        }
        
        fn get_status(&self, id: &str) -> Option<IdempotencyStatus> {
            self.state.lock().unwrap().get(id).copied()
        }
        
        fn current_time_secs() -> u64 {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }
    }

    impl IdempotencyStore for MockStore {
        type Error = io::Error;

        async fn check_and_lock(&self, id: &str) -> Result<IdempotencyStatus, Self::Error> {
            let mut state = self.state.lock().unwrap();
            let current = state.get(id).copied().unwrap_or(IdempotencyStatus::New);
            
            // If it's new, we lock it immediately with a 15-minute TTL
            if matches!(current, IdempotencyStatus::New) {
                let expires_at = Self::current_time_secs() + 900;
                state.insert(id.to_string(), IdempotencyStatus::InProgress { expires_at });
            }
            Ok(current)
        }

        async fn mark_success(&self, id: &str) -> Result<(), Self::Error> {
            let mut state = self.state.lock().unwrap();
            state.insert(id.to_string(), IdempotencyStatus::Completed);
            Ok(())
        }

        async fn release_lock(&self, id: &str) -> Result<(), Self::Error> {
            let mut state = self.state.lock().unwrap();
            state.insert(id.to_string(), IdempotencyStatus::New);
            Ok(())
        }
        
        // Added the steal_lock method required to fix the fall-through bug
        async fn steal_lock(&self, id: &str) -> Result<(), Self::Error> {
            let mut state = self.state.lock().unwrap();
            let expires_at = Self::current_time_secs() + 900;
            state.insert(id.to_string(), IdempotencyStatus::InProgress { expires_at });
            Ok(())
        }
    }

    // --- Tests ---

    #[tokio::test]
    async fn test_new_item_processes_and_marks_completed() {
        let store = MockStore::new();
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_1".into(), should_fail: false };

        let result = node.execute(&input).await.unwrap();

        assert_eq!(result, vec!["Processed item_1"]);
        assert_eq!(store.get_status("item_1"), Some(IdempotencyStatus::Completed));
    }

    #[tokio::test]
    async fn test_node_failure_releases_lock() {
        let store = MockStore::new();
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_2".into(), should_fail: true };

        let result = node.execute(&input).await;

        assert!(result.is_err());
        assert_eq!(store.get_status("item_2"), Some(IdempotencyStatus::New));
    }

    #[tokio::test]
    async fn test_completed_item_is_skipped() {
        let store = MockStore::new();
        store.seed("item_3", IdempotencyStatus::Completed);
        
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_3".into(), should_fail: false };

        let result = node.execute(&input).await.unwrap();

        assert!(result.is_empty());
        assert_eq!(store.get_status("item_3"), Some(IdempotencyStatus::Completed));
    }

    #[tokio::test]
    async fn test_active_in_progress_item_returns_would_block() {
        let store = MockStore::new();
        // Seed an ACTIVE lock (expires 1 hour in the future)
        let future_time = MockStore::current_time_secs() + 3600;
        store.seed("item_4", IdempotencyStatus::InProgress { expires_at: future_time });
        
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_4".into(), should_fail: false };

        let result = node.execute(&input).await;

        // Should return a WouldBlock error to prevent duplicate execution
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::WouldBlock);
        
        // Should remain InProgress with the original future time
        assert_eq!(store.get_status("item_4"), Some(IdempotencyStatus::InProgress { expires_at: future_time }));
    }

    #[tokio::test]
    async fn test_stale_in_progress_item_steals_lock_and_processes() {
        let store = MockStore::new();
        // Seed a STALE lock (expired 1 hour ago)
        let past_time = MockStore::current_time_secs() - 3600;
        store.seed("item_5", IdempotencyStatus::InProgress { expires_at: past_time });
        
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_5".into(), should_fail: false };

        let result = node.execute(&input).await.unwrap();

        // Because it was stale, it should steal the lock, process successfully, and return the data
        assert_eq!(result, vec!["Processed item_5"]);
        
        // It should finally be marked as completed
        assert_eq!(store.get_status("item_5"), Some(IdempotencyStatus::Completed));
    }
}