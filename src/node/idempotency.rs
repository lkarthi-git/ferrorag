use crate::idempotency::{IdempotencyStore, IdempotencyStatus};
use crate::node::Node;
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
            IdempotencyStatus::InProgress => {
                metrics::counter!("pipeline_node_idempotency_skipped_total", "node" => node_name, "reason" => "in_progress").increment(1);
                // NOTE: This drops the item if another task is working on it.
                // If strict consistency is required, you might want to implement a 
                // retry loop/delay here to wait for the other task to finish.
                warn!(
                    item_id = %id, 
                    "Item currently in progress elsewhere. Dropping to prevent duplication."
                );
                return Err(std::io::Error::new(
                        std::io::ErrorKind::WouldBlock, 
                        "Item is currently in progress by another worker"
                    ).into());
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

mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;
    use std::io;

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
    }

    /// An in-memory mock store to verify the node interacts with the store correctly.
    #[derive(Clone)]
    struct MockStore {
        // Shared state for the store mapping ID -> Status
        state: Arc<Mutex<HashMap<String, IdempotencyStatus>>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                state: Arc::new(Mutex::new(HashMap::new())),
            }
        }
        
        /// Helper to seed the store with a specific status before a test
        fn seed(&self, id: &str, status: IdempotencyStatus) {
            self.state.lock().unwrap().insert(id.to_string(), status);
        }
        
        /// Helper to verify the final status of an item
        fn get_status(&self, id: &str) -> Option<IdempotencyStatus> {
            self.state.lock().unwrap().get(id).copied()
        }
    }

    // You might need to adjust this depending on if your actual trait uses `#[async_trait]` 
    // or native async fns.
    impl IdempotencyStore for MockStore {
        type Error = io::Error;

        async fn check_and_lock(&self, id: &str) -> Result<IdempotencyStatus, Self::Error> {
            let mut state = self.state.lock().unwrap();
            let current = state.get(id).copied().unwrap_or(IdempotencyStatus::New);
            
            // If it's new, we lock it immediately
            if matches!(current, IdempotencyStatus::New) {
                state.insert(id.to_string(), IdempotencyStatus::InProgress);
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
            // In a real system, you might delete the key or mark it as New
            state.insert(id.to_string(), IdempotencyStatus::New);
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
        // Since it failed, the lock should have been released (set back to New)
        assert_eq!(store.get_status("item_2"), Some(IdempotencyStatus::New));
    }

    #[tokio::test]
    async fn test_completed_item_is_skipped() {
        let store = MockStore::new();
        store.seed("item_3", IdempotencyStatus::Completed);
        
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_3".into(), should_fail: false };

        let result = node.execute(&input).await.unwrap();

        // Should return empty vec, NOT "Processed item_3"
        assert!(result.is_empty());
        assert_eq!(store.get_status("item_3"), Some(IdempotencyStatus::Completed));
    }

    #[tokio::test]
    async fn test_in_progress_item_is_skipped() {
        let store = MockStore::new();
        store.seed("item_4", IdempotencyStatus::InProgress);
        
        let node = IdempotencyNode::new(MockNode, store.clone());
        let input = TestInput { id: "item_4".into(), should_fail: false };

        let result = node.execute(&input).await.unwrap();

        // Should drop the item (return empty vec)
        assert!(result.is_empty());
        // Should remain InProgress
        assert_eq!(store.get_status("item_4"), Some(IdempotencyStatus::InProgress));
    }
}