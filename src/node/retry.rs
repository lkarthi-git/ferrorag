use std::time::Duration;
use crate::node::Node;
use std::fmt::Display;
use tracing::{info, warn,error};

/// A trait for classifying whether an error is transient (temporary) 
/// and should be retried, or fatal and should fail immediately.
pub trait ClassifyRetry<E> {
    fn is_transient(&self, error: &E) -> bool;
}

// Blanket implementation allowing closures/functions to be used as retry policies.
impl<E, F> ClassifyRetry<E> for F 
where
    F: Fn(&E) -> bool,
{
    fn is_transient(&self, error: &E) -> bool {
        (self)(error)
    }
}

/// A node wrapper that automatically retries the inner node upon failure.
///
/// It uses a provided `policy` to determine if an error is transient.
/// If transient, it backs off exponentially with jitter before attempting again,
/// up to the configured `retry_count`.
pub struct RetryNode<N, C> {
    pub node: N,
    pub retry_count: usize,
    pub policy: C, 
}

impl<N, C> RetryNode<N, C> {
    /// Creates a new `RetryNode`.
    ///
    /// # Panics
    /// Panics if `retry_count` is 0, as a retry node must be able to perform 
    /// at least one retry to be useful.
    pub fn new(node: N, retry_count: usize, policy: C) -> Self {
        assert!(
            retry_count > 0,
            "RetryNode requires a retry_count greater than 0"
        );
        Self {
            node,
            retry_count,
            policy,
        }
    }
}

impl<N, C, I, O, E> Node for RetryNode<N, C> 
where 
    N: Node<Input = I, Output = O, Error = E> + Send + Sync,
    I: Sync,
    O: Default + Send,
    C: ClassifyRetry<E> + Send + Sync,
    E: Send + Display,
{
    type Input = I;
    type Output = O;
    type Error = E;
    
    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
            let node_name = std::any::type_name::<N>();
            let mut remaining_retries = self.retry_count;
            loop {
                // Calculate this before the match so it represents the CURRENT attempt
                let current_attempt = self.retry_count - remaining_retries + 1;
                
                match self.node.execute(input).await {
                    Ok(output) => {
                        if current_attempt > 1 {
                            metrics::counter!("pipeline_node_retry_success_total", "node" => node_name).increment(1);
                            info!(
                                total_attempts = current_attempt,
                                "Operation succeeded after retries"
                            );
                        } 
                        return Ok(output); 
                    }
                    Err(error) => {
                        if remaining_retries > 0 && self.policy.is_transient(&error) {
                            remaining_retries -= 1;
                            
                            metrics::counter!("pipeline_node_retries_total", "node" => node_name).increment(1);
                            // We use `current_attempt` for the backoff math so the first retry
                            // sleeps up to 2^1 (2s), the second up to 2^2 (4s), etc.
                            let base_delay = 2_u64.saturating_pow(current_attempt as u32).min(120);
                            let jittered_delay = rand::random_range(0..=base_delay);

                            warn!(
                                    error = %error,
                                    attempt = current_attempt,
                                    next_retry_in_secs = jittered_delay,
                                    retries_left = remaining_retries,
                                    "Transient error encountered, backing off and retrying"
                            );
                            
                            tokio::time::sleep(Duration::from_secs(jittered_delay)).await;
                        } else {
                            metrics::counter!("pipeline_node_retry_exhausted_total", "node" => node_name).increment(1);
                            error!(
                                    error = %error,
                                    total_attempts = current_attempt,
                                    "Retries exhausted (or fatal error), failing operation"
                            );
                            return Err(error);
                        }
                    }
                }
            }
        }
    
    fn flush(&self) -> Result<Self::Output, Self::Error> {
        self.node.flush()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // --- Mock Node Setup ---
    
    #[derive(Clone)]
    struct MockNode {
        execution_count: Arc<AtomicUsize>,
        // Number of times it should fail before succeeding
        fails_until: usize, 
        // The type of error it yields when failing
        error_kind: std::io::ErrorKind, 
    }

    impl MockNode {
        fn new(fails_until: usize, error_kind: std::io::ErrorKind) -> Self {
            Self {
                execution_count: Arc::new(AtomicUsize::new(0)),
                fails_until,
                error_kind,
            }
        }
    }

    // Implementing your crate's Node trait for the mock
    impl Node for MockNode {
        type Input = ();
        type Output = String;
        type Error = std::io::Error;

        async fn execute(&self, _input: &Self::Input) -> Result<Self::Output, Self::Error> {
            let current = self.execution_count.fetch_add(1, Ordering::SeqCst);
            
            if current < self.fails_until {
                Err(std::io::Error::new(self.error_kind, "Mock failure"))
            } else {
                Ok("Success".to_string())
            }
        }

        fn flush(&self) -> Result<Self::Output, Self::Error> {
            Ok("Flushed".to_string())
        }
    }

    // --- Tests ---

    #[test]
    #[should_panic(expected = "requires a retry_count greater than 0")]
    fn test_new_validates_retry_count() {
        let mock = MockNode::new(0, std::io::ErrorKind::Other);
        // Using a simple closure as the policy
        let policy = |_: &std::io::Error| true;
        RetryNode::new(mock, 0, policy); // Should panic
    }

    #[tokio::test]
    async fn test_successful_execution_first_try() {
        // Node succeeds immediately
        let mock = MockNode::new(0, std::io::ErrorKind::TimedOut);
        let policy = |_: &std::io::Error| true;
        let retry_node = RetryNode::new(mock.clone(), 3, policy);

        let res = retry_node.execute(&()).await;
        
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "Success");
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_transient_error_retries_and_succeeds() {
        // Pause time so the sleep(jittered_delay) completes instantly in tests
        tokio::time::pause();

        // Node fails 2 times with TimedOut (transient), succeeds on 3rd try
        let mock = MockNode::new(2, std::io::ErrorKind::TimedOut);
        
        // Policy: Only TimedOut is transient
        let policy = |e: &std::io::Error| e.kind() == std::io::ErrorKind::TimedOut;
        
        // Allow up to 3 retries (4 total attempts)
        let retry_node = RetryNode::new(mock.clone(), 3, policy);

        let res = retry_node.execute(&()).await;
        
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "Success");
        // Execution count should be 3 (2 failures + 1 success)
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_transient_error_exhausts_retries() {
        tokio::time::pause();

        // Node fails 5 times, but we only allow 2 retries
        let mock: MockNode = MockNode::new(5, std::io::ErrorKind::TimedOut);
        let policy = |e: &std::io::Error| e.kind() == std::io::ErrorKind::TimedOut;
        let retry_node = RetryNode::new(mock.clone(), 2, policy);

        let res = retry_node.execute(&()).await;
        
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::TimedOut);
        
        // Initial attempt (1) + retries (2) = 3 total executions
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_fatal_error_fails_immediately() {
        tokio::time::pause();

        // Node fails with PermissionDenied
        let mock = MockNode::new(3, std::io::ErrorKind::PermissionDenied);
        
        // Policy: Only TimedOut is transient. PermissionDenied is FATAL.
        let policy = |e: &std::io::Error| e.kind() == std::io::ErrorKind::TimedOut;
        let retry_node = RetryNode::new(mock.clone(), 3, policy);

        let res = retry_node.execute(&()).await;
        
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::PermissionDenied);
        
        // Should only execute exactly once. No retries should happen because the policy returned false.
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 1);
    }
}