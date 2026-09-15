use crate::node::Node;
use std::time::Duration;
use tracing::{warn};

/// A node wrapper that enforces a strict time limit on the execution of the inner node.
///
/// If the inner node takes longer than the configured `timeout` duration to complete,
/// the future is canceled (aborted) and a `TimedOut` error is immediately returned.
///
/// # ⚠️ Cancellation Safety Warning
/// Because this node relies on dropping the underlying future when the timeout elapses, 
/// the inner node **must be cancellation safe**. If the inner node performs partial I/O 
/// operations, state mutations across `.await` points, or relies on executing cleanup 
/// code after an `.await`, dropping it may leave the system in an inconsistent state.
pub struct TimeoutNode<N> {
    pub node: N,
    pub timeout: Duration,
}

impl<N> TimeoutNode<N> {
    /// Creates a new `TimeoutNode`.
    ///
    /// # Panics
    /// Panics if the `timeout` is zero. A zero-duration timeout would cause every 
    /// single request to instantly fail without ever giving the inner node a chance to run.
    pub fn new(node: N, timeout: Duration) -> Self {
        assert!(
            !timeout.is_zero(),
            "TimeoutNode requires a timeout duration greater than zero"
        );
        Self { node, timeout }
    }
}

impl<N, I, O, E> Node for TimeoutNode<N> 
where 
    N: Node<Input = I, Output = O, Error = E> + Send + Sync,
    I: Sync,
    O: Default + Send,
    E: Send + From<std::io::Error>,
{
    type Input = I;
    type Output = O;
    type Error = E;

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        // tokio::time::timeout wraps the future and cancels it if the duration elapses.
        match tokio::time::timeout(self.timeout, self.node.execute(input)).await {
            Ok(inner_result) => inner_result,
            Err(_) => {
                // Emitted on WARN because the pipeline had to actively step in 
                // and abort the execution to protect system resources.
                warn!(
                    timeout_duration_ms = self.timeout.as_millis(),
                    "Node execution timed out and was forcefully aborted"
                );
                
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut, 
                    "Node execution exceeded the configured timeout"
                ).into())
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // --- Mock Node Setup ---
    
    #[derive(Clone)]
    struct MockNode {
        // How long the node takes to execute
        execution_time: Duration,
        // Flag to verify if the node actually finished its work
        finished: Arc<AtomicBool>,
    }

    impl MockNode {
        fn new(execution_time: Duration) -> Self {
            Self {
                execution_time,
                finished: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    // Implementing your crate's Node trait for the mock
    impl Node for MockNode {
        type Input = ();
        type Output = String;
        type Error = std::io::Error;

        async fn execute(&self, _input: &Self::Input) -> Result<Self::Output, Self::Error> {
            // Reset the finished flag on each execution
            self.finished.store(false, Ordering::SeqCst);
            
            // Simulate work that takes time
            tokio::time::sleep(self.execution_time).await;
            
            // If the future was cancelled by a timeout, it will never reach this line
            self.finished.store(true, Ordering::SeqCst);
            
            Ok("Success".to_string())
        }

        fn flush(&self) -> Result<Self::Output, Self::Error> {
            Ok("Flushed".to_string())
        }
    }

    // --- Tests ---

    #[test]
    #[should_panic(expected = "requires a timeout duration greater than zero")]
    fn test_new_validates_timeout() {
        let mock = MockNode::new(Duration::from_millis(10));
        TimeoutNode::new(mock, Duration::ZERO); // Should panic
    }

    #[tokio::test]
    async fn test_successful_execution_within_timeout() {
        // Automatically fast-forward Tokio time to keep tests instant
        tokio::time::pause();

        // Node takes 5ms to run, timeout is 50ms (plenty of time)
        let mock = MockNode::new(Duration::from_millis(5));
        let timeout_node = TimeoutNode::new(mock.clone(), Duration::from_millis(50));

        let res = timeout_node.execute(&()).await;
        
        // Assert the execution was successful and didn't time out
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "Success");
        
        // Ensure the inner node actually completed its execution
        assert!(mock.finished.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_execution_aborted_on_timeout() {
        tokio::time::pause();

        // Node takes 100ms to run, timeout is only 10ms (will fail)
        let mock = MockNode::new(Duration::from_millis(100));
        let timeout_node = TimeoutNode::new(mock.clone(), Duration::from_millis(10));

        let res = timeout_node.execute(&()).await;
        
        // Assert the execution failed with our expected error
        assert!(res.is_err());
        
        let err = res.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
        assert_eq!(err.to_string(), "Node execution exceeded the configured timeout");
        
        // CRITICAL: Ensure the inner future was actually canceled and stopped running.
        // It should never have reached the `finished.store(true)` line.
        assert!(!mock.finished.load(Ordering::SeqCst));
    }
}