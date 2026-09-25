use std::sync::Arc;
use tokio::sync::Semaphore;
use crate::node::Node;
use tracing::{error, trace};

/// A node wrapper that limits the number of concurrent executions.
///
/// This uses a Tokio `Semaphore` to manage execution permits, protecting downstream 
/// services or databases from being overwhelmed.
///
/// # ⚠️ Composition Guidelines
/// When composing this node with a `RetryNode`, **always place the `ConcurrencyLimitNode` 
/// INSIDE the `RetryNode`**.
///
/// - **Good:** `RetryNode::new(ConcurrencyLimitNode::new(Node))`
/// - **Bad:** `ConcurrencyLimitNode::new(RetryNode::new(Node))`
///
/// If placed on the outside, a failing request will hold the concurrency permit hostage 
/// while it sleeps during its retry backoff. If enough requests fail, the entire 
/// pipeline will deadlock until the sleep timers expire.
pub struct ConcurrencyLimitNode<N> {
    /// The underlying node to execute once a permit is acquired.
    pub node: N,
    /// The semaphore used to track and limit concurrent requests.
    pub no_of_requests: Arc<Semaphore>,
}

impl<N> ConcurrencyLimitNode<N> {
    /// Creates a new `ConcurrencyLimitNode`.
    ///
    /// # Panics
    /// Panics if `limit` is 0, as this would permanently block all requests from executing.
    pub fn new(node: N, limit: usize) -> Self {
        assert!(
            limit > 0,
            "ConcurrencyLimitNode requires a limit greater than 0"
        );

        Self {
            node,
            no_of_requests: Arc::new(Semaphore::new(limit)),
        }
    }
}

impl<N, I, O, E> Node for ConcurrencyLimitNode<N> 
where 
    N: Node<Input = I, Output = O, Error = E> + Send + Sync,
    I: Sync,
    O: Default + Send,
    E: Send + From<std::io::Error>,
{
    type Input = I;
    type Output = O;
    type Error = E;

    fn name(&self) -> &'static str {
        "ConcurrencyLimitNode"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let node_name = self.name();
        // Trace-level logging helps diagnose pipeline bottlenecks locally
        // without flooding production I/O.
        trace!(
            available_permits = self.no_of_requests.available_permits(),
            "Waiting to acquire concurrency permit"
        );

        let wait_start = std::time::Instant::now();
        // Wait to acquire a permit before allowing execution.
        // The _permit is automatically dropped (released) when this function exits,
        // whether the inner node succeeds or returns an error.
        let _permit = self.no_of_requests.acquire().await.map_err(|e| {
            error!(
                error = %e, 
                "Concurrency semaphore closed unexpectedly"
            );
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe, 
                "Concurrency semaphore closed"
            )
        })?;

        metrics::histogram!("pipeline_node_concurrency_wait_duration_seconds", "node" => node_name)
            .record(wait_start.elapsed().as_secs_f64());
        // METRIC: Increment the active request gauge
        metrics::gauge!("pipeline_node_concurrency_active_requests", "node" => node_name).increment(1.0);

        let result = self.node.execute(input).await;

        // METRIC: Decrement the active request gauge when execution finishes
        metrics::gauge!("pipeline_node_concurrency_active_requests", "node" => node_name).decrement(1.0);
        
        result
    }
    
    fn flush(&self) -> Result<Self::Output, Self::Error> {
        self.node.flush()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // --- Mock Node Setup ---
    
    #[derive(Clone)]
    struct MockNode {
        // Tracks currently active executions
        active_requests: Arc<AtomicUsize>,
        // Tracks the maximum number of concurrent executions observed
        max_concurrent_requests: Arc<AtomicUsize>,
        // How long the mock execution should block to simulate work
        sleep_duration: Duration,
        // Configurable flag to simulate an error
        should_fail: bool,
    }

    impl MockNode {
        fn new(sleep_duration: Duration, should_fail: bool) -> Self {
            Self {
                active_requests: Arc::new(AtomicUsize::new(0)),
                max_concurrent_requests: Arc::new(AtomicUsize::new(0)),
                sleep_duration,
                should_fail,
            }
        }
    }

    // Implementing your crate's Node trait for the mock
    impl Node for MockNode {
        type Input = ();
        type Output = String;
        type Error = std::io::Error;

        async fn execute(&self, _input: &Self::Input) -> Result<Self::Output, Self::Error> {
            // Increment active requests
            let current = self.active_requests.fetch_add(1, Ordering::SeqCst) + 1;
            
            // Update the maximum concurrent counter safely
            let mut max = self.max_concurrent_requests.load(Ordering::SeqCst);
            while current > max {
                match self.max_concurrent_requests.compare_exchange_weak(
                    max, current, Ordering::SeqCst, Ordering::SeqCst
                ) {
                    Ok(_) => break,
                    Err(actual) => max = actual,
                }
            }

            // Simulate I/O or processing time
            tokio::time::sleep(self.sleep_duration).await;

            // Decrement active requests on exit
            self.active_requests.fetch_sub(1, Ordering::SeqCst);

            if self.should_fail {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "Mock error"))
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
    #[should_panic(expected = "requires a limit greater than 0")]
    fn test_new_validates_limit() {
        let mock = MockNode::new(Duration::from_millis(1), false);
        ConcurrencyLimitNode::new(mock, 0);
    }

    #[tokio::test]
    async fn test_successful_execution() {
        let mock = MockNode::new(Duration::from_millis(5), false);
        let limited_node = ConcurrencyLimitNode::new(mock, 2);

        let res = limited_node.execute(&()).await;
        
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "Success");
    }

    #[tokio::test]
    async fn test_error_propagation() {
        let mock = MockNode::new(Duration::from_millis(5), true);
        let limited_node = ConcurrencyLimitNode::new(mock, 2);

        let res = limited_node.execute(&()).await;
        
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Mock error");
    }

    #[tokio::test]
    async fn test_concurrency_limit_is_enforced() {
        // 100ms simulated work, so requests queue up
        let mock = MockNode::new(Duration::from_millis(100), false);
        
        // We only allow 2 concurrent executions
        let max_allowed_concurrency = 2;
        let limited_node = Arc::new(ConcurrencyLimitNode::new(mock.clone(), max_allowed_concurrency));

        let mut handles = Vec::new();

        // Spawn 10 simultaneous tasks
        for _ in 0..10 {
            let node_clone = limited_node.clone();
            handles.push(tokio::spawn(async move {
                node_clone.execute(&()).await
            }));
        }

        // Wait for all 10 tasks to complete
        for handle in handles {
            let res = handle.await.unwrap();
            assert!(res.is_ok());
        }

        // Verify that at no point were there more than `max_allowed_concurrency` tasks running inside the mock
        let max_observed = mock.max_concurrent_requests.load(Ordering::SeqCst);
        assert_eq!(max_observed, max_allowed_concurrency);
    }

    #[tokio::test]
    async fn test_semaphore_closure_yields_broken_pipe() {
        let mock = MockNode::new(Duration::from_millis(5), false);
        let limited_node = ConcurrencyLimitNode::new(mock, 1);

        // Manually close the semaphore to simulate shutdown/cancellation
        limited_node.no_of_requests.close();

        let res = limited_node.execute(&()).await;
        
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
        assert_eq!(err.to_string(), "Concurrency semaphore closed");
    }
}