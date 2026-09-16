use std::sync::Mutex;
use std::time::Instant;
use crate::node::Node;
use std::time::Duration;
use tracing::{trace, warn};

/// Tracks the current state of the token bucket for rate limiting.
pub struct RateLimitState {
    last_updated: Instant,
    tokens: f64,
}

pub enum RateLimitStrategy {
    /// Instantly reject requests that exceed the limit.
    Reject,
    /// Pause the current Tokio task until a token regenerates.
    Sleep,
}

/// A node wrapper that enforces a rate limit using a Token Bucket algorithm.
///
/// Requests that exceed the configured `tokens_per_minute` are immediately rejected.
/// Tokens continuously regenerate over time based on the elapsed time between requests,
/// up to the maximum capacity (`tokens_per_minute`).
pub struct RateLimitNode<N> {
    pub node: N,
    pub strategy: RateLimitStrategy,
    pub tokens_per_minute: f64,
    pub state: Mutex<RateLimitState>,
}

impl<N> RateLimitNode<N> {
    /// Creates a new `RateLimitNode` with a fully charged token bucket.
    ///
    /// # Panics
    /// Panics if `tokens_per_minute` is less than or equal to 0.0.
    pub fn new(node: N, tokens_per_minute: f64, strategy: RateLimitStrategy) -> Self {
        assert!(
            tokens_per_minute > 0.0,
            "RateLimitNode requires tokens_per_minute to be greater than 0.0"
        );
        Self {
            node,
            tokens_per_minute,
            strategy,
            state: Mutex::new(RateLimitState {
                last_updated: Instant::now(),
                tokens: tokens_per_minute,
            }),
        }
    }
}

impl<N, I, O, E> Node for RateLimitNode<N> 
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
            let node_name = std::any::type_name::<N>();

            let wait_time = {
                let mut state = self.state.lock().unwrap(); // Using std::sync::Mutex
                
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_updated).as_secs_f64();
                let regeneration_rate = self.tokens_per_minute / 60.0;
                
                let new_tokens = state.tokens + (elapsed * regeneration_rate);
                state.tokens = new_tokens.min(self.tokens_per_minute);
                state.last_updated = now;

                metrics::gauge!("pipeline_node_rate_limit_tokens", "node" => node_name).set(state.tokens);
                if state.tokens >= 1.0 {
                    // We have a token! Consume it and return no wait time.
                    state.tokens -= 1.0;
                    None
                } else {
                    // We don't have enough tokens. 
                    match self.strategy {
                        RateLimitStrategy::Reject => {
                            // Special marker to indicate instant rejection
                            Some(Duration::MAX) 
                        }
                        RateLimitStrategy::Sleep => {
                            // Calculate EXACTLY how many seconds until 1 full token exists
                            let deficit = 1.0 - state.tokens;
                            let seconds_to_wait = deficit / regeneration_rate;
                            
                            // We must pre-consume the token we are about to wait for,
                            // so other concurrent threads don't steal it!
                            state.tokens -= 1.0; 
                            
                            Some(Duration::from_secs_f64(seconds_to_wait))
                        }
                    }
                }
            }; // Lock drops here

            if let Some(duration) = wait_time {
                if duration == Duration::MAX {
                    metrics::counter!("pipeline_node_rate_limit_rejected_total", "node" => node_name).increment(1);
                    warn!(limit = self.tokens_per_minute, "Rate limit exceeded, rejecting request");
                    return Err(std::io::Error::new(std::io::ErrorKind::Other, "Rate limit exceeded").into());
                } else {
                    metrics::counter!("pipeline_node_rate_limit_delayed_total", "node" => node_name).increment(1);
                    metrics::histogram!("pipeline_node_rate_limit_delay_duration_seconds", "node" => node_name)
                    .record(duration.as_secs_f64());
                    trace!(wait_ms = duration.as_millis(), "Rate limit reached, sleeping until token regenerates");
                    tokio::time::sleep(duration).await;
                }
            }
            
            self.node.execute(input).await
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
    use std::time::Duration;
    use tokio::time::Instant as TokioInstant;

    // --- Mock Node Setup ---
    
    #[derive(Clone)]
    struct MockNode {
        execution_count: Arc<AtomicUsize>,
    }

    impl MockNode {
        fn new() -> Self {
            Self {
                execution_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Node for MockNode {
        type Input = ();
        type Output = String;
        type Error = std::io::Error;

        async fn execute(&self, _input: &Self::Input) -> Result<Self::Output, Self::Error> {
            self.execution_count.fetch_add(1, Ordering::SeqCst);
            Ok("Success".to_string())
        }

        fn flush(&self) -> Result<Self::Output, Self::Error> {
            Ok("Flushed".to_string())
        }
    }

    // --- Original Tests (Using Reject Strategy) ---

    #[test]
    #[should_panic(expected = "requires tokens_per_minute to be greater than 0.0")]
    fn test_new_validates_tokens_per_minute() {
        let mock = MockNode::new();
        RateLimitNode::new(mock, 0.0, RateLimitStrategy::Reject); // Should panic
    }

    #[tokio::test]
    async fn test_initial_burst_allowed() {
        let mock = MockNode::new();
        let node = RateLimitNode::new(mock.clone(), 60.0, RateLimitStrategy::Reject);

        // We should be able to execute 60 times immediately
        for i in 0..60 {
            let res = node.execute(&()).await;
            assert!(res.is_ok(), "Failed on iteration {}", i);
        }
        
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 60);
    }

    #[tokio::test]
    async fn test_rate_limit_rejection() {
        let mock = MockNode::new();
        let node = RateLimitNode::new(mock, 5.0, RateLimitStrategy::Reject);

        // Drain the 5 initial tokens
        for _ in 0..5 {
            assert!(node.execute(&()).await.is_ok());
        }

        // The 6th request should fail immediately because no time has passed
        let res = node.execute(&()).await;
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Rate limit exceeded");
    }

    #[tokio::test]
    async fn test_tokens_regenerate_over_time() {
        let mock = MockNode::new();
        let node = RateLimitNode::new(mock.clone(), 60.0, RateLimitStrategy::Reject);

        for _ in 0..60 {
            assert!(node.execute(&()).await.is_ok());
        }

        assert!(node.execute(&()).await.is_err());

        // Wait slightly more than 1 second to let 1 token regenerate
        tokio::time::sleep(Duration::from_millis(1100)).await;

        assert!(node.execute(&()).await.is_ok());
        assert!(node.execute(&()).await.is_err());
    }

    #[tokio::test]
    async fn test_tokens_do_not_exceed_capacity() {
        let mock = MockNode::new();
        let node = RateLimitNode::new(mock.clone(), 600.0, RateLimitStrategy::Reject);

        for _ in 0..600 {
            assert!(node.execute(&()).await.is_ok());
        }
        
        tokio::time::sleep(Duration::from_millis(110)).await;
        
        {
            // USING .unwrap() INSTEAD OF .await
            let mut state = node.state.lock().unwrap();
            state.last_updated -= Duration::from_secs(3600);
        }

        assert!(node.execute(&()).await.is_ok());

        let state = node.state.lock().unwrap();
        assert!(state.tokens <= 600.0);
    }

    // --- New Tests (Using Sleep Strategy) ---

    #[tokio::test]
    async fn test_sleep_strategy_waits_for_tokens() {
        let mock = MockNode::new();
        // 600 tokens/min = 10 tokens/sec (100ms per token)
        let node = RateLimitNode::new(mock.clone(), 600.0, RateLimitStrategy::Sleep);

        // Drain the initial burst completely
        for _ in 0..600 {
            assert!(node.execute(&()).await.is_ok());
        }

        let start = TokioInstant::now();
        
        // This should NOT reject. It should sleep for ~100ms.
        let res = node.execute(&()).await;
        
        let elapsed = start.elapsed();

        assert!(res.is_ok());
        // Verify it actually waited for the token
        assert!(elapsed >= Duration::from_millis(90), "Node did not sleep long enough");
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 601);
    }

    #[tokio::test]
    async fn test_sleep_strategy_concurrent_staggering() {
        let mock = MockNode::new();
        // 600 tokens/min = 10 tokens/sec (100ms per token)
        let node = Arc::new(RateLimitNode::new(mock.clone(), 600.0, RateLimitStrategy::Sleep));

        for _ in 0..600 {
            let _ = node.execute(&()).await;
        }

        let mut handles = Vec::new();
        let start = TokioInstant::now();

        // Spawn 3 concurrent tasks on an empty bucket.
        // Because of our pre-consumption math, they should calculate perfectly 
        // staggered sleep durations: ~100ms, ~200ms, and ~300ms.
        for _ in 0..3 {
            let node_clone = node.clone();
            handles.push(tokio::spawn(async move {
                node_clone.execute(&()).await
            }));
        }

        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }
        
        let elapsed = start.elapsed();
        
        // Total time for all 3 tasks to finish should be ~300ms.
        // If they blocked each other, it would take much longer.
        // If they didn't stagger, they would all fire at 100ms and break the rate limit.
        assert!(elapsed >= Duration::from_millis(290));
        assert_eq!(mock.execution_count.load(Ordering::SeqCst), 603);
    }
}