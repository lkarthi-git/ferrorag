use std::sync::Mutex;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::node::Node;
use std::fmt::Display;
use tracing::{info, warn, error, trace, debug};

/// Represents the internal state of the Circuit Breaker.
#[derive(PartialEq, Debug)]
enum BreakerState {
    /// The circuit is operating normally. Requests are passed to the inner node.
    Closed { failures: usize },
    /// The circuit has tripped. Requests fail immediately without hitting the inner node.
    Open { until: Instant },
    /// The circuit is testing if the inner node has recovered. 
    /// Only a single probe request is allowed through.
    HalfOpen { probe_id: u64 },
}

/// A node wrapper that implements the Circuit Breaker pattern to protect downstream services.
/// 
/// If the inner node fails consecutively beyond `allowed_failures`, the breaker opens,
/// fast-failing subsequent requests for the duration of `reset_after`. After this duration,
/// it transitions to a half-open state to probe the service with a single request.
pub struct CircuitBreakerNode<N> {
    pub node: N,
    state: Mutex<BreakerState>,
    probe_id: AtomicU64,
    pub allowed_failures: usize,
    pub reset_after: Duration,
}

impl<N> CircuitBreakerNode<N> {
    /// Creates a new `CircuitBreakerNode`.
    ///
    /// # Panics
    /// Panics if `allowed_failures` is set to 0. A circuit breaker must allow 
    /// at least one attempt before it can track failures and open.
    pub fn new(node: N, allowed_failures: usize, reset_after: Duration) -> Self {
        assert!(
            allowed_failures > 0,
            "CircuitBreakerNode requires allowed_failures to be greater than 0"
        );
        
        Self {
            node,
            state: Mutex::new(BreakerState::Closed { failures: 0 }),
            probe_id: AtomicU64::new(0),
            allowed_failures,
            reset_after,
        }
    }
}

impl<N, I, O, E> Node for CircuitBreakerNode<N>
where 
    N: Node<Input = I, Output = O, Error = E> + Send + Sync,
    I: Sync,
    O: Default + Send,
    E: Display + Send + From<std::io::Error>,
{
    type Input = I;
    type Output = O;
    type Error = E;

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
       let node_name = std::any::type_name::<N>();
       let probe_id = {
            // std::sync::Mutex doesn't need .await
            let mut state = self.state.lock().unwrap(); 
            match *state {
                BreakerState::Closed { .. } => None,
                BreakerState::Open { until } => {
                    if Instant::now() < until {
                        metrics::counter!("pipeline_node_circuit_breaker_rejected_total", "node" => node_name).increment(1);
                        trace!("Circuit breaker is OPEN. Fast-failing request.");
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other, 
                            "Circuit breaker tripped"
                        ).into());
                    } 
                    let id = self.probe_id.fetch_add(1, Ordering::Relaxed);
                    *state = BreakerState::HalfOpen { probe_id: id };
                    metrics::gauge!("pipeline_node_circuit_breaker_state", "node" => node_name).set(1.0);
                    info!(
                        probe_id = id,
                        "Circuit breaker reset timeout reached. Transitioning to HALF-OPEN state and sending probe."
                    );
                    Some(id)
                }
                BreakerState::HalfOpen { .. } => {
                    metrics::counter!("pipeline_node_circuit_breaker_rejected_total", "node" => node_name).increment(1);
                    trace!("Circuit breaker is HALF-OPEN. Fast-failing non-probe request.");
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other, 
                        "Circuit breaker is half-open"
                    ).into());
                }
            }
        }; // Lock is dropped here!

        let result: Result<O, E> = self.node.execute(input).await;

        {
            let mut state = self.state.lock().unwrap();
            match &result {
                Ok(_)  => {
                    if let BreakerState::HalfOpen { probe_id: current_probe } = *state {
                        if Some(current_probe) == probe_id {
                            info!("Circuit breaker probe SUCCEEDED. Transitioning back to CLOSED state.");
                            *state = BreakerState::Closed { failures: 0 };
                            metrics::gauge!("pipeline_node_circuit_breaker_state", "node" => node_name).set(0.0);
                        }
                    } else if let BreakerState::Closed { ref mut failures } = *state {
                        debug!("Request succeeded. Resetting consecutive failure count to 0.");
                        *failures = 0; 
                    }
                }
                Err(error) => {
                    match *state {
                        BreakerState::Closed { ref mut failures } => {
                            *failures += 1;
                            if *failures >= self.allowed_failures {
                                error!(
                                    error = %error,
                                    failures = *failures,
                                    allowed_failures = self.allowed_failures,
                                    "Circuit breaker threshold reached! Transitioning to OPEN state."
                                );
                                *state = BreakerState::Open { 
                                    until: Instant::now() + self.reset_after 
                                };
                                metrics::counter!("pipeline_node_circuit_breaker_tripped_total", "node" => node_name).increment(1);
                                metrics::gauge!("pipeline_node_circuit_breaker_state", "node" => node_name).set(2.0);
                            }
                            else{
                                warn!(
                                    error = %error,
                                    failures = *failures,
                                    allowed_failures = self.allowed_failures,
                                    "Inner node failed. Tracking consecutive failure."
                                );
                            }
                        }
                        BreakerState::HalfOpen { probe_id: current_probe } => {
                            // CRITICAL FIX: Only let the actual probe transition us back to Open!
                            if Some(current_probe) == probe_id {
                                warn!(
                                    error = %error,
                                    "Circuit breaker probe FAILED. Transitioning back to OPEN state."
                                );
                                *state = BreakerState::Open { 
                                    until: Instant::now() + self.reset_after 
                                };
                                metrics::counter!("pipeline_node_circuit_breaker_tripped_total", "node" => node_name).increment(1);
                                metrics::gauge!("pipeline_node_circuit_breaker_state", "node" => node_name).set(2.0);
                            } else {
                                trace!("Lingering request failed while in HalfOpen state. Ignoring.");
                            }
                        }
                        BreakerState::Open { .. } => {
                            // Lingering request failed after circuit was already open. Safe to ignore.
                            trace!("Lingering request failed while in Open state. Ignoring.");
                        }
                    }
                }
            }
        }

        result
    }

    fn flush(&self) -> Result<Self::Output, Self::Error> {
        self.node.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*; // This brings CircuitBreakerNode and your real `crate::node::Node` into scope
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone)]
    struct MockNode {
        should_fail: Arc<AtomicBool>,
    }

    impl MockNode {
        fn new(should_fail: bool) -> Self {
            Self {
                should_fail: Arc::new(AtomicBool::new(should_fail)),
            }
        }
        
        fn set_fail(&self, fail: bool) {
            self.should_fail.store(fail, Ordering::SeqCst);
        }
    }

    // Implement your actual crate::node::Node trait for the mock
    impl Node for MockNode {
        type Input = ();
        type Output = String;
        type Error = std::io::Error;

        async fn execute(&self, _input: &Self::Input) -> Result<Self::Output, Self::Error> {
            if self.should_fail.load(Ordering::SeqCst) {
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
    #[should_panic(expected = "allowed_failures to be greater than 0")]
    fn test_new_validates_failures() {
        let mock = MockNode::new(false);
        CircuitBreakerNode::new(mock, 0, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn test_successful_requests_stay_closed() {
        let mock = MockNode::new(false);
        let breaker = CircuitBreakerNode::new(mock, 2, Duration::from_secs(1));

        for _ in 0..5 {
            let res = breaker.execute(&()).await;
            assert!(res.is_ok());
        }
        
        let state = breaker.state.lock().unwrap();
        assert_eq!(*state, BreakerState::Closed { failures: 0 });
    }

    #[tokio::test]
    async fn test_breaker_trips_open() {
        let mock = MockNode::new(true);
        let breaker = CircuitBreakerNode::new(mock, 2, Duration::from_secs(10));

        // Attempt 1: Fails, but breaker remains closed (failures = 1)
        let res1 = breaker.execute(&()).await;
        assert_eq!(res1.unwrap_err().to_string(), "Mock error");

        // Attempt 2: Fails, breaker trips Open (failures = 2)
        let res2 = breaker.execute(&()).await;
        assert_eq!(res2.unwrap_err().to_string(), "Mock error");

        // Attempt 3: Fast fails due to open breaker
        let res3 = breaker.execute(&()).await;
        assert_eq!(res3.unwrap_err().to_string(), "Circuit breaker tripped");
    }

    #[tokio::test]
    async fn test_half_open_success_resets_breaker() {
        let mock = MockNode::new(true);
        let breaker = CircuitBreakerNode::new(mock.clone(), 1, Duration::from_millis(10));

        // Trip the breaker
        let _ = breaker.execute(&()).await;

        // Wait for reset duration to pass
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Service recovers
        mock.set_fail(false);

        // Probe request (Half-Open) -> should succeed and reset to Closed
        let res = breaker.execute(&()).await;
        assert!(res.is_ok());

        // Verify state is Closed
        let state = breaker.state.lock().unwrap();
        assert_eq!(*state, BreakerState::Closed { failures: 0 });
    }

    #[tokio::test]
    async fn test_half_open_failure_reopens_breaker() {
        let mock = MockNode::new(true);
        let breaker = CircuitBreakerNode::new(mock, 1, Duration::from_millis(10));

        // Trip the breaker
        let _ = breaker.execute(&()).await;

        // Wait for reset duration to pass
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Probe request (Half-Open) -> fails again -> goes back to Open
        let res1 = breaker.execute(&()).await;
        assert_eq!(res1.unwrap_err().to_string(), "Mock error");

        // Subsequent request immediately fast-fails
        let res2 = breaker.execute(&()).await;
        assert_eq!(res2.unwrap_err().to_string(), "Circuit breaker tripped");
    }

    #[tokio::test]
    async fn test_concurrent_half_open_rejection() {
        let mock = MockNode::new(true);
        let breaker = Arc::new(CircuitBreakerNode::new(mock, 1, Duration::from_millis(10)));

        // Trip the breaker
        let _ = breaker.execute(&()).await;
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Manually set the state to HalfOpen to test rejection of secondary requests.
        {
            let mut state = breaker.state.lock().unwrap();
            *state = BreakerState::HalfOpen { probe_id: 1 };
        }

        let res = breaker.execute(&()).await;
        assert_eq!(res.unwrap_err().to_string(), "Circuit breaker is half-open");
    }
}