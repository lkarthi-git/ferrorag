mod retry;
mod timeout;
mod concurrency;
mod circuit_breaker;
mod rate_limit;
mod chunk;
mod idempotency;
mod ordering;

// 2. Re-export the structs so they are available at the `node::` level
pub use retry::{RetryNode, ClassifyRetry};
pub use timeout::TimeoutNode;
pub use concurrency::ConcurrencyLimitNode;
pub use circuit_breaker::CircuitBreakerNode;
pub use rate_limit::RateLimitNode;
pub use chunk::ChunkNode;
pub use idempotency::IdempotencyNode;
pub use ordering::OrderingNode;
// 3. Define the core traits here (since everything relies on them)
use std::future::Future;

pub trait Node {
    type Input;
    type Output: Default;
    type Error;
    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
    fn flush(&self) -> Result<Self::Output,Self::Error> {
        Result::Ok(Self::Output::default())
    }
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .split("::")
            .last()
            .unwrap_or("UnknownNode")
    }
}