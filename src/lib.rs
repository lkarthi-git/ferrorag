//! # Concurrent Data Processing Pipeline
//! 
//! A highly resilient, metric-rich, concurrent DAG framework for Rust.
//!
//! ## Best Practices & Architectural Notes
//! 
//! 1. **Errors are routed, not dropped:** The pipeline catches all node errors and routes 
//!    them to a background Dead Letter Queue (DLQ). Terminal methods like `.execute()` and `.collect()` 
//!    return an `ExecutionSummary` containing the exact count of successful items and DLQ errors, 
//!    allowing you to easily programmatically verify if a batch job was 100% successful.
//! 2. **Middleware Ordering Matters:** Middlewares wrap each other. Always wrap `RetryNode` 
//!    *around* `ConcurrencyLimitNode` so that sleeping retries do not hold concurrency permits hostage.
//! 3. **Avoid Deep Cloning:** `ChunkNode` requires `T: Clone`. To avoid massive memory allocation 
//!    overheads, wrap large data structs in `std::sync::Arc<T>` or `bytes::Bytes` before 
//!    pushing them into the pipeline.

pub mod pipeline;
pub mod graph;
pub mod source;
mod dlq;
pub mod node;
pub mod cancel;
pub mod idempotency;

