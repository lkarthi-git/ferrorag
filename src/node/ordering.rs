use std::collections::BTreeMap;
use std::sync::Mutex;
use crate::node::Node;
use std::fmt::Display;
use tracing::{debug, trace, warn};


/// Trait to extract a strictly monotonically increasing sequence ID from an input.
/// This is required for the `OrderingNode` to reconstruct the original order.
pub trait Sequenced {
    fn get_sequence_id(&self) -> u64;
}

/// Internal state for the ordering buffer.
struct OrderState<O> {
    /// The next sequence ID we are permitted to emit downstream.
    next_expected_id: u64,
    /// A buffer holding items that completed faster than their predecessors.
    buffer: BTreeMap<u64, Vec<O>>,
}

/// A middleware node that restores sequential ordering in a concurrent pipeline.
///
/// When data is processed concurrently (e.g., via `pipe_concurrently`), the output 
/// order is typically scrambled. This node buffers out-of-order results and emits 
/// them strictly in the order of their sequence IDs.
///
/// # ⚠️ Critical Warning: Gapless Sequence Requirement
/// This node fundamentally relies on strictly contiguous sequence IDs (`next_expected_id += 1`).
/// **You must not place filtering nodes (e.g., a validator that drops items) upstream 
/// of this node.** If an item is dropped before reaching this node, the sequence 
/// will break, and this node will buffer all subsequent items indefinitely until 
/// the application crashes with an Out-Of-Memory (OOM) error.
/// 
/// # Latency Note on Errors
/// If an inner node execution fails, a tombstone is inserted to prevent deadlocks. 
/// However, due to Rust's `Result` signature, already-buffered successful items cannot 
/// be yielded simultaneously with the error. They will be released milliseconds later 
/// when the *next* successful item passes through. This causes no data loss.
pub struct OrderingNode<N, O> {
    pub inner_node: N,
    state: Mutex<OrderState<O>>,
}

impl<N, O> OrderingNode<N, O> {
    /// Creates a new `OrderingNode`.
    /// 
    /// `start_id` should be the sequence ID of the very first item expected to 
    /// flow through the pipeline (usually 0 or 1).
    pub fn new(inner_node: N, start_id: u64) -> Self {
        Self {
            inner_node,
            state: Mutex::new(OrderState {
                next_expected_id: start_id,
                buffer: BTreeMap::new(),
            }),
        }
    }
}

impl<N, I, O, E> Node for OrderingNode<N, O>
where
    N: Node<Input = I, Output = Vec<O>, Error = E> + Send + Sync,
    I: Sequenced + Send + Sync + 'static,
    O: Send + Sync + 'static,
    E: Send + Display + Sync + 'static,
{
    type Input = I;
    type Output = Vec<O>;
    type Error = E;

    fn name(&self) -> &'static str {
        "OrderingNode"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let node_name = self.name();
        let seq_id = input.get_sequence_id();
        trace!(seq_id = seq_id, "Executing ordered item");

        match self.inner_node.execute(input).await {
            Ok(output) => {
                let mut state = self.state.lock().unwrap();
                
                // Buffer the current result
                state.buffer.entry(seq_id).or_default().extend(output);

                let mut results_to_emit = Vec::new();
                let mut ids_emitted = 0;

                // Drain the buffer of any contiguous ready items
                loop {
                    let current_id = state.next_expected_id;
                    if let Some(ready_output) = state.buffer.remove(&current_id) {
                        results_to_emit.extend(ready_output);
                        state.next_expected_id += 1;
                        ids_emitted += 1;
                    } else {
                        break; // The next expected ID is not ready yet
                    }
                }

                metrics::gauge!("pipeline_node_ordering_buffer_depth", "node" => node_name)
                    .set(state.buffer.len() as f64);

                if ids_emitted == 0 {
                    trace!(
                        seq_id = seq_id,
                        waiting_for = state.next_expected_id,
                        "Item buffered (out of order)"
                    );
                } else {
                    metrics::counter!("pipeline_node_ordering_released_total", "node" => node_name)
                        .increment(ids_emitted as u64);
                    debug!(
                        seq_id = seq_id,
                        items_released = ids_emitted,
                        next_expected_id = state.next_expected_id,
                        "Advanced sequence"
                    );
                }

                Ok(results_to_emit)
            }
            Err(e) => {
                let mut state = self.state.lock().unwrap();
                
                metrics::counter!("pipeline_node_ordering_tombstones_total", "node" => node_name).increment(1);

                warn!(
                    seq_id = seq_id,
                    "Item failed. Inserting tombstone to prevent sequence deadlock."
                );
                
                // CRITICAL FIX: Insert an empty vector as a tombstone. 
                // This guarantees that when `next_expected_id` reaches this failed item, 
                // it will simply pop the empty vector, increment, and move on to the next items,
                // rather than deadlocking the pipeline waiting for a failed item.
                state.buffer.insert(seq_id, Vec::new());

                metrics::gauge!("pipeline_node_ordering_buffer_depth", "node" => node_name).set(state.buffer.len() as f64);
                
                Err(e)
            }
        }
    }

    fn flush(&self) -> Result<Self::Output, Self::Error> {
        let mut state = self.state.lock().unwrap();
        let mut results_to_emit = Vec::new();

        let remaining = state.buffer.len();
        if remaining > 0 {
            warn!(
                remaining_items = remaining,
                expected_id = state.next_expected_id,
                "Flushing ordering node with incomplete sequences. Force-emitting remaining buffer."
            );
        }

        match self.inner_node.flush() {
            Ok(inner_output) => {
                results_to_emit.extend(inner_output);
            }
            Err(e) => {
                warn!(error = %e, "Inner node flush failed, but preserving already buffered items");
            }
        }

        // Force-drain everything remaining in the buffer, ignoring sequence gaps
        while let Some((_, ready_output)) = state.buffer.pop_first() {
            results_to_emit.extend(ready_output);
        }

        Ok(results_to_emit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    // --- Mock Setup ---

    struct TestInput {
        id: u64,
        val: String,
        should_fail: bool,
    }

    impl Sequenced for TestInput {
        fn get_sequence_id(&self) -> u64 {
            self.id
        }
    }

    struct MockNode;
    impl Node for MockNode {
        type Input = TestInput;
        type Output = Vec<String>;
        type Error = io::Error;

        async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
            if input.should_fail {
                Err(io::Error::new(io::ErrorKind::Other, "Mock Failure"))
            } else {
                Ok(vec![input.val.clone()])
            }
        }
    }

    // --- Tests ---

    #[tokio::test]
    async fn test_in_order_execution() {
        let node = OrderingNode::new(MockNode, 1);

        // Execute 1
        let res1 = node.execute(&TestInput { id: 1, val: "A".into(), should_fail: false }).await.unwrap();
        assert_eq!(res1, vec!["A"], "Item 1 should emit immediately");

        // Execute 2
        let res2 = node.execute(&TestInput { id: 2, val: "B".into(), should_fail: false }).await.unwrap();
        assert_eq!(res2, vec!["B"], "Item 2 should emit immediately");
    }

    #[tokio::test]
    async fn test_out_of_order_execution() {
        let node = OrderingNode::new(MockNode, 1);

        // Execute 3 first (out of order)
        let res3 = node.execute(&TestInput { id: 3, val: "C".into(), should_fail: false }).await.unwrap();
        assert!(res3.is_empty(), "Item 3 should be buffered");

        // Execute 2 (out of order)
        let res2 = node.execute(&TestInput { id: 2, val: "B".into(), should_fail: false }).await.unwrap();
        assert!(res2.is_empty(), "Item 2 should be buffered");

        // Execute 1 (in order) - this should unblock 2 and 3!
        let res1 = node.execute(&TestInput { id: 1, val: "A".into(), should_fail: false }).await.unwrap();
        
        assert_eq!(res1, vec!["A", "B", "C"], "Item 1 should unblock and emit buffered items");
        
        let state = node.state.lock().unwrap();
        assert_eq!(state.next_expected_id, 4);
        assert!(state.buffer.is_empty());
    }

    #[tokio::test]
    async fn test_failure_prevents_deadlock() {
        let node = OrderingNode::new(MockNode, 1);

        // Execute 3 (buffers)
        let _ = node.execute(&TestInput { id: 3, val: "C".into(), should_fail: false }).await;
        
        // Execute 2 (FAILS). It should drop a tombstone so sequence can proceed later.
        let err2 = node.execute(&TestInput { id: 2, val: "B".into(), should_fail: true }).await;
        assert!(err2.is_err());

        // Execute 1 (succeeds). It should unblock 2 (tombstone) and 3 (valid).
        let res1 = node.execute(&TestInput { id: 1, val: "A".into(), should_fail: false }).await.unwrap();
        
        // Since 2 failed, we only get A and C back out.
        assert_eq!(res1, vec!["A", "C"]);
        
        let state = node.state.lock().unwrap();
        assert_eq!(state.next_expected_id, 4);
    }

    #[tokio::test]
    async fn test_flush_force_emits_buffer() {
        let node = OrderingNode::new(MockNode, 1);

        // Missing item 1! We only process 2 and 3.
        let _ = node.execute(&TestInput { id: 2, val: "B".into(), should_fail: false }).await;
        let _ = node.execute(&TestInput { id: 3, val: "C".into(), should_fail: false }).await;

        let state = node.state.lock().unwrap();
        assert_eq!(state.buffer.len(), 2, "Items are stuck waiting for ID 1");
        drop(state); // Drop lock before flushing

        let flush_res = node.flush().unwrap();
        
        assert_eq!(flush_res, vec!["B", "C"], "Flush should force emit the remaining items");
    }
}