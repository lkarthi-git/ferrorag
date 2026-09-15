use crate::node::Node;
use std::sync::Mutex;
use tracing::{debug, trace};

/// A middleware node that groups individual items into fixed-size batches.
///
/// This is highly useful before nodes that perform I/O operations, such as 
/// bulk database inserts or batch API requests, reducing the number of calls.
pub struct ChunkNode<T> {
    batch_size: usize,
    buffer: Mutex<Vec<T>>,
}

impl<T> ChunkNode<T> {
    /// Creates a new `ChunkNode`.
    ///
    /// # Panics
    /// Panics if `batch_size` is 0, as a chunk size of 0 would never emit.
    pub fn new(batch_size: usize) -> Self {
        assert!(batch_size > 0, "Batch size must be greater than 0");
        Self {
            batch_size,
            buffer: Mutex::new(Vec::with_capacity(batch_size)),
        }
    }
}

impl<T: Clone + Send + Sync> Node for ChunkNode<T> {
    type Input = T;
    type Output = Vec<Vec<T>>;
    type Error = std::io::Error;

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let mut buf = self.buffer.lock().unwrap();
        buf.push(input.clone());
        
        if buf.len() >= self.batch_size {
            // Batch is full! Swap it out for a fresh, empty buffer.
            let chunk = std::mem::take(&mut *buf);
            
            debug!(
                chunk_size = chunk.len(), 
                "Batch size reached. Emitting chunk downstream."
            );
            
            // Wrap the chunk in a Vec so the pipeline flattens it correctly.
            Ok(vec![chunk])
        } else {
            // Batch is still filling.
            trace!(
                current_size = buf.len(), 
                target_size = self.batch_size, 
                "Item buffered."
            );
            
            // Emit nothing for now.
            Ok(Vec::new())
        }
    }

    fn flush(&self) -> Result<Self::Output, Self::Error> {
        let mut buf = self.buffer.lock().unwrap();
        let remainder = std::mem::take(&mut *buf);
        
        if !remainder.is_empty() {
            debug!(
                chunk_size = remainder.len(), 
                "Flushing partial chunk at pipeline shutdown."
            );
            Ok(vec![remainder])
        } else {
            trace!("Flush called, but buffer is already empty.");
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunks_at_exact_batch_size() {
        let node = ChunkNode::new(3);

        // Feed 1: buffers
        let res1 = node.execute(&1).await.unwrap();
        assert!(res1.is_empty());

        // Feed 2: buffers
        let res2 = node.execute(&2).await.unwrap();
        assert!(res2.is_empty());

        // Feed 3: emits the chunk
        let res3 = node.execute(&3).await.unwrap();
        assert_eq!(res3, vec![vec![1, 2, 3]]);
    }

    #[tokio::test]
    async fn test_multiple_chunks() {
        let node = ChunkNode::new(2);

        let _ = node.execute(&1).await.unwrap();
        let chunk1 = node.execute(&2).await.unwrap();
        assert_eq!(chunk1, vec![vec![1, 2]]);

        let _ = node.execute(&3).await.unwrap();
        let chunk2 = node.execute(&4).await.unwrap();
        assert_eq!(chunk2, vec![vec![3, 4]]);
    }

    #[tokio::test]
    async fn test_flush_emits_partial_chunk() {
        let node = ChunkNode::new(5);

        let _ = node.execute(&1).await.unwrap();
        let _ = node.execute(&2).await.unwrap();

        // Buffer has 2 items. Flush should emit them even though size < 5.
        let flush_res = node.flush().unwrap();
        assert_eq!(flush_res, vec![vec![1, 2]]);
        
        // Second flush should be empty
        let flush2 = node.flush().unwrap();
        assert!(flush2.is_empty());
    }

    #[test]
    #[should_panic(expected = "Batch size must be greater than 0")]
    fn test_zero_batch_size_panics() {
        ChunkNode::<i32>::new(0);
    }
}