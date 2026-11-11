use candle_core::{Device, Tensor};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct TokenEmbeddings {
    pub hidden_states: Tensor,
    pub device: Device,
}

impl TokenEmbeddings {
    /// Pools specific token spans directly on the GPU.
    /// `spans` are pairs of (start_token_index, end_token_index)
    pub fn pool_spans(&self, spans: &[(usize, usize)]) -> Result<Vec<Vec<f32>>, BoxError> {
        let mut final_vectors = Vec::with_capacity(spans.len());

        for &(start, end) in spans {
            let chunk_len = end - start;
            if chunk_len == 0 { continue; }
            
            // 1. Slice the 256-token window from the full 8000-token tensor
            let span_tensor = self.hidden_states.narrow(1, start, chunk_len)?;
            
            // 2. Mean Pool the span
            let sum_embeddings = span_tensor.sum(1)?;
            let mean_pooled = (sum_embeddings / (chunk_len as f64))?;
            
            // 3. L2 Normalize (Required for Nomic)
            let l2_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
            let normalized = mean_pooled.broadcast_div(&l2_norm)?;
            
            // 4. Move only the final 768-dimensional chunk vector to CPU
            let vec2d: Vec<Vec<f32>> = normalized.to_vec2()?;
            if let Some(vec) = vec2d.into_iter().next() {
                final_vectors.push(vec);
            }
        }

        Ok(final_vectors)
    }

    pub fn pool_late_chunks(&self, prefix_len: usize, spans: &[(usize, usize)]) -> Result<Vec<Vec<f32>>, BoxError> {
        let mut final_vectors = Vec::with_capacity(spans.len());

        // 1. Extract and sum the instruction prefix tokens (e.g., [CLS] + "search_document: ")
        let prefix_tensor = if prefix_len > 0 {
            Some(self.hidden_states.narrow(1, 0, prefix_len)?)
        } else {
            None
        };
        let prefix_sum = match &prefix_tensor {
            Some(pt) => Some(pt.sum(1)?),
            None => None,
        };

        for &(start, end) in spans {
            let chunk_len = end - start;
            if chunk_len == 0 { continue; }
            
            // 2. Extract and sum the chunk's specific text tokens
            let span_tensor = self.hidden_states.narrow(1, start, chunk_len)?;
            let mut chunk_sum = span_tensor.sum(1)?;
            let mut total_tokens = chunk_len as f64;
            
            // 3. Combine the chunk with the global prefix instructions
            if let Some(psum) = &prefix_sum {
                chunk_sum = chunk_sum.broadcast_add(psum)?;
                total_tokens += prefix_len as f64;
            }
            
            // 4. Mean Pool over (Prefix Tokens + Chunk Tokens)
            let mean_pooled = (chunk_sum / total_tokens)?;
            
            // 5. L2 Normalize (Required for Nomic)
            let l2_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
            let normalized = mean_pooled.broadcast_div(&l2_norm)?;
            
            let vec2d: Vec<Vec<f32>> = normalized.to_vec2()?;
            if let Some(vec) = vec2d.into_iter().next() {
                final_vectors.push(vec);
            }
        }

        Ok(final_vectors)
    }
}