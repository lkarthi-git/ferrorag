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
            if chunk_len == 0 {
                continue;
            }

            // 1. Slice the window from the full tensor
            let span_tensor = self.hidden_states.narrow(1, start, chunk_len)?;

            // 2. Mean Pool the span
            let sum_embeddings = span_tensor.sum(1)?;
            let mean_pooled = (sum_embeddings / (chunk_len as f64))?;

            // 3. L2 Normalize (Standard practice for Jina / Cosine Similarity)
            let l2_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
            
            // Ensure eps matches the dtype of mean_pooled (f32 or bf16)
            let eps = Tensor::new(1e-12f32, &self.device)?.to_dtype(mean_pooled.dtype())?;

            let safe_l2_norm = l2_norm.broadcast_add(&eps)?;
            let normalized = mean_pooled.broadcast_div(&safe_l2_norm)?;

            // 4. Move only the final 768-dimensional chunk vector to CPU
            let vec2d: Vec<Vec<f32>> = normalized.to_vec2()?;
            if let Some(vec) = vec2d.into_iter().next() {
                final_vectors.push(vec);
            }
        }

        Ok(final_vectors)
    }
}