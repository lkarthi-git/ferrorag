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
        let seq_len = self.hidden_states.dim(1)?;

        for &(start, end) in spans {
            let chunk_len = end.saturating_sub(start);
            if chunk_len == 0 || end > seq_len {
                continue;
            }

            let span_tensor = self
                .hidden_states
                .narrow(1, start, chunk_len)?
                .contiguous()?;
            let sum_embeddings = span_tensor.sum(1)?;
            let mean_pooled = (sum_embeddings / (chunk_len as f64))?;

            let l2_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
            let eps = Tensor::new(1e-12f32, &Device::Cpu)?;
            let safe_l2_norm = l2_norm.broadcast_add(&eps)?;
            let normalized = mean_pooled.broadcast_div(&safe_l2_norm)?;

            if let Some(vec) = normalized.to_vec2::<f32>()?.into_iter().next() {
                if vec.iter().any(|v| !v.is_finite()) {
                    return Err(format!("Non-finite embedding in span ({start}..{end})").into());
                }
                final_vectors.push(vec);
            }
        }

        Ok(final_vectors)
    }
}
