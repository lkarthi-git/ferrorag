pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct TokenEmbeddings {
    pub data: Vec<f32>,
    pub seq_len: usize,
    pub hidden_dim: usize,
}

impl TokenEmbeddings {
    pub fn pool_spans(&self, spans: &[(usize, usize)]) -> Result<Vec<Vec<f32>>, BoxError> {
        let mut final_vectors = Vec::with_capacity(spans.len());

        for &(start, end) in spans {
            let chunk_len = end.saturating_sub(start);
            if chunk_len == 0 || end > self.seq_len {
                continue;
            }

            let mut pooled = vec![0.0f32; self.hidden_dim];
            for token_idx in start..end {
                let row_start = token_idx * self.hidden_dim;
                let row = &self.data[row_start..row_start + self.hidden_dim];
                for (acc, &val) in pooled.iter_mut().zip(row.iter()) {
                    *acc += val;
                }
            }

            let inv_len = 1.0 / (chunk_len as f32);
            let mut sq_sum = 0.0f32;
            for val in &mut pooled {
                *val *= inv_len;
                sq_sum += *val * *val;
            }

            let norm = (sq_sum.sqrt() + 1e-12).max(1e-12);
            for val in &mut pooled {
                *val /= norm;
                if !val.is_finite() {
                    return Err(format!("Non-finite embedding in span ({start}..{end})").into());
                }
            }

            final_vectors.push(pooled);
        }

        Ok(final_vectors)
    }
}