
use std::sync::{Arc, Mutex};
use crate::reranker_backend::RerankerBackend;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use tokenizers::Tokenizer;
use crate::util::hf_download;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct CandleReranker {
    model: Arc<XLMRobertaForSequenceClassification>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    device: Device,
}

impl CandleReranker {
    pub fn new(repo_id: &str, revision: &str) -> Result<Self, BoxError> {
        let device = Device::new_cuda(0)
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu);

        let config_path = hf_download(repo_id, revision, "config.json")?;
        let weights_path = hf_download(repo_id, revision, "model.safetensors")?;
        let tokenizer_path = hf_download(repo_id, revision, "tokenizer.json")?;

        let config: Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer error: {}", e))?;

        let dtype = if device.is_cuda() || device.is_metal() {
            candle_core::DType::BF16
        } else {
            candle_core::DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)? };

        let model = XLMRobertaForSequenceClassification::new(1, &config, vb)
            .map_err(|e| format!("Failed to load Reranker Model: {}", e))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            device,
        })
    }

    pub fn predict_scores_sync(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, BoxError> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenize and immediately release the mutex before GPU inference
        let encodings = {
            let mut tokenizer = self.tokenizer.lock().unwrap();

            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

            let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: 512,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            }));

            let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (query, *d)).collect();
            let res = tokenizer
                .encode_batch(pairs, true)
                .map_err(|e| format!("Batch tokenization error: {}", e));

            tokenizer.with_padding(None);
            let _ = tokenizer.with_truncation(None);
            res?
        };

        let seq_len = encodings[0].get_ids().len();
        let chunk_size = if self.device.is_cuda() {
            32
        } else if self.device.is_metal() {
            16
        } else {
            8
        };

        let mut all_scores = Vec::with_capacity(encodings.len());

        // 2. Mini-batch GPU execution
        for chunk_encodings in encodings.chunks(chunk_size) {
            let cur_bs = chunk_encodings.len();
            let mut ids = Vec::with_capacity(cur_bs * seq_len);
            let mut type_ids = Vec::with_capacity(cur_bs * seq_len);
            let mut masks = Vec::with_capacity(cur_bs * seq_len);

            for enc in chunk_encodings {
                ids.extend_from_slice(enc.get_ids());
                type_ids.extend(std::iter::repeat_n(0u32, seq_len));
                masks.extend(enc.get_attention_mask().iter().map(|&x| x as u8));
            }

            let chunk_input = Tensor::from_vec(ids, (cur_bs, seq_len), &self.device)?;
            let chunk_type = Tensor::from_vec(type_ids, (cur_bs, seq_len), &self.device)?;
            let chunk_mask = Tensor::from_vec(masks, (cur_bs, seq_len), &self.device)?;

            let logits = self.model.forward(&chunk_input, &chunk_mask, &chunk_type)?;

            let scores = logits
                .contiguous()?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            all_scores.extend(scores);
        }

        Ok(all_scores)
    }
}


#[async_trait]
impl RerankerBackend for CandleReranker {
    async fn predict_scores(&self, query: &str, docs: Vec<String>) -> Result<Vec<f32>, BoxError> {
        let reranker = self.clone();
        let query = query.to_string();

        tokio::task::spawn_blocking(move || {
            let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            reranker.predict_scores_sync(&query, &doc_refs)
        })
        .await?
    }
}