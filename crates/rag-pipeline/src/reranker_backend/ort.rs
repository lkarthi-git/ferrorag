
use std::sync::{Arc, Mutex};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tokenizers::Tokenizer;
use crate::util::hf_download;
use crate::reranker_backend::RerankerBackend;
use async_trait::async_trait;


pub type BoxError = Box<dyn std::error::Error + Send + Sync>;


#[derive(Clone)]
pub struct OrtReranker {
    session: Arc<Mutex<ort::session::Session>>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    has_token_type_ids: bool,
    batch_size: usize,
}


impl OrtReranker {
    pub fn new(
        repo_id: &str,
        revision: Option<&str>,
        onnx_filename: &str, // e.g., "onnx/model.onnx"
        batch_size: Option<usize>,
    ) -> Result<Self, BoxError> {
        let rev = revision.unwrap_or("main");
        let model_path = hf_download(repo_id, rev, onnx_filename)?;
        let tokenizer_path = hf_download(repo_id, rev, "tokenizer.json")?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer error: {e}"))?;

        let _ = ort::init().with_name("ferro-reranker").commit();

        let mut builder = Session::builder()
            .map_err(|e| format!("Failed to create ORT session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {e}"))?;

        #[cfg(feature = "cuda")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build()])
                .map_err(|e| format!("Failed to register CUDA EP: {e}"))?;
        }

        #[cfg(feature = "metal")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CoreML::default().build()])
                .map_err(|e| format!("Failed to register CoreML EP: {e}"))?;
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX reranker model: {e}"))?;

        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let default_batch_size = if cfg!(feature = "cuda") {
            32
        } else if cfg!(feature = "metal") {
            16
        } else {
            8
        };

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            has_token_type_ids,
            batch_size: batch_size.unwrap_or(default_batch_size),
        })
    }

    pub fn bge_reranker_v2_m3() -> Result<Self, BoxError> {
        Self::new(
            "onnx-community/bge-reranker-v2-m3-ONNX",
            Some("main"),
            "onnx/model_fp16.onnx",
            None,
        )
    }
}

#[async_trait]
impl RerankerBackend for OrtReranker {
    async fn predict_scores(&self, query: &str, docs: Vec<String>) -> Result<Vec<f32>, BoxError> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        let reranker = self.clone();
        let query = query.to_string();

        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, BoxError> {
            let encodings = {
                let mut tokenizer = reranker.tokenizer.lock().unwrap();
                tokenizer.with_padding(Some(tokenizers::PaddingParams {
                    strategy: tokenizers::PaddingStrategy::BatchLongest,
                    ..Default::default()
                }));
                let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                    max_length: 512,
                    strategy: tokenizers::TruncationStrategy::LongestFirst,
                    ..Default::default()
                }));

                let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (query.as_str(), d.as_str())).collect();
                let res = tokenizer
                    .encode_batch(pairs, true)
                    .map_err(|e| format!("Batch tokenization error: {}", e));

                tokenizer.with_padding(None);
                let _ = tokenizer.with_truncation(None);
                res?
            };

            let seq_len = encodings[0].get_ids().len();
            let mut all_scores = Vec::with_capacity(encodings.len());

            for chunk in encodings.chunks(reranker.batch_size) {
                let cur_bs = chunk.len();
                let mut ids = Vec::with_capacity(cur_bs * seq_len);
                let mut masks = Vec::with_capacity(cur_bs * seq_len);

                for enc in chunk {
                    ids.extend(enc.get_ids().iter().map(|&x| x as i64));
                    masks.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
                }

                let ids_tensor = ort::value::Tensor::from_array(([cur_bs, seq_len], ids))?;
                let mask_tensor = ort::value::Tensor::from_array(([cur_bs, seq_len], masks))?;

                let mut session = reranker.session.lock().unwrap();
                let outputs = if reranker.has_token_type_ids {
                    let type_ids = vec![0i64; cur_bs * seq_len];
                    let type_tensor = ort::value::Tensor::from_array(([cur_bs, seq_len], type_ids))?;
                    session.run(ort::inputs![
                        "input_ids" => ids_tensor,
                        "attention_mask" => mask_tensor,
                        "token_type_ids" => type_tensor,
                    ])?
                } else {
                    session.run(ort::inputs![
                        "input_ids" => ids_tensor,
                        "attention_mask" => mask_tensor,
                    ])?
                };

                // Logits shape: [cur_bs, 1] or [cur_bs]
                let (_shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
                all_scores.extend_from_slice(logits);
            }

            Ok(all_scores)
        })
        .await?
    }
}