use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use hf_hub::{HFClientSync, RepoTypeAny};
use tokenizers::Tokenizer;

use crate::search::{EmbeddedQuery, vector::VectorSearchNode};
use crate::store::SearchResult;
use ferro_core::node::Node;

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

        // Using your exact HFClientSync and RepoTypeAny pattern
        let client = HFClientSync::new().map_err(|e| format!("Failed to init HF API: {}", e))?;

        let parts: Vec<&str> = repo_id.split('/').collect();
        let (owner, name) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            ("".to_string(), repo_id.to_string())
        };

        let repo = client.repository(RepoTypeAny::Model, owner, name);

        let config_path = repo
            .download_file()
            .filename("config.json")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download config: {}", e))?;

        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download weights: {}", e))?;

        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download tokenizer: {}", e))?;

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

// ---------------------------------------------------------------------------
// 3. Updated Node Implementation
// ---------------------------------------------------------------------------

pub struct SearchAndRerankNode {
    search_node: Arc<VectorSearchNode>, // Wrapped in Arc to avoid requiring Clone on VectorSearchNode
    reranker: Arc<CandleReranker>,
    query_text_map: Arc<HashMap<String, String>>,
    final_top_k: usize,
}

impl SearchAndRerankNode {
    pub fn new(
        search_node: VectorSearchNode,
        reranker: Arc<CandleReranker>,
        query_text_map: Arc<HashMap<String, String>>,
        final_top_k: usize,
    ) -> Result<Self, BoxError> {
        Ok(Self {
            search_node: Arc::new(search_node),
            reranker,
            query_text_map,
            final_top_k,
        })
    }
}

impl Node for SearchAndRerankNode {
    type Input = EmbeddedQuery;
    type Output = Vec<SearchResult>;
    type Error = BoxError;

    fn execute(
        &self,
        input: &Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        // Clone Arc handles rather than requiring direct object clones
        let search_node = Arc::clone(&self.search_node);
        let input = input.clone();

        let reranker = self.reranker.clone();
        let query_text_map = self.query_text_map.clone();
        let final_top_k = self.final_top_k;

        async move {
            let mut results = search_node.execute(&input).await?;

            if results.is_empty() {
                return Ok(results);
            }

            let q_id = &results[0].query_metadata.query_id;
            let query_text = match query_text_map.get(q_id) {
                Some(text) => text.clone(),
                None => {
                    tracing::warn!("Query text missing for reranking query_id: {}", q_id);
                    results.truncate(final_top_k);
                    return Ok(results);
                }
            };

            let docs: Vec<String> = results.iter().map(|r| r.text.to_string()).collect();

            let scores_result = tokio::task::spawn_blocking(move || {
                let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
                reranker.predict_scores_sync(&query_text, &doc_refs)
            })
            .await;

            let scores = match scores_result {
                Ok(Ok(s)) => s,
                Ok(Err(e)) => {
                    // Exposes the actual Candle tensor/shape error
                    tracing::error!("Reranking inference failed with error: {:?}", e);
                    results.truncate(final_top_k);
                    return Ok(results);
                }
                Err(join_err) => {
                    // Exposes thread panics (e.g., Out of Memory)
                    tracing::error!("Reranking thread panicked: {:?}", join_err);
                    results.truncate(final_top_k);
                    return Ok(results);
                }
            };

            if scores.len() != results.len() {
                tracing::error!(
                    "Reranker score count ({}) != result count ({})",
                    scores.len(),
                    results.len()
                );
                results.truncate(final_top_k);
                return Ok(results);
            }

            for (i, score) in scores.into_iter().enumerate() {
                results[i].score = score;
            }

            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.document_uri.cmp(&b.document_uri))
            });

            results.truncate(final_top_k);
            Ok(results)
        }
    }
}
