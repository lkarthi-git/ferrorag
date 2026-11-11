use std::future::Future;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{HFClientSync, RepoTypeAny};
use tokenizers::Tokenizer;

use crate::search::{EmbeddedQuery, vector::VectorSearchNode};
use crate::store::SearchResult;
use ferro_core::node::Node;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ---------------------------------------------------------------------------
// 1. Model Definition for Sequence Classification
// ---------------------------------------------------------------------------
pub struct BertForSequenceClassification {
    pub bert: BertModel,
    pub classifier: Linear,
}

impl BertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let bert = BertModel::load(vb.clone(), config)?;
        let num_labels = 1;
        let hidden_size = config.hidden_size; 
        
        let classifier = candle_nn::linear(hidden_size, num_labels, vb.pp("classifier"))?;
        
        Ok(Self { bert, classifier })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let embeddings = self.bert.forward(input_ids, token_type_ids, None)?;
        let cls_token = embeddings.i((.., 0, ..))?;
        self.classifier.forward(&cls_token)
    }
}

// ---------------------------------------------------------------------------
// 2. The Candle Cross-Encoder Reranker
// ---------------------------------------------------------------------------

pub struct CandleReranker {
    model: Arc<BertForSequenceClassification>,
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

        let config_path = repo.download_file()
            .filename("config.json")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download config: {}", e))?;
            
        let weights_path = repo.download_file()
            .filename("model.safetensors")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download weights: {}", e))?;
            
        let tokenizer_path = repo.download_file()
            .filename("tokenizer.json")
            .revision(revision.to_string())
            .send()
            .map_err(|e| format!("Failed to download tokenizer: {}", e))?;

        let config: Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer error: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
        };
        
        let model = BertForSequenceClassification::load(vb, &config)
            .map_err(|e| format!("Failed to load Reranker Model: {}", e))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            device,
        })
    }

    pub fn predict_scores_sync(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, BoxError> {
        let tokenizer = self.tokenizer.lock().unwrap();
        let mut scores = Vec::with_capacity(docs.len());

        for doc in docs {
            let encoding = tokenizer.encode((query, *doc), true)
                .map_err(|e| format!("Tokenization error: {}", e))?;

            let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
            let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;
            let type_ids = Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input_ids, &attention_mask, &type_ids)?;
            let score = logits.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()?;
            scores.push(score);
        }

        Ok(scores)
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
            }).await;

            let scores = match scores_result {
                Ok(Ok(s)) => s,
                _ => {
                    tracing::error!("Reranking inference failed, returning un-reranked results");
                    results.truncate(final_top_k);
                    return Ok(results);
                }
            };

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