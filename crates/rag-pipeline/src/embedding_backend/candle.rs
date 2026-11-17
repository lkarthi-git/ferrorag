use crate::embedding::token::TokenEmbeddings;
use crate::embedding_backend::AdvancedEmbeddingBackend;
use async_trait::async_trait;
use candle_core::{Device, IndexOp, Module, Tensor};

// use candle_nn::VarBuilder;
// use candle_transformers::models::nomic_bert::{Config, NomicBertModel};
use candle_transformers::models::jina_bert::{BertModel as JinaModel, Config as JinaConfig};
use hf_hub::{HFClientSync, RepoTypeAny};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct CandleEmbedder {
    model: Arc<JinaModel>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    device: Device,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    vector_dimension: usize,
    max_batch_size: usize,
    max_batch_tokens: usize,
}

// Helper function used by embed_documents to process safe-sized batches
fn forward_batch(
    embedder: &CandleEmbedder,
    encodings: &[tokenizers::Encoding],
    max_len: usize,
) -> Result<Vec<TokenEmbeddings>, BoxError> {
    let batch_size = encodings.len();
    let total_elements = batch_size * max_len;

    // 1. Pre-allocate and zero-initialize the entire flattened arrays once
    let mut all_ids = vec![0u32; total_elements];

    // 2. Overwrite the non-padded sections with actual token data
    for (i, enc) in encodings.iter().enumerate() {
        let ids = enc.get_ids();
        let seq_len = ids.len();

        // Calculate the flat array boundaries for this specific document
        let start = i * max_len;
        let end = start + seq_len;

        // Fast slice copying (C-style memcopy) for contiguous memory
        all_ids[start..end].copy_from_slice(ids);
    }

    // 3. Tensor creation and inference
    let token_tensor = Tensor::from_vec(all_ids, (batch_size, max_len), &embedder.device)?;

    let hidden_states = embedder.model.forward(&token_tensor)?;

    let hidden_states = if hidden_states.dtype() != candle_core::DType::F32 {
        hidden_states.to_dtype(candle_core::DType::F32)?
    } else {
        hidden_states
    };

    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let doc_hidden = hidden_states.i((i, .., ..))?.contiguous()?;

        results.push(TokenEmbeddings {
            hidden_states: doc_hidden.unsqueeze(0)?,
            device: embedder.device.clone(),
        });
    }

    Ok(results)
}

impl CandleEmbedder {
    pub fn new(
        repo_id: &str,
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
        vector_dimension: usize,
        max_batch_size: usize,
        max_batch_tokens: usize,
    ) -> Result<Self, BoxError> {
        let device = Device::new_cuda(0)
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu);
        let client = HFClientSync::new().map_err(|e| format!("Failed to init HF API: {}", e))?;

        let parts: Vec<&str> = repo_id.split('/').collect();
        let (owner, name) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            ("".to_string(), repo_id.to_string()) // Flat repo name with no explicit owner/namespace
        };

        // 4. Construct Repo Handle
        // If you always target models, prefer: client.model(owner, name)
        // If you need runtime dynamic types using RepoTypeAny:
        let repo_type = RepoTypeAny::Model;
        let repo = client.repository(repo_type, owner, name);

        let config_path = repo
            .download_file()
            .filename("config.json")
            .revision("main".to_string()) // <--- Set revision here
            .send()
            .map_err(|e| format!("Failed to download config: {}", e))?;

        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .revision("main".to_string()) // <--- Set revision here
            .send()
            .map_err(|e| format!("Failed to download weights: {}", e))?;

        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .revision("main".to_string()) // <--- Set revision here
            .send()
            .map_err(|e| format!("Failed to download tokenizer: {}", e))?;

        let config: JinaConfig = serde_json::from_reader(std::fs::File::open(config_path)?)
            .map_err(|e| format!("Failed to parse Jina config: {}", e))?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer error: {}", e))?;

        let dtype = candle_core::DType::F32;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)?
        };

        let model =
            JinaModel::new(vb, &config).map_err(|e| format!("Failed to load BertModel: {}", e))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            device,
            query_prefix: query_prefix.map(|s| s.to_string()),
            document_prefix: document_prefix.map(|s| s.to_string()),
            vector_dimension,
            max_batch_size,
            max_batch_tokens,
        })
    }

    pub fn jina_v2_base_en() -> Result<Self, BoxError> {
        Self::new(
            "jinaai/jina-embeddings-v2-base-en",
            None,
            None,
            768,
            256,
            80_000,
        )
    }

    fn encode_tokens_sync(&self, text: &str) -> Result<TokenEmbeddings, BoxError> {
        let token_ids = {
            let tokenizer = self.tokenizer.lock().unwrap();
            let tokens = tokenizer
                .encode(text, true)
                .map_err(|e| format!("Token error: {}", e))?;
            tokens.get_ids().to_vec()
        };

        let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let hidden_states = self.model.forward(&token_tensor)?;

        // Synchronize, make contiguous, and move to CPU just like embed_document_tokens
        let cpu_hidden = hidden_states
            .contiguous()?
            .to_dtype(candle_core::DType::F32)?
            .to_device(&Device::Cpu)?;

        Ok(TokenEmbeddings {
            hidden_states: cpu_hidden,
            device: Device::Cpu,
        })
    }
}

#[async_trait]
impl AdvancedEmbeddingBackend for CandleEmbedder {
    async fn embed_document_tokens(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<TokenEmbeddings>, BoxError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let embedder = self.clone();

        let embeddings =
            tokio::task::spawn_blocking(move || -> Result<Vec<TokenEmbeddings>, BoxError> {
                let mut tokenizer = embedder.tokenizer.lock().unwrap();

                let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                    max_length: 8192,
                    ..Default::default()
                }));

                let mut all_results = Vec::with_capacity(documents.len());

                // Process macro-chunks 1-by-1.
                // At 8,192 tokens, a single attention matrix is ~805 million elements.
                // Batching them causes 32-bit CUDA index overflows.
                for text in documents {
                    tokenizer.with_padding(None); // No padding needed for BS=1

                    let encoding = tokenizer
                        .encode(text, true)
                        .map_err(|e| format!("Tokenization error: {}", e))?;

                    let seq_len = encoding.get_ids().len();
                    if seq_len == 0 {
                        continue;
                    }

                    let all_ids = encoding.get_ids();

                    let token_tensor =
                        Tensor::from_vec(all_ids.to_vec(), (1, seq_len), &embedder.device)?;

                    let hidden_states = embedder.model.forward(&token_tensor)?;

                    let hidden_states = hidden_states
                        .contiguous()?
                        .to_dtype(candle_core::DType::F32)?;

                    let cpu_hidden = hidden_states.to_device(&Device::Cpu)?;

                    all_results.push(TokenEmbeddings {
                        hidden_states: cpu_hidden,
                        device: Device::Cpu,
                    });
                }

                let _ = tokenizer.with_truncation(None);

                Ok(all_results)
            })
            .await??;

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError> {
        let query_text = match &self.query_prefix {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        };

        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();

        let embedding = tokio::task::spawn_blocking(move || {
            let local_adapter = CandleEmbedder {
                model,
                tokenizer,
                device,
                query_prefix: None,
                document_prefix: None,
                vector_dimension: 0,
                max_batch_size: 0,
                max_batch_tokens: 0,
            };

            // Execute forward pass
            let tokens = local_adapter.encode_tokens_sync(&query_text)?;

            // Pool the entire sequence (0 to seq_len)
            let seq_len = tokens.hidden_states.dim(1)?;
            let mut vectors = tokens.pool_spans(&[(0, seq_len)])?;

            vectors
                .pop()
                .ok_or_else(|| BoxError::from("Failed to extract vector"))
        })
        .await??;

        Ok(embedding)
    }

    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        let docs_to_embed: Vec<String> = match &self.document_prefix {
            Some(prefix) => texts
                .into_iter()
                .map(|t| format!("{}{}", prefix, t))
                .collect(),
            None => texts,
        };

        if docs_to_embed.is_empty() {
            return Ok(vec![]);
        }

        let embedder = self.clone();

        let embeddings = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, BoxError> {
            let mut tokenizer = embedder.tokenizer.lock().unwrap();

            let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: 8192,
                ..Default::default()
            }));

            // Pre-tokenize to safely group batches based on actual token lengths
            let encodings = tokenizer
                .encode_batch(docs_to_embed, true)
                .map_err(|e| format!("Batch tokenization error: {}", e))?;

            let mut all_results = Vec::with_capacity(encodings.len());
            let mut current_batch = Vec::new();
            let mut current_max_len = 0;

            let max_vram_tokens = embedder.max_batch_tokens;

            for encoding in encodings {
                let seq_len = encoding.get_ids().len();
                if seq_len == 0 {
                    all_results.push(vec![0.0; embedder.vector_dimension()]);
                    continue;
                }

                let next_max_len = std::cmp::max(current_max_len, seq_len);
                let potential_batch_size = current_batch.len() + 1;

                // Memory with Flash Attention scales linearly: Batch Size * Max Length
                let total_batch_tokens = potential_batch_size * next_max_len;

                // Check against BOTH struct limits: raw batch size AND total token volume
                let hits_token_limit = total_batch_tokens > max_vram_tokens;
                let hits_batch_limit = potential_batch_size > embedder.max_batch_size();

                if (hits_token_limit || hits_batch_limit) && !current_batch.is_empty() {
                    let batch_embeddings =
                        forward_batch(&embedder, &current_batch, current_max_len)?;

                    for (dummy_tokens, enc) in batch_embeddings.into_iter().zip(&current_batch) {
                        let actual_seq_len = enc.get_ids().len();
                        let mut vectors = dummy_tokens.pool_spans(&[(0, actual_seq_len)])?;
                        all_results.push(vectors.pop().unwrap());
                    }

                    current_batch.clear();
                    current_max_len = 0;
                }

                current_max_len = std::cmp::max(current_max_len, seq_len);
                current_batch.push(encoding);
            }

            // Catch any remaining sequences
            if !current_batch.is_empty() {
                let batch_embeddings = forward_batch(&embedder, &current_batch, current_max_len)?;
                for (dummy_tokens, enc) in batch_embeddings.into_iter().zip(&current_batch) {
                    let actual_seq_len = enc.get_ids().len();
                    let mut vectors = dummy_tokens.pool_spans(&[(0, actual_seq_len)])?;
                    all_results.push(vectors.pop().unwrap());
                }

                current_batch.clear();
            }

            tokenizer.with_padding(None);
            let _ = tokenizer.with_truncation(None);

            Ok(all_results)
        })
        .await??;

        Ok(embeddings)
    }

    fn vector_dimension(&self) -> usize {
        self.vector_dimension
    }
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}
