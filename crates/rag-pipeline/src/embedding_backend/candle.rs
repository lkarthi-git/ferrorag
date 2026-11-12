use async_trait::async_trait;
use candle_core::{Device, Tensor,IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel};
use hf_hub::{HFClientSync,RepoTypeAny};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use crate::embedding::token::TokenEmbeddings;
use crate::embedding_backend::AdvancedEmbeddingBackend;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct CandleEmbedder {
    model: Arc<NomicBertModel>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    device: Device,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    vector_dimension: usize,
    max_batch_size: usize,
}

impl CandleEmbedder {
    pub fn new(
        repo_id: &str,
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
        vector_dimension: usize,
        max_batch_size: usize,
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

        let config: Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Tokenizer error: {}", e))?;

        let dtype = if device.is_cuda() || device.is_metal() {
            candle_core::DType::F16 
        } else {
            candle_core::DType::F32
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)?
        };
        
        let model = NomicBertModel::load(vb, &config)
            .map_err(|e| format!("Failed to load BertModel: {}", e))?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            device,
            query_prefix: query_prefix.map(|s| s.to_string()),
            document_prefix: document_prefix.map(|s| s.to_string()),
            vector_dimension,
            max_batch_size,
        })
    }

    pub fn nomic_v1_5() -> Result<Self, BoxError> {
        Self::new(
            "nomic-ai/nomic-embed-text-v1.5",
            Some("search_query: "),
            Some("search_document: "),
            768,
            256,
        )
    }

    fn encode_tokens_sync(&self, text: &str) -> Result<TokenEmbeddings, BoxError> {
        let (token_ids, mask_ids) = {
            let tokenizer = self.tokenizer.lock().unwrap();
            let tokens = tokenizer.encode(text, true).map_err(|e| format!("Token error: {}", e))?;
            (tokens.get_ids().to_vec(), tokens.get_attention_mask().to_vec())
        };

        let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_tensor.zeros_like()?;

        let mask_u8: Vec<u8> = mask_ids.into_iter().map(|x| x as u8).collect();
        let attention_mask = Tensor::new(mask_u8.as_slice(), &self.device)?.unsqueeze(0)?;

        let hidden_states = self.model.forward(&token_tensor, Some(&token_type_ids), Some(&attention_mask))?;

        let hidden_states = if hidden_states.dtype() != candle_core::DType::F32 {
            hidden_states.to_dtype(candle_core::DType::F32)?
        } else {
            hidden_states
        };

        Ok(TokenEmbeddings {
            hidden_states,
            device: self.device.clone(),
        })
    }
}

#[async_trait]
impl AdvancedEmbeddingBackend for CandleEmbedder {
    async fn embed_document_tokens(&self, documents: Vec<String>) -> Result<Vec<TokenEmbeddings>, BoxError> {
            if documents.is_empty() {
                return Ok(vec![]);
            }
            
        let embedder = self.clone();

        let embeddings = tokio::task::spawn_blocking(move || -> Result<Vec<TokenEmbeddings>, BoxError> {
            let mut tokenizer = embedder.tokenizer.lock().unwrap();
            
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

            let encodings = tokenizer.encode_batch(documents, true)
                .map_err(|e| format!("Batch tokenization error: {}", e))?;

            let batch_size = encodings.len();
            if batch_size == 0 {
                return Ok(vec![]);
            }
            
            let seq_len = encodings[0].get_ids().len(); 
            let mut all_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_type_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_masks_u8 = Vec::with_capacity(batch_size * seq_len);
            for encoding in &encodings {
                all_ids.extend_from_slice(encoding.get_ids());
                all_type_ids.extend_from_slice(encoding.get_type_ids());
                all_masks_u8.extend(encoding.get_attention_mask().iter().map(|&x| x as u8));
            }

            // Send entire batch to GPU
            let token_tensor = Tensor::from_vec(all_ids, (batch_size, seq_len), &embedder.device)?;
            let token_type_ids = Tensor::from_vec(all_type_ids, (batch_size, seq_len), &embedder.device)?;
            let attention_mask = Tensor::from_vec(all_masks_u8, (batch_size, seq_len), &embedder.device)?;
            
            // Single forward pass for the whole batch
            let hidden_states = embedder.model.forward(&token_tensor, Some(&token_type_ids), Some(&attention_mask))?;
            
            let hidden_states = if hidden_states.dtype() != candle_core::DType::F32 {
                hidden_states.to_dtype(candle_core::DType::F32)?
            } else {
                hidden_states
            };

            // Slice the 3D tensor (batch_size, seq_len, hidden_dim) back down to 
            // individual TokenEmbeddings so Late Chunking span logic works transparently.
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let doc_hidden = hidden_states.i((i, .., ..))?;
                results.push(TokenEmbeddings {
                    hidden_states: doc_hidden.unsqueeze(0)?, 
                    device: embedder.device.clone(),
                });
            }

            tokenizer.with_padding(None);

            Ok(results)
        }).await??;

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
                model, tokenizer, device, query_prefix: None, document_prefix: None,
                vector_dimension: 0, max_batch_size: 0,
            };
            
            // Execute forward pass
            let tokens = local_adapter.encode_tokens_sync(&query_text)?;
            
            // Pool the entire sequence (0 to seq_len)
            let seq_len = tokens.hidden_states.dim(1)?;
            let mut vectors = tokens.pool_spans(&[(0, seq_len)])?;
            
            vectors.pop().ok_or_else(|| BoxError::from("Failed to extract vector"))
        }).await??;

        Ok(embedding)
    }

    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        let docs_to_embed: Vec<String> = match &self.document_prefix {
            Some(prefix) => texts.into_iter().map(|t| format!("{}{}", prefix, t)).collect(),
            None => texts,
        };

        let embedder = self.clone();

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut tokenizer = embedder.tokenizer.lock().unwrap();
            
            // 1. Enable padding to equalize all sequence lengths in the batch
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

            // 2. Batch encode
            let encodings = tokenizer.encode_batch(docs_to_embed, true)
                .map_err(|e| format!("Batch tokenization error: {}", e))?;

            let batch_size = encodings.len();
            if batch_size == 0 {
                return Ok::<Vec<Vec<f32>>, BoxError>(vec![]);
            }
            
            let seq_len = encodings[0].get_ids().len(); 
            
            // 3. Pre-allocate 1D vectors for the flattened batch
            let mut all_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_type_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_masks_u8 = Vec::with_capacity(batch_size * seq_len);

            for encoding in &encodings {
                all_ids.extend_from_slice(encoding.get_ids());
                all_type_ids.extend_from_slice(encoding.get_type_ids());
                // U8 cast for Metal/CUDA where_cond compatibility
                all_masks_u8.extend(encoding.get_attention_mask().iter().map(|&x| x as u8));
            }

            // 4. Create 2D (batch_size, seq_len) Tensors
            let token_tensor = Tensor::from_vec(all_ids, (batch_size, seq_len), &embedder.device)?;
            let token_type_ids = Tensor::from_vec(all_type_ids, (batch_size, seq_len), &embedder.device)?;
            let attention_mask = Tensor::from_vec(all_masks_u8, (batch_size, seq_len), &embedder.device)?;
            
            // 5. ONE massive forward pass for the whole batch
            let hidden_states = embedder.model.forward(&token_tensor, Some(&token_type_ids), Some(&attention_mask))?;
            
            // 6. Immediately cast to F32
            let hidden_states = if hidden_states.dtype() != candle_core::DType::F32 {
                hidden_states.to_dtype(candle_core::DType::F32)?
            } else {
                hidden_states
            };

            // 7. Slice the 3D tensor back into individual documents for pooling
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let doc_hidden = hidden_states.i((i, .., ..))?;
                
                let dummy_tokens = TokenEmbeddings {
                    hidden_states: doc_hidden.unsqueeze(0)?, 
                    device: embedder.device.clone(),
                };
                
                let mut vectors = dummy_tokens.pool_spans(&[(0, seq_len)])?;
                results.push(vectors.pop().unwrap());
            }

            // Disable padding to not pollute single-query calls later
            tokenizer.with_padding(None);
            
            Ok(results)
        }).await??;

        Ok(embeddings)
    }


    

    fn vector_dimension(&self) -> usize { self.vector_dimension }
    fn max_batch_size(&self) -> usize { self.max_batch_size }
}