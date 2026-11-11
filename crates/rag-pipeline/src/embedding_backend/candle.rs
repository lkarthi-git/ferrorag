use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel};
use hf_hub::{HFClientSync,RepoTypeAny};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use crate::embedding::token::TokenEmbeddings;
use crate::embedding_backend::AdvancedEmbeddingBackend;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

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

        // 3. Load Model to Device
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
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
        let token_ids = {
            let tokenizer = self.tokenizer.lock().unwrap();
            let tokens = tokenizer.encode(text, true).map_err(|e| format!("Token error: {}", e))?;
            tokens.get_ids().to_vec()
        };

        let token_tensor = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_tensor.zeros_like()?;

        let hidden_states = self.model.forward(&token_tensor, Some(&token_type_ids), None)?;

        Ok(TokenEmbeddings {
            hidden_states,
            device: self.device.clone(),
        })
    }
}

#[async_trait]
impl AdvancedEmbeddingBackend for CandleEmbedder {
async fn embed_document_tokens(&self, document: &str) -> Result<TokenEmbeddings, BoxError> {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        
        let text = document.to_string();

        let embeddings = tokio::task::spawn_blocking(move || {
            let local_adapter = CandleEmbedder {
                model, 
                tokenizer, 
                device, 
                query_prefix: None, 
                document_prefix: None,
                vector_dimension: 0, 
                max_batch_size: 0,
            };
            local_adapter.encode_tokens_sync(&text)
        }).await??; // The ?? unwraps both the JoinError and the inner BoxError

        // Wrap the final raw struct in Ok()
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

        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();

        let embeddings = tokio::task::spawn_blocking(move || {
            let local_adapter = CandleEmbedder {
                model, tokenizer, device, query_prefix: None, document_prefix: None,
                vector_dimension: 0, max_batch_size: 0,
            };
            
            let mut results = Vec::with_capacity(docs_to_embed.len());
            for doc in docs_to_embed {
                let tokens = local_adapter.encode_tokens_sync(&doc)?;
                let seq_len = tokens.hidden_states.dim(1)?;
                let mut vectors = tokens.pool_spans(&[(0, seq_len)])?;
                results.push(vectors.pop().unwrap());
            }
            Ok::<Vec<Vec<f32>>, BoxError>(results)
        }).await??;

        Ok(embeddings)
    }

    fn vector_dimension(&self) -> usize { self.vector_dimension }
    fn max_batch_size(&self) -> usize { self.max_batch_size }
}