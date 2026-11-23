use crate::embedding::token::TokenEmbeddings;
use crate::embedding_backend::AdvancedEmbeddingBackend;
use async_trait::async_trait;
use crate::util::hf_download;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
pub struct OrtEmbedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    has_token_type_ids: bool,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    vector_dimension: usize,
    max_batch_size: usize,
    max_batch_tokens: usize,
}

impl OrtEmbedder {
    pub fn new(
        repo_id: &str,
        revision: &str,
        onnx_filename: &str, // e.g., "onnx/model.onnx"
        query_prefix: Option<&str>,
        document_prefix: Option<&str>,
        vector_dimension: usize,
        max_batch_size: usize,
        max_batch_tokens: usize,
    ) -> Result<Self, BoxError> {
        let model_path = hf_download(repo_id, revision, onnx_filename)?;
        let tokenizer_path = hf_download(repo_id, revision, "tokenizer.json")?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Tokenizer error: {}", e))?;

        // Initialize ONNX Runtime environment & session
        ort::init().with_name("ferro-embedder").commit();
        let mut builder = Session::builder()?
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
            .map_err(|e| format!("Failed to load ONNX model: {e}"))?;

        // Some BERT/Jina ONNX exports require `token_type_ids`, others only take `input_ids` + `attention_mask`
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
            has_token_type_ids,
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
            "main",
            "model.onnx",
            None,
            None,
            768,
            256,
            80_000,
        )
    }

    /// Runs a padded batch through the ONNX session and returns per-document TokenEmbeddings.
    fn forward_batch(
        &self,
        encodings: &[tokenizers::Encoding],
        max_len: usize,
    ) -> Result<Vec<TokenEmbeddings>, BoxError> {
        let batch_size = encodings.len();
        let total_elements = batch_size * max_len;

        // ONNX transformer models expect i64 for token IDs and attention masks
        let mut input_ids = vec![0i64; total_elements];
        let mut attention_mask = vec![0i64; total_elements];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let start = i * max_len;
            for (j, &id) in ids.iter().enumerate() {
                input_ids[start + j] = id as i64;
                attention_mask[start + j] = 1;
            }
        }

        let ids_tensor = Tensor::from_array(([batch_size, max_len], input_ids))?;
        let mask_tensor = Tensor::from_array(([batch_size, max_len], attention_mask))?;

        let mut session = self.session.lock().unwrap();
        let outputs = if self.has_token_type_ids {
            let type_ids = vec![0i64; total_elements];
            let type_tensor = Tensor::from_array(([batch_size, max_len], type_ids))?;
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

        // Output 0 is typically `last_hidden_state` with shape [batch_size, max_len, hidden_dim]
        let (shape, raw_data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden_dim = shape[2] as usize;
        let doc_stride = max_len * hidden_dim;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * doc_stride;
            let end = start + doc_stride;
            results.push(TokenEmbeddings {
                data: raw_data[start..end].to_vec(),
                seq_len: max_len,
                hidden_dim,
            });
        }

        Ok(results)
    }
}

#[async_trait]
impl AdvancedEmbeddingBackend for OrtEmbedder {
    async fn embed_document_tokens(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<TokenEmbeddings>, BoxError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let embedder = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<TokenEmbeddings>, BoxError> {
            let mut tokenizer = embedder.tokenizer.lock().unwrap();
            let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: 8192,
                ..Default::default()
            }));
            tokenizer.with_padding(None);

            let mut all_results = Vec::with_capacity(documents.len());

            // Process 1-by-1 for 8192-token macro-chunks to avoid quadratic attention OOMs
            for text in documents {
                let encoding = tokenizer
                    .encode(text, true)
                    .map_err(|e| format!("Tokenization error: {}", e))?;

                let seq_len = encoding.get_ids().len();
                if seq_len == 0 {
                    continue;
                }

                let mut batch = embedder.forward_batch(&[encoding], seq_len)?;
                if let Some(tokens) = batch.pop() {
                    all_results.push(tokens);
                }
            }

            let _ = tokenizer.with_truncation(None);
            Ok(all_results)
        })
        .await?
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError> {
        let query_text = match &self.query_prefix {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        };

        let embedder = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, BoxError> {
            let encoding = {
                let tokenizer = embedder.tokenizer.lock().unwrap();
                tokenizer
                    .encode(query_text, true)
                    .map_err(|e| format!("Token error: {}", e))?
            };

            let seq_len = encoding.get_ids().len();
            let tokens = embedder
                .forward_batch(&[encoding], seq_len)?
                .pop()
                .ok_or_else(|| BoxError::from("Empty batch output"))?;

            tokens
                .pool_spans(&[(0, seq_len)])?
                .pop()
                .ok_or_else(|| BoxError::from("Failed to extract query vector"))
        })
        .await?
    }

    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        let docs_to_embed: Vec<String> = match &self.document_prefix {
            Some(prefix) => texts.into_iter().map(|t| format!("{}{}", prefix, t)).collect(),
            None => texts,
        };

        if docs_to_embed.is_empty() {
            return Ok(vec![]);
        }

        let embedder = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, BoxError> {
            let encodings = {
                let mut tokenizer = embedder.tokenizer.lock().unwrap();
                let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                    max_length: 8192,
                    ..Default::default()
                }));
                let encs = tokenizer
                    .encode_batch(docs_to_embed, true)
                    .map_err(|e| format!("Batch tokenization error: {}", e))?;
                let _ = tokenizer.with_truncation(None);
                encs
            };

            let mut all_results = Vec::with_capacity(encodings.len());
            let mut current_batch = Vec::new();
            let mut current_max_len = 0;

            for encoding in encodings {
                let seq_len = encoding.get_ids().len();
                if seq_len == 0 {
                    all_results.push(vec![0.0; embedder.vector_dimension()]);
                    continue;
                }

                let next_max_len = current_max_len.max(seq_len);
                let potential_batch_size = current_batch.len() + 1;
                let hits_token_limit = (potential_batch_size * next_max_len) > embedder.max_batch_tokens;
                let hits_batch_limit = potential_batch_size > embedder.max_batch_size();

                if (hits_token_limit || hits_batch_limit) && !current_batch.is_empty() {
                    let batch_embeddings = embedder.forward_batch(&current_batch, current_max_len)?;
                    for (tokens, enc) in batch_embeddings.into_iter().zip(&current_batch) {
                        let actual_seq_len = enc.get_ids().len();
                        let mut vectors = tokens.pool_spans(&[(0, actual_seq_len)])?;
                        all_results.push(vectors.pop().unwrap());
                    }
                    current_batch.clear();
                    current_max_len = 0;
                }

                current_max_len = current_max_len.max(seq_len);
                current_batch.push(encoding);
            }

            if !current_batch.is_empty() {
                let batch_embeddings = embedder.forward_batch(&current_batch, current_max_len)?;
                for (tokens, enc) in batch_embeddings.into_iter().zip(&current_batch) {
                    let actual_seq_len = enc.get_ids().len();
                    let mut vectors = tokens.pool_spans(&[(0, actual_seq_len)])?;
                    all_results.push(vectors.pop().unwrap());
                }
            }

            Ok(all_results)
        })
        .await?
    }

    fn vector_dimension(&self) -> usize {
        self.vector_dimension
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}