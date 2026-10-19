use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use crate::embedding_backend::{EmbeddingBackend};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct LocalEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    query_prefix: Option<String>,
    document_prefix: Option<String>,
    vector_dimension: usize,
    max_batch_size: usize,
}

impl LocalEmbedder {
    /// The base constructor allows full customization
    pub fn new(
        model_type: EmbeddingModel, 
        query_prefix: Option<&str>, 
        document_prefix: Option<&str>,
        vector_dimension: usize,
        max_batch_size: usize
    ) -> Result<Self, BoxError> {
        let mut execution_providers = vec![];

        // Safely enable CUDA if the OS supports it
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            execution_providers.push(ort::ep::CUDA::default().build().into());
        }

        let mut options = TextInitOptions::new(model_type)
            .with_show_download_progress(true)
            .with_execution_providers(execution_providers);

        // Optimize intra-threads for Apple Silicon / CPU inference
        #[cfg(target_os = "macos")]
        {
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            options = options.with_intra_threads(threads);
        }

        let model = TextEmbedding::try_new(options)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            query_prefix: query_prefix.map(|s| s.to_string()),
            document_prefix: document_prefix.map(|s| s.to_string()),
            vector_dimension,
            max_batch_size,
        })
    }

    /// Convenience constructor for Nomic v1.5 (Handles the prefixes automatically)
    pub fn nomic_v1_5() -> Result<Self, BoxError> {
        Self::new(
            EmbeddingModel::NomicEmbedTextV15, 
            Some("search_query: "), 
            Some("search_document: "),
            768,
            256
        )
    }

}

#[async_trait]
impl EmbeddingBackend for LocalEmbedder {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError> {
        let query_text = match &self.query_prefix {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        };

        let model_arc = Arc::clone(&self.model);
        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model_guard = model_arc.lock().unwrap();
            model_guard.embed(vec![query_text], None)
        }).await??;

        embeddings.into_iter().next().ok_or_else(|| "No embedding returned".into())
    }

    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError> {
        let docs_to_embed: Vec<String> = match &self.document_prefix {
            Some(prefix) => texts.into_iter().map(|t| format!("{}{}", prefix, t)).collect(),
            None => texts,
        };

        let model_arc = Arc::clone(&self.model);
        let embeddings = tokio::task::spawn_blocking(move || {
            let mut model_guard = model_arc.lock().unwrap();
            model_guard.embed(docs_to_embed, None)
        }).await??;

        Ok(embeddings)
    }

    fn vector_dimension(&self) -> usize {
        self.vector_dimension
    }

    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}