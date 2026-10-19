pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[cfg(feature = "local-embeddings")]
pub mod local;

#[async_trait::async_trait]
pub trait EmbeddingBackend: Send + Sync {
    /// Embeds a single query string.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError>;
    
    /// Embeds a batch of document strings.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError>;

    fn vector_dimension(&self) -> usize;

    fn max_batch_size(&self) -> usize;
}

