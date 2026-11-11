pub type BoxError = Box<dyn std::error::Error + Send + Sync>;


#[cfg(feature = "candle")]
pub mod candle;


#[async_trait::async_trait]
pub trait EmbeddingBackend: Send + Sync {
    /// Embeds a single query string.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError>;
    
    /// Embeds a batch of document strings.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError>;

    fn vector_dimension(&self) -> usize;

    fn max_batch_size(&self) -> usize;
}


#[async_trait::async_trait]
pub trait AdvancedEmbeddingBackend: Send + Sync {
    /// Traditional naive embedding for queries
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, BoxError>;
    
    /// Traditional naive embedding for multiple chunks
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BoxError>;
    
    /// Advanced Late Chunking: Returns the unpooled GPU token embeddings
    async fn embed_document_tokens(&self, document: &str) -> Result<crate::embedding::token::TokenEmbeddings, BoxError>;
    
    fn vector_dimension(&self) -> usize;
    fn max_batch_size(&self) -> usize;
}
