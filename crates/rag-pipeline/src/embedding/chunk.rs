use ferro_core::node::Node;
// Adjust these imports based on your actual workspace structure
use crate::chunk::{Chunk, EmbeddedChunk}; 
use std::sync::Arc;
use std::future::Future;
use crate::embedding_backend::{AdvancedEmbeddingBackend};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct ChunkEmbeddingNode {
    backend: Arc<dyn AdvancedEmbeddingBackend>,
}

impl ChunkEmbeddingNode {
    pub fn new(backend: Arc<dyn AdvancedEmbeddingBackend>) -> Self {
        Self { backend }
    }
}

impl Node for ChunkEmbeddingNode {
    type Input = Vec<Chunk>;
    type Output = Vec<EmbeddedChunk>;
    type Error = BoxError;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        // Clone data to move it safely into the async execution block
        let input = input.clone();
        let backend = self.backend.clone();

        async move {
            // 1. Prepare the input data
            // Extract the texts using your closure/iterator mapping.
            // Note: If you moved the "search_document: " prefix logic inside the LocalEmbedder's 
            // EmbedderConfig (as discussed earlier), you can just use `chunk.text.to_string()` here.
            // If you want the node to handle it, keep the format! macro.
            let texts_to_embed: Vec<String> = input
                .iter()
                .map(|chunk| chunk.text.to_string()) 
                .collect();

            // 2. Execute the heavy embedding strategy
            let vectors = backend.embed_documents(texts_to_embed).await?;

            // 3. Ensure the lengths match to prevent panic on zip
            if input.len() != vectors.len() {
                return Err(format!(
                    "Mismatch: sent {} texts to embedder, received {} vectors",
                    input.len(),
                    vectors.len()
                )
                .into());
            }

            // 4. Re-zip the original chunks with their newly computed vectors
            let embedded_chunks: Vec<EmbeddedChunk> = input
                .into_iter()
                .zip(vectors)
                .map(|(chunk, vector)| EmbeddedChunk { chunk, vector })
                .collect();

            Ok(embedded_chunks)
        }
    }
}