use ferro_core::node::Node;
use crate::search::{Query, EmbeddedQuery}; 
use crate::embedding_backend::{EmbeddingBackend};
use std::sync::Arc;
use std::future::Future;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ========================================================================
// The Query Embedding Node
// ========================================================================
pub struct QueryEmbeddingNode {
    backend: Arc<dyn EmbeddingBackend>,
}

impl QueryEmbeddingNode {
    pub fn new(backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self { backend }
    }
}

impl Node for QueryEmbeddingNode {
    type Input = Query;
    type Output = Vec<EmbeddedQuery>; // Changed from Option to Vec
    type Error = BoxError;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let text = input.text.clone();
        let limit = input.limit;
        let backend = self.backend.clone();

        async move {
            match backend.embed_query(&text).await {
                Ok(vector) => {
                    // Wrap the successful result in a Vec
                    Ok(vec![EmbeddedQuery {
                        text,
                        limit,
                        vector,
                    }])
                }
                Err(e) => {
                    tracing::error!("Failed to embed query: {}", e);
                    // Return an empty Vec so the pipeline gracefully drops it
                    Ok(Vec::new()) 
                }
            }
        }
    }
}