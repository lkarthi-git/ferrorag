use std::sync::Arc; 
use async_trait::async_trait;
use ferro_core::node::Node;
// Adjust paths to your actual structs
use crate::document::{Document, DocumentError};


pub mod token_greedy;

#[derive(Debug, Clone, Default)]
pub struct Chunk {
    pub document_uri: Arc<str>,
    pub text: String,
    pub token_count: usize,
    pub pages: Vec<u32>,
}


#[async_trait]
pub trait ChunkerStrategy: Send + Sync + 'static {
    async fn chunk(&self, documents: &Document) -> Result<Vec<Chunk>, DocumentError>;
}

/// The Framework Node that executes ANY strategy
pub struct ChunkerNode<S: ChunkerStrategy> {
    strategy: S,
}

impl<S: ChunkerStrategy> ChunkerNode<S> {
    pub fn new(strategy: S) -> Self {
        Self { strategy }
    }
}

impl<S: ChunkerStrategy> Node for ChunkerNode<S> {
    type Input = Document;
    type Output = Vec<Chunk>;
    type Error = DocumentError;

    fn name(&self) -> &'static str {
        "ChunkerNode"
    }

    async fn execute(&self, document: &Self::Input) -> Result<Self::Output, Self::Error> {
        self.strategy.chunk(document).await
    }
}