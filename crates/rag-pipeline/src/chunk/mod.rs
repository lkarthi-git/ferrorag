use std::sync::Arc; 
use async_trait::async_trait;
use ferro_core::node::Node;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
// Adjust paths to your actual structs
use crate::document::{Document, DocumentError, DocumentMetadata};

#[cfg(feature = "token-greedy-chunker")]
pub mod token_greedy;

#[cfg(feature = "structural-chunker")]
pub mod structural;


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub local_id: usize,
    pub text: String,
    pub label: String,
    pub token_start: usize, 
    pub token_end: usize,
    pub canonical_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Triplet {
    pub head_id: usize,
    pub relation: String,
    pub tail_id: usize,
}


#[derive(Debug, Clone, Default,Serialize, Deserialize)]
pub struct Chunk {
    pub document_uri: Arc<str>,
    pub document_hash: Arc<str>,
    pub metadata: Arc<DocumentMetadata>,
    pub chunk_index: u32,
    pub text: Arc<str>,
    pub token_count: usize,
    pub pages: Vec<u32>,
    pub entities: Vec<ExtractedEntity>,
    pub triplets: Vec<Triplet>,
    pub chunk_attributes: HashMap<String, String>
}

#[derive(Clone)]
struct ChunkBlock {
    header_context: String, // Stored separately so we only print it when it changes!
    text: String,
    token_count: usize,
    page: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub vector: Vec<f32>,
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