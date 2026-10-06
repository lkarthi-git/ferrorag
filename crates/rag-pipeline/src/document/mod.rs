use std::collections::HashMap;
use async_trait::async_trait;
use thiserror::Error;
use ferro_core::node::Node;
use crate::source::Record;
use std::sync::Arc; 


pub mod pdf;
// 1. Define a Framework-Specific Error Type
#[derive(Error, Debug)]
pub enum DocumentError {
    #[error("No parser strategy registered for mime_type: {0}")]
    StrategyNotFound(String),
    
    #[error("I/O error during parsing: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parsing failed: {0}")]
    ParseFailed(String),
}

// 2. Data Structures (Derive Debug and Clone for framework usability)
#[derive(Debug, Clone)]
pub struct Document {
    pub uri: Arc<str>, 
    pub metadata: Arc<DocumentMetadata>,
    pub elements: Vec<Element>, 
}

#[derive(Debug, Clone, Default)]
pub struct DocumentMetadata {
    pub author: Option<String>,
    pub created_at: Option<String>, 
    pub modified_at: Option<String>,
    pub allowed_groups: Vec<String>, 
    pub custom_attributes: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Element {
    pub element_type: ElementType,
    pub text: String,
    pub provenance: Option<ProvenanceData>, 
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    Title,
    Header(u8),
    Paragraph,
    ListItem,
    Table(String),    
    ImageCaption,
    Caption
}

#[derive(Debug, Clone)]
pub struct ProvenanceData {
    pub page_number: Option<u32>,
    pub bounding_box: Option<BoundingBox>,
    pub row_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

// 3. Object-Safe Strategy Trait
// The #[async_trait] macro wraps the return type in a Pin<Box<dyn Future>> 
// automatically so it can be stored in the Box<dyn ParserStrategy> below.
#[async_trait]
pub trait ParserStrategy: Send + Sync + 'static {
    async fn parse(&self, record: &Record) -> Result<Vec<Document>, DocumentError>;
}

// 4. The Node Router
pub struct DocumentParserNode {
    strategies: HashMap<String, Box<dyn ParserStrategy>>,
}

impl DocumentParserNode {
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }

    pub fn with_strategy<S: ParserStrategy>(mut self, mime_type: &str, strategy: S) -> Self {
        self.strategies.insert(mime_type.to_string(), Box::new(strategy));
        self
    }
}

// 5. Node Implementation
impl Node for DocumentParserNode {
    type Input = Record;
    type Output = Vec<Document>; 
    type Error = DocumentError; // Using our custom framework error!

    fn name(&self) -> &'static str {
        "DocumentParserNode"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        // Find the strategy or return our specific enum variant
        let strategy = self.strategies.get(&*input.mime_type).ok_or_else(|| {
            DocumentError::StrategyNotFound(input.mime_type.to_string())
        })?;

        let documents = strategy.parse(input).await?;
        
        Ok(documents)
    }
}