use std::collections::HashMap;
use std::sync::Arc;

pub mod vector;
pub mod print;
pub mod parent;

#[cfg(feature = "candle")]
pub mod rerank;


#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid { rrf_k: f32 },
}

#[derive(Debug, Clone)]
pub enum MetadataFilter {
    Eq(String, String),           
    GreaterThan(String, usize),   
    Contains(String, String),     
    In(String, Vec<String>),      
    And(Box<MetadataFilter>, Box<MetadataFilter>),
    Or(Box<MetadataFilter>, Box<MetadataFilter>),
}

#[derive(Debug, Clone, Default)]
pub struct QueryMetadata {
    pub query_id: String,           // Essential for evaluation mapping
    pub trace_id: String,           // Essential for OpenTelemetry/logging
    pub filter: Option<MetadataFilter>, // For LanceDB pre-filtering (e.g., tenant_id: "123")
    pub custom: HashMap<String, String>,  // For any framework extensions
}

#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub limit: usize,
    pub metadata: Arc<QueryMetadata>,
}

#[derive(Debug, Clone)]
pub struct EmbeddedQuery {
    pub text: String,
    pub limit: usize,
    pub vector: Vec<f32>,
    pub metadata: Arc<QueryMetadata>,
}



impl MetadataFilter {
    pub fn to_sql(&self) -> String {
        match self {
            MetadataFilter::Eq(col, val) => {
                let escaped_val = val.replace('\'', "''");
                format!("{} = '{}'", col, escaped_val)
            }
            MetadataFilter::GreaterThan(col, val) => {
                format!("{} > {}", col, val)
            }
            MetadataFilter::Contains(col, val) => {
                // Perfect for searching inside your serialized JSON strings
                let escaped_val = val.replace('\'', "''");
                format!("{} LIKE '%{}%'", col, escaped_val)
            }
            MetadataFilter::In(col, vals) => {
                let formatted_vals: Vec<String> = vals
                    .iter()
                    .map(|v| format!("'{}'", v.replace('\'', "''")))
                    .collect();
                format!("{} IN ({})", col, formatted_vals.join(", "))
            }
            MetadataFilter::And(left, right) => {
                format!("({} AND {})", left.to_sql(), right.to_sql())
            }
            MetadataFilter::Or(left, right) => {
                format!("({} OR {})", left.to_sql(), right.to_sql())
            }
        }
    }
}