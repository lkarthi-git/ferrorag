use ferro_core::node::Node;
use crate::store::{ VectorStore, SearchResult};
use super::{EmbeddedQuery,SearchMode};
use std::sync::Arc;
use std::future::Future;

pub struct VectorSearchNode {
    store: Arc<dyn VectorStore>,
    table_name: String,
    mode: SearchMode,
}

impl VectorSearchNode {
    pub fn new(store: Arc<dyn VectorStore>, table_name: impl Into<String>, mode: SearchMode) -> Self {
        Self {
            store,
            table_name: table_name.into(),
            mode
        }
    }
}

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;


impl Node for VectorSearchNode {
    type Input = EmbeddedQuery;
    type Output = Vec<SearchResult>; 
    type Error = BoxError;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let input = input.clone();
        let store = self.store.clone();
        let table_name = self.table_name.clone();
        let mode = self.mode;
        async move {
            let results = match mode {
                SearchMode::Vector => {
                    // Requires only the vector
                    store.search(&table_name, &input.vector, input.limit).await?
                },
                SearchMode::FullText => {
                    // Requires only the raw text
                    store.full_text_search(&table_name, &input.text, input.limit).await?
                },
                SearchMode::Hybrid => {
                    // Requires both text and vector
                    store.hybrid_search(&table_name, &input.text, &input.vector, input.limit).await?
                }
            };
            Ok(results)
        }
    }
}
