use ferro_core::node::Node;
use crate::document::{Document};
use crate::source::Record;
use crate::store::{VectorStore, FileStatus};
use std::sync::Arc;
use crate::document::DocumentError;
pub struct DeduplicatorNode<N>{
    store: Arc<dyn VectorStore>,
    table_name: String,
    inner_node: N,
}

pub type GenericError = Box<dyn std::error::Error + Send + Sync>;


impl<N> DeduplicatorNode<N> {
    pub fn new(store: Arc<dyn VectorStore>, table_name: &str,node: N) -> Self {
        Self {
            store,
            table_name: table_name.to_string(),
            inner_node: node,
        }
    }
}

impl<N> Node for DeduplicatorNode<N> where  N: Node<Input = Record, Output = Vec<Document>, Error = DocumentError> + Send + Sync, {
    type Input = Record;
    type Output = Vec<Document>;
    type Error = DocumentError;

    fn name(&self) -> &'static str {
        "DeduplicatorNode"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> { 
          let documents =  match self.store.check_file_status(&self.table_name, &input.uri, &input.hash).await.map_err(|e| DocumentError::ParseFailed(e.to_string()))? {
                FileStatus::Unchanged => {
                    tracing::info!("File {} is unchanged. Skipping ingestion.", input.uri);
                    Vec::new()
                },
                FileStatus::Modified => {
                    tracing::info!("File {} was modified. Deleting old chunks...", input.uri);
                    self.store.delete_by_uri(&self.table_name, &input.uri).await.map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
                    let documents = self.inner_node.execute(input).await?;
                    documents
                    // Clone the document to pass it downstream (cheap since fields are Arcs)
                },
                FileStatus::New => {
                    tracing::info!("New file detected: {}", input.uri);
                    let documents = self.inner_node.execute(input).await?;
                    documents
                }
            };

            Result::Ok(documents)
        }
}