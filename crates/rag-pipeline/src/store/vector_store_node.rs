use std::sync::Arc;
use std::future::Future;

// Import your existing types and VectorStore
use crate::chunk::EmbeddedChunk;
use crate::store::VectorStore;
use ferro_core::node::Node;



pub struct VectorStorageNode {
    /// Use an Arc to share the connection safely across async boundaries
    store: Arc<VectorStore>,
    table_name: String,
}

impl VectorStorageNode {
    pub fn new(store: Arc<VectorStore>, table_name: &str) -> Self {
        Self {
            store,
            table_name: table_name.to_string(),
        }
    }
}

impl Node for VectorStorageNode {
    // We expect a Vec of embedded chunks as input from the previous node
    type Input = Vec<EmbeddedChunk>;
    
    // The Unit type () implements Default. 
    // You could also use a custom struct if you wanted to output a success metric (e.g., records inserted)
    type Output = Vec<EmbeddedChunk>; 
    
    // Ensure our error type is thread-safe
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        // Because of the `+ Send` constraint on the trait, an async block natively captures 
        // `&self` and `input` references and guarantees they are safe to cross thread boundaries.
        async {
            self.store
                .store_chunks(&self.table_name, input)
                .await
                .map_err(|e| {
                    e.to_string() 
                })?;

            Ok(input.clone()) 
        }
    }
}