use std::sync::Arc;
use futures::StreamExt;
use lancedb::{connect, Connection};

// 1. IMPORT FROM LANCEDB'S RE-EXPORTS to guarantee version compatibility!
use lancedb::arrow::arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder, StringBuilder};
use lancedb::arrow::arrow_array::types::Float32Type;
use lancedb::arrow::arrow_array::RecordBatch;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema as ArrowSchema};

use lancedb::query::{ExecutableQuery, QueryBase};

use uuid::Uuid;

// Import your existing types (adjust path as needed)
use crate::chunk::{Chunk, EmbeddedChunk};

pub mod vector_store_node;
pub use vector_store_node::VectorStorageNode;

pub struct VectorStore {
    db: Connection,
    vector_dimension: i32,
}

impl VectorStore {
    /// Initializes the connection to LanceDB
    pub async fn new(db_uri: &str, vector_dimension: i32) -> Result<Self, lancedb::Error> {
        let db = connect(db_uri).execute().await?;
        Ok(Self {
            db,
            vector_dimension,
        })
    }

    /// Helper to get the Arrow Schema
    fn get_schema(&self) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document_uri", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("pages", DataType::Utf8, false),
            Field::new("vector", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                self.vector_dimension,
            ), false),
        ]))
    }

    /// Stores a batch of embedded chunks into the specified table
    pub async fn store_chunks(&self, table_name: &str, chunks: &[EmbeddedChunk]) -> Result<(), Box<dyn std::error::Error>> {
        let schema = self.get_schema();

        // 1. Initialize Arrow Builders
        let mut id_builder = StringBuilder::new();
        let mut uri_builder = StringBuilder::new();
        let mut text_builder = StringBuilder::new();
        let mut pages_builder = StringBuilder::new();
        
        let values_builder = PrimitiveBuilder::<Float32Type>::new();
        let mut vector_builder = FixedSizeListBuilder::new(values_builder, self.vector_dimension);

        // 2. Populate Builders
        for item in chunks {
            id_builder.append_value(Uuid::new_v4().to_string());
            uri_builder.append_value(item.chunk.document_uri.as_ref());
            text_builder.append_value(item.chunk.text.as_str());
            
            let pages_str = item.chunk.pages.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",");
            pages_builder.append_value(pages_str);

            for &val in &item.vector {
                vector_builder.values().append_value(val);
            }
            vector_builder.append(true);
        }

        // 3. Finalize Arrow Arrays
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(uri_builder.finish()),
                Arc::new(text_builder.finish()),
                Arc::new(pages_builder.finish()),
                Arc::new(vector_builder.finish()),
            ],
        )?;

        // 4. Open or Create Table
        if self.db.table_names().execute().await?.contains(&table_name.to_string()) {
            let table = self.db.open_table(table_name).execute().await?;
            // SIMPLIFIED: Vec<RecordBatch> naturally implements Scannable
            table.add(vec![batch]).execute().await?;
        } else {
            // SIMPLIFIED: Vec<RecordBatch> naturally implements Scannable
            self.db.create_table(table_name, vec![batch]).execute().await?;
        }

        Ok(())
    }

    /// Retrieves the top-k most similar chunks for a given query vector
    pub async fn search(&self, table_name: &str, query_vector: &[f32], limit: usize) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let table = self.db.open_table(table_name).execute().await?;
        
        // Execute the vector search
        let mut stream = table
            .query()
            .nearest_to(query_vector)? // <-- Re-added the '?' 
            .limit(limit)
            .execute()
            .await?;

        let mut results = Vec::new();
        
        // Stream out the batches of results
        while let Some(batch) = stream.next().await {
            results.push(batch?);
        }

        Ok(results)
    }
}