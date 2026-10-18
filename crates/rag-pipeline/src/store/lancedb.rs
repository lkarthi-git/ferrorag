use std::sync::Arc;
use futures::StreamExt;
use super::{BoxError, FileStatus, SearchResult, VectorStore};
use crate::chunk::EmbeddedChunk;
use lancedb::{connect, Connection};
use lancedb::arrow::arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder, StringBuilder};
use lancedb::arrow::arrow_array::types::{Float32Type, UInt32Type};
use lancedb::arrow::arrow_array::{Float32Array, RecordBatch, StringArray, UInt32Array};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::OptimizeAction;
use uuid::Uuid;

pub struct LanceDbStore {
    db: Connection,
    vector_dimension: i32,
}

impl LanceDbStore {
    pub async fn new(db_uri: &str, vector_dimension: i32) -> Result<Self, lancedb::Error> {
        let db = connect(db_uri).execute().await?;
        Ok(Self { db, vector_dimension })
    }

    fn get_schema(&self) -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document_uri", DataType::Utf8, false),
            Field::new("document_hash", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("pages", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.vector_dimension,
                ),
                false,
            ),
        ]))
    }

    async fn collect_results(
        mut stream: lancedb::arrow::SendableRecordBatchStream,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let mut results = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();

            let ids = batch.column_by_name("id").ok_or("Missing id")?.as_any().downcast_ref::<StringArray>().unwrap();
            let uris = batch.column_by_name("document_uri").ok_or("Missing uri")?.as_any().downcast_ref::<StringArray>().unwrap();
            let hashes = batch.column_by_name("document_hash").ok_or("Missing hash")?.as_any().downcast_ref::<StringArray>().unwrap();
            let indices = batch.column_by_name("chunk_index").ok_or("Missing index")?.as_any().downcast_ref::<UInt32Array>().unwrap();
            let texts = batch.column_by_name("text").ok_or("Missing text")?.as_any().downcast_ref::<StringArray>().unwrap();
            let pages_col = batch.column_by_name("pages").ok_or("Missing pages")?.as_any().downcast_ref::<StringArray>().unwrap();

            // Vector search populates `_distance`, FTS populates `_score`
            let scores = batch
                .column_by_name("_distance")
                .or_else(|| batch.column_by_name("_score"))
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..num_rows {
                let pages = pages_col
                    .value(i)
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<u32>().ok())
                    .collect();

                results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    document_uri: uris.value(i).to_string(),
                    document_hash: hashes.value(i).to_string(),
                    chunk_index: indices.value(i),
                    text: texts.value(i).to_string(),
                    pages,
                    score: scores.map(|s| s.value(i)).unwrap_or(0.0),
                });
            }
        }
        Ok(results)
    }
}

#[async_trait::async_trait]
impl VectorStore for LanceDbStore {
    async fn store_chunks(&self, table_name: &str, chunks: &[EmbeddedChunk]) -> Result<(), BoxError> {
        if chunks.is_empty() {
            return Ok(());
        }
        let schema = self.get_schema();

        let mut id_builder = StringBuilder::new();
        let mut uri_builder = StringBuilder::new();
        let mut hash_builder = StringBuilder::new();
        let mut chunk_index_builder = PrimitiveBuilder::<UInt32Type>::new();
        let mut text_builder = StringBuilder::new();
        let mut pages_builder = StringBuilder::new();

        let values_builder = PrimitiveBuilder::<Float32Type>::new();
        let mut vector_builder = FixedSizeListBuilder::new(values_builder, self.vector_dimension);

        for item in chunks {
            id_builder.append_value(Uuid::new_v4().to_string());
            uri_builder.append_value(item.chunk.document_uri.as_ref());
            hash_builder.append_value(item.chunk.document_hash.as_ref());
            chunk_index_builder.append_value(item.chunk.chunk_index as u32);
            text_builder.append_value(item.chunk.text.as_ref());

            let pages_str = item.chunk.pages.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",");
            pages_builder.append_value(pages_str);

            for &val in &item.vector {
                vector_builder.values().append_value(val);
            }
            vector_builder.append(true);
        }

        // Fixed order to match get_schema() exactly
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(uri_builder.finish()),
                Arc::new(hash_builder.finish()),
                Arc::new(chunk_index_builder.finish()),
                Arc::new(text_builder.finish()),
                Arc::new(pages_builder.finish()),
                Arc::new(vector_builder.finish()),
            ],
        )?;

        if self.db.table_names().execute().await?.contains(&table_name.to_string()) {
            let table = self.db.open_table(table_name).execute().await?;
            table.add(vec![batch]).execute().await?;
        } else {
            self.db.create_table(table_name, vec![batch]).execute().await?;
        }

        Ok(())
    }

    async fn search(
        &self,
        table_name: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let table = self.db.open_table(table_name).execute().await?;
        let stream = table
            .query()
            .nearest_to(query_vector)?
            .limit(limit)
            .execute()
            .await?;

        Self::collect_results(stream).await
    }

    async fn full_text_search(
        &self,
        table_name: &str,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let table = self.db.open_table(table_name).execute().await?;
        let fts_query = FullTextSearchQuery::new(query_text.to_owned());

        let stream = table
            .query()
            .full_text_search(fts_query)
            .limit(limit)
            .execute()
            .await?;

        Self::collect_results(stream).await
    }

    async fn check_file_status(
        &self,
        table_name: &str,
        file_uri: &str,
        new_hash: &str,
    ) -> Result<FileStatus, BoxError> {
        if !self.db.table_names().execute().await?.contains(&table_name.to_string()) {
            return Ok(FileStatus::New);
        }

        let table = self.db.open_table(table_name).execute().await?;
        let escaped_uri = file_uri.replace('\'', "''"); // Prevent SQL injection on single quotes
        let filter = format!("document_uri = '{}'", escaped_uri);

        let mut stream = table.query().only_if(filter).limit(1).execute().await?;

        if let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            if batch.num_rows() > 0 {
                let hash_column = batch.column_by_name("document_hash").ok_or("Missing document_hash")?;
                let hash_array = hash_column.as_any().downcast_ref::<StringArray>().ok_or("Invalid StringArray")?;

                return if hash_array.value(0) == new_hash {
                    Ok(FileStatus::Unchanged)
                } else {
                    Ok(FileStatus::Modified)
                };
            }
        }

        Ok(FileStatus::New)
    }

    async fn delete_by_uri(&self, table_name: &str, file_uri: &str) -> Result<(), BoxError> {
        if self.db.table_names().execute().await?.contains(&table_name.to_string()) {
            let table = self.db.open_table(table_name).execute().await?;
            let escaped_uri = file_uri.replace('\'', "''");
            table.delete(&format!("document_uri = '{}'", escaped_uri)).await?;
        }
        Ok(())
    }

    async fn optimize(&self, table_name: &str) -> Result<(), BoxError> {
        let table = self.db.open_table(table_name).execute().await?;
        table.optimize(OptimizeAction::All).await?;
        Ok(())
    }

    async fn ensure_indexes(&self, table_name: &str) -> Result<(), BoxError> {
        // 1. Check if the table exists
        if !self.db.table_names().execute().await?.contains(&table_name.to_string()) {
            tracing::info!("Table '{}' does not exist yet. Skipping index creation.", table_name);
            return Ok(());
        }

        let table = self.db.open_table(table_name).execute().await?;
        
        // 2. Check row count (None means count all rows)
        let row_count = table.count_rows(None).await?;
        if row_count == 0 {
            tracing::info!("Table '{}' is empty. Skipping index creation.", table_name);
            return Ok(());
        }
        
        // 3. Build BTree scalar index on 'document_uri'
        tracing::info!("Building BTree scalar index on 'document_uri' for table '{}'...", table_name);
        table.create_index(&["document_uri"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;

        // 4. Build BTree scalar index on 'chunk_index'
        tracing::info!("Building BTree scalar index on 'chunk_index' for table '{}'...", table_name);
        table.create_index(&["chunk_index"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;

        // 5. Build Full-Text Search (FTS) index on 'text'
        tracing::info!("Building FTS index on 'text' column for table '{}'...", table_name);
        table.create_index(&["text"], Index::FTS(Default::default()))
            .execute()
            .await?;
        
        // 6. Build IVF-PQ vector index only if row count exceeds the 100k threshold
        if row_count > 100_000 {
            let index_builder = IvfPqIndexBuilder::default();
            
            table.create_index(&["vector"], Index::IvfPq(index_builder))
                .execute()
                .await?;
                
            tracing::info!("IVF-PQ index successfully built on table '{}' ({} rows).", table_name, row_count);
        } else {
            tracing::info!(
                "Skipping IVF-PQ index creation for '{}'. Row count ({}) is below the 100,000 threshold.", 
                table_name, 
                row_count
            );
        }

        Ok(())
    }
}