use std::sync::Arc;
use futures::StreamExt;
use super::{BoxError, FileStatus, SearchResult, VectorStore};
use crate::search::QueryMetadata;
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
use lancedb::arrow::arrow_array::Array;
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
            Field::new("token_count", DataType::UInt32, false), 
            Field::new("text", DataType::Utf8, false),
            Field::new("pages", DataType::Utf8, false),
            Field::new("entities", DataType::Utf8, true),       
            Field::new("triplets", DataType::Utf8, true),       
            Field::new("chunk_attributes", DataType::Utf8, true), 
            Field::new("metadata", DataType::Utf8, true),       
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
        query_metadata: Arc<QueryMetadata>,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let mut results = Vec::new();

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let num_rows = batch.num_rows();

            let ids = batch.column_by_name("id").ok_or("Missing id")?.as_any().downcast_ref::<StringArray>().unwrap();
            let uris = batch.column_by_name("document_uri").ok_or("Missing uri")?.as_any().downcast_ref::<StringArray>().unwrap();
            let hashes = batch.column_by_name("document_hash").ok_or("Missing hash")?.as_any().downcast_ref::<StringArray>().unwrap();
            let indices = batch.column_by_name("chunk_index").ok_or("Missing index")?.as_any().downcast_ref::<UInt32Array>().unwrap();
            let token_counts = batch.column_by_name("token_count").ok_or("Missing token_count")?.as_any().downcast_ref::<UInt32Array>().unwrap();
            let texts = batch.column_by_name("text").ok_or("Missing text")?.as_any().downcast_ref::<StringArray>().unwrap();
            let pages_col = batch.column_by_name("pages").ok_or("Missing pages")?.as_any().downcast_ref::<StringArray>().unwrap();
            
            // JSON Columns (These are nullable in the schema)
            let entities_col = batch.column_by_name("entities").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let triplets_col = batch.column_by_name("triplets").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let attributes_col = batch.column_by_name("chunk_attributes").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let metadata_col = batch.column_by_name("metadata").and_then(|c| c.as_any().downcast_ref::<StringArray>());

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

                // Safely deserialize JSON or fall back to defaults
                let entities = entities_col
                    .filter(|c| c.is_valid(i))
                    .and_then(|c| serde_json::from_str(c.value(i)).ok())
                    .unwrap_or_default();

                let triplets = triplets_col
                    .filter(|c| c.is_valid(i))
                    .and_then(|c| serde_json::from_str(c.value(i)).ok())
                    .unwrap_or_default();

                let chunk_attributes = attributes_col
                    .filter(|c| c.is_valid(i))
                    .and_then(|c| serde_json::from_str(c.value(i)).ok())
                    .unwrap_or_default();

                let metadata = metadata_col
                    .filter(|c| c.is_valid(i))
                    .and_then(|c| serde_json::from_str(c.value(i)).ok())
                    .unwrap_or_default();

                results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    document_uri: uris.value(i).to_string(),
                    document_hash: hashes.value(i).to_string(),
                    chunk_index: indices.value(i),
                    token_count: token_counts.value(i) as usize,
                    text: texts.value(i).to_string(),
                    pages,
                    entities,
                    triplets,
                    chunk_attributes,
                    metadata,
                    score: scores.map(|s| s.value(i)).unwrap_or(0.0),
                    query_metadata: query_metadata.clone(),
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
        let mut token_count_builder = PrimitiveBuilder::<UInt32Type>::new();
        let values_builder = PrimitiveBuilder::<Float32Type>::new();
        let mut vector_builder = FixedSizeListBuilder::new(values_builder, self.vector_dimension);

        let mut entities_builder = StringBuilder::new();
        let mut triplets_builder = StringBuilder::new();
        let mut attributes_builder = StringBuilder::new();
        let mut metadata_builder = StringBuilder::new();

        for item in chunks {
            id_builder.append_value(Uuid::new_v4().to_string());
            uri_builder.append_value(item.chunk.document_uri.as_ref());
            hash_builder.append_value(item.chunk.document_hash.as_ref());
            chunk_index_builder.append_value(item.chunk.chunk_index as u32);
            token_count_builder.append_value(item.chunk.token_count as u32);
            text_builder.append_value(item.chunk.text.as_ref());

            let pages_str = item.chunk.pages.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",");
            pages_builder.append_value(pages_str);

            entities_builder.append_value(serde_json::to_string(&item.chunk.entities).unwrap_or_else(|_| "[]".to_string()));
            triplets_builder.append_value(serde_json::to_string(&item.chunk.triplets).unwrap_or_else(|_| "[]".to_string()));
            attributes_builder.append_value(serde_json::to_string(&item.chunk.chunk_attributes).unwrap_or_else(|_| "{}".to_string()));
            metadata_builder.append_value(serde_json::to_string(&*item.chunk.metadata).unwrap_or_else(|_| "{}".to_string()));

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
                Arc::new(token_count_builder.finish()),
                Arc::new(text_builder.finish()),
                Arc::new(pages_builder.finish()),
                Arc::new(entities_builder.finish()),
                Arc::new(triplets_builder.finish()),
                Arc::new(attributes_builder.finish()),
                Arc::new(metadata_builder.finish()),
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

    async fn drop_table(&self,table_name: &str) -> Result<(),BoxError>{
       let _ = self.db.drop_table(table_name,&[]).await?;
       Ok(())
    }

    async fn search(
        &self,
        table_name: &str,
        query_vector: &[f32],
        limit: usize,
        query_metadata: Arc<QueryMetadata>,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let table = self.db.open_table(table_name).execute().await?;

        let mut query = table.query().nearest_to(query_vector)?;

        // Apply the dynamic enum filter
        if let Some(filter) = &query_metadata.filter {
            let sql_filter = filter.to_sql();
            query = query.only_if(sql_filter);
        }

        let stream = query.limit(limit).execute().await?;

        Self::collect_results(stream,query_metadata).await
    }

    async fn full_text_search(
        &self,
        table_name: &str,
        query_text: &str,
        limit: usize,
        query_metadata: Arc<QueryMetadata>,
    ) -> Result<Vec<SearchResult>, BoxError> {
        let table = self.db.open_table(table_name).execute().await?;
        let fts_query = FullTextSearchQuery::new(query_text.to_owned());
        let mut query = table
            .query()
            .full_text_search(fts_query);

        if let Some(filter) = &query_metadata.filter {
            let sql_filter = filter.to_sql();
            query = query.only_if(sql_filter);
        }

        let stream = query.limit(limit).execute().await?;

        Self::collect_results(stream, query_metadata).await
    }

    async fn expand_context(
        &self,
        table_name: &str,
        candidates: &[SearchResult],
        window_size: i32,
        query_metadata: Arc<QueryMetadata>,
    ) -> Result<Vec<SearchResult>, BoxError> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // 1. Group by document_uri and compute the combined min/max window
        let mut doc_bounds: std::collections::HashMap<String, (i32, i32)> = std::collections::HashMap::new();
        
        for candidate in candidates {
            let uri = candidate.document_uri.clone();
            let min_idx = 0.max(candidate.chunk_index as i32 - window_size) as i32;
            let max_idx = candidate.chunk_index as i32 + window_size as i32;

            doc_bounds.entry(uri)
                .and_modify(|(curr_min, curr_max)| {
                    *curr_min = (*curr_min).min(min_idx);
                    *curr_max = (*curr_max).max(max_idx);
                })
                .or_insert((min_idx, max_idx));
        }

        // 2. Build the batched SQL filter using DataFusion syntax
        // Result looks like: "(document_uri = 'A' AND chunk_index >= 1 AND chunk_index <= 3) OR (document_uri = 'B'...)"
        let mut predicates = Vec::with_capacity(doc_bounds.len());
        for (uri, (min_idx, max_idx)) in doc_bounds {
            predicates.push(format!(
                "(document_uri = '{}' AND chunk_index >= {} AND chunk_index <= {})",
                uri, min_idx, max_idx
            ));
        }
        let bounds_sql = predicates.join(" OR ");

        // 3. Merge the bounds SQL with the MetadataFilter (Security / Context preservation)
        let final_sql = if let Some(filter) = &query_metadata.filter {
            format!("({}) AND ({})", bounds_sql, filter.to_sql())
        } else {
            bounds_sql
        };

        // 3. Execute exactly ONE query against LanceDB
        let table = self.db.open_table(table_name).execute().await?;
        let stream = table
            .query()
            .only_if(final_sql)
            .execute()
            .await?;

        // 4. Collect the raw expanded rows
        Self::collect_results(stream, query_metadata).await
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

        // 5. Build BTree scalar index on 'token_count'
        // Allows lightning-fast context window filtering (e.g., token_count < 500)
        tracing::info!("Building BTree scalar index on 'token_count' for table '{}'...", table_name);
        table.create_index(&["token_count"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;

        // 6. Build Full-Text Search (FTS) index on 'text'
        tracing::info!("Building FTS index on 'text' column for table '{}'...", table_name);
        table.create_index(&["text"], Index::FTS(Default::default()))
            .execute()
            .await?;
        
        // 7. Build IVF-PQ vector index only if row count exceeds the 100k threshold
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