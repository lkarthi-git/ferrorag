use std::error::Error;
use crate::chunk::EmbeddedChunk;
use std::collections::HashMap;

#[cfg(feature = "lance-db")]
pub mod lancedb;

pub mod dedup;
pub mod vector_store_node;

// Standard thread-safe boxed error for async traits
pub type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    New,       // URI not found, process normally
    Unchanged, // URI found & hash matches, skip entirely
    Modified,  // URI found & hash differs, delete old chunks and re-process
}

/// Backend-agnostic search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub document_uri: String,
    pub document_hash: String,
    pub chunk_index: u32,
    pub text: String,
    pub pages: Vec<u32>,
    pub score: f32, // Distance or similarity score
}

#[async_trait::async_trait]
pub trait VectorStore: Send + Sync {
    async fn store_chunks(&self, collection: &str, chunks: &[EmbeddedChunk]) -> Result<(), BoxError>;

    async fn search(
        &self,
        collection: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, BoxError>;

    async fn full_text_search(
        &self,
        collection: &str,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, BoxError>;

    /// Hybrid Search (Vector + FTS combined via Reciprocal Rank Fusion)
    async fn hybrid_search(
        &self,
        collection: &str,
        query_text: &str,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, BoxError> {
        // Fetch 2x limit from each retriever concurrently, then fuse with RRF
        let fetch_k = limit * 2;
        let (vec_res, fts_res) = tokio::try_join!(
            self.search(collection, query_vector, fetch_k),
            self.full_text_search(collection, query_text, fetch_k)
        )?;
        Ok(reciprocal_rank_fusion(vec_res, fts_res, limit, 60.0))
    }

    async fn check_file_status(
        &self,
        collection: &str,
        file_uri: &str,
        new_hash: &str,
    ) -> Result<FileStatus, BoxError>;

    async fn delete_by_uri(&self, collection: &str, file_uri: &str) -> Result<(), BoxError>;

    /// Optional maintenance hooks: Default to no-op for managed cloud DBs like Supabase
    async fn optimize(&self, _collection: &str) -> Result<(), BoxError> {
        Ok(())
    }

    async fn ensure_indexes(&self, _collection: &str) -> Result<(), BoxError> {
        Ok(())
    }
}


pub fn reciprocal_rank_fusion(
    vector_results: Vec<SearchResult>,
    fts_results: Vec<SearchResult>,
    limit: usize,
    k: f32, // Usually 60.0
) -> Vec<SearchResult> {
    let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

    for (rank, item) in vector_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank as f32) + 1.0);
        scores.insert(item.id.clone(), (rrf_score, item));
    }

    for (rank, item) in fts_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank as f32) + 1.0);
        scores
            .entry(item.id.clone())
            .and_modify(|(score, _)| *score += rrf_score)
            .or_insert((rrf_score, item));
    }

    let mut fused: Vec<SearchResult> = scores
        .into_values()
        .map(|(score, mut item)| {
            item.score = score;
            item
        })
        .collect();

    // Sort highest RRF score first
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);
    fused
}