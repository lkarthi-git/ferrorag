use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc};

use crate::search::{EmbeddedQuery, vector::VectorSearchNode};
use crate::store::SearchResult;
use ferro_core::node::Node;
use crate::reranker_backend::RerankerBackend;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;


pub struct SearchAndRerankNode {
    search_node: Arc<VectorSearchNode>, // Wrapped in Arc to avoid requiring Clone on VectorSearchNode
    reranker: Arc<dyn RerankerBackend>,
    query_text_map: Arc<HashMap<String, String>>,
    final_top_k: usize,
}

impl SearchAndRerankNode {
    pub fn new(
        search_node: VectorSearchNode,
        reranker: Arc<dyn RerankerBackend>,
        query_text_map: Arc<HashMap<String, String>>,
        final_top_k: usize,
    ) -> Result<Self, BoxError> {
        Ok(Self {
            search_node: Arc::new(search_node),
            reranker,
            query_text_map,
            final_top_k,
        })
    }
}

impl Node for SearchAndRerankNode {
    type Input = EmbeddedQuery;
    type Output = Vec<SearchResult>;
    type Error = BoxError;

    fn execute(
        &self,
        input: &Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        // Clone Arc handles rather than requiring direct object clones
        let search_node = Arc::clone(&self.search_node);
        let input = input.clone();

        let reranker = self.reranker.clone();
        let query_text_map = self.query_text_map.clone();
        let final_top_k = self.final_top_k;

        async move {
            let mut results = search_node.execute(&input).await?;

            if results.is_empty() {
                return Ok(results);
            }

            let q_id = &results[0].query_metadata.query_id;
            let query_text = match query_text_map.get(q_id) {
                Some(text) => text.clone(),
                None => {
                    tracing::warn!("Query text missing for reranking query_id: {}", q_id);
                    results.truncate(final_top_k);
                    return Ok(results);
                }
            };

            let docs: Vec<String> = results.iter().map(|r| r.text.to_string()).collect();

            let scores = match reranker.predict_scores(&query_text, docs).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Reranking inference failed: {:?}", e);
                    results.truncate(final_top_k);
                    return Ok(results);
                }
            };

            if scores.len() != results.len() {
                tracing::error!(
                    "Reranker score count ({}) != result count ({})",
                    scores.len(),
                    results.len()
                );
                results.truncate(final_top_k);
                return Ok(results);
            }

            for (i, score) in scores.into_iter().enumerate() {
                results[i].score = score;
            }

            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.document_uri.cmp(&b.document_uri))
            });

            results.truncate(final_top_k);
            Ok(results)
        }
    }
}
