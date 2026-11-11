use ferro_core::node::Node;
use crate::store::{VectorStore, SearchResult};
use std::future::Future;
use std::collections::{HashMap, HashSet};
use super::{EmbeddedQuery, vector::VectorSearchNode};
use std::sync::Arc;

pub struct VectorAndParentSearchNode {
    search_node: VectorSearchNode,
    store: Arc<dyn VectorStore>,
    table_name: String,
    window_size: i32,
}

impl VectorAndParentSearchNode {
    pub fn new(
        search_node: VectorSearchNode,
        store: Arc<dyn VectorStore>,
        table_name: impl Into<String>,
        window_size: i32,
    ) -> Self {
        Self {
            search_node,
            store,
            table_name: table_name.into(),
            window_size,
        }
    }

    fn stitch_chunks(raw_rows: Vec<SearchResult>, original_candidates: &[SearchResult]) -> Vec<SearchResult> {
        let mut grouped: HashMap<String, Vec<SearchResult>> = HashMap::new();
        for row in raw_rows {
            grouped.entry(row.document_uri.clone()).or_default().push(row);
        }

        // 1. Memorize the correct ranking order from the original RRF hits
        let mut doc_order = Vec::new();
        let mut seen = HashSet::new();
        for candidate in original_candidates {
            if seen.insert(candidate.document_uri.clone()) {
                doc_order.push(candidate.document_uri.clone());
            }
        }

        // 2. Rebuild the stitched array in the correct rank order
        let mut stitched = Vec::with_capacity(doc_order.len());
        for uri in doc_order {
            if let Some(mut chunks) = grouped.remove(&uri) {
                // Sort chunks chronologically by index for readable text
                chunks.sort_by_key(|c| c.chunk_index);

                // Preserve the highest RRF score for this document
                let best_score = chunks.iter()
                    .filter_map(|c| Some(c.score))
                    .fold(f32::NEG_INFINITY, |a, b| a.max(b));

                let merged_text = chunks
                    .iter()
                    .map(|c| c.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");

                let mut representative = chunks.remove(0);
                representative.text = merged_text;
                
                if best_score > f32::NEG_INFINITY {
                    representative.score = best_score;
                }
                
                stitched.push(representative);
            }
        }

        stitched
    }
}

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

impl Node for VectorAndParentSearchNode {
    type Input = EmbeddedQuery;
    type Output = Vec<SearchResult>;
    type Error = BoxError;

    fn execute(
        &self,
        input: &Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        async move {
            // 1. Initial Vector Search 
            let candidates = self.search_node.execute(input).await?;

            if candidates.is_empty() {
                return Ok(vec![]);
            }

            // 2. Fetch the sliding window of surrounding chunks
            let expanded_rows = self.store.expand_context(
                &self.table_name, 
                &candidates, 
                self.window_size, 
                input.metadata.clone()
            ).await?;
            
            // 3. Stitch chunks while passing the candidates to maintain ranking
            let merged_results = Self::stitch_chunks(expanded_rows, &candidates);

            Ok(merged_results)
        }
    }
}