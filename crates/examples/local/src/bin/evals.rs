use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ferro_rag::embedding_backend::AdvancedEmbeddingBackend;
use ferro_rag::embedding_backend::candle::CandleEmbedder;
use ferro_rag::{
    embedding::query::QueryEmbeddingNode,
    search::BaseSearchNode,
    search::parent::VectorAndParentSearchNode,
    search::rerank::{CandleReranker, SearchAndRerankNode},
    search::{SearchMode, vector::VectorSearchNode},
    source::memory_source::MemorySource,
    store::{self, SearchResult},
};

use ferro_core::{dlq::NoOpDLQ, pipeline::Pipeline};
use local::telemetry::init_telemetry;

use local::rag_evals::{
    BeirDatasetLoader, CsvReporter, DatasetLoader, Evaluator, ResultReporter,
    StandardDocumentEvaluator,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_telemetry();

    // 1. Load Data via Trait
    tracing::info!("Loading BEIR dataset...");
    let dataset = BeirDatasetLoader::new("data/fiqa/qrels/test.tsv", "data/fiqa/queries.jsonl");
    let (queries, ground_truth, arc_query_text_map) = dataset.load()?;
    tracing::info!(
        "Loaded {} ground truth targets and filtered test queries.",
        ground_truth.len()
    );

    // 2. Initialize Pipeline Architecture
    let local_embedder = Arc::new(CandleEmbedder::nomic_v1_5().unwrap());
    let db_path = "data/lancedb_eval";
    let store = Arc::new(
        store::lancedb::LanceDbStore::new(db_path, local_embedder.vector_dimension() as i32)
            .await?,
    );

    // 3. Initialize Evaluator Components
    let evaluator = StandardDocumentEvaluator::new(5);
    let bge_reranker_model = Arc::new(CandleReranker::new("BAAI/bge-reranker-base", "main")?);

    let k_values = [10.0, 30.0, 60.0, 100.0, 150.0];
    for rrf_k in k_values {
        let eval_source = MemorySource::new(queries.clone(), Duration::ZERO);
        let query_node = QueryEmbeddingNode::new(local_embedder.clone());

        let search_node =
            VectorSearchNode::new(store.clone(), "eval_chunks", SearchMode::Hybrid { rrf_k });

        // let search_and_rerank_node = SearchAndRerankNode::new(
        //     search_node,
        //     bge_reranker_model.clone(),
        //     arc_query_text_map.clone(),
        //     10,
        // )?;

        let parent_search_node = VectorAndParentSearchNode::new(
            BaseSearchNode::Vector(search_node),
            store.clone(),
            "eval_chunks",
            2,
        );

        // let parent_search_node = VectorAndParentSearchNode::new(
        //     BaseSearchNode::SearchAndRerank(search_and_rerank_node),
        //     store.clone(),
        //     "eval_chunks",
        //     2
        // );
        let eval_pipeline = Pipeline::from_source(eval_source, NoOpDLQ, 16)
            .pipe(query_node, 16)
            .pipe(parent_search_node, 16);

        let mut grouped_results: HashMap<String, Vec<SearchResult>> = HashMap::new();
        tracing::info!("Executing evaluation pipeline for RRF_K={}...", rrf_k);

        // Run Pipeline
        let _eval_summary = eval_pipeline
            .for_each(|result: SearchResult| {
                let q_id = result.query_metadata.query_id.clone();
                grouped_results.entry(q_id).or_default().push(result);
            })
            .await;

        // Initialize Reporter
        let mut reporter = CsvReporter::new(&format!("evaluation_results_k{}.csv", rrf_k))?;

        let mut total_evaluated = 0;
        let mut hits_at_k = 0;
        let mut mrr_sum = 0.0;
        let mut ndcg_sum = 0.0;

        // Evaluate and Report via Traits
        for (q_id, results) in grouped_results {
            if let Some(expected_docs) = ground_truth.get(&q_id) {
                total_evaluated += 1;

                let metrics = evaluator.evaluate_query(&results, expected_docs);

                if metrics.hit_at_k {
                    hits_at_k += 1;
                }
                mrr_sum += metrics.reciprocal_rank;
                ndcg_sum += metrics.ndcg;

                let query_text = arc_query_text_map
                    .get(&q_id)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                reporter.report(&q_id, query_text, &metrics)?;
            }
        }

        reporter.flush()?;

        let hit_rate = (hits_at_k as f64 / total_evaluated as f64) * 100.0;
        let mrr = mrr_sum / total_evaluated as f64;
        let ndcg = if total_evaluated > 0 {
            ndcg_sum / total_evaluated as f64
        } else {
            0.0
        };

        println!(
            "RRF k={:<5} | Hit Rate @ 5: {:.2}% | MRR: {:.4} | NDCG: {:.4}",
            rrf_k, hit_rate, mrr, ndcg
        );
    }

    Ok(())
}
