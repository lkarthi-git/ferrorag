use ferro_core::pipeline::Pipeline;
use ferro_core::dlq::NoOpDLQ;
use ferro_rag::source::memory_source::MemorySource;
use ferro_rag::store::{self, VectorStore};
use ferro_rag::embedding::query::QueryEmbeddingNode;
use ferro_rag::embedding_backend::candle::CandleEmbedder;
use ferro_rag::embedding_backend::AdvancedEmbeddingBackend;
use ferro_rag::search::{Query,SearchMode,QueryMetadata};
use ferro_rag::search::vector::VectorSearchNode;
use ferro_rag::search::print::PrintResultsNode;
use std::time::Duration;

use local::telemetry::init_telemetry;
use std::sync::Arc;

// ========================================================================
// 3. Main Execution
// ========================================================================
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_telemetry();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    
    if args.is_empty() {
        eprintln!("Usage: cargo run -p local --bin search [--mode vector|fulltext|hybrid] \"<your search query>\"");
        std::process::exit(1);
    }

    // 2. Parse the --mode flag if it exists
    let mut mode = SearchMode::Vector; // Default mode
    if let Some(pos) = args.iter().position(|a| a == "--mode") {
        if pos + 1 < args.len() {
            let mode_str = &args[pos + 1];
            mode = match mode_str.to_lowercase().as_str() {
                "fulltext" => SearchMode::FullText,
                "hybrid" => SearchMode::Hybrid { rrf_k: 60.0 },
                "vector" => SearchMode::Vector,
                _ => {
                    tracing::warn!("Unknown mode '{}', defaulting to vector", mode_str);
                    SearchMode::Vector
                }
            };
            // Remove the flag and its value from the arguments so it doesn't get mixed into the query
            args.remove(pos); 
            args.remove(pos); 
        }
    }

    // 3. Join the remaining arguments as the actual search query
    let query_text = args.join(" ");
    if query_text.trim().is_empty() {
        eprintln!("Error: Search query cannot be empty.");
        std::process::exit(1);
    }

    tracing::info!("Searching for: '{}' using mode: {:?}", query_text, mode);

    // 1. Initialize Backend & Extract Dimension
    let local_embedder = Arc::new(CandleEmbedder::jina_v2_base_en().unwrap());
    let dimension = local_embedder.vector_dimension();

    // 2. Initialize Store using dynamic dimension
    let db_path = "/Users/sensehawk/Documents/ferrogent/ferrogent/data/lancedb";
    let store: Arc<dyn VectorStore> = Arc::new(
        store::lancedb::LanceDbStore::new(db_path, dimension as i32).await?
    );

    // 3. Initialize Nodes
    let query_node = QueryEmbeddingNode::new(local_embedder.clone());
    let search_node = VectorSearchNode::new(store, "chunks", mode);
    let print_node = PrintResultsNode;
    let metadata: QueryMetadata = QueryMetadata::default();
    // 4. Memory Source
    let source_queries = vec![Query {
        text: query_text,
        limit: 5,
        metadata: Arc::new(metadata),
    }]; 
    let memory_source = MemorySource::new(source_queries, Duration::ZERO);
    // 5. Execute Pipeline
    let pipeline = Pipeline::from_source(memory_source, NoOpDLQ, 1)
        .pipe(query_node, 4)
        .pipe(search_node, 4)
        .pipe(print_node, 1); 

    pipeline.execute().await;
    
    Ok(())
}