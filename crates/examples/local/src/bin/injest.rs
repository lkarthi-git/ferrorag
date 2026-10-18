use ferro_rag::store::{self, VectorStore};
use tokio;
use ferro_rag::source::FileSource;
use ferro_rag::document::pdf::PdfInspectorStrategy;
use ferro_rag::document::DocumentParserNode;
use ferro_rag::chunk::ChunkerNode;
use ferro_rag::embedding::LocalEmbeddingNode;
use ferro_rag::chunk::structural::StructuralChunkerStrategy;
use ferro_core::pipeline::Pipeline;
use ferro_core::node::ChunkNode;
use local::telemetry::init_telemetry;
use ferro_rag::store::dedup::DeduplicatorNode;
use std::sync::{Arc};



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_telemetry();
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("Please provide a file");
        std::process::exit(1);
    }
    let file_path: String = args[1].clone();
    let file = FileSource::new(file_path).await.unwrap();
    let parser = DocumentParserNode::new().with_strategy("application/pdf", PdfInspectorStrategy::new());
    let chunker = ChunkerNode::new(StructuralChunkerStrategy::new("nomic-ai/nomic-embed-text-v1.5", 1000,50));
    let chunk_node = ChunkNode::new(2);
    let embedder = LocalEmbeddingNode::new().unwrap();
    let store = Arc::new(store::lancedb::LanceDbStore::new("/Users/sensehawk/Documents/ferrogent/ferrogent/data/lancedb", 768).await?);
    let deduplicator_node = DeduplicatorNode::new(store.clone(), "chunks",parser);
    let vector_storage = store::vector_store_node::VectorStorageNode::new(store.clone(), "chunks");
    let batch_node = ChunkNode::new(500);

    let pipeline: Pipeline<ferro_rag::chunk::EmbeddedChunk> = Pipeline::from_source(file, ferro_core::dlq::NoOpDLQ, 16)
        .pipe(deduplicator_node, 16)
        .pipe(chunker, 16)
        .pipe(chunk_node, 16)
        .pipe(embedder, 16)
        .pipe(batch_node, 16)
        .pipe(vector_storage, 16);

    pipeline.execute().await;
    tracing::info!("Evaluating vector index threshold...");
    if let Err(e) = store.ensure_indexes("chunks").await {
        tracing::error!("Failed to create/update IVF-PQ index: {}", e);
    }
    tracing::info!("Optimizing table...");
    if let Err(e) = store.optimize("chunks").await {
        tracing::error!("Failed to optimize table: {}", e);
    }
    tracing::info!("Pipeline finished. Metrics server remains open on port 9000.");
    tracing::info!("Press Ctrl+C to exit...");
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down.");
    Ok(())
}