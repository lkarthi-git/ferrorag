use ferro_rag::store;
use tokio;
use ferro_rag::source::FileSource;
use ferro_rag::document::pdf::PdfInspectorStrategy;
use ferro_rag::document::DocumentParserNode;
use ferro_rag::chunk::ChunkerNode;
use ferro_rag::embedding::LocalEmbeddingNode;
use ferro_rag::chunk::structural::StructuralChunkerStrategy;
use ferro_core::pipeline::Pipeline;
use ferro_core::node::ChunkNode;
use ferro_rag::telemetry::init_telemetry;
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
    let store = store::VectorStore::new("/Users/sensehawk/Documents/ferrogent/ferrogent/crates/rag-pipeline/data/lancedb", 768).await.unwrap();
    let vector_storage = store::VectorStorageNode::new(Arc::new(store), "chunks");
    let batch_node = ChunkNode::new(500);

    let pipeline = Pipeline::from_source(file, ferro_core::dlq::NoOpDLQ, 16)
        .pipe(parser, 16)
        .pipe(chunker, 16)
        .pipe(chunk_node, 16)
        .pipe(embedder, 16)
        .pipe(batch_node, 16)
        .pipe(vector_storage, 16);

    pipeline.for_each(|record| {
        println!("{:?}", record);
    }).await;
    tracing::info!("Pipeline finished. Metrics server remains open on port 9000.");
    tracing::info!("Press Ctrl+C to exit...");
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down.");
    Ok(())
}