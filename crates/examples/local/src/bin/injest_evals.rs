use std::sync::Arc;
use ferro_rag::{
    chunk::{ChunkerNode, structural::StructuralChunkerStrategy},
    document::{DocumentParserNode,json::JsonlParserStrategy},
    embedding::chunk::ChunkEmbeddingNode,
    embedding::late_chunk::LateChunkingNode,
    extraction::GraphExtractionNode,
    embedding_backend::{AdvancedEmbeddingBackend, candle::CandleEmbedder},
    source::FileSource,
    store::{self, VectorStore, dedup::DeduplicatorNode, vector_store_node::VectorStorageNode},
};
use ferro_core::{dlq::NoOpDLQ, node::ChunkNode, pipeline::Pipeline};
use local::telemetry::init_telemetry;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_telemetry();
    
    // Hardcoded to the downloaded SciFact corpus
    let file_path = "data/fiqa/corpus.jsonl".to_string();
    let file = FileSource::new(file_path).await.unwrap();
    
    // Parse JSON lines mapping "title" and "text" to your Document schema
    let parser = DocumentParserNode::new().with_strategy("application/x-ndjson", JsonlParserStrategy);
    let chunker = ChunkerNode::new(StructuralChunkerStrategy::new("nomic-ai/nomic-embed-text-v1.5", 8000, 256));
    
    let local_embedder = Arc::new(CandleEmbedder::nomic_v1_5().unwrap());
    let db_path = "data/lancedb_eval";
    let store = Arc::new(store::lancedb::LanceDbStore::new(db_path, local_embedder.vector_dimension() as i32).await?);
    
    // Reset table for clean eval runs
    let _ = store.drop_table("eval_chunks").await;

    let extraction_labels: Vec<&str> = vec![
        "company",
        "organization",
        "person",
        "product",
        "location",
        "metric",     // Important for financial data (e.g., "gross margin", "revenue")
        "currency",   // (e.g., "USD", "EUR")
    ];

    let entity_extractor = GraphExtractionNode::new(extraction_labels,Vec::new()).map_err(|e| e as Box<dyn std::error::Error>)?;
    let deduplicator_node = DeduplicatorNode::new(store.clone(), "eval_chunks", parser);
    let batch_node  = ChunkNode::new(4);
    let embedder = LateChunkingNode::new(local_embedder.clone(),"nomic-ai/nomic-embed-text-v1.5",256,50);
    // let batch_node_2  = ChunkNode::new(16);
    let batch_node_3 = ChunkNode::new(500);
    let vector_storage = VectorStorageNode::new(store.clone(), "eval_chunks");

    let pipeline = Pipeline::from_source(file, NoOpDLQ, 16)
        .pipe(deduplicator_node, 16)
        .pipe(chunker, 16)
        .pipe(batch_node, 16)
        // .pipe(entity_extractor, 16)
        // .pipe(batch_node_2, 16)
        .pipe(embedder, 16)
        .pipe(batch_node_3, 16)
        .pipe(vector_storage, 16);

    tracing::info!("Starting BEIR Ingestion Pipeline...");
    let summary = pipeline.execute().await;
    tracing::info!("Ingestion complete. Processed items: {}", summary.successful_items);

    tracing::info!("Building LanceDB Indexes...");
    if let Err(e) = store.ensure_indexes("eval_chunks").await {
        tracing::error!("Failed to create IVF-PQ index: {}", e);
    }
    tracing::info!("Ready for evaluation.");
    
    Ok(())
}