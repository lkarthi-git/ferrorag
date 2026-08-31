use std::env;
use tokio::fs::File;
use ferrorag::pipeline::{ Pipeline};
use ferrorag::LengthChunker;
use ferrorag::MockEmbedder;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file");
        std::process::exit(1);
    }
        let file_path: String = args[1].clone();
        let file: File = File::open(file_path).await.expect("file not found");
        let length_chunker = LengthChunker {
            chunk_size: 10_000,
            counter: 0,
            chunks: Vec::new(),
        };
        let mock_embedder = MockEmbedder {
            embeddings: Vec::new(),
        };
        Pipeline::ingest(file).pipe(length_chunker).pipe(mock_embedder).execute().await;
}