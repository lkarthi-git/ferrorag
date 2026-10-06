use tokio;
use ferro_rag::source::FileSource;
use ferro_rag::document::pdf::PdfInspectorStrategy;
use ferro_rag::document::DocumentParserNode;
use ferro_rag::chunk::ChunkerNode;
use ferro_rag::chunk::token_greedy::TokenGreedyStrategy;
use ferro_core::pipeline::Pipeline;
use ferro_rag::telemetry::init_telemetry;



#[tokio::main]
async fn main() {
    init_telemetry();
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("Please provide a file");
        std::process::exit(1);
    }
    let file_path: String = args[1].clone();
    let file = FileSource::new(file_path).await.unwrap();
    let parser = DocumentParserNode::new().with_strategy("application/pdf", PdfInspectorStrategy::new());
    let chunker = ChunkerNode::new(TokenGreedyStrategy::new("nomic-ai/nomic-embed-text-v1.5", 1000));


    let pipeline = Pipeline::from_source(file, ferro_core::dlq::NoOpDLQ, 100)
        .pipe(parser, 100)
        .pipe(chunker, 100);
    
    pipeline.for_each(|x| {
        println!("{:?}", x);
    }).await;
}