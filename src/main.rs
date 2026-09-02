use std::env;
use tokio::fs::File;
use ferrorag::pipeline::{ Pipeline};
use ferrorag::source::FileSource;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file");
        std::process::exit(1);
    }
        let file_path: String = args[1].clone();
        let file: File = File::open(file_path).await.expect("file not found");
        let file_source = FileSource::new(file);
        Pipeline::from_source(file_source);
}