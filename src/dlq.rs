use std::fmt::Display;
use tokio::fs::File;
use tokio::io::{BufWriter, AsyncWriteExt};


pub struct ErrorPayload {
    pub node_name: String,
    pub error_message: String,
    pub raw_data: String, 
}

impl Display for ErrorPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.node_name, self.error_message, self.raw_data)
    }
}

pub struct DLQueue{
    pub writer: BufWriter<File>,
}

impl DLQueue {
    pub async fn new() -> Self {
        let file = File::create("errors.log").await.expect("Unable to create errors.log");
        let writer = BufWriter::new( file);
        Self {
            writer,
        }
    } 
    pub async fn write(&mut self, error_payload: ErrorPayload) {
        if let Err(e) = self.writer.write_all(format!("{}\n", error_payload).as_bytes()).await {
            eprintln!("CRITICAL DISK ERROR: Could not write to DLQ: {}", e);
        }
    }
}

