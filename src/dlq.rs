use std::fmt::Display;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::io::{BufWriter, AsyncWriteExt};

const PATH:&str = "errors.log";

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
    pub async fn new() ->   Result<Self, std::io::Error> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(PATH)
            .await?;
        Ok(Self { writer: BufWriter::new(file) })

    } 
    pub async fn write(&mut self, error_payload: ErrorPayload) -> Result<(), std::io::Error> {
        self.writer.write_all(format!("{}\n", error_payload).as_bytes()).await?;
        self.writer.flush().await
    }
    pub async fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush().await
    }
}

