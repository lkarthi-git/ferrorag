use bytes::Bytes;
use std::path::PathBuf;
use std::sync::Arc;

pub mod file_source;
pub mod kafka_source;


pub use file_source::FileSource;

#[derive(Debug)]
pub enum Payload {
    Inline(Bytes),
    Pointer(PathBuf), 
}

#[derive(Debug)]
pub struct Record {
    pub uri: Arc<str>,
    pub mime_type: Arc<str>,
    pub payload: Payload
}