use std::collections::{ VecDeque};
use std::path::{Path, PathBuf};
use tokio::fs::{self, ReadDir};

use ferro_core::source::Source;
use super::{Payload, Record};
use std::sync::Arc;
use tracing::{ error};


pub struct FileSource {
    dir_stack: Vec<ReadDir>,
    pending_files: VecDeque<PathBuf>,
}

impl FileSource {
    pub async fn new(uri: impl Into<String>) -> std::io::Result<Self> {
        let uri_str = uri.into();

        // 1. Reject network schemes early with a clear error message
        if uri_str.starts_with("http://") || uri_str.starts_with("https://") || uri_str.starts_with("s3://") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("FileSource only supports local paths. Unsupported URI scheme: {}", uri_str)
            ));
        }

        // 2. Strip the formal "file://" prefix if the user included it
        let clean_path = uri_str.strip_prefix("file://").unwrap_or(&uri_str);
        
        let root_path = PathBuf::from(clean_path);
        let mut dir_stack = Vec::new();
        let mut pending_files = VecDeque::new();
        let metadata = fs::metadata(&root_path).await?;
        if metadata.is_file() { pending_files.push_back(root_path); }
        else if metadata.is_dir() { 
            dir_stack.push(fs::read_dir(root_path).await?);
        }
        Ok(Self { dir_stack, pending_files })
    }


    fn infer_mime_type(path: &Path) -> &'static str {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") => "application/pdf",
            Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            Some("md") => "text/markdown",
            Some("html") | Some("htm") => "text/html",
            Some("jsonl") | Some("ndjson") => "application/x-ndjson",
            Some("csv") => "text/csv",
            Some("json") => "application/json",
            Some("txt") => "text/plain",
            _ => "application/octet-stream",
        }
    }

}

impl Source for FileSource {
    type Item = Record;
    type Error = std::io::Error;

async fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        loop {

            if let Some(path) = self.pending_files.pop_front() {
                // Arc::from cleanly allocates the Arc<str>. 
                // to_string_lossy() is safer than unwrap() to prevent panics on invalid UTF-8 paths.
                let uri: Arc<str> = Arc::from(path.to_string_lossy().as_ref());
                let mime_type: Arc<str> = Arc::from(Self::infer_mime_type(&path)); 

                return Ok(Some(Record { 
                    uri, 
                    mime_type, 
                    payload: Payload::Pointer(path) 
                }));
            }

            let Some(mut current_dir) = self.dir_stack.pop() else {
                // Both queue and stack are empty: pipeline is complete.
                return Ok(None);
            };

            // Pull exactly one entry from the active directory iterator
            if let Some(entry) = current_dir.next_entry().await? {
                // Push the directory BACK onto the stack to process its remaining items later
                self.dir_stack.push(current_dir);
                
                let path = entry.path();
                let file_type = entry.file_type().await?;

                if file_type.is_dir() {
                    // Open the subdirectory and push it to the top of the stack
                    if let Ok(new_dir) = tokio::fs::read_dir(&path).await {
                        self.dir_stack.push(new_dir);
                    } else {
                        error!("Failed to open directory {}", path.display());
                        continue;
                    }
                } else if file_type.is_file() {
                    // Queue the file for processing on the next loop iteration
                    self.pending_files.push_back(path);
                }
            }
            // If next_entry() returns None, the directory is exhausted. 
            // We intentionally do not push it back, naturally dropping it.
        } 
    }
}