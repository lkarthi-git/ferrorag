use async_trait::async_trait;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use crate::source::{Payload, Record};
use super::{
    Document, DocumentMetadata, DocumentError, Element, ElementType, ParserStrategy
};
use serde_json::Value;

pub struct JsonlParserStrategy;

#[async_trait]
impl ParserStrategy for JsonlParserStrategy {
    async fn parse(&self, record: &Record) -> Result<Vec<Document>, DocumentError> {
        let Payload::Pointer(path) = &record.payload else {
            return Err(DocumentError::ParseFailed("Expected file pointer for JSONL".into()));
        };

        let file = File::open(path).await.map_err(DocumentError::Io)?;
        let mut lines = BufReader::new(file).lines();
        let mut documents = Vec::new();
        let mut line_number = 0;

        let default_metadata = Arc::new(DocumentMetadata::default());

        while let Some(line) = lines.next_line().await.map_err(DocumentError::Io)? {
            line_number += 1;
            if line.trim().is_empty() { 
                continue; 
            }

            // 1. Deserialize the JSON line
            let parsed_json: Value = serde_json::from_str(&line)
                .unwrap_or_else(|_| Value::String(line.clone()));

            let mut elements = Vec::new();
            let mut doc_id = format!("{}#line={}", record.uri, line_number);

            // 2. Map standard BEIR fields to your Structural Elements
            if let Value::Object(map) = &parsed_json {
                // Override the URI if the dataset provides a distinct document ID
                if let Some(id) = map.get("_id").and_then(|v| v.as_str()) {
                    doc_id = format!("{}#id={}", record.uri, id);
                }

                // If a title exists, make it a Title element so your chunker catches it
                if let Some(title) = map.get("title").and_then(|v| v.as_str()) {
                    if !title.is_empty() {
                        elements.push(Element {
                            element_type: ElementType::Title, // Triggers your update_headers()
                            text: title.to_string(),
                            provenance: None,
                        });
                    }
                }

                // Extract the main body text
                if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                    elements.push(Element {
                        element_type: ElementType::Paragraph,
                        text: text.to_string(),
                        provenance: None,
                    });
                } else {
                    // Fallback if no "text" key exists
                    elements.push(Element {
                        element_type: ElementType::Paragraph,
                        text: line, 
                        provenance: None,
                    });
                }
            } else {
                 elements.push(Element {
                    element_type: ElementType::Paragraph,
                    text: line, 
                    provenance: None,
                });
            }

            // 3. Construct the document
            documents.push(Document {
                uri: Arc::from(doc_id),
                metadata: default_metadata.clone(),
                elements,
                hash: record.hash.clone(), 
            });
        }

        Ok(documents)
    }
}