use std::sync::Arc;
use async_trait::async_trait;
use tokio::task;

use pdf_inspector::extract_text_with_positions;

use crate::source::{Payload, Record};
use super::{
    Document, DocumentMetadata, DocumentError, Element, ElementType, 
    ProvenanceData, BoundingBox, ParserStrategy
};

pub struct PdfInspectorStrategy{
    max_elements_per_document: usize,
}

impl PdfInspectorStrategy {
    pub fn new() -> Self {
        Self {
            max_elements_per_document: 100,
        }
    }

    pub fn with_max_elements_per_document(mut self, max_elements: usize) -> Self {
        self.max_elements_per_document = max_elements;
        self
    }
}

#[async_trait]
impl ParserStrategy for PdfInspectorStrategy {
    async fn parse(&self, record: &Record) -> Result<Vec<Document>, DocumentError> {
        let uri = record.uri.clone();
        let metadata = Arc::new(DocumentMetadata::default());
        let path = match &record.payload {
            Payload::Pointer(p) => p.clone(),
            Payload::Inline(_) => {
                return Err(DocumentError::ParseFailed(
                    "PdfInspectorStrategy requires Payload::Pointer".into()
                ));
            }
        };

        // Offload the CPU-heavy spatial parsing to a blocking thread
        let parsed_document = task::spawn_blocking(move || -> Result<Vec<Document>, DocumentError> {
            let positioned_items = extract_text_with_positions(&path)
                .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;

            let mut partitioned_documents = Vec::new();
            let mut elements = Vec::with_capacity(100);
            for item in positioned_items {
                let text = item.text.trim().to_string();
                if text.is_empty() {
                    continue;
                }

                // Use the natively extracted font metadata directly
                let element_type = if text.starts_with("•") || text.starts_with("-") || text.starts_with("*") {
                    ElementType::ListItem // Bullet lists detection
                } else if text.starts_with("Figure") || text.starts_with("Table") {
                    ElementType::Caption  // Caption prefix detection
                } else if item.is_bold {
                    // Bold formatting
                    ElementType::Header(3) 
                } else if item.font_size >= 18.0 { 
                    // Emulate the "font size tiers relative to body text" logic
                    ElementType::Header(1)
                } else if item.font_size >= 14.0 {
                    ElementType::Header(2)
                } else {
                    ElementType::Paragraph
                };

                let bbox = BoundingBox {
                    x_min: item.x,
                    y_min: item.y,
                    x_max: item.x + item.width,
                    y_max: item.y + item.height,
                };


                elements.push(Element {
                    element_type,
                    text,
                    provenance: Some(ProvenanceData {
                        page_number: Some(item.page as u32),
                        bounding_box: Some(bbox),
                        row_index: None,
                    }),
                });

                if elements.len() >= 100 {
                    partitioned_documents.push(Document {
                        uri: Arc::clone(&uri),
                        metadata: Arc::clone(&metadata), // Cheap Arc clone!
                        elements: std::mem::take(&mut elements), // Flushes and resets
                    });
                }
            }

            if !elements.is_empty() {
                partitioned_documents.push(Document {
                    uri: Arc::clone(&uri),
                    metadata: Arc::clone(&metadata),
                    elements: elements,
                });
            }

            Ok(partitioned_documents)
        }).await.map_err(|e| DocumentError::ParseFailed(format!("Task panicked: {}", e)))??;

        Ok(parsed_document)
    }
}
