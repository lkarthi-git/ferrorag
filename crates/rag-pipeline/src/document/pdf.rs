use std::sync::Arc;
use async_trait::async_trait;
use tokio::task;
use sha2::{Sha256, Digest}; 
use std::fs;

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
        Self { max_elements_per_document: 100 }
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
            Payload::Inline(_) => return Err(DocumentError::ParseFailed("Requires Payload::Pointer".into())),
        };

        let max_elements = self.max_elements_per_document;

        let parsed_document = task::spawn_blocking(move || -> Result<Vec<Document>, DocumentError> {
            let file_bytes = fs::read(&path)
                .map_err(|e| DocumentError::ParseFailed(format!("Failed to read file: {}", e)))?;
            
            let mut hasher = Sha256::new();
            hasher.update(&file_bytes);
            let file_hash = hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect::<String>();

            let positioned_items = extract_text_with_positions(&path)
                .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;

            let mut partitioned_documents = Vec::new();
            let mut elements = Vec::with_capacity(max_elements);
            
            // Buffer to hold the active paragraph so we can append adjacent lines to it
            let mut current_element: Option<Element> = None;
            
            for item in positioned_items {
                let mut text = item.text.trim().to_string();

                // 1. FILTER: Skip lone page numbers or short numeric headers/footers
                if text.chars().all(|c| c.is_numeric() || c.is_ascii_punctuation()) && text.len() < 5 {
                    continue;
                }

                // 2. CLEAN: Strip stray bullet points so they don't sit on a line by themselves
                let is_list_item = text.starts_with("•") || text.starts_with("-") || text.starts_with("*");
                if is_list_item {
                    text = text.trim_start_matches(&['•', '-', '*', ' '][..]).to_string();
                }
                
                if text.is_empty() {
                    continue; // Skip if it was just a standalone bullet character or empty space
                }

                let element_type = if is_list_item {
                    ElementType::ListItem
                } else if text.starts_with("Figure") || text.starts_with("Table") {
                    ElementType::Caption
                } else if item.is_bold {
                    ElementType::Header(3) 
                } else if item.font_size >= 18.0 { 
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

                let new_element = Element {
                    element_type: element_type.clone(),
                    text: text.clone(),
                    provenance: Some(ProvenanceData {
                        page_number: Some(item.page as u32),
                        bounding_box: Some(bbox.clone()),
                        row_index: None,
                    }),
                };

                // 3. MERGE: Check if we should merge this line with the previous one
                if let Some(mut prev) = current_element.take() {
                    let prev_prov = prev.provenance.as_ref().unwrap();
                    let is_same_page = prev_prov.page_number == Some(item.page as u32);
                    
                    let is_mergeable_type = matches!((&prev.element_type, &new_element.element_type), 
                        (ElementType::Paragraph, ElementType::Paragraph) | 
                        (ElementType::ListItem, ElementType::Paragraph) 
                    );

                    // FIX: Compare baselines (y_min to y_min) instead of y_max
                    let prev_y = prev_prov.bounding_box.as_ref().unwrap().y_min;
                    let baseline_gap = (item.y - prev_y).abs(); 
                    
                    // A standard line break is usually 1.2x to 1.5x the font size.
                    // We use 2.5x to generously catch double-spaced lines, 
                    // ensuring sentences don't get broken in half.
                    let is_adjacent = baseline_gap < (item.font_size * 2.5);

                    if is_same_page && is_mergeable_type && is_adjacent {
                        // Merge them together!
                        prev.text.push(' ');
                        prev.text.push_str(&new_element.text);
                        
                        // Expand the bounding box to cover the new line
                        let mut prev_bbox = prev.provenance.as_mut().unwrap().bounding_box.as_mut().unwrap();
                        
                        // Safely expand the box regardless of coordinate direction
                        prev_bbox.y_max = prev_bbox.y_max.max(bbox.y_max); 
                        prev_bbox.y_min = prev_bbox.y_min.min(bbox.y_min); 
                        prev_bbox.x_max = prev_bbox.x_max.max(bbox.x_max); 
                        prev_bbox.x_min = prev_bbox.x_min.min(bbox.x_min); 

                        current_element = Some(prev);
                    } else {
                        // Not mergeable. Push the previous element to the array and start a new buffer
                        elements.push(prev);
                        current_element = Some(new_element);
                    }
                } else {
                    current_element = Some(new_element);
                }

                // 4. FLUSH to Document if we hit max_elements
                if elements.len() >= max_elements {
                    partitioned_documents.push(Document {
                        uri: Arc::clone(&uri),
                        metadata: Arc::clone(&metadata),
                        elements: std::mem::take(&mut elements),
                        hash: file_hash.clone(),
                    });
                }
            }

            // Flush the very last remaining element
            if let Some(last_element) = current_element {
                elements.push(last_element);
            }

            if !elements.is_empty() {
                partitioned_documents.push(Document {
                    uri: Arc::clone(&uri),
                    metadata: Arc::clone(&metadata),
                    elements, 
                    hash: file_hash,
                });
            }

            Ok(partitioned_documents)
        }).await.map_err(|e| DocumentError::ParseFailed(format!("Task panicked: {}", e)))??;

        Ok(parsed_document)
    }
}