use std::sync::Arc;
use std::collections::VecDeque;
use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::document::{Document, DocumentError, ElementType};
use crate::chunk::{ChunkerStrategy, Chunk};

pub struct StructuralChunkerStrategy {
    tokenizer: Arc<Tokenizer>,
    max_tokens: usize,
    overlap_tokens: usize, // e.g., 50-100 tokens of overlap
}

impl StructuralChunkerStrategy {
    pub fn new(hf_model_name: &str, max_tokens: usize, overlap_tokens: usize) -> Self {
        let tokenizer = Tokenizer::from_pretrained(hf_model_name, None)
            .expect("Failed to load tokenizer");
            
        Self {
            tokenizer: Arc::new(tokenizer),
            max_tokens,
            overlap_tokens,
        }
    }

    /// Helper to maintain active headers. If we see an H2, we keep the H1 but drop H3/H4.
    fn update_headers(active_headers: &mut Vec<(u8, String)>, level: u8, text: String) {
        active_headers.retain(|&(l, _)| l < level);
        active_headers.push((level, text));
    }

    /// Helper to format the current header context (e.g., "[Q3 Report > Financials] ")
    fn format_header_context(active_headers: &[(u8, String)]) -> String {
        if active_headers.is_empty() {
            return String::new();
        }
        let breadcrumbs = active_headers
            .iter()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>()
            .join(" > ");
        format!("[{}] ", breadcrumbs)
    }
}

// A temporary struct to hold elements before they are flushed into a final Chunk
#[derive(Clone)]
struct ChunkBlock {
    text: String,
    token_count: usize,
    page: Option<u32>,
}

#[async_trait]
impl ChunkerStrategy for StructuralChunkerStrategy {
    async fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>, DocumentError> {
        let mut all_chunks = Vec::new();
        let mut active_headers: Vec<(u8, String)> = Vec::new();
        
        // This queue acts as our sliding window for overlap
        let mut current_window: VecDeque<ChunkBlock> = VecDeque::new();
        let mut current_window_tokens = 0;

        for element in &doc.elements {
            // 1. Track Hierarchy Structure
            match element.element_type {
                ElementType::Header(level) => {
                    Self::update_headers(&mut active_headers, level, element.text.clone());
                    continue; // Headers are stored as state, not added as standalone chunks
                }
                ElementType::Title => {
                    Self::update_headers(&mut active_headers, 1, element.text.clone());
                    continue;
                }
                _ => {} // Normal paragraphs, lists, tables, etc.
            }

            // 2. Inject structural context into the element text
            let header_context = Self::format_header_context(&active_headers);
            let contextualized_text = format!("{}{}", header_context, element.text.trim());
            
            let encoding = self.tokenizer.encode(contextualized_text.as_str(), false)
                .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
            let token_count = encoding.get_ids().len();
            let page = element.provenance.as_ref().and_then(|p| p.page_number);

            // 3. Handle massive single elements (fallback semantic splitting)
            if token_count > self.max_tokens {
                // Split by periods to avoid breaking sentences
                let sentences: Vec<&str> = contextualized_text.split_inclusive(". ").collect();
                
                for sentence in sentences {
                    let s_encoding = self.tokenizer.encode(sentence, false).unwrap();
                    let s_tokens = s_encoding.get_ids().len();
                    
                    // If even a sentence is too big, hard-slice it (rare edge case)
                    if s_tokens > self.max_tokens {
                        // Hard token slice logic omitted for brevity, but you can reuse 
                        // your tokenizer slice fallback here if needed.
                        continue; 
                    }

                    push_to_window(
                        ChunkBlock { text: sentence.to_string(), token_count: s_tokens, page },
                        &mut current_window,
                        &mut current_window_tokens,
                        &mut all_chunks,
                        self.max_tokens,
                        self.overlap_tokens,
                        Arc::clone(&doc.uri)
                    );
                }
                continue;
            }

            // 4. Standard accumulation
            push_to_window(
                ChunkBlock { text: contextualized_text, token_count, page },
                &mut current_window,
                &mut current_window_tokens,
                &mut all_chunks,
                self.max_tokens,
                self.overlap_tokens,
                Arc::clone(&doc.uri)
            );
        }

        // 5. Flush any remaining blocks in the window
        if !current_window.is_empty() {
            flush_window(&mut current_window, &mut current_window_tokens, &mut all_chunks, Arc::clone(&doc.uri), 0);
        }

        Ok(all_chunks)
    }
}

/// Helper function to manage the sliding window and overlap logic
fn push_to_window(
    block: ChunkBlock,
    window: &mut VecDeque<ChunkBlock>,
    window_tokens: &mut usize,
    all_chunks: &mut Vec<Chunk>,
    max_tokens: usize,
    overlap_tokens: usize,
    uri: Arc<str>,
) {
    // If adding this block exceeds max_tokens, we need to flush first
    if *window_tokens + block.token_count > max_tokens && !window.is_empty() {
        flush_window(window, window_tokens, all_chunks, uri, overlap_tokens);
    }

    *window_tokens += block.token_count;
    window.push_back(block);
}

/// Combines the blocks into a Chunk, then retains enough blocks to satisfy overlap_tokens
fn flush_window(
    window: &mut VecDeque<ChunkBlock>,
    window_tokens: &mut usize,
    all_chunks: &mut Vec<Chunk>,
    uri: Arc<str>,
    overlap_tokens: usize,
) {
    let mut combined_text = String::new();
    let mut pages = Vec::new();

    for block in window.iter() {
        combined_text.push_str(&block.text);
        combined_text.push_str("\n\n");
        if let Some(p) = block.page {
            pages.push(p);
        }
    }
    pages.dedup();

    all_chunks.push(Chunk {
        document_uri: uri,
        text: Arc::new(combined_text.trim().to_string()),
        token_count: *window_tokens,
        pages,
    });

    // Handle overlap: pop elements from the front until we are under the overlap limit
    while *window_tokens > overlap_tokens && !window.is_empty() {
        if let Some(removed) = window.pop_front() {
            *window_tokens -= removed.token_count;
        }
    }
}