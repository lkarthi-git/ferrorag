use std::sync::Arc;
use std::collections::VecDeque;
use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::document::{Document, DocumentError, ElementType};
use crate::chunk::{ChunkerStrategy, Chunk,ChunkBlock};

pub struct StructuralChunkerStrategy {
    tokenizer: Arc<Tokenizer>,
    max_tokens: usize,
    overlap_tokens: usize,
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

    fn update_headers(active_headers: &mut Vec<(u8, String)>, level: u8, text: String) {
        active_headers.retain(|&(l, _)| l < level);
        active_headers.push((level, text));
    }

    fn format_header_context(active_headers: &[(u8, String)]) -> String {
        if active_headers.is_empty() {
            return String::new();
        }
        let breadcrumbs = active_headers
            .iter()
            .map(|(_, text)| text.trim())
            .collect::<Vec<_>>()
            .join(" - ");
        format!("[Context: {}]", breadcrumbs)
    }

    fn count_tokens(&self, text: &str) -> Result<usize, DocumentError> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
        Ok(encoding.get_ids().len())
    }

    /// Safely slices a massive token sequence into max_token pieces without dropping data
    fn slice_by_tokens(&self, text: &str, limit: usize) -> Result<Vec<(String, usize)>, DocumentError> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
        let ids = encoding.get_ids();
        let mut slices = Vec::new();

        for chunk_ids in ids.chunks(limit) {
            let decoded = self.tokenizer
                .decode(chunk_ids, true)
                .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
            slices.push((decoded, chunk_ids.len()));
        }
        Ok(slices)
    }
}

#[async_trait]
impl ChunkerStrategy for StructuralChunkerStrategy {
    async fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>, DocumentError> {
        let mut all_chunks = Vec::new();
        let mut active_headers: Vec<(u8, String)> = Vec::new();
        
        let mut current_window: VecDeque<ChunkBlock> = VecDeque::new();
        let mut current_window_tokens = 0;

        for element in &doc.elements {
            // 1. Track Hierarchy & Flush on Major Section Boundaries (H1/H2)
            match element.element_type {
                ElementType::Header(level) => {
                    if level <= 2 && !current_window.is_empty() {
                        // Cleanly flush before starting a new major section (0 overlap)
                        flush_window(
                            &mut current_window,
                            &mut current_window_tokens,
                            &mut all_chunks,
                            Arc::clone(&doc.uri),
                            0,
                            Arc::clone(&doc.hash),
                        );
                    }
                    Self::update_headers(&mut active_headers, level, element.text.clone());
                    continue;
                }
                ElementType::Title => {
                    Self::update_headers(&mut active_headers, 1, element.text.clone());
                    continue;
                }
                _ => {}
            }

            let raw_text = element.text.trim();
            if raw_text.is_empty() {
                continue;
            }

            let header_context = Self::format_header_context(&active_headers);
            let header_tokens = if header_context.is_empty() {
                0
            } else {
                self.count_tokens(&header_context)?
            };

            // Budget available for body text after accounting for the header once
            let effective_max = self.max_tokens.saturating_sub(header_tokens + 4).max(64);
            let token_count = self.count_tokens(raw_text)?;
            let page = element.provenance.as_ref().and_then(|p| p.page_number);

            // 2. Handle massive single elements (Tables, long paragraphs)
            if token_count > effective_max {
                let sentences: Vec<&str> = raw_text.split_inclusive(". ").collect();
                
                for sentence in sentences {
                    let s_tokens = self.count_tokens(sentence)?;
                    
                    // Fallback: Hard token slice if a single sentence/table exceeds effective_max
                    if s_tokens > effective_max {
                        for (sliced_text, sliced_len) in self.slice_by_tokens(sentence, effective_max)? {
                            push_to_window(
                                ChunkBlock {
                                    header_context: header_context.clone(),
                                    text: sliced_text,
                                    token_count: sliced_len,
                                    page,
                                },
                                &mut current_window,
                                &mut current_window_tokens,
                                &mut all_chunks,
                                effective_max,
                                self.overlap_tokens,
                                Arc::clone(&doc.uri),
                                Arc::clone(&doc.hash),
                            );
                        }
                        continue;
                    }

                    push_to_window(
                        ChunkBlock {
                            header_context: header_context.clone(),
                            text: sentence.to_string(),
                            token_count: s_tokens,
                            page,
                        },
                        &mut current_window,
                        &mut current_window_tokens,
                        &mut all_chunks,
                        effective_max,
                        self.overlap_tokens,
                        Arc::clone(&doc.uri),
                        Arc::clone(&doc.hash),
                    );
                }
                continue;
            }

            // 3. Standard accumulation
            push_to_window(
                ChunkBlock {
                    header_context,
                    text: raw_text.to_string(),
                    token_count,
                    page,
                },
                &mut current_window,
                &mut current_window_tokens,
                &mut all_chunks,
                effective_max,
                self.overlap_tokens,
                Arc::clone(&doc.uri),
                Arc::clone(&doc.hash),
            );
        }

        // 4. Final flush
        if !current_window.is_empty() {
            flush_window(
                &mut current_window,
                &mut current_window_tokens,
                &mut all_chunks,
                Arc::clone(&doc.uri),
                0,
                Arc::clone(&doc.hash),
            );
        }

        Ok(all_chunks)
    }
}

fn push_to_window(
    block: ChunkBlock,
    window: &mut VecDeque<ChunkBlock>,
    window_tokens: &mut usize,
    all_chunks: &mut Vec<Chunk>,
    max_tokens: usize,
    overlap_tokens: usize,
    uri: Arc<str>,
    hash: Arc<str>,
) {
    if *window_tokens + block.token_count > max_tokens && !window.is_empty() {
        flush_window(window, window_tokens, all_chunks, uri, overlap_tokens, hash);
    }

    *window_tokens += block.token_count;
    window.push_back(block);
}

fn flush_window(
    window: &mut VecDeque<ChunkBlock>,
    window_tokens: &mut usize,
    all_chunks: &mut Vec<Chunk>,
    uri: Arc<str>,
    overlap_tokens: usize,
    hash: Arc<str>,
) {
    let mut combined_text = String::new();
    let mut pages = Vec::new();
    let mut last_header: Option<&str> = None;

    for block in window.iter() {
        // Only emit the header context once at the top of the chunk (or if sub-header changed)
        if !block.header_context.is_empty() && last_header != Some(block.header_context.as_str()) {
            combined_text.push_str(&block.header_context);
            combined_text.push('\n');
            last_header = Some(&block.header_context);
        }

        combined_text.push_str(&block.text);
        combined_text.push_str("\n\n");
        if let Some(p) = block.page {
            pages.push(p);
        }
    }
    pages.dedup();

    let chunk_index = all_chunks.len() as u32;

    all_chunks.push(Chunk {
        document_uri: uri,
        document_hash: hash,
        chunk_index, // Sequential ID (0, 1, 2...)
        text: Arc::from(combined_text.trim().to_string()),
        token_count: *window_tokens,
        pages,
    });

    while *window_tokens > overlap_tokens && !window.is_empty() {
        if let Some(removed) = window.pop_front() {
            *window_tokens -= removed.token_count;
        }
    }
}