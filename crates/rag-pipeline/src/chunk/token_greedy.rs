use std::sync::Arc;
use async_trait::async_trait;
use tokenizers::Tokenizer;

use crate::document::{Document, DocumentError, ElementType};
use crate::chunk::{ChunkerStrategy, Chunk};

pub struct TokenGreedyStrategy {
    tokenizer: Arc<Tokenizer>,
    max_tokens: usize,
}

impl TokenGreedyStrategy {
    /// Initialize with the Hugging Face model ID matching your fastembed model.
    /// Example: "BAAI/bge-small-en-v1.5" or "nomic-ai/nomic-embed-text-v1.5"
    pub fn new(hf_model_name: &str, max_tokens: usize) -> Self {
        // Automatically fetches the exact tokenizer rules for your chosen model
        let tokenizer = Tokenizer::from_pretrained(hf_model_name, None)
            .expect("Failed to download or load tokenizer from Hugging Face");
            
        Self {
            tokenizer: Arc::new(tokenizer),
            max_tokens,
        }
    }
}

#[async_trait]
impl ChunkerStrategy for TokenGreedyStrategy {
    async fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>, DocumentError> {
        let mut all_chunks = Vec::new();

            let mut current_chunk_text = String::new();
            let mut current_chunk_tokens = 0;
            let mut current_pages = Vec::new();
            
            for element in &doc.elements {
                if matches!(element.element_type, ElementType::Title | ElementType::Header(_)) {
                    continue; 
                }

                // 2. Encode text using the Hugging Face Tokenizer
                // The `false` flag tells it NOT to add special tokens (like [CLS] or [SEP]) yet,
                // which is exactly what we want for raw length counting.
                let encoding = self.tokenizer.encode(element.text.as_str(), false)
                    .map_err(|e| DocumentError::ParseFailed(e.to_string()))?;
                
                let element_ids = encoding.get_ids();
                let element_token_count = element_ids.len();

                // 3. Flush condition
                if current_chunk_tokens > 0 && (current_chunk_tokens + element_token_count > self.max_tokens) {
                    current_pages.dedup();
                    all_chunks.push(Chunk {
                        document_uri: Arc::clone(&doc.uri),
                        text: current_chunk_text.trim().to_string(),
                        token_count: current_chunk_tokens,
                        pages: std::mem::take(&mut current_pages),
                    });
                    
                    current_chunk_text.clear();
                    current_chunk_tokens = 0;
                }

                let page_number = element.provenance.as_ref().and_then(|p| p.page_number);

                // 4. Fallback for massive elements
                if element_token_count > self.max_tokens {
                    for token_slice in element_ids.chunks(self.max_tokens) {
                        
                        // Decode the chunked slice back into a String
                        // `false` means don't skip special tokens (since we didn't add them anyway)
                        if let Ok(decoded_text) = self.tokenizer.decode(token_slice, false) {
                            all_chunks.push(Chunk {
                                document_uri: Arc::clone(&doc.uri),
                                text: decoded_text.trim().to_string(),
                                token_count: token_slice.len(),
                                pages: page_number.map(|p| vec![p]).unwrap_or_default(),
                            });
                        }
                    }
                    continue;
                }

                // 5. Accumulate normal elements
                current_chunk_text.push_str(&element.text);
                current_chunk_text.push_str("\n\n");
                
                // +2 tokens approximate the double newline we just appended
                current_chunk_tokens += element_token_count + 2; 
                
                if let Some(page) = page_number {
                    current_pages.push(page);
                }
            }

            // 6. Flush the final remaining buffer
            if current_chunk_tokens > 0 {
                current_pages.dedup();
                all_chunks.push(Chunk {
                    document_uri: Arc::clone(&doc.uri),
                    text: current_chunk_text.trim().to_string(),
                    token_count: current_chunk_tokens,
                    pages: current_pages,
                });
            }

        Ok(all_chunks)
    }
}