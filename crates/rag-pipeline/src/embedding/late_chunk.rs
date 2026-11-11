use ferro_core::node::Node;
use std::sync::Arc;
use std::future::Future;
use tokenizers::Tokenizer;

// Adjust imports based on your actual workspace
use crate::embedding_backend::AdvancedEmbeddingBackend; 
use crate::chunk::{Chunk, EmbeddedChunk};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct LateChunkingNode {
    backend: Arc<dyn AdvancedEmbeddingBackend>,
    tokenizer: Arc<Tokenizer>,
    target_chunk_size: usize,
    overlap_tokens: usize,
}

impl LateChunkingNode {
    pub fn new(
        backend: Arc<dyn AdvancedEmbeddingBackend>, 
        hf_model_name: &str,
        target_chunk_size: usize,
        overlap_tokens: usize,
    ) -> Self {
        let tokenizer = Tokenizer::from_pretrained(hf_model_name, None)
            .expect("Failed to load tokenizer for LateChunkingNode");
        Self { backend, tokenizer:Arc::new(tokenizer), target_chunk_size, overlap_tokens }
    }
}

impl Node for LateChunkingNode {
    // Accepts a batch of large 8,000-token macro-chunks
    type Input = Vec<Chunk>; 
    type Output = Vec<EmbeddedChunk>;
    type Error = BoxError;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let backend = self.backend.clone();
        let tokenizer = self.tokenizer.clone();
        let parent_chunks = input.clone();
        
        let chunk_size = self.target_chunk_size;
        let overlap = self.overlap_tokens;

        async move {
            let mut all_embedded_chunks = Vec::new();
            
            let prefix = "search_document: ";
            
            // Fix 1: Safely calculate prefix tokens including the invisible [CLS] token (+1)
            let prefix_token_count = {
                let prefix_encoding = tokenizer.encode(prefix, false)
                    .map_err(|e| e.to_string())?;
                prefix_encoding.get_ids().len() + 1 
            };

            // Iterate over the incoming 8,000-token chunks
            for parent_chunk in parent_chunks {
                
                // Fix 2: Move tokenization and sliding window math to a blocking thread
                let tokenizer_clone = tokenizer.clone();
                let text_clone = parent_chunk.text.to_string();
                
                let (chunk_text_spans, sub_chunks) = tokio::task::spawn_blocking(move || {
                    let document_encoding = tokenizer_clone.encode(text_clone.as_str(), false)
                        .map_err(|e| e.to_string())?;
                    let tokens = document_encoding.get_ids().to_vec();
                    
                    let mut spans = Vec::new();
                    let mut chunks = Vec::new();
                    let mut start = 0;
                    let mut sub_chunk_index = 0;

                    while start < tokens.len() {
                        let end = std::cmp::min(start + chunk_size, tokens.len());
                        
                        // Offset by exact prefix length for GPU tensor slicing
                        spans.push((prefix_token_count + start, prefix_token_count + end));
                        
                        // Decode exact tokens 
                        let chunk_token_ids = &tokens[start..end];
                        let chunk_text = tokenizer_clone.decode(chunk_token_ids, true).map_err(|e| e.to_string())?;
                        
                        chunks.push(Chunk {
                            document_uri: parent_chunk.document_uri.clone(),
                            document_hash: parent_chunk.document_hash.clone(),
                            chunk_index: (parent_chunk.chunk_index * 10_000) + sub_chunk_index as u32,
                            text: Arc::from(chunk_text),
                            token_count: end - start,
                            pages: parent_chunk.pages.clone(),
                            metadata:parent_chunk.metadata.clone(),
                            entities:parent_chunk.entities.clone(),
                            triplets:parent_chunk.triplets.clone(),
                            chunk_attributes:parent_chunk.chunk_attributes.clone() 
                        });
                        
                        sub_chunk_index += 1;
                        if end == tokens.len() { break; }
                        
                        start += chunk_size - overlap;
                    }
                    Ok::<_, BoxError>((spans, chunks))
                }).await??;

                // 2. Format with prefix and get the [1, seq_len, 768] tensor on GPU
                // (Already async/blocking-safe inside backend implementation)
                let formatted_input = format!("{}{}", prefix, parent_chunk.text);
                let token_embeddings = backend.embed_document_tokens(&formatted_input).await?;

                // Fix 3: Execute the NEW Span Pooling on a blocking thread
                // We pass `prefix_token_count` so the GPU knows exactly how many instruction tokens to add to every chunk pool
                let span_vectors = tokio::task::spawn_blocking(move || {
                    token_embeddings.pool_late_chunks(prefix_token_count, &chunk_text_spans)
                }).await??;

                // 4. Zip and collect results for this macro-chunk
                for (sub_chunk, vector) in sub_chunks.into_iter().zip(span_vectors) {
                    all_embedded_chunks.push(EmbeddedChunk { 
                        chunk: sub_chunk, 
                        vector 
                    });
                }
            }

            Ok(all_embedded_chunks)
        }
    }
}