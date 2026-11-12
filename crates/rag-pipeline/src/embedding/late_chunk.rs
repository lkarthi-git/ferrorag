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
            let prefix = "search_document: ";
            
            let prefix_token_count = {
                let prefix_encoding = tokenizer.encode(prefix, false).map_err(|e| e.to_string())?;
                prefix_encoding.get_ids().len() + 1 
            };

            // PHASE 1: Batch Preparation (CPU)
            // Tokenize and calculate spans for ALL documents in one blocking thread
            let tokenizer_clone = tokenizer.clone();
            let (all_formatted_inputs, batch_metadata) = tokio::task::spawn_blocking(move || {
                let mut formatted_inputs = Vec::with_capacity(parent_chunks.len());
                // Store spans and sub_chunks for each parent chunk
                let mut batch_meta = Vec::with_capacity(parent_chunks.len()); 

                for parent_chunk in parent_chunks {
                    let text = parent_chunk.text.to_string();
                    formatted_inputs.push(format!("{}{}", prefix, text));

                    let document_encoding = tokenizer_clone.encode(text.as_str(), false).map_err(|e| e.to_string())?;
                    let tokens = document_encoding.get_ids().to_vec();
                    
                    let mut spans = Vec::new();
                    let mut sub_chunks = Vec::new();
                    let mut start = 0;
                    let mut sub_chunk_index = 0;

                    while start < tokens.len() {
                        let end = std::cmp::min(start + chunk_size, tokens.len());
                        spans.push((prefix_token_count + start, prefix_token_count + end));
                        
                        let chunk_token_ids = &tokens[start..end];
                        let chunk_text = tokenizer_clone.decode(chunk_token_ids, true).map_err(|e| e.to_string())?;
                        
                        sub_chunks.push(Chunk {
                            document_uri: parent_chunk.document_uri.clone(),
                            document_hash: parent_chunk.document_hash.clone(),
                            chunk_index: (parent_chunk.chunk_index * 10_000) + sub_chunk_index as u32,
                            text: Arc::from(chunk_text),
                            token_count: end - start,
                            pages: parent_chunk.pages.clone(),
                            metadata: parent_chunk.metadata.clone(),
                            entities: parent_chunk.entities.clone(),
                            triplets: parent_chunk.triplets.clone(),
                            chunk_attributes: parent_chunk.chunk_attributes.clone() 
                        });
                        
                        sub_chunk_index += 1;
                        if end == tokens.len() { break; }
                        start += chunk_size - overlap;
                    }
                    batch_meta.push((spans, sub_chunks));
                }
                Ok::<_, BoxError>((formatted_inputs, batch_meta))
            }).await??;

            // PHASE 2: Batch Inference (GPU)
            // ONE call to the backend for the entire batch.
            // (You will need to add this method to your AdvancedEmbeddingBackend trait)
            let batch_token_embeddings = backend.embed_document_tokens(all_formatted_inputs).await?;

            // PHASE 3: Batch Pooling (CPU)
            let mut all_embedded_chunks = Vec::new();
            
            // Pool the chunks for all documents in a single blocking thread
            let embedded_results = tokio::task::spawn_blocking(move || {
                let mut final_chunks = Vec::new();
                
                // Zip the unpadded token embeddings returned by the model with the span metadata
                for (token_embeddings, (spans, sub_chunks)) in batch_token_embeddings.into_iter().zip(batch_metadata) {
                    let span_vectors = token_embeddings.pool_late_chunks(prefix_token_count, &spans)?;
                    
                    for (sub_chunk, vector) in sub_chunks.into_iter().zip(span_vectors) {
                        final_chunks.push(EmbeddedChunk { chunk: sub_chunk, vector });
                    }
                }
                Ok::<_, BoxError>(final_chunks)
            }).await??;

            all_embedded_chunks.extend(embedded_results);

            Ok(all_embedded_chunks)
        }
    }
}