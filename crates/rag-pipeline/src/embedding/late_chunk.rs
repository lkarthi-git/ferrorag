use ferro_core::node::Node;
use std::future::Future;
use std::sync::Arc;
use tokenizers::Tokenizer;

// Adjust imports based on your actual workspace
use crate::chunk::{Chunk, EmbeddedChunk};
use crate::embedding_backend::AdvancedEmbeddingBackend;

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
        Self {
            backend,
            tokenizer: Arc::new(tokenizer),
            target_chunk_size,
            overlap_tokens,
        }
    }
}

impl Node for LateChunkingNode {
    type Input = Vec<Chunk>;
    type Output = Vec<EmbeddedChunk>;
    type Error = BoxError;

    fn execute(
        &self,
        input: &Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let backend = self.backend.clone();
        let tokenizer = self.tokenizer.clone();
        let parent_chunks = input.clone();

        let chunk_size = self.target_chunk_size;
        let overlap = self.overlap_tokens;

        async move {
            let prefix = "search_document: ";
            let max_seq_len = 8192; // Must match the backend's truncation limit

            // Calculate prefix tokens without special tokens, then add 1 for the [CLS] token
            let prefix_token_count = tokenizer.encode(prefix, false).unwrap().get_ids().len() + 1;

            // PHASE 1: Batch Preparation (CPU)
            let tokenizer_clone = tokenizer.clone();
            let (all_formatted_inputs, batch_metadata) = tokio::task::spawn_blocking(move || {
                let mut formatted_inputs = Vec::with_capacity(parent_chunks.len());
                let mut batch_meta = Vec::with_capacity(parent_chunks.len());

                for parent_chunk in parent_chunks {
                    let text = parent_chunk.text.to_string();
                    let formatted_text = format!("{}{}", prefix, text);
                    formatted_inputs.push(formatted_text.clone());

                    // Tokenize EXACTLY as the backend will (with special tokens)
                    let encoding = tokenizer_clone
                        .encode(formatted_text, true)
                        .map_err(|e| e.to_string())?;
                    let all_ids = encoding.get_ids();

                    let mut total_len = all_ids.len();

                    // 1. Enforce truncation to match the backend
                    if total_len > max_seq_len {
                        total_len = max_seq_len;
                    }

                    // 2. Drop the [SEP] token at the end (if it wasn't truncated away)
                    let doc_end_idx = if total_len < max_seq_len {
                        total_len - 1
                    } else {
                        total_len
                    };

                    // 3. Calculate how many tokens actually belong to the document
                    let doc_token_count = doc_end_idx.saturating_sub(prefix_token_count);

                    let mut spans = Vec::new();
                    let mut sub_chunks = Vec::new();
                    let mut start = 0;
                    let mut sub_chunk_index = 0;

                    while start < doc_token_count {
                        let end = std::cmp::min(start + chunk_size, doc_token_count);

                        // Calculate absolute indices in the GPU tensor
                        let absolute_start = prefix_token_count + start;
                        let absolute_end = prefix_token_count + end;

                        spans.push((absolute_start, absolute_end));

                        // Decode text exactly from the token bounds
                        let chunk_token_ids = &all_ids[absolute_start..absolute_end];
                        let chunk_text = tokenizer_clone
                            .decode(chunk_token_ids, true)
                            .map_err(|e| e.to_string())?;

                        sub_chunks.push(Chunk {
                            document_uri: parent_chunk.document_uri.clone(),
                            document_hash: parent_chunk.document_hash.clone(),
                            chunk_index: (parent_chunk.chunk_index * 10_000)
                                + sub_chunk_index as u32,
                            text: Arc::from(chunk_text),
                            token_count: end - start,
                            pages: parent_chunk.pages.clone(),
                            metadata: parent_chunk.metadata.clone(),
                            entities: parent_chunk.entities.clone(),
                            triplets: parent_chunk.triplets.clone(),
                            chunk_attributes: parent_chunk.chunk_attributes.clone(),
                        });

                        sub_chunk_index += 1;
                        if end == doc_token_count {
                            break;
                        }
                        start += chunk_size - overlap;
                    }
                    batch_meta.push((spans, sub_chunks));
                }
                Ok::<_, BoxError>((formatted_inputs, batch_meta))
            })
            .await??;

            // PHASE 2: Batch Inference (GPU)
            let batch_token_embeddings =
                backend.embed_document_tokens(all_formatted_inputs).await?;

            // PHASE 3: Batch Pooling (CPU)
            let mut all_embedded_chunks = Vec::new();

            let embedded_results = tokio::task::spawn_blocking(move || {
                let mut final_chunks = Vec::new();

                for (token_embeddings, (spans, sub_chunks)) in
                    batch_token_embeddings.into_iter().zip(batch_metadata)
                {
                    let span_vectors = token_embeddings.pool_spans(&spans)?;

                    for (sub_chunk, vector) in sub_chunks.into_iter().zip(span_vectors) {
                        final_chunks.push(EmbeddedChunk {
                            chunk: sub_chunk,
                            vector,
                        });
                    }
                }
                Ok::<_, BoxError>(final_chunks)
            })
            .await??;

            all_embedded_chunks.extend(embedded_results);

            Ok(all_embedded_chunks)
        }
    }
}
