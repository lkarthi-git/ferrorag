use fastembed::{TextEmbedding, TextInitOptions, EmbeddingModel};
use std::sync::{Arc, Mutex};
use crate::chunk::{Chunk,EmbeddedChunk};
use ferro_core::node::Node;


pub struct LocalEmbeddingNode {
    model: Arc<Mutex<TextEmbedding>>,
}

pub type GenericError = Box<dyn std::error::Error + Send + Sync>;

impl LocalEmbeddingNode {
    pub fn new() -> Result<Self,GenericError> {
        // 1. Initialize a vector for execution providers
        let mut execution_providers = vec![];


        // 2. Initialize CUDA on NVIDIA-supported operating systems
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            execution_providers.push(
                ort::ep::CUDA::default()
                    .build() // 💡 Required builder step in ort 2.x
                    .into()
            );
        }


        // 4. Construct using the modern builder pattern
        let mut options = TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15)
            .with_show_download_progress(true)
            .with_execution_providers(execution_providers);

        #[cfg(target_os = "macos")]
        {
            // Tailor the internal thread pool based on your system cores (e.g., 4 or 8)
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            options = options.with_intra_threads(threads); 
        }


        // 5. Initialize the embedding instance
        let model = TextEmbedding::try_new(options)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}


impl Node for LocalEmbeddingNode {
    type Input = Vec<Chunk>; 
    type Output = Vec<EmbeddedChunk>;
    type Error = GenericError;

    fn name(&self) -> &'static str {
        "LocalEmbeddingNodeBatch"
    }

    async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        let texts_to_embed: Vec<String> = input
            .iter()
            .map(|chunk| format!("search_document: {}", chunk.text))
            .collect();

        let model_arc = Arc::clone(&self.model);

        // Process the entire batch in one massive CoreML/CUDA operation
        let embeddings = tokio::task::spawn_blocking(move || {
            // 1. Explicitly handle the Mutex poisoning error
            let mut model_guard = model_arc.lock().map_err(|e| {
                Self::Error::from(format!("Failed to acquire model lock due to poisoning: {}", e))
            })?;
            
            // 2. Explicitly map fastembed::Error into your node's Self::Error
            model_guard.embed(texts_to_embed, None).map_err(|e| {
                Self::Error::from(format!("FastEmbed inference failed: {}", e))
            })
        })
        .await
        // 3. Handle tokio join errors (if the background thread panicked)
        .map_err(|e| Self::Error::from(format!("Tokio thread pool error: {}", e)))??;

        let embedded_chunks: Vec<EmbeddedChunk> = input
            .iter()
            .zip(embeddings)
            .map(|(original_chunk, vector)| EmbeddedChunk {
                chunk: original_chunk.clone(),
                vector,
            })
            .collect();

        Ok(embedded_chunks)
    }
}