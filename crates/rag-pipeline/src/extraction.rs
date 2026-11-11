use ferro_core::node::Node;
use std::future::Future;
use std::thread;
use tokio::sync::{mpsc, oneshot};
use crate::chunk::{Chunk, ExtractedEntity};

use gliner25_rs::{BoundaryConfig, BoundaryEngine, SchemaTask, Spanned, hub};
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// 1. Task now carries a batch of chunks
struct ExtractionTask {
    chunks: Vec<Chunk>,
    responder: oneshot::Sender<Result<Vec<Chunk>, String>>, 
}

pub struct GraphExtractionNode {
    sender: mpsc::Sender<ExtractionTask>,
}

impl GraphExtractionNode {
    pub fn new(
        entity_types: Vec<&str>, 
        classification_schemas: Vec<(&str, Vec<&str>)>
    ) -> Result<Self, BoxError> {
        
        let (sender, mut receiver) = mpsc::channel::<ExtractionTask>(32);
        
        let mut tasks = vec![SchemaTask::Entities(
            entity_types.into_iter().map(|s| s.into()).collect()
        )];

        for (feature_name, labels) in classification_schemas {
            tasks.push(SchemaTask::classification(
                feature_name,
                labels.into_iter().map(|s| s.into()).collect()
            ));
        }

        thread::spawn(move || {
            let config = BoundaryConfig::from_hub(hub::GLINER25_MULTI_V1);
            let mut engine = match BoundaryEngine::new(config) {
                Ok(e) => e,
                Err(err) => {
                    tracing::error!("Failed to initialize GLiNER engine: {}", err);
                    return;
                }
            };

        while let Some(task) = receiver.blocking_recv() {
                // 1. Destructure the task to take ownership of its fields
                let ExtractionTask { chunks, responder } = task;
                
                let mut enriched_chunks = Vec::with_capacity(chunks.len());
                let mut error_msg = None;

                // 2. Process the chunks without touching the responder
                for mut chunk in chunks {
                    match engine.extract_long(&chunk.text, &tasks) {
                        Ok(out) => {
                            for (id, m) in out.mentions.into_iter().enumerate() {
                                if m.score > 0.65 {
                                    chunk.entities.push(ExtractedEntity {
                                        local_id: id,
                                        text: m.text.to_string(),
                                        label: m.field.to_string(),
                                        token_start: m.start(), 
                                        token_end: m.end(),     
                                        canonical_id: None,
                                    });
                                }
                            }

                            for c in out.classifications {
                                chunk.chunk_attributes.insert(c.task, c.label);
                            }

                            enriched_chunks.push(chunk);
                        }
                        Err(e) => {
                            // Capture the error and halt the batch
                            error_msg = Some(e.to_string());
                            break; 
                        }
                    }
                }

                // 3. Evaluate the result and send exactly once
                let response = match error_msg {
                    Some(err) => Err(err),
                    None => Ok(enriched_chunks),
                };

                let _ = responder.send(response);
            }
        });

        Ok(Self { sender })
    }
}

impl Node for GraphExtractionNode {
    type Input = Vec<Chunk>;
    type Output = Vec<Chunk>;
    type Error = BoxError;

    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let chunks = input.clone();
        let sender = self.sender.clone();

        async move {
            let (resp_tx, resp_rx) = oneshot::channel();

            let task = ExtractionTask {
                chunks,
                responder: resp_tx,
            };

            sender.send(task).await.map_err(|_| "Failed to send batch to ONNX thread")?;

            match resp_rx.await {
                Ok(Ok(enriched_chunks)) => Ok(enriched_chunks),
                Ok(Err(e)) => Err(e.into()),
                Err(e) => Err(e.into()),
            }
        }
    }
}