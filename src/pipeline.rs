use std::fmt::{Debug, Display};
use tokio::sync::mpsc;
use crate::source::Source;
use crate::dlq::ErrorPayload;
use crate::dlq::DLQueue;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinSet;
use crate::node::Node;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use crate::cancel::ShutdownSignal;
use tracing::{debug, error, info, info_span, trace, warn, Instrument};

const BUFFER_SIZE: usize = 100;

/// A concurrent data processing pipeline.
/// 
/// The `Pipeline` handles pulling data from a source, routing it through
/// concurrent or sequential processing nodes, validating data, and 
/// automatically funneling any errors to a Dead Letter Queue (DLQ).
pub struct Pipeline<T> {
    receiver: mpsc::Receiver<T>,
    error_sender: mpsc::Sender<ErrorPayload>,
    dlq_handle: Option<JoinHandle<()>>,
}

impl<T: Debug + Send + Sync + 'static> Pipeline<T> {
    /// Creates a new Pipeline from a given source.
    /// 
    /// Initializes a background task for the Dead Letter Queue to handle errors.
    pub fn from_source<S, E>(source: S) -> Self 
    where 
        S: Source<Item = T, Error = E> + Send + 'static,
        E: Display + Send + 'static,
    {
        Self::from_source_with_signal(source, ShutdownSignal::default())
    }

    /// Creates a new Pipeline from a given source with a shutdown signal.
    /// 
    /// This allows for graceful shutdown of the pipeline when a cancellation 
    /// token is triggered.
    pub fn from_source_with_signal<S,E>(mut source: S, shutdown_signal: ShutdownSignal) -> Self 
    where 
        S: Source<Item = T, Error = E> + Send + 'static,
        E: Display + Send + 'static,
    {
        let (send_raw, recv_raw) = mpsc::channel(BUFFER_SIZE);
        let (send_error, mut recv_error) = mpsc::channel::<ErrorPayload>(BUFFER_SIZE);
        let cancel_token_opt = shutdown_signal.into_token();

        let dlq_handle = tokio::spawn(async move {
            info!("DLQ background task started");
            let mut dlq: DLQueue = DLQueue::new().await;
            
            while let Some(input) = recv_error.recv().await {
                trace!(node = %input.node_name, "Writing error to DLQ");
                dlq.write(input).await;
            }
            
            if let Err(e) = dlq.writer.flush().await {
                error!(error = %e, "Failed to flush DLQ writer during shutdown");
            }
            debug!("DLQ background task shutting down");
        }.instrument(info_span!("dlq_worker")));

        let send_error_clone = send_error.clone();
        tokio::spawn(async move {   
            let source_name = std::any::type_name_of_val(&source).to_string();
            info!(source = %source_name, "Starting source polling task");
            
            loop {
                let should_break = if let Some(token) = &cancel_token_opt {
                    tokio::select! {
                        _ = token.cancelled() => {
                            info!("Graceful shutdown initiated via cancellation token");
                            break; 
                        },
                        res = source.next() => {
                            Self::handle_source_result(res, &send_raw, &send_error_clone, &source_name).await
                        }
                    }
                } else {
                    let res: Result<Option<T>, E> = source.next().await;
                    Self::handle_source_result(res, &send_raw, &send_error_clone, &source_name).await
                };

                if should_break {
                    debug!("Exiting source polling loop");
                    break;
                }
            }
        }.instrument(info_span!("source_polling")));

        Pipeline {
            receiver: recv_raw,
            error_sender: send_error,
            dlq_handle: Some(dlq_handle),
        }
    }

    async fn handle_source_result<E>(
        res: Result<Option<T>, E>, 
        send_raw: &mpsc::Sender<T>, 
        send_error_clone: &mpsc::Sender<ErrorPayload>,
        source_name: &str
    ) -> bool
    where E: Display + Send + 'static,
    {
        match res {
            Ok(Some(line)) => {
                if send_raw.send(line).await.is_err() {
                    warn!("Downstream channel closed, stopping source polling");
                    return true;
                }
                false
            },
            Ok(None) => {
                info!("Source exhausted, ending polling");
                true
            },
            Err(error) => {
                error!(error = %error, source = %source_name, "Source encountered an error");
                let payload = ErrorPayload {
                    node_name: source_name.to_string(),
                    error_message: error.to_string(),
                    raw_data: "".to_string(),
                };
                
                if send_error_clone.send(payload).await.is_err() {
                    error!(
                        target: "pipeline::critical",
                        severity = "CRITICAL",
                        "The background logging task died. Halting pipeline to prevent data loss."
                    );
                }
                false
            }
        }
    }

    /// Adds a validation step to the pipeline.
    pub fn validate<F, E>(self, validator: F) -> Pipeline<T>
    where
        F: Fn(&T) -> Result<(), E> + Send + Sync + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        let send_error_clone = self.error_sender.clone();

        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = "ValidationNode";

            while let Some(input) = receiver.recv().await {
                if send.is_closed() {
                    warn!("Downstream channel closed, halting validation");
                    break;
                }
                
                let span = info_span!("node_validate", node.name = node_name);

                let validation_result = async {
                    validator(&input).inspect_err(|e| {
                        error!(error = %e, "Node validation failed");
                    })
                }
                .instrument(span)
                .await;

                match validation_result {
                    Ok(_) => {
                        if send.send(input).await.is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let payload = ErrorPayload {
                            node_name: node_name.to_string(),
                            error_message: error.to_string(),
                            raw_data: format!("{:?}", input), 
                        };
                        if let Err(send_err) = send_error_clone.send(payload).await {
                            error!(
                                target: "pipeline::critical",
                                severity = "CRITICAL",
                                error = %send_err,
                                original_node = %node_name,
                                "The background DLQ task died. Halting pipeline to prevent data loss."
                            );
                        }
                    }
                }
            }
        });

        Pipeline {
            receiver: recv,
            error_sender: self.error_sender,
            dlq_handle: self.dlq_handle,
        }
    }

    /// Pipes the data sequentially through a processing node.
    pub fn pipe<O,E>(
        self, 
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + Sync + 'static
    ) -> Pipeline<O> 
    where 
        O: Display + Send + Sync + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        let send_error_clone = self.error_sender.clone();
        
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = std::any::type_name_of_val(&node);
            
            while let Some(input) = receiver.recv().await {
                if send.is_closed() {
                    warn!("Downstream channel closed, halting sequential pipe execution");
                    break; 
                }
                let span = info_span!("node_execute", node.name = node_name);

                let output = async {
                    node.execute(&input)
                        .await
                        .inspect_err(|e| error!(error = %e, "Node execution failed"))
                }
                .instrument(span) 
                .await;                
                
                if Self::handle_node_result(output, &send, &send_error_clone, node_name, Some(&input)).await{
                    break;
                }
            }
            
            info!(node = %node_name, "Flushing sequential node");
            let flush_output = node.flush();
            Self::handle_node_result(flush_output, &send, &send_error_clone, node_name, None).await;
        });
        
        Pipeline {
            receiver: recv,
            error_sender: self.error_sender,
            dlq_handle: self.dlq_handle,
        }
    }       

    /// Pipes the data concurrently through a processing node.
    pub async fn pipe_concurrently<O,E>(
        self, 
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + Sync + 'static,
        concurrency_limit: usize,
    ) -> Pipeline<O> 
    where 
        O: Display + Send + Sync  + 'static,
        E: Display + Send  + 'static,
    {
        assert!(
                    concurrency_limit <= BUFFER_SIZE, 
                    "Concurrency limit ({}) must be less than or equal to BUFFER_SIZE ({})",
                    concurrency_limit,
                    BUFFER_SIZE
        );        
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        let send_error_clone = self.error_sender.clone();
        
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = std::any::type_name_of_val(&node);
            let shared_node= Arc::new(node);
            let mut set: JoinSet<bool> = JoinSet::new();

            while let Some(input) = receiver.recv().await {
                if send.is_closed() {
                    warn!("Downstream channel closed, aborting concurrent tasks");
                    set.abort_all();
                    break; 
                }

                let mut should_stop = false;
                while set.len() >= concurrency_limit {
                   if let Some(res) = set.join_next().await {
                        match res {
                            Ok(continue_loop) => {
                                if !continue_loop {
                                    should_stop = true;
                                    break;
                                }
                            }
                            Err(error) => {
                                if error.is_panic() {
                                    error!(
                                        target: "pipeline::critical", 
                                        severity = "CRITICAL", 
                                        "A concurrent pipeline task panicked!"
                                    );
                                } else if error.is_cancelled() {
                                    warn!("Pipeline task was cancelled");
                                }
                            }
                        }
                   }
                }

                if should_stop {
                    warn!("A critical error occurred; aborting all concurrent tasks");
                    set.abort_all();
                    break;
                }

                let send_clone = send.clone();
                let error_send_clone: mpsc::Sender<ErrorPayload> = send_error_clone.clone();
                let node_ref = Arc::clone(&shared_node);
                
                set.spawn(async move {
                    let span = info_span!("node_execute_concurrent", node.name = node_name);
                    async {
                        let output: Result<Vec<O>, E> = node_ref.execute(&input)
                            .await
                            .inspect_err(|e| error!(error = %e, "Concurrent node execution failed"));
                        Self::handle_node_result(output, &send_clone, &error_send_clone, node_name, Some(&input)).await
                    }
                    .instrument(span)
                    .await
                });
            }
            
            // Wait for remaining tasks to complete
            while let Some(res) = set.join_next().await {
                if let Ok(false) = res {
                    set.abort_all();
                    break;
                }
            }
            
            info!(node = %node_name, "Flushing concurrent node");
            let flush_output = shared_node.flush();
            Self::handle_node_result(flush_output, &send, &send_error_clone, node_name, None).await;
        });
        
        Pipeline {
            receiver: recv,
            error_sender: self.error_sender,
            dlq_handle: self.dlq_handle,
        }
    }

    async fn handle_node_result<O, E>(
        result: Result<Vec<O>, E>,
        send_channel: &mpsc::Sender<O>,
        error_channel: &mpsc::Sender<ErrorPayload>,
        node_name: &'static str,
        input: Option<&T>,
    ) -> bool
    where 
        O: Send,
        E: std::fmt::Display,
    {
        match result {
            Ok(items) => {
                for value in items {
                    if send_channel.send(value).await.is_err() {
                        return true; 
                    }
                }
                false
            },
            Err(error) => {
                let raw_data = match input {
                    Some(input) => format!("{:?}", input),
                    None => "Flush".to_string(),
                };

                let payload = ErrorPayload {
                    node_name: node_name.to_string(),
                    error_message: error.to_string(),
                    raw_data,
                };
                
                if error_channel.send(payload).await.is_err() {
                    error!(
                        target: "pipeline::critical", 
                        severity = "CRITICAL", 
                        node = %node_name,
                        "The background DLQ task died. Halting pipeline to prevent data loss."
                    );
                    return true
                }
                false
            }
        }
    }

    /// Private helper to handle the graceful shutdown of the DLQ
    async fn close_and_wait(mut self) {
        // Drop the sender to close the DLQ channel
        drop(self.error_sender);
        
        if let Some(handle) = self.dlq_handle.take() {
            // Configure a grace period for the DLQ to flush its final logs.
            // You can also make this a configurable field on the Pipeline struct if desired.
            let grace_period = Duration::from_secs(10);

            match tokio::time::timeout(grace_period, handle).await {
                Ok(Ok(_)) => {
                    info!("Pipeline shutdown gracefully. All errors logged to DLQ.");
                }
                Ok(Err(e)) => {
                    error!(
                        target: "pipeline::critical", 
                        severity = "CRITICAL", 
                        error = %e, 
                        "DLQ task panicked during shutdown"
                    );
                }
                Err(_) => {
                    error!(
                        target: "pipeline::critical", 
                        severity = "CRITICAL", 
                        "DLQ shutdown timed out after {} seconds. Some final error logs may be lost.",
                        grace_period.as_secs()
                    );
                    // The handle is dropped here, which detaches the background task.
                    // If the main tokio runtime drops shortly after this, the stuck task is killed.
                }
            }
        }
    }

    /// 1. DRAIN / EXECUTE
    pub async fn execute(mut self) {
        info!("Executing pipeline as drain...");
        while let Some(_) = self.receiver.recv().await {}
        
        self.close_and_wait().await;
    }

    /// 2. COLLECT
    pub async fn collect(mut self) -> Vec<T> {
        info!("Executing pipeline as collect...");
        let mut results = Vec::new();
        while let Some(item) = self.receiver.recv().await {
            results.push(item);
        }
        
        self.close_and_wait().await;
        results
    }

    /// 3. FOR EACH
    pub async fn for_each<F>(mut self, mut action: F)
    where
        F: FnMut(T),
    {
        info!("Executing pipeline as for_each...");
        while let Some(item) = self.receiver.recv().await {
            action(item);
        }
        
        self.close_and_wait().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Mock Implementations ---

    struct MockSource {
        items: Vec<String>,
    }

    impl Source for MockSource {
        type Item = String;
        type Error = String;

        async fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
            if self.items.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.items.remove(0)))
            }
        }
    }

    struct MockNode;

    impl Node for MockNode {
        type Input = String;
        type Output = Vec<String>;
        type Error = String;

        async fn execute(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
            if input == "error" {
                Err("Node error triggered".to_string())
            } else {
                Ok(vec![format!("processed_{}", input)])
            }
        }

        fn flush(&self) -> Result<Self::Output, Self::Error> {
            Ok(vec![])
        }
    }

    // --- Tests ---

    #[tokio::test]
    async fn test_pipeline_validate_success() {
        let source = MockSource {
            items: vec!["valid_data".to_string(), "invalid_data".to_string()],
        };

        // Note: In a real test environment, DLQueue should ideally write to memory
        // or a temp file to avoid polluting the actual DLQ on disk.
        let mut pipeline = Pipeline::from_source(source)
            .validate(|item| {
                if item == "invalid_data" {
                    Err("Invalid data detected".to_string())
                } else {
                    Ok(())
                }
            });

        // Pull processed items directly off the receiver to verify
        let mut results = Vec::new();
        while let Some(item) = pipeline.receiver.recv().await {
            results.push(item);
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "valid_data");
    }

    #[tokio::test]
    async fn test_pipeline_pipe() {
        let source = MockSource {
            items: vec!["data1".to_string(), "error".to_string(), "data2".to_string()],
        };

        let mut pipeline = Pipeline::from_source(source).pipe(MockNode);

        let mut results = Vec::new();
        while let Some(item) = pipeline.receiver.recv().await {
            results.push(item);
        }

        // We expect "data1" and "data2" to process successfully. 
        // "error" should drop and funnel to the DLQ internally.
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"processed_data1".to_string()));
        assert!(results.contains(&"processed_data2".to_string()));
    }
}