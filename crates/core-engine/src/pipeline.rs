use std::fmt::{Debug, Display};
use tokio::sync::mpsc;
use crate::source::Source;
use crate::dlq::ErrorPayload;
use crate::dlq::DeadLetterQueue;
use tokio::task::JoinSet;
use crate::node::Node;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use crate::cancel::ShutdownSignal;
use tracing::{debug, error, info, info_span, trace, warn, Instrument};


#[derive(Debug, Clone, Copy)]
pub struct ExecutionSummary {
    /// The number of items that successfully made it to the end of the pipeline.
    pub successful_items: usize,
    /// The total number of errors caught and routed to the DLQ.
    pub error_count: usize,
}

/// A concurrent data processing pipeline.
/// 
/// The `Pipeline` handles pulling data from a source, routing it through
/// concurrent or sequential processing nodes, validating data, and 
/// automatically funneling any errors to a Dead Letter Queue (DLQ).
pub struct Pipeline<T> {
    receiver: mpsc::Receiver<T>,
    error_sender: mpsc::Sender<ErrorPayload>,
    dlq_handle: Option<JoinHandle<usize>>,
}


pub struct PipelineSource<T>{
    receiver: mpsc::Receiver<T>,
}

impl<T: Send> Source for PipelineSource<T> {
    type Item = T;
    type Error = std::convert::Infallible;

    async fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        Ok(self.receiver.recv().await)
    }
}


impl<T: Debug + Send + Sync + 'static> Pipeline<T> {
    pub fn into_source(mut self) -> PipelineSource<T> {
        // drop(self.error_sender); // signal DLQ to stop
        // if let Some(handle) = self.dlq_handle {
        //     handle.await.ok(); // wait for flush
        // }
        let _ = self.dlq_handle.take();
        PipelineSource {
            receiver: self.receiver,
        }
    }

    /// Creates a new Pipeline from a given source.
    ///
    /// # DLQ Backpressure Warning
    /// This pipeline bounds its internal channels using `buffer_size`. If the configured 
    /// Dead Letter Queue (DLQ) writer is slow (e.g., slow disk or network I/O), and a 
    /// massive spike of errors occurs, the error channel will fill up. 
    ///
    /// When the error channel is full, the active worker nodes will block while trying 
    /// to send errors. This means a slow DLQ will actively throttle and pause your 
    /// data ingestion pipeline. Size your `buffer_size` accordingly.
    pub fn from_source<S, E>(source: S, dlq: impl DeadLetterQueue + Send + 'static,buffer_size:usize) -> Self 
    where 
        S: Source<Item = T, Error = E> + Send + 'static,
        E: Display + Send + 'static,
    {
        Self::from_source_with_signal(source, dlq,ShutdownSignal::default(),buffer_size)
    }


    /// Creates a new Pipeline from a given source.
    ///
    /// # DLQ Backpressure Warning
    /// This pipeline bounds its internal channels using `buffer_size`. If the configured 
    /// Dead Letter Queue (DLQ) writer is slow (e.g., slow disk or network I/O), and a 
    /// massive spike of errors occurs, the error channel will fill up. 
    ///
    /// When the error channel is full, the active worker nodes will block while trying 
    /// to send errors. This means a slow DLQ will actively throttle and pause your 
    /// data ingestion pipeline. Size your `buffer_size` accordingly.
    pub fn from_source_with_signal<S,E>(mut source: S, mut dlq: impl DeadLetterQueue + Send + 'static, shutdown_signal: ShutdownSignal, buffer_size:usize) -> Self 
    where 
        S: Source<Item = T, Error = E> + Send + 'static,
        E: Display + Send + 'static,
    {
        let (send_raw, recv_raw) = mpsc::channel(buffer_size);
        let (send_error, mut recv_error) = mpsc::channel::<ErrorPayload>(buffer_size);
        let cancel_token_opt = shutdown_signal.into_token();

        let dlq_handle = tokio::spawn(async move {
            info!("DLQ background task started");
            let mut error_count = 0;

            while let Some(input) = recv_error.recv().await {
                error_count += 1;
                trace!(node = %input.node_name, "Writing error to DLQ");
                metrics::counter!("pipeline_dlq_events_total", "node" => input.node_name.clone()).increment(1);
                let start = std::time::Instant::now();
                if let Err(e) = dlq.write(input).await {
                    error!(error = %e, "Failed to write to DLQ. Disk full? Halting DLQ worker.");
                    break; // Drops the channel, halting upstream nodes safely
                }
                metrics::histogram!("pipeline_dlq_write_duration_seconds").record(start.elapsed().as_secs_f64());
            }
            
            if let Err(e) = dlq.flush().await {
                error!(error = %e, "Failed to flush DLQ writer during shutdown");
            }
            debug!("DLQ background task shutting down");
            error_count
        }.instrument(info_span!("dlq_worker")));

        let send_error_clone = send_error.clone();
        tokio::spawn(async move {   
            let source_name = source.name();
            info!(source = %source_name, "Starting source polling task");
            
            loop {
                let should_break = if let Some(token) = &cancel_token_opt {
                    tokio::select! {
                        _ = token.cancelled() => {
                            info!("Graceful shutdown initiated via cancellation token");
                            source.close().await.ok();
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
                metrics::counter!("pipeline_source_pulled_total", "source" => source_name.to_string()).increment(1);

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
                    return true
                }
                true
            }
        }
    }

    /// Adds a validation step to the pipeline.
    pub fn validate<F, E>(self, validator: F,buffer_size:usize) -> Pipeline<T>
    where
        F: Fn(&T) -> Result<(), E> + Send + Sync + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(buffer_size);
        let send_error_clone = self.error_sender.clone();

        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = "ValidationNode";

            while let Some(input) = receiver.recv().await {
                metrics::counter!("pipeline_node_received_total", "node" => node_name).increment(1);
                if send.is_closed() {
                    warn!("Downstream channel closed, halting validation");
                    break;
                }
                
                let span = info_span!("node_validate", node.name = node_name);
                let start = std::time::Instant::now();
                let validation_result = async {
                    validator(&input).inspect_err(|e| {
                        error!(error = %e, "Node validation failed");
                    })
                }
                .instrument(span)
                .await;
                metrics::histogram!("pipeline_node_execution_duration_seconds", "node" => node_name).record(start.elapsed().as_secs_f64());

                match validation_result {
                    Ok(_) => {
                        metrics::counter!("pipeline_node_emitted_total", "node" => node_name).increment(1);
                        if send.send(input).await.is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        metrics::counter!("pipeline_node_errors_total", "node" => node_name).increment(1);
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
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + Sync + 'static,
        buffer_size:usize
    ) -> Pipeline<O> 
    where 
        O: Debug + Send + Sync + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(buffer_size);
        let send_error_clone = self.error_sender.clone();
        
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = node.name();
            
            while let Some(input) = receiver.recv().await {
                metrics::counter!("pipeline_node_received_total", "node" => node_name).increment(1);
                if send.is_closed() {
                    warn!("Downstream channel closed, halting sequential pipe execution");
                    break; 
                }
                let span = info_span!("node_execute", node.name = node_name);
                let start = std::time::Instant::now();
                let output = async {
                    node.execute(&input)
                        .await
                        .inspect_err(|e| error!(error = %e, "Node execution failed"))
                }
                .instrument(span) 
                .await;                
                metrics::histogram!("pipeline_node_execution_duration_seconds", "node" => node_name).record(start.elapsed().as_secs_f64());
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
   pub fn pipe_concurrently<O, E>(
        self, 
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + Sync + 'static,
        concurrency_limit: usize,
        buffer_size:usize
    ) -> Pipeline<O> 
    where 
        O: Debug +Send + Sync  + 'static,
        E: Display + Send  + 'static,
    {
 
        let (send, recv) = mpsc::channel(buffer_size);
        let send_error_clone = self.error_sender.clone();
        
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let node_name = node.name();
            let shared_node = Arc::new(node);
            let mut set: JoinSet<bool> = JoinSet::new();
            let mut is_upstream_closed = false;
            while !is_upstream_closed || !set.is_empty()  {
                tokio::select! {
                    // Branch 1: Pull new items ONLY if we have capacity and upstream is open
                    res = receiver.recv(), if !is_upstream_closed && set.len() < concurrency_limit => {
                        match res {
                            Some(input) => {
                                metrics::counter!("pipeline_node_received_total", "node" => node_name).increment(1);
                                
                                if send.is_closed() {
                                    warn!("Downstream channel closed, aborting concurrent tasks");
                                    set.abort_all();
                                    break; 
                                }

                                let send_clone = send.clone();
                                let error_send_clone = send_error_clone.clone();
                                let node_ref = Arc::clone(&shared_node);
                                
                                set.spawn(async move {
                                    let span = info_span!("node_execute_concurrent", node.name = node_name);
                                    metrics::gauge!("pipeline_concurrent_tasks_active", "node" => node_name).increment(1.0);
                                    let start = std::time::Instant::now();
                                    
                                    let result = async {
                                        let output: Result<Vec<O>, E> = node_ref.execute(&input)
                                            .await
                                            .inspect_err(|e| error!(error = %e, "Concurrent node execution failed"));
                                            
                                        metrics::histogram!("pipeline_node_execution_duration_seconds", "node" => node_name).record(start.elapsed().as_secs_f64());
                                        Self::handle_node_result(output, &send_clone, &error_send_clone, node_name, Some(&input)).await
                                    }
                                    .instrument(span)
                                    .await;
                                    
                                    metrics::gauge!("pipeline_concurrent_tasks_active", "node" => node_name).decrement(1.0);
                                    result
                                });
                            }
                            None => {
                                // Upstream is exhausted. Mark it closed so this branch stops polling.
                                is_upstream_closed = true;
                            }
                        }
                    }

                    // Branch 2: Actively reap completed tasks (even when upstream is paused or closed)
                    Some(res) = set.join_next() => {
                        match res {
                            Ok(is_fatal_error) => {
                                if is_fatal_error {
                                    warn!("Fatal error downstream, aborting concurrent node.");
                                    set.abort_all();
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
                                    // Send to DLQ
                                    let payload = ErrorPayload {
                                        node_name: node_name.to_string(),
                                        error_message: "Task panicked".to_string(),
                                        raw_data: "unknown — lost to panic".to_string(),
                                    };
                                    send_error_clone.send(payload).await.ok();
                                    // Decrement gauge since task didn't get to do it
                                    metrics::gauge!("pipeline_concurrent_tasks_active", "node" => node_name).decrement(1.0);
                                    set.abort_all();
                                    break;
                                } else if error.is_cancelled() {
                                    warn!("Pipeline task was cancelled");
                                    metrics::gauge!("pipeline_concurrent_tasks_active", "node" => node_name).decrement(1.0);
                                }
                            }
                        }
                    }
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
                metrics::counter!("pipeline_node_emitted_total", "node" => node_name).increment(items.len() as u64);
                for value in items {
                    if send_channel.send(value).await.is_err() {
                        return true; 
                    }
                }
                false
            },
            Err(error) => {
                metrics::counter!("pipeline_node_errors_total", "node" => node_name).increment(1);
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
    async fn close_and_wait(mut self) -> usize {
        // Drop the sender to close the DLQ channel
        drop(self.error_sender);
        let mut final_error_count = 0;
        if let Some(handle) = self.dlq_handle.take() {
            // Configure a grace period for the DLQ to flush its final logs.
            // You can also make this a configurable field on the Pipeline struct if desired.
            let grace_period = Duration::from_secs(10);

            match tokio::time::timeout(grace_period, handle).await {
                Ok(Ok(errors)) => {
                    info!("Pipeline shutdown gracefully. All errors logged to DLQ.");
                    final_error_count = errors;
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
        final_error_count
    }

    /// 1. DRAIN / EXECUTE
    pub async fn execute(mut self) -> ExecutionSummary {
        info!("Executing pipeline as drain...");
        let mut successful_items = 0;

        while let Some(_) = self.receiver.recv().await {
            metrics::counter!("pipeline_completed_total").increment(1);
            successful_items += 1;
        }
        
        let error_count = self.close_and_wait().await;
        ExecutionSummary { successful_items, error_count }
    }

    /// 2. COLLECT
    pub async fn collect(mut self) -> (Vec<T>, ExecutionSummary) {
        info!("Executing pipeline as collect...");
        let mut results = Vec::new();
        while let Some(item) = self.receiver.recv().await {
            metrics::counter!("pipeline_completed_total").increment(1);
            results.push(item);
        }
        let successful_items = results.len();
        let error_count = self.close_and_wait().await;
        (results, ExecutionSummary { successful_items, error_count })
    }

    /// 3. FOR EACH
    pub async fn for_each<F>(mut self, mut action: F) -> ExecutionSummary
    where
        F: FnMut(T),
    {
        info!("Executing pipeline as for_each...");
        let mut successful_items = 0;

        while let Some(item) = self.receiver.recv().await {
            metrics::counter!("pipeline_completed_total").increment(1);
            action(item);
            successful_items += 1;
        }
        
        let error_count = self.close_and_wait().await;
        ExecutionSummary { successful_items, error_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlq::NoOpDLQ;
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

        let dlq = NoOpDLQ;

        // Note: In a real test environment, DLQueue should ideally write to memory
        // or a temp file to avoid polluting the actual DLQ on disk.
        let (results,_) = Pipeline::from_source(source,dlq,100)
            .validate(|item| {
                if item == "invalid_data" {
                    Err("Invalid data detected".to_string())
                } else {
                    Ok(())
                }
            },100).collect().await;


        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "valid_data");
    }

    #[tokio::test]
    async fn test_pipeline_pipe() {
        let source = MockSource {
            items: vec!["data1".to_string(), "error".to_string(), "data2".to_string()],
        };

        let dlq = NoOpDLQ;

        let (results,_) = Pipeline::from_source(source,dlq,100).pipe(MockNode,100).collect().await;

        // We expect "data1" and "data2" to process successfully. 
        // "error" should drop and funnel to the DLQ internally.
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"processed_data1".to_string()));
        assert!(results.contains(&"processed_data2".to_string()));
    }
}