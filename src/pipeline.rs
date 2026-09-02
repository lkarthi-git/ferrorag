use std::fmt::Display;
use tokio::sync::mpsc;
use crate::source::Source;
use crate::dlq::ErrorPayload;
use crate::dlq::DLQueue;
use tokio::io::AsyncWriteExt;
use tokio::task::JoinSet;
use crate::node::Node;
use std::sync::Arc;

const BUFFER_SIZE: usize = 100;



pub struct Pipeline<T> {
    receiver: mpsc::Receiver<T>,
    error_sender: mpsc::Sender<ErrorPayload>,
}

impl<T: Display + Send + 'static> Pipeline<T> {

    pub fn from_source<S,E>(mut source:S) -> Self 
    where S: Source<Item = T, Error = E> + Send + 'static,
          E: Display + Send + 'static,
    {
        let (send_raw, recv_raw) = mpsc::channel(BUFFER_SIZE);
        let (send_error, mut recv_error) = mpsc::channel::<ErrorPayload>(BUFFER_SIZE);
        tokio::spawn(async move {
            let mut dlq = DLQueue::new().await;
            while let Some(input) = recv_error.recv().await {
                dlq.write(input).await;
            }
            let _ = dlq.writer.flush().await;
        });

        let send_error_clone = send_error.clone();
        tokio::spawn(async move {   
            loop {
                match source.next().await {
                    Ok(Some(line)) => {
                        if send_raw.send(line).await.is_err() {
                            break;
                        }
                    },
                    Ok(None) => {
                        break;
                    },
                    Err(error) => {
                        if send_error_clone.send(ErrorPayload {
                            node_name: "FileSource".to_string(),
                            error_message: error.to_string(),
                            raw_data: "".to_string(),
                        }).await.is_err() {
                            eprintln!("CRITICAL: The background logging task died. Halting pipeline to prevent data loss.");
                        }
                    }
                }
            }
        });
        Pipeline {
            receiver: recv_raw,
            error_sender: send_error,
        }
    }

    pub async fn run(mut self) -> () {
        while let Some(_) = self.receiver.recv().await {

        }
    }


    pub async fn pipe_concurrently<O,E>(
        self, 
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + Sync + 'static,
        concurrency_limit: usize,
    ) -> Pipeline<O> 
    where 
        O: Send + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        let send_error_clone = self.error_sender.clone();
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            let mut set = JoinSet::new();
            let node_name = std::any::type_name_of_val(&node).to_string();
            let shared_node = Arc::new(node);
            while let Some(input) = receiver.recv().await {
                if send.is_closed() {
                    break; 
                }
                while set.len() >= concurrency_limit {
                   if let Some(Err(error)) =  set.join_next().await{
                        if error.is_panic(){
                            eprintln!("CRITICAL: A concurrent pipeline task panicked!");
                        }
                   }
                }
                let send_clone = send.clone();
                let error_send_clone: mpsc::Sender<ErrorPayload> = send_error_clone.clone();
                let node_ref = Arc::clone(&shared_node);
                let task_node_name = node_name.clone();
                set.spawn(async move {
                    let output: Result<Vec<O>, E> = node_ref.execute(&input).await;
                    Self::handle_node_result(output, &send_clone, &error_send_clone, task_node_name , input.to_string()).await;
                });
            }
            while let Some(_) = set.join_next().await {};
            let flush_output = shared_node.flush();
            Self::handle_node_result(flush_output, &send, &send_error_clone, node_name, "FLUSH".to_string()).await;
        });
        Pipeline {
            receiver: recv,
            error_sender: self.error_sender,
        }
    }

    async fn handle_node_result<O, E>(
        result: Result<Vec<O>, E>,
        send_channel: &mpsc::Sender<O>,
        error_channel: &mpsc::Sender<ErrorPayload>,
        node_name: String,
        raw_data: String,
    ) 
    where 
        O: Send,
        E: std::fmt::Display,
    {
        match result {
            Ok(items) => {
                for value in items {
                    if send_channel.send(value).await.is_err() {
                        break; // Downstream is closed
                    }
                }
            },
            Err(error) => {
                let payload = ErrorPayload {
                    node_name,
                    error_message: error.to_string(),
                    raw_data,
                };
                if error_channel.send(payload).await.is_err() {
                    eprintln!("CRITICAL: The background DLQ task died. Halting pipeline to prevent data loss.");
                }
            }
        }
    }

    pub fn pipe<O,E>(
        self, 
        node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + 'static
    ) -> Pipeline<O> 
    where 
        O: Send + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        let send_error_clone = self.error_sender.clone();
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            'main_loop: while let Some(input) = receiver.recv().await {
                let output: Result<Vec<O>, E> = node.execute(&input).await;
                match output {
                    Ok(output) => {
                        for value in output {
                             if send.send(value).await.is_err() {
                                break 'main_loop;
                            }
                        }
                    },
                    Err(error) => {
                        if send_error_clone.send(ErrorPayload {
                            node_name: std::any::type_name_of_val(&node).to_string(),
                            error_message: error.to_string(),
                            raw_data: input.to_string(),
                        }).await.is_err() {
                            eprintln!("CRITICAL: The background logging task died. Halting pipeline to prevent data loss.");
                            break 'main_loop;
                        }
                    }
                }
            }
            let output = node.flush();
            match output {
                Ok(output) => {
                    for value in output {
                        if send.send(value).await.is_err() {
                            break;
                        }
                    }
                },
                Err(error) => {
                   if send_error_clone.send(ErrorPayload {
                            node_name: std::any::type_name_of_val(&node).to_string(),
                            error_message: error.to_string(),
                            raw_data: " ".to_string(),
                        }).await.is_err() {
                            eprintln!("CRITICAL: The background logging task died. Halting pipeline to prevent data loss.");
                        }
                }
            }
        });
        
        Pipeline {
            receiver: recv,
            error_sender: self.error_sender,
        }
    }       
}   
