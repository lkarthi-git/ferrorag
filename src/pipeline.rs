use std::fmt::Display;

use tokio::sync::mpsc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::fs::File;

const BUFFER_SIZE: usize = 100;

pub trait Node {
    type Input;
    type Output: Default;
    type Error;
    fn execute(&mut self, input: Self::Input) -> impl Future<Output = Result<Self::Output,Self::Error>> + Send;
    fn flush(&mut self) -> Result<Self::Output,Self::Error> {
        Result::Ok(Self::Output::default())
    }
}

pub struct Pipeline<T> {
    receiver: mpsc::Receiver<T>,
}

impl Pipeline<String> {
    pub fn ingest(file:File) -> Self {
        let (send_raw, recv_raw) = mpsc::channel(BUFFER_SIZE);
        tokio::spawn(async move {
            let reader: BufReader<File> = BufReader::new(file);
            let mut lines: tokio::io::Lines<BufReader<File>> = reader.lines();      
            while let Ok(Some(line)) = lines.next_line().await {
               let _ = send_raw.send(line).await;
            }
        });
        Pipeline {
            receiver: recv_raw,
        }
    }

    pub async fn execute(mut self) -> () {
            while let Some(input) = self.receiver.recv().await {
                println!("Received input: {}", input);
            }
    }
}

impl<T: Send + 'static> Pipeline<T> {
    pub fn pipe<O,E>(
        self, 
        mut node: impl Node<Input = T, Output = Vec<O>, Error = E> + Send + 'static
    ) -> Pipeline<O> 
    where 
        O: Send + 'static,
        E: Display + Send + 'static,
    {
        let (send, recv) = mpsc::channel(BUFFER_SIZE);
        
        tokio::spawn(async move {
            let mut receiver = self.receiver;
            while let Some(input) = receiver.recv().await {
                let output = node.execute(input).await;
                match output {
                    Ok(output) => {
                        for value in output {
                            let _ = send.send(value).await.is_err();
                        }
                    },
                    Err(error) => {
                        println!("Error: {}", error);
                    }
                }
            }
            let output = node.flush();
            match output {
                Ok(output) => {
                    for value in output {
                        let _ = send.send(value).await.is_err();
                    }
                },
                Err(error) => {
                    println!("Error: {}", error);
                }
            }
        });
        
        Pipeline {
            receiver: recv,
        }
    }       
}   
