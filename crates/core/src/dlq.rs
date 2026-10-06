use std::fmt::Display;

pub struct ErrorPayload {
    pub node_name: String,
    pub error_message: String,
    pub raw_data: String, 
}

pub trait DeadLetterQueue: Send + 'static {
    type Error: Display + Send;
    
    fn write(&mut self, payload: ErrorPayload) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn flush(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send;
}



pub struct NoOpDLQ;

impl DeadLetterQueue for NoOpDLQ {
    type Error = std::convert::Infallible;
    
    fn write(&mut self, _: ErrorPayload) -> impl Future<Output = Result<(), Self::Error>> + Send {
        std::future::ready(Ok(()))
    }
    
    fn flush(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        std::future::ready(Ok(()))
    }
}