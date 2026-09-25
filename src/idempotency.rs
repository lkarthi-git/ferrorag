use std::future::Future;
use std::fmt::Display;

// The states our store needs to track
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdempotencyStatus {
    New,        
    InProgress, 
    Completed,   
}

pub trait IdempotencyStore: Send + Sync + Clone {
    type Error: Send + Display;

    // Checks the status and locks it if it's New
    fn check_and_lock(&self, id: &str) -> impl Future<Output = Result<IdempotencyStatus, Self::Error>> + Send;
    
    // Marks the ID as permanently successful
    fn mark_success(&self, id: &str) -> impl Future<Output = Result<(), Self::Error>> + Send;
    
    // Deletes the ID so it can be retried later
    fn release_lock(&self, id: &str) -> impl Future<Output = Result<(), Self::Error>> + Send;
}