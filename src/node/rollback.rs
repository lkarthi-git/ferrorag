use std::sync::Arc;
use tokio::sync::Mutex;
use std::pin::Pin;
use std::future::Future;

type BoxedRollback = Pin<Box<dyn Future<Output=Result<(),String>> + Send + Sync>>;

#[derive(Clone)]
pub struct RollbackContext<T> {
    pub inner: T, // The user's actual data
    stack: Arc<Mutex<Vec<BoxedRollback>>>,
}

impl<T> RollbackContext<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            stack: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn push(&self, rollback: BoxedRollback) {
        self.stack.lock().await.push(rollback);
    }

    pub async fn execute_rollbacks(&self) {
        let mut stack = self.stack.lock().await;
        tracing::info!("Executing {} item-level rollbacks", stack.len());
        while let Some(rollback) = stack.pop() {
            if let Err(e) = rollback.await {
                tracing::error!("Pipeline rollback failed: {}", e);
            }
        }
    }
}