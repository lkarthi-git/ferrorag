use async_trait::async_trait;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;


#[cfg(feature = "candle")]
pub mod candle;


#[cfg(feature = "ort")]
pub mod ort;



#[async_trait]
pub trait RerankerBackend: Send + Sync {
    async fn predict_scores(&self, query: &str, docs: Vec<String>) -> Result<Vec<f32>, BoxError>;
}