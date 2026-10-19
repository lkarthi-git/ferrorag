use ferro_core::source::Source;
use std::collections::VecDeque;
use std::time::Duration;

pub struct MemorySource<T> {
    items: VecDeque<T>,
    interval: Duration,
}

impl<T> MemorySource<T> {
    pub fn new(items: Vec<T>, interval: Duration) -> Self {
        Self {
            items: VecDeque::from(items),
            interval,
        }
    }
}

// Ensure T is Send if your pipeline requires items to be sent across threads
impl<T: Send + Unpin> Source for MemorySource<T> {
    type Item = T;
    type Error = std::io::Error;

    async fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.items.is_empty() {
            return Ok(None);
        }
        
        let item = self.items.pop_front().unwrap();
        
        // Optimize: skip the await machinery entirely if interval is zero
        if !self.interval.is_zero() {
            tokio::time::sleep(self.interval).await;
        }
        
        Ok(Some(item))
    }
}