pub trait Node {
    type Input;
    type Output: Default;
    type Error;
    fn execute(&self, input: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
    fn flush(&self) -> Result<Self::Output,Self::Error> {
        Result::Ok(Self::Output::default())
    }
}

pub struct RetryNode<N> {
    pub node: N,
    pub retry_count: usize,
}

impl<N,I,O,E> Node for RetryNode<N> 
where N: Node<Input = I, Output = O, Error = E> + Send + Sync,
      I: Sync,
      O: Default + Send,
      E: Send,
{
    type Input = I;
    type Output = O;
    type Error = E;
    async fn execute(&self, input: &Self::Input) -> Result<Self::Output,Self::Error> {
        let mut remaining_retries = self.retry_count;
        loop {
            match self.node.execute(input).await {
                Ok(output) => return Ok(output),
                Err(_) if remaining_retries > 0 => {
                        remaining_retries -= 1;
                        let exponent = (self.retry_count - remaining_retries) as f64;
                        let duration = 120_f64.min(2_f64.powf(exponent));
                        tokio::time::sleep(std::time::Duration::from_secs_f64(duration)).await;
                }
                Err(error) => {
                    return Err(error);
                }
            };
        }
    }
    
    fn flush(&self) -> Result<Self::Output,Self::Error> {
        self.node.flush()
    }
}
