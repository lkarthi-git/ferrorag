pub trait Source {
    type Item: Send;
    type Error: Send;

    fn next(&mut self) -> impl Future<Output = Result<Option<Self::Item>, Self::Error>> + Send;

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        std::future::ready(Ok(()))
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
            .split("::")
            .last()
            .unwrap_or("Unknown Source Node")
    }
}