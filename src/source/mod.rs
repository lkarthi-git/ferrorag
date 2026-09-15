mod file_source;

pub use file_source::FileSource;

pub trait Source {
    type Item;
    type Error;
    fn next(&mut self) -> impl Future<Output = Result<Option<Self::Item>, Self::Error>> + Send;
}

