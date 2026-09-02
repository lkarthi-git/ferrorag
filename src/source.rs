use tokio::io::{AsyncBufReadExt, BufReader,Error};
use tokio::fs::File;

pub trait Source {
    type Item;
    type Error;
    fn next(&mut self) -> impl Future<Output = Result<Option<Self::Item>, Self::Error>> + Send;
}

pub struct FileSource {
    lines: tokio::io::Lines<BufReader<File>>,
}

impl FileSource {
    pub fn new(file: File) -> Self {
        let reader: BufReader<File> = BufReader::new(file);
        let lines: tokio::io::Lines<BufReader<File>> = reader.lines();
        Self {
            lines,
        }
    }
}

impl Source for FileSource {
    type Item = String;
    type Error = std::io::Error;
    async fn next(&mut self) -> Result<Option<String>, self::Error>  {
        self.lines.next_line().await
    }
}