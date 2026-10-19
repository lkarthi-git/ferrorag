use ferro_core::node::Node;
use crate::store::{SearchResult};
use std::future::Future;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

pub struct PrintResultsNode;

impl Node for PrintResultsNode {
    type Input = SearchResult;
    // The unit type () implements Default
    type Output = Vec<()>; 
    type Error = BoxError;

    fn execute(&self, res: &Self::Input) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let res = res.clone();
        async move {
                    println!("Score: {:.4} | File: {}", res.score, res.document_uri);
                    println!("Pages: {:?}", res.pages);
                    println!("{}\n", res.text.trim());
                    println!("--------------------------------------------------");
            Ok(Vec::new())
        }
    }
}