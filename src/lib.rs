use crate::pipeline::Node;

pub mod pipeline;

pub struct LengthChunker {
    pub chunks: Vec<String>,
    pub counter: usize,
    pub chunk_size: usize,
}


impl Node for LengthChunker {
    type Input = String;
    type Output = Vec<String>;
    type Error = String;

    async fn execute(&mut self, input: Self::Input) -> Result<Self::Output,Self::Error> {
        self.counter += input.len();
        self.chunks.push(input);
        
        if self.counter >= self.chunk_size {
            // Create the chunk, reset state, and return it
            let completed_chunk = self.chunks.join("\n");
            self.chunks.clear();
            self.counter = 0;
            Ok((vec![completed_chunk]))
        } else {
            // Buffer isn't full yet, return nothing to the pipeline
            Ok(vec![])
        }
    }

    fn flush(&mut self) -> Result<Self::Output,Self::Error> {
            Ok(vec![self.chunks.join("\n")])
    }
}


pub struct MockEmbedder {
     pub embeddings: Vec<String>,
}

impl Node for MockEmbedder {
    type Input = String;
    type Output = Vec<String>;
    type Error = String;

    async fn execute(&mut self, input: Self::Input) -> Result<Self::Output,Self::Error> 
    {
        let mock_embedding = format!("EMBEDDING_FOR: {}", input);
        self.embeddings.push(mock_embedding.clone());
        Ok(vec![mock_embedding])
    }
}




