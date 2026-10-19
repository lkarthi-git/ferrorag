pub mod vector;
pub mod print;


#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct EmbeddedQuery {
    pub text: String,
    pub limit: usize,
    pub vector: Vec<f32>,
}

