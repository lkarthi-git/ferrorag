use csv::Writer;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use ferro_rag::search::{Query, QueryMetadata};
use ferro_rag::store::SearchResult;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// ==========================================
// 1. DATASET LOADER
// ==========================================
pub trait DatasetLoader {
    fn load(
        &self,
    ) -> Result<
        (
            Vec<Query>,
            HashMap<String, Vec<String>>,
            Arc<HashMap<String, String>>,
        ),
        BoxError,
    >;
}

pub struct BeirDatasetLoader {
    qrels_path: String,
    queries_path: String,
}

impl BeirDatasetLoader {
    pub fn new(qrels_path: impl Into<String>, queries_path: impl Into<String>) -> Self {
        Self {
            qrels_path: qrels_path.into(),
            queries_path: queries_path.into(),
        }
    }
}

impl DatasetLoader for BeirDatasetLoader {
    fn load(
        &self,
    ) -> Result<
        (
            Vec<Query>,
            HashMap<String, Vec<String>>,
            Arc<HashMap<String, String>>,
        ),
        BoxError,
    > {
        // 1. Load Ground Truth (qrels)
        let mut ground_truth: HashMap<String, Vec<String>> = HashMap::new();
        let qrels_file = File::open(&self.qrels_path)?;
        for line in BufReader::new(qrels_file).lines().skip(1).flatten() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let score: i32 = parts[2].parse().unwrap_or(0);
                if score > 0 {
                    ground_truth
                        .entry(parts[0].to_string())
                        .or_default()
                        .push(parts[1].to_string());
                }
            }
        }

        // 2. Load Queries
        let mut all_queries = Vec::new();
        let queries_file = File::open(&self.queries_path)?;
        for line in BufReader::new(queries_file).lines().flatten() {
            if let Ok(parsed) = serde_json::from_str::<Value>(&line) {
                if let (Some(q_id), Some(text)) = (parsed["_id"].as_str(), parsed["text"].as_str())
                {
                    all_queries.push(Query {
                        text: text.to_string(),
                        limit: 15,
                        metadata: Arc::new(QueryMetadata {
                            query_id: q_id.to_string(),
                            trace_id: format!("eval-{}", q_id),
                            filter: Option::None,
                            custom: HashMap::new(),
                        }),
                    });
                }
            }
        }

        // 3. Filter and Map
        let queries: Vec<Query> = all_queries
            .into_iter()
            .filter(|q| ground_truth.contains_key(&q.metadata.query_id))
            .collect();

        let query_text_map: HashMap<String, String> = queries
            .iter()
            .map(|q| (q.metadata.query_id.clone(), q.text.clone()))
            .collect();

        Ok((queries, ground_truth, Arc::new(query_text_map)))
    }
}

// ==========================================
// 2. EVALUATOR
// ==========================================
pub struct QueryMetrics {
    pub hit_at_k: bool,
    pub reciprocal_rank: f64,
    pub ndcg: f64,
}

pub trait Evaluator {
    fn evaluate_query(
        &self,
        raw_results: &[SearchResult],
        expected_docs: &[String],
    ) -> QueryMetrics;
}

pub struct StandardDocumentEvaluator {
    pub k: usize,
}

impl StandardDocumentEvaluator {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    fn calculate_ndcg_at_k(results: &[&SearchResult], expected_docs: &[String], k: usize) -> f64 {
        let mut dcg = 0.0;
        for (i, result) in results.iter().take(k).enumerate() {
            if expected_docs
                .iter()
                .any(|exp| result.document_uri.contains(exp))
            {
                let rank = i + 1;
                dcg += 1.0 / ((rank + 1) as f64).log2();
            }
        }

        let mut idcg = 0.0;
        let num_ideal_hits = expected_docs.len().min(k);
        for i in 0..num_ideal_hits {
            let rank = i + 1;
            idcg += 1.0 / ((rank + 1) as f64).log2();
        }

        if idcg == 0.0 { 0.0 } else { dcg / idcg }
    }
}

impl Evaluator for StandardDocumentEvaluator {
    fn evaluate_query(
        &self,
        raw_results: &[SearchResult],
        expected_docs: &[String],
    ) -> QueryMetrics {
        // 1. DEDUPLICATE: Convert chunk-level results to document-level results
        let mut seen_uris = HashSet::new();
        let doc_level_results: Vec<&SearchResult> = raw_results
            .iter()
            .filter(|res| seen_uris.insert(&res.document_uri))
            .collect();

        // 2. Calculate Rank
        let mut rank = 0;
        for (i, result) in doc_level_results.iter().enumerate() {
            if expected_docs
                .iter()
                .any(|exp| result.document_uri.contains(exp))
            {
                rank = i + 1;
                break;
            }
        }

        let reciprocal_rank = if rank > 0 { 1.0 / (rank as f64) } else { 0.0 };
        let hit_at_k = rank > 0 && rank <= self.k;
        let ndcg = Self::calculate_ndcg_at_k(&doc_level_results, expected_docs, self.k);

        QueryMetrics {
            hit_at_k,
            reciprocal_rank,
            ndcg,
        }
    }
}

// ==========================================
// 3. REPORTER
// ==========================================
pub trait ResultReporter {
    fn report(
        &mut self,
        query_id: &str,
        query_text: &str,
        metrics: &QueryMetrics,
    ) -> Result<(), BoxError>;
    fn flush(&mut self) -> Result<(), BoxError>;
}

#[derive(Serialize)]
struct CsvEvalRow<'a> {
    query_id: &'a str,
    query_text: &'a str,
    hit_at_k: bool,
    reciprocal_rank: f64,
    ndcg: f64,
}

pub struct CsvReporter {
    wtr: Writer<File>,
}

impl CsvReporter {
    pub fn new(path: &str) -> Result<Self, BoxError> {
        Ok(Self {
            wtr: Writer::from_path(path)?,
        })
    }
}

impl ResultReporter for CsvReporter {
    fn report(
        &mut self,
        query_id: &str,
        query_text: &str,
        metrics: &QueryMetrics,
    ) -> Result<(), BoxError> {
        self.wtr.serialize(CsvEvalRow {
            query_id,
            query_text,
            hit_at_k: metrics.hit_at_k,
            reciprocal_rank: metrics.reciprocal_rank,
            ndcg: metrics.ndcg,
        })?;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), BoxError> {
        self.wtr.flush()?;
        Ok(())
    }
}
