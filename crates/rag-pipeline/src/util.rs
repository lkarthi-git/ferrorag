use hf_hub::{HFClientSync, RepoTypeAny};
use std::path::PathBuf;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Downloads a single file from a Hugging Face model repository.
pub fn hf_download(repo_id: &str, revision: &str, filename: &str) -> Result<PathBuf, BoxError> {
    let client = HFClientSync::new().map_err(|e| format!("Failed to init HF API: {}", e))?;

    let (owner, name) = match repo_id.split_once('/') {
        Some((owner, name)) => (owner.to_string(), name.to_string()),
        None => (String::new(), repo_id.to_string()),
    };

    let repo = client.repository(RepoTypeAny::Model, owner, name);

    let path = repo
        .download_file()
        .filename(filename)
        .revision(revision.to_string())
        .send()
        .map_err(|e| format!("Failed to download '{}' from '{}': {}", filename, repo_id, e))?;

    Ok(path)
}