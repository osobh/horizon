//! Snapshot metadata
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub description: String,
    pub tags: std::collections::HashMap<String, String>,
    pub checksum: String,
    pub compressed: bool,
    pub encrypted: bool,
    pub state: SnapshotState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotState {
    Creating,
    Available,
    Failed,
    Deleting,
}
