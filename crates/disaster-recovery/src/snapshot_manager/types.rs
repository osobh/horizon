//! Snapshot types and structures
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub size_bytes: u64,
    pub snapshot_type: SnapshotType,
    pub metadata: super::metadata::SnapshotMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotType {
    Full,
    Incremental,
    Differential,
}
