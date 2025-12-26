//! Storage backends for snapshots
#[derive(Debug, Clone)]
pub enum StorageBackend {
    Local(String),
    S3(String),
    Azure(String),
    GCS(String),
}

pub struct SnapshotStorage {
    backend: StorageBackend,
}

impl SnapshotStorage {
    pub fn new(backend: StorageBackend) -> Self {
        Self { backend }
    }
}
