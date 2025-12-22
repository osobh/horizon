//! Snapshot configuration
pub struct SnapshotConfig {
    pub retention_days: u32,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub max_snapshots: usize,
    pub storage_backend: super::storage::StorageBackend,
}
