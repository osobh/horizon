//! Snapshot management for disaster recovery

mod config;
mod manager;
mod metadata;
mod storage;
mod types;
mod verification;

pub use config::SnapshotConfig;
pub use manager::SnapshotManager;
pub use metadata::{SnapshotMetadata, SnapshotState};
pub use storage::{SnapshotStorage, StorageBackend};
pub use types::{Snapshot, SnapshotType};
pub use verification::SnapshotVerifier;
