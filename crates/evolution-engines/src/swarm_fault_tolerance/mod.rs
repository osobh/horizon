//! Fault tolerance capabilities for distributed SwarmAgentic systems
//!
//! This module provides node failure detection, particle state recovery,
//! distributed checkpointing, and backup/restoration mechanisms.

mod checkpoint_manager;
mod fault_detector;
mod recovery_executors;
mod recovery_manager;
mod storage;
mod types;

#[cfg(test)]
mod tests;

pub use checkpoint_manager::CheckpointManager;
pub use fault_detector::{FaultDetector, FaultToleranceConfig};
pub use recovery_executors::{
    CheckpointRecovery, HybridRecovery, RecoveryExecutor, RedistributeRecovery,
};
pub use recovery_manager::RecoveryManager;
pub use storage::{CheckpointStorage, StorageType};
pub use types::{
    AlertThresholds, CheckpointMetadata, CheckpointSnapshot, CompressionAlgorithm,
    FailureDetectionAlgorithm, HealthStatus, NodeHealth, NodeState, RecoveryEvent, RecoveryStatus,
};
