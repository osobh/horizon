//! Disaster recovery and business continuity framework with automated failover
//!
//! This crate provides comprehensive disaster recovery capabilities for:
//! - Automated backup and restore operations with scheduling and retention
//! - Multi-site failover orchestration and health monitoring
//! - Real-time data replication with conflict resolution
//! - Service health monitoring with cascade failure detection
//! - RTO/RPO planning with dependency resolution
//! - Point-in-time snapshots with space optimization
//! - Data integrity verification with automated repair
//! - Automated runbook execution with rollback capabilities
//! - Zero data loss objectives and minimal downtime requirements
//! - Geographic redundancy and compliance standards

#![warn(missing_docs)]

pub mod backup_manager;
pub mod data_integrity;
pub mod error;
pub mod failover_coordinator;
pub mod health_monitor;
pub mod recovery_planner;
pub mod replication_manager;
pub mod runbook_executor;
pub mod snapshot_manager;

#[cfg(test)]
mod edge_case_tests;

// Core error types and results
pub use error::{DisasterRecoveryError, DisasterRecoveryResult};

// Backup management exports
pub use backup_manager::{
    BackupConfig, BackupFilters, BackupManager, BackupMetadata, BackupState, BackupStats,
    BackupType,
};

// Failover coordination exports
pub use failover_coordinator::{
    FailoverConfig, FailoverCoordinator, FailoverEvent, FailoverMetrics, FailoverPlan,
    FailoverState, FailoverTrigger, Site, SiteRole, SiteState,
};

// Replication management exports
pub use replication_manager::{
    ConflictResolution, ReplicationConfig, ReplicationManager, ReplicationMetrics, ReplicationMode,
    ReplicationNode, ReplicationState, ReplicationStream,
};

// Health monitoring exports
pub use health_monitor::{
    AlertSeverity, AlertType, CircuitBreakerState, HealthAlert, HealthMetrics, HealthMonitor,
    HealthMonitorConfig, HealthStatus, ServiceHealth,
};

// Recovery planning exports
pub use recovery_planner::{
    RecoveryObjective, RecoveryPlanner, RecoveryPlannerConfig, RecoveryPlannerMetrics,
    RecoveryStrategy, RecoveryTier, RecoveryTimeline, ServiceRecoveryPlan,
};

// Snapshot management exports
pub use snapshot_manager::{
    ConsistencyLevel, Snapshot, SnapshotChain, SnapshotManager, SnapshotManagerConfig,
    SnapshotMetrics, SnapshotState, SnapshotType,
};

// Data integrity exports
pub use data_integrity::{
    ChecksumAlgorithm, CorruptionDetection, DataIntegrityConfig, DataIntegrityManager, DataObject,
    IntegrityMetrics, IntegrityStatus, RepairStrategy,
};

// Runbook execution exports
pub use runbook_executor::{
    ExecutionState, Runbook, RunbookCategory, RunbookExecution, RunbookExecutor,
    RunbookExecutorConfig, RunbookExecutorMetrics, StepType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_manager_creation() {
        let config = BackupConfig::default();
        let manager = BackupManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_failover_coordinator_creation() {
        let config = FailoverConfig::default();
        let coordinator = FailoverCoordinator::new(config);
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_replication_manager_creation() {
        let config = ReplicationConfig::default();
        let manager = ReplicationManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let monitor = HealthMonitor::new(config);
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_recovery_planner_creation() {
        let config = RecoveryPlannerConfig::default();
        let planner = RecoveryPlanner::new(config);
        assert!(planner.is_ok());
    }

    #[test]
    fn test_snapshot_manager_creation() {
        let config = SnapshotManagerConfig::default();
        let manager = SnapshotManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_data_integrity_manager_creation() {
        let config = DataIntegrityConfig::default();
        let manager = DataIntegrityManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_runbook_executor_creation() {
        let config = RunbookExecutorConfig::default();
        let executor = RunbookExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_disaster_recovery_error_types() {
        let error = DisasterRecoveryError::BackupFailed {
            reason: "test error".to_string(),
        };

        match error {
            DisasterRecoveryError::BackupFailed { reason } => {
                assert_eq!(reason, "test error");
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_recovery_tiers() {
        assert_eq!(RecoveryTier::Critical.default_rto_minutes(), 60);
        assert_eq!(RecoveryTier::Essential.default_rto_minutes(), 240);
        assert_eq!(RecoveryTier::Important.default_rto_minutes(), 1440);
        assert_eq!(RecoveryTier::Standard.default_rto_minutes(), 4320);
        assert_eq!(RecoveryTier::NonCritical.default_rto_minutes(), 10080);
    }

    #[test]
    fn test_backup_types() {
        let types = vec![
            BackupType::Full,
            BackupType::Incremental,
            BackupType::Differential,
            BackupType::Continuous,
        ];

        for backup_type in types {
            let serialized = serde_json::to_string(&backup_type)?;
            let deserialized: BackupType = serde_json::from_str(&serialized)?;
            assert_eq!(backup_type, deserialized);
        }
    }

    #[test]
    fn test_health_status_values() {
        let statuses = vec![
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
            HealthStatus::Unknown,
            HealthStatus::Maintenance,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status)?;
            let deserialized: HealthStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }
}
