//! Node migration between subnets
//!
//! Provides zero-downtime migration of nodes between subnets using a dual-stack
//! approach where nodes temporarily maintain connectivity on both old and new
//! IP addresses during the transition.
//!
//! # Migration Strategy
//!
//! 1. **Plan**: Validate migration feasibility and allocate target resources
//! 2. **Prepare**: Allocate target IP, generate new WireGuard config
//! 3. **Dual-Stack**: Node responds on both old and new IPs
//! 4. **Propagate**: Push new address to all peers
//! 5. **Verify**: Confirm connectivity through new address
//! 6. **Cutover**: Switch primary traffic to new address
//! 7. **Cleanup**: Remove old assignment and peer configs
//!
//! # Components
//!
//! - **Planner**: Validates and plans migrations
//! - **Executor**: Executes migrations with state machine
//! - **Metrics**: Tracks migration progress and success rates

mod executor;
mod planner;

pub use executor::{
    MigrationExecutor, MigrationHandle, MigrationOptions, MigrationProgress, MigrationStateMachine,
    MigrationStep,
};
pub use planner::{
    BulkMigrationPlan, MigrationConstraint, MigrationPlan, MigrationPlanner, MigrationReason,
    MigrationValidation, ValidationIssue,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use uuid::Uuid;

/// Status of a migration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationStatus {
    /// Migration is planned but not started
    Pending,
    /// Migration is in progress
    InProgress,
    /// Dual-stack phase - node on both IPs
    DualStack,
    /// Verifying connectivity on new address
    Verifying,
    /// Migration completed successfully
    Completed,
    /// Migration failed
    Failed,
    /// Migration was cancelled
    Cancelled,
    /// Migration is rolling back
    RollingBack,
    /// Rollback completed
    RolledBack,
}

impl MigrationStatus {
    /// Check if this is a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            MigrationStatus::Completed
                | MigrationStatus::Failed
                | MigrationStatus::Cancelled
                | MigrationStatus::RolledBack
        )
    }

    /// Check if migration can be cancelled
    pub fn can_cancel(&self) -> bool {
        matches!(
            self,
            MigrationStatus::Pending | MigrationStatus::InProgress | MigrationStatus::DualStack
        )
    }

    /// Check if migration can be rolled back
    pub fn can_rollback(&self) -> bool {
        matches!(
            self,
            MigrationStatus::InProgress
                | MigrationStatus::DualStack
                | MigrationStatus::Verifying
                | MigrationStatus::Failed
        )
    }
}

/// A node migration record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Unique migration ID
    pub id: Uuid,
    /// Node being migrated
    pub node_id: Uuid,
    /// Source subnet
    pub source_subnet_id: Uuid,
    /// Target subnet
    pub target_subnet_id: Uuid,
    /// Source IP address
    pub source_ip: Ipv4Addr,
    /// Target IP address (allocated during planning)
    pub target_ip: Option<Ipv4Addr>,
    /// Current status
    pub status: MigrationStatus,
    /// Reason for migration
    pub reason: MigrationReason,
    /// Priority (higher = more urgent)
    pub priority: i32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Current step in the migration
    pub current_step: Option<MigrationStep>,
    /// Number of retry attempts
    pub retry_count: u32,
    /// Maximum retries allowed
    pub max_retries: u32,
    /// Last error message
    pub last_error: Option<String>,
    /// Initiated by (user ID or "system")
    pub initiated_by: Option<String>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

impl Migration {
    /// Create a new migration
    pub fn new(
        node_id: Uuid,
        source_subnet_id: Uuid,
        target_subnet_id: Uuid,
        source_ip: Ipv4Addr,
        reason: MigrationReason,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            node_id,
            source_subnet_id,
            target_subnet_id,
            source_ip,
            target_ip: None,
            status: MigrationStatus::Pending,
            reason,
            priority: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            current_step: None,
            retry_count: 0,
            max_retries: 3,
            last_error: None,
            initiated_by: None,
            metadata: None,
        }
    }

    /// Set the target IP
    pub fn with_target_ip(mut self, ip: Ipv4Addr) -> Self {
        self.target_ip = Some(ip);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set initiated by
    pub fn with_initiated_by(mut self, by: impl Into<String>) -> Self {
        self.initiated_by = Some(by.into());
        self
    }

    /// Mark migration as started
    pub fn start(&mut self) {
        self.status = MigrationStatus::InProgress;
        self.started_at = Some(Utc::now());
    }

    /// Mark migration as completed
    pub fn complete(&mut self) {
        self.status = MigrationStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark migration as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = MigrationStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.last_error = Some(error.into());
    }

    /// Record a retry attempt
    pub fn record_retry(&mut self, error: impl Into<String>) {
        self.retry_count += 1;
        self.last_error = Some(error.into());
    }

    /// Check if migration can retry
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Get duration of the migration
    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            (Some(start), None) => Some(Utc::now() - start),
            _ => None,
        }
    }
}

/// Summary statistics for migrations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MigrationStats {
    /// Total migrations
    pub total: usize,
    /// Pending migrations
    pub pending: usize,
    /// In-progress migrations
    pub in_progress: usize,
    /// Completed migrations
    pub completed: usize,
    /// Failed migrations
    pub failed: usize,
    /// Cancelled migrations
    pub cancelled: usize,
    /// Rolled back migrations
    pub rolled_back: usize,
    /// Average duration in seconds
    pub avg_duration_secs: Option<f64>,
    /// Success rate (0.0 - 1.0)
    pub success_rate: Option<f64>,
}

impl MigrationStats {
    /// Calculate statistics from a list of migrations
    pub fn from_migrations(migrations: &[Migration]) -> Self {
        let mut stats = Self::default();
        stats.total = migrations.len();

        let mut durations = Vec::new();
        let mut completed_or_failed = 0usize;

        for m in migrations {
            match m.status {
                MigrationStatus::Pending => stats.pending += 1,
                MigrationStatus::InProgress
                | MigrationStatus::DualStack
                | MigrationStatus::Verifying => stats.in_progress += 1,
                MigrationStatus::Completed => {
                    stats.completed += 1;
                    completed_or_failed += 1;
                    if let Some(d) = m.duration() {
                        durations.push(d.num_seconds() as f64);
                    }
                }
                MigrationStatus::Failed => {
                    stats.failed += 1;
                    completed_or_failed += 1;
                }
                MigrationStatus::Cancelled => stats.cancelled += 1,
                MigrationStatus::RollingBack | MigrationStatus::RolledBack => {
                    stats.rolled_back += 1
                }
            }
        }

        if !durations.is_empty() {
            stats.avg_duration_secs = Some(durations.iter().sum::<f64>() / durations.len() as f64);
        }

        if completed_or_failed > 0 {
            stats.success_rate = Some(stats.completed as f64 / completed_or_failed as f64);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_creation() {
        let m = Migration::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Ipv4Addr::new(10, 100, 0, 10),
            MigrationReason::PolicyChange,
        );

        assert_eq!(m.status, MigrationStatus::Pending);
        assert!(m.target_ip.is_none());
        assert!(m.started_at.is_none());
    }

    #[test]
    fn test_migration_lifecycle() {
        let mut m = Migration::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Ipv4Addr::new(10, 100, 0, 10),
            MigrationReason::Manual,
        );

        assert!(m.status.can_cancel());
        assert!(!m.status.is_terminal());

        m.start();
        assert_eq!(m.status, MigrationStatus::InProgress);
        assert!(m.started_at.is_some());

        m.complete();
        assert_eq!(m.status, MigrationStatus::Completed);
        assert!(m.completed_at.is_some());
        assert!(m.status.is_terminal());
    }

    #[test]
    fn test_migration_failure_and_retry() {
        let mut m = Migration::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Ipv4Addr::new(10, 100, 0, 10),
            MigrationReason::Manual,
        );
        m.max_retries = 3;

        assert!(m.can_retry());

        m.record_retry("Connection timeout");
        assert_eq!(m.retry_count, 1);
        assert!(m.can_retry());

        m.record_retry("Connection refused");
        m.record_retry("Peer unreachable");
        assert!(!m.can_retry());

        m.fail("Max retries exceeded");
        assert_eq!(m.status, MigrationStatus::Failed);
    }

    #[test]
    fn test_migration_status_transitions() {
        assert!(MigrationStatus::Pending.can_cancel());
        assert!(MigrationStatus::InProgress.can_cancel());
        assert!(MigrationStatus::DualStack.can_cancel());
        assert!(!MigrationStatus::Completed.can_cancel());

        assert!(MigrationStatus::InProgress.can_rollback());
        assert!(MigrationStatus::DualStack.can_rollback());
        assert!(MigrationStatus::Failed.can_rollback());
        assert!(!MigrationStatus::Completed.can_rollback());
        assert!(!MigrationStatus::Pending.can_rollback());
    }

    #[test]
    fn test_migration_stats() {
        let migrations = vec![
            {
                let mut m = Migration::new(
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Ipv4Addr::new(10, 100, 0, 1),
                    MigrationReason::Manual,
                );
                m.complete();
                m
            },
            {
                let mut m = Migration::new(
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Ipv4Addr::new(10, 100, 0, 2),
                    MigrationReason::Manual,
                );
                m.complete();
                m
            },
            {
                let mut m = Migration::new(
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    Ipv4Addr::new(10, 100, 0, 3),
                    MigrationReason::Manual,
                );
                m.fail("Error");
                m
            },
            Migration::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
                Ipv4Addr::new(10, 100, 0, 4),
                MigrationReason::Manual,
            ),
        ];

        let stats = MigrationStats::from_migrations(&migrations);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.pending, 1);

        // Success rate: 2 completed / 3 (completed + failed)
        assert!((stats.success_rate.unwrap() - 0.666).abs() < 0.01);
    }
}
