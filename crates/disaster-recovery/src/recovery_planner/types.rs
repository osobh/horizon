//! Core types for recovery planning

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Recovery tier priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryTier {
    /// Tier 0 - Critical systems (RTO < 1 hour)
    Critical,
    /// Tier 1 - Essential systems (RTO < 4 hours)
    Essential,
    /// Tier 2 - Important systems (RTO < 24 hours)
    Important,
    /// Tier 3 - Standard systems (RTO < 72 hours)
    Standard,
    /// Tier 4 - Non-critical systems (RTO > 72 hours)
    NonCritical,
}

impl RecoveryTier {
    /// Get default RTO for tier in minutes
    pub fn default_rto_minutes(&self) -> u64 {
        match self {
            RecoveryTier::Critical => 60,       // 1 hour
            RecoveryTier::Essential => 240,     // 4 hours
            RecoveryTier::Important => 1440,    // 24 hours
            RecoveryTier::Standard => 4320,     // 72 hours
            RecoveryTier::NonCritical => 10080, // 7 days
        }
    }

    /// Get default RPO for tier in minutes
    pub fn default_rpo_minutes(&self) -> u64 {
        match self {
            RecoveryTier::Critical => 0,       // Zero data loss
            RecoveryTier::Essential => 15,     // 15 minutes
            RecoveryTier::Important => 60,     // 1 hour
            RecoveryTier::Standard => 240,     // 4 hours
            RecoveryTier::NonCritical => 1440, // 24 hours
        }
    }
}

/// Recovery step types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepType {
    /// Validate prerequisites
    ValidatePrerequisites,
    /// Allocate resources
    AllocateResources,
    /// Start backup services
    StartBackupServices,
    /// Switch traffic
    SwitchTraffic,
    /// Verify health
    VerifyHealth,
    /// Rollback changes
    Rollback,
    /// Cleanup resources
    Cleanup,
    /// Send notifications
    SendNotifications,
    /// Custom script execution
    CustomScript { script_path: String },
    /// Wait for duration
    Wait { duration_seconds: u64 },
}

/// Recovery step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    /// Step identifier
    pub id: Uuid,
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: StepType,
    /// Expected duration in minutes
    pub estimated_duration_minutes: u64,
    /// Prerequisites that must be completed first
    pub prerequisites: Vec<Uuid>,
    /// Whether step can run in parallel with others
    pub parallel: bool,
    /// Whether step is mandatory for recovery
    pub mandatory: bool,
    /// Retry configuration for this step
    pub retry_config: Option<super::config::RetryConfig>,
    /// Rollback steps if this fails
    pub rollback_steps: Vec<Uuid>,
    /// Custom parameters for step execution
    pub parameters: HashMap<String, String>,
}

/// Plan update information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanUpdate {
    /// Update ID
    pub id: Uuid,
    /// Plan ID being updated
    pub plan_id: Uuid,
    /// Update timestamp
    pub timestamp: DateTime<Utc>,
    /// Update reason
    pub reason: String,
    /// Changes made
    pub changes: Vec<PlanChange>,
}

/// Individual plan change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanChange {
    /// Change type
    pub change_type: ChangeType,
    /// Field that changed
    pub field: String,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: String,
}

/// Types of plan changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Field was added
    Added,
    /// Field was modified
    Modified,
    /// Field was removed
    Removed,
}
