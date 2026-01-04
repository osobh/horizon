//! Data repair and recovery operations

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::corruption::RepairStrategy;

/// Repair record for tracking repair operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairRecord {
    /// Record ID
    pub id: Uuid,
    /// Corruption detection ID
    pub detection_id: Uuid,
    /// Object ID being repaired
    pub object_id: Uuid,
    /// Repair strategy used
    pub strategy: RepairStrategy,
    /// Repair status
    pub status: RepairStatus,
    /// Started timestamp
    pub started_at: DateTime<Utc>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Repair duration
    pub duration: Option<Duration>,
    /// Bytes repaired
    pub bytes_repaired: u64,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Validation performed after repair
    pub validated: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Repair status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairStatus {
    /// Repair pending
    Pending,
    /// Repair in progress
    InProgress,
    /// Repair completed successfully
    Completed,
    /// Repair failed
    Failed,
    /// Repair cancelled
    Cancelled,
    /// Validation in progress
    Validating,
    /// Validated successfully
    Validated,
}

impl RepairRecord {
    /// Create new repair record
    pub fn new(detection_id: Uuid, object_id: Uuid, strategy: RepairStrategy) -> Self {
        Self {
            id: Uuid::new_v4(),
            detection_id,
            object_id,
            strategy,
            status: RepairStatus::Pending,
            started_at: Utc::now(),
            completed_at: None,
            duration: None,
            bytes_repaired: 0,
            success: false,
            error_message: None,
            validated: false,
            metadata: HashMap::new(),
        }
    }

    /// Start repair operation
    pub fn start(&mut self) {
        self.status = RepairStatus::InProgress;
        self.started_at = Utc::now();
    }

    /// Complete repair operation
    pub fn complete(&mut self, success: bool, bytes_repaired: u64) {
        self.status = if success {
            RepairStatus::Completed
        } else {
            RepairStatus::Failed
        };
        self.success = success;
        self.bytes_repaired = bytes_repaired;
        self.completed_at = Some(Utc::now());
        if let Some(completed) = self.completed_at {
            self.duration = Some(completed - self.started_at);
        }
    }

    /// Mark as failed with error
    pub fn fail(&mut self, error: String) {
        self.status = RepairStatus::Failed;
        self.success = false;
        self.error_message = Some(error);
        self.completed_at = Some(Utc::now());
        if let Some(completed) = self.completed_at {
            self.duration = Some(completed - self.started_at);
        }
    }

    /// Start validation
    pub fn start_validation(&mut self) {
        self.status = RepairStatus::Validating;
    }

    /// Complete validation
    pub fn complete_validation(&mut self, success: bool) {
        self.validated = success;
        if success {
            self.status = RepairStatus::Validated;
        }
    }

    /// Get repair duration in seconds
    pub fn duration_seconds(&self) -> Option<i64> {
        self.duration.map(|d| d.num_seconds())
    }
}

/// Repair configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConfig {
    /// Maximum repair attempts
    pub max_attempts: u32,
    /// Retry delay between attempts
    pub retry_delay: Duration,
    /// Validate after repair
    pub validate_after_repair: bool,
    /// Parallel repair operations allowed
    pub max_parallel_repairs: usize,
    /// Timeout for repair operations
    pub repair_timeout: Duration,
    /// Preferred repair strategies in order
    pub preferred_strategies: Vec<RepairStrategy>,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            retry_delay: Duration::seconds(30),
            validate_after_repair: true,
            max_parallel_repairs: 4,
            repair_timeout: Duration::minutes(30),
            preferred_strategies: vec![
                RepairStrategy::UseRedundantCopy {
                    copy_location: String::new(),
                },
                RepairStrategy::ReconstructFromParity,
                RepairStrategy::ForwardErrorCorrection,
            ],
        }
    }
}
