//! Verification tasks and scheduling

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::types::{ChecksumAlgorithm, DataObject, IntegrityCheckType, VerificationSchedule};

/// Verification task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationTask {
    /// Task ID
    pub id: Uuid,
    /// Object to verify
    pub object: DataObject,
    /// Check type
    pub check_type: IntegrityCheckType,
    /// Algorithm to use
    pub algorithm: ChecksumAlgorithm,
    /// Schedule
    pub schedule: VerificationSchedule,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last run timestamp
    pub last_run: Option<DateTime<Utc>>,
    /// Next scheduled run
    pub next_run: Option<DateTime<Utc>>,
    /// Run count
    pub run_count: u64,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
    /// Enabled flag
    pub enabled: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl VerificationTask {
    /// Create new verification task
    pub fn new(
        object: DataObject,
        check_type: IntegrityCheckType,
        algorithm: ChecksumAlgorithm,
        schedule: VerificationSchedule,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            object,
            check_type,
            algorithm,
            schedule,
            priority: 5,
            created_at: Utc::now(),
            last_run: None,
            next_run: None,
            run_count: 0,
            success_count: 0,
            failure_count: 0,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Record task execution
    pub fn record_execution(&mut self, success: bool) {
        self.run_count += 1;
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
        self.last_run = Some(Utc::now());
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.run_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.run_count as f64
        }
    }

    /// Check if task should run now
    pub fn should_run_now(&self) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.next_run {
            Some(next) => Utc::now() >= *next,
            None => true,
        }
    }

    /// Calculate next run time
    pub fn calculate_next_run(&mut self) {
        use chrono::{Datelike, Duration, Timelike};

        let now = Utc::now();

        self.next_run = Some(match &self.schedule {
            VerificationSchedule::Continuous => now + Duration::seconds(1),
            VerificationSchedule::Hourly => {
                let next = now + Duration::hours(1);
                next.with_minute(0)
                    .and_then(|t| t.with_second(0))
                    .unwrap_or(next)
            }
            VerificationSchedule::Daily { hour } => {
                let next = now + Duration::days(1);
                next.with_hour(*hour)
                    .and_then(|t| t.with_minute(0))
                    .and_then(|t| t.with_second(0))
                    .unwrap_or(next)
            }
            VerificationSchedule::Weekly { day_of_week, hour } => {
                let days_until =
                    (*day_of_week as i64 - now.weekday().num_days_from_monday() as i64 + 7) % 7;
                let next = now + Duration::days(if days_until == 0 { 7 } else { days_until });
                next.with_hour(*hour)
                    .and_then(|t| t.with_minute(0))
                    .and_then(|t| t.with_second(0))
                    .unwrap_or(next)
            }
            VerificationSchedule::Monthly { day_of_month, hour } => {
                let next = if now.day() >= *day_of_month {
                    // Next month
                    let next_month = now
                        .with_day(1)
                        .and_then(|d| d.checked_add_signed(Duration::days(32)))
                        .and_then(|d| d.with_day(1))
                        .unwrap_or(now);
                    next_month.with_day(*day_of_month).unwrap_or(next_month)
                } else {
                    // This month
                    now.with_day(*day_of_month).unwrap_or(now)
                };
                next.with_hour(*hour)
                    .and_then(|t| t.with_minute(0))
                    .and_then(|t| t.with_second(0))
                    .unwrap_or(next)
            }
            VerificationSchedule::Cron(_) => {
                // For cron, would need a cron parser
                now + Duration::hours(1)
            }
        });
    }
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Task ID
    pub task_id: Uuid,
    /// Object ID
    pub object_id: Uuid,
    /// Verification timestamp
    pub timestamp: DateTime<Utc>,
    /// Success flag
    pub success: bool,
    /// Expected checksum
    pub expected_checksum: Option<Vec<u8>>,
    /// Actual checksum
    pub actual_checksum: Option<Vec<u8>>,
    /// Error message if failed
    pub error: Option<String>,
    /// Verification duration in milliseconds
    pub duration_ms: u64,
}
