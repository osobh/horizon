//! Integrity metrics and statistics

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::corruption::CorruptionSeverity;
use super::types::ChecksumAlgorithm;

/// Integrity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityMetrics {
    /// Total objects monitored
    pub total_objects: u64,
    /// Total bytes monitored
    pub total_bytes: u64,
    /// Total verifications performed
    pub total_verifications: u64,
    /// Successful verifications
    pub successful_verifications: u64,
    /// Failed verifications
    pub failed_verifications: u64,
    /// Corruptions detected
    pub corruptions_detected: u64,
    /// Corruptions by severity
    pub corruptions_by_severity: HashMap<CorruptionSeverity, u64>,
    /// Repairs attempted
    pub repairs_attempted: u64,
    /// Successful repairs
    pub successful_repairs: u64,
    /// Failed repairs
    pub failed_repairs: u64,
    /// Average verification time
    pub avg_verification_time: Duration,
    /// Average repair time
    pub avg_repair_time: Duration,
    /// Bytes repaired
    pub bytes_repaired: u64,
    /// Algorithm usage count
    pub algorithm_usage: HashMap<ChecksumAlgorithm, u64>,
    /// Last verification timestamp
    pub last_verification: Option<DateTime<Utc>>,
    /// Last corruption detected
    pub last_corruption: Option<DateTime<Utc>>,
    /// Metrics start time
    pub started_at: DateTime<Utc>,
}

impl Default for IntegrityMetrics {
    fn default() -> Self {
        Self {
            total_objects: 0,
            total_bytes: 0,
            total_verifications: 0,
            successful_verifications: 0,
            failed_verifications: 0,
            corruptions_detected: 0,
            corruptions_by_severity: HashMap::new(),
            repairs_attempted: 0,
            successful_repairs: 0,
            failed_repairs: 0,
            avg_verification_time: Duration::zero(),
            avg_repair_time: Duration::zero(),
            bytes_repaired: 0,
            algorithm_usage: HashMap::new(),
            last_verification: None,
            last_corruption: None,
            started_at: Utc::now(),
        }
    }
}

impl IntegrityMetrics {
    /// Record verification
    pub fn record_verification(&mut self, success: bool, duration: Duration) {
        self.total_verifications += 1;
        if success {
            self.successful_verifications += 1;
        } else {
            self.failed_verifications += 1;
        }

        // Update average verification time
        let total_time = self.avg_verification_time * self.total_verifications as i32;
        self.avg_verification_time = (total_time + duration) / self.total_verifications as i32;

        self.last_verification = Some(Utc::now());
    }

    /// Record corruption detection
    pub fn record_corruption(&mut self, severity: CorruptionSeverity) {
        self.corruptions_detected += 1;
        *self.corruptions_by_severity.entry(severity).or_insert(0) += 1;
        self.last_corruption = Some(Utc::now());
    }

    /// Record repair attempt
    pub fn record_repair(&mut self, success: bool, bytes: u64, duration: Duration) {
        self.repairs_attempted += 1;
        if success {
            self.successful_repairs += 1;
            self.bytes_repaired += bytes;
        } else {
            self.failed_repairs += 1;
        }

        // Update average repair time
        let total_time = self.avg_repair_time * self.repairs_attempted as i32;
        self.avg_repair_time = (total_time + duration) / self.repairs_attempted as i32;
    }

    /// Record algorithm usage
    pub fn record_algorithm_usage(&mut self, algorithm: ChecksumAlgorithm) {
        *self.algorithm_usage.entry(algorithm).or_insert(0) += 1;
    }

    /// Get verification success rate
    pub fn verification_success_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.successful_verifications as f64 / self.total_verifications as f64
        }
    }

    /// Get repair success rate
    pub fn repair_success_rate(&self) -> f64 {
        if self.repairs_attempted == 0 {
            0.0
        } else {
            self.successful_repairs as f64 / self.repairs_attempted as f64
        }
    }

    /// Get corruption rate
    pub fn corruption_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.corruptions_detected as f64 / self.total_verifications as f64
        }
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        Utc::now() - self.started_at
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
