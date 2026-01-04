//! Data integrity configuration

use chrono::Duration;
use serde::{Deserialize, Serialize};

use super::repair::RepairConfig;
use super::types::{ChecksumAlgorithm, IntegrityCheckType, VerificationSchedule};

/// Data integrity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIntegrityConfig {
    /// Default checksum algorithm
    pub default_algorithm: ChecksumAlgorithm,
    /// Enabled checksum algorithms
    pub enabled_algorithms: Vec<ChecksumAlgorithm>,
    /// Default check type
    pub default_check_type: IntegrityCheckType,
    /// Verification schedule
    pub verification_schedule: VerificationSchedule,
    /// Block size for block-level checksums (bytes)
    pub block_size: usize,
    /// Enable real-time corruption detection
    pub realtime_detection: bool,
    /// Enable automatic repair
    pub auto_repair: bool,
    /// Repair configuration
    pub repair_config: RepairConfig,
    /// Alert on corruption detection
    pub alert_on_corruption: bool,
    /// Maximum parallel verifications
    pub max_parallel_verifications: usize,
    /// Verification timeout
    pub verification_timeout: Duration,
    /// Retention period for audit logs
    pub audit_retention_days: u32,
}

impl Default for DataIntegrityConfig {
    fn default() -> Self {
        Self {
            default_algorithm: ChecksumAlgorithm::SHA256,
            enabled_algorithms: vec![
                ChecksumAlgorithm::SHA256,
                ChecksumAlgorithm::SHA512,
                ChecksumAlgorithm::Blake2b,
            ],
            default_check_type: IntegrityCheckType::Full,
            verification_schedule: VerificationSchedule::Daily { hour: 2 },
            block_size: 4096,
            realtime_detection: true,
            auto_repair: false,
            repair_config: RepairConfig::default(),
            alert_on_corruption: true,
            max_parallel_verifications: 4,
            verification_timeout: Duration::minutes(30),
            audit_retention_days: 90,
        }
    }
}

impl DataIntegrityConfig {
    /// Create config for high security environments
    pub fn high_security() -> Self {
        Self {
            default_algorithm: ChecksumAlgorithm::SHA512,
            enabled_algorithms: vec![ChecksumAlgorithm::SHA512, ChecksumAlgorithm::Blake2b],
            default_check_type: IntegrityCheckType::Full,
            verification_schedule: VerificationSchedule::Continuous,
            block_size: 1024,
            realtime_detection: true,
            auto_repair: false,
            repair_config: RepairConfig {
                validate_after_repair: true,
                max_attempts: 5,
                ..RepairConfig::default()
            },
            alert_on_corruption: true,
            max_parallel_verifications: 8,
            verification_timeout: Duration::minutes(60),
            audit_retention_days: 365,
        }
    }

    /// Create config for high performance environments
    pub fn high_performance() -> Self {
        Self {
            default_algorithm: ChecksumAlgorithm::XXHash,
            enabled_algorithms: vec![ChecksumAlgorithm::XXHash, ChecksumAlgorithm::CRC32],
            default_check_type: IntegrityCheckType::Block,
            verification_schedule: VerificationSchedule::Weekly {
                day_of_week: 0,
                hour: 3,
            },
            block_size: 16384,
            realtime_detection: false,
            auto_repair: true,
            repair_config: RepairConfig {
                validate_after_repair: false,
                max_parallel_repairs: 8,
                ..RepairConfig::default()
            },
            alert_on_corruption: false,
            max_parallel_verifications: 16,
            verification_timeout: Duration::minutes(15),
            audit_retention_days: 30,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled_algorithms.is_empty() {
            return Err("At least one algorithm must be enabled".to_string());
        }

        if !self.enabled_algorithms.contains(&self.default_algorithm) {
            return Err("Default algorithm must be in enabled algorithms".to_string());
        }

        if self.block_size == 0 {
            return Err("Block size must be greater than 0".to_string());
        }

        if self.max_parallel_verifications == 0 {
            return Err("Must allow at least one parallel verification".to_string());
        }

        Ok(())
    }
}
