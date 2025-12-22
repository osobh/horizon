//! Corruption detection and classification

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::types::ChecksumAlgorithm;

/// Corruption severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorruptionSeverity {
    /// Critical corruption - data loss imminent
    Critical,
    /// High severity - significant data corruption
    High,
    /// Medium severity - recoverable corruption
    Medium,
    /// Low severity - minor inconsistencies
    Low,
    /// Information only - potential issues detected
    Info,
}

/// Corruption detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptionDetection {
    /// Detection ID
    pub id: Uuid,
    /// Object ID
    pub object_id: Uuid,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Corruption type
    pub corruption_type: CorruptionType,
    /// Severity level
    pub severity: CorruptionSeverity,
    /// Affected byte ranges
    pub affected_ranges: Vec<ByteRange>,
    /// Detection method used
    pub detection_method: DetectionMethod,
    /// Expected checksum
    pub expected_checksum: Option<Vec<u8>>,
    /// Actual checksum
    pub actual_checksum: Option<Vec<u8>>,
    /// Error details
    pub error_details: String,
    /// Suggested repair strategies
    pub repair_strategies: Vec<RepairStrategy>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Corruption type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorruptionType {
    /// Checksum mismatch
    ChecksumMismatch,
    /// Bit rot detected
    BitRot,
    /// File truncation
    Truncation,
    /// Header corruption
    HeaderCorruption,
    /// Block corruption
    BlockCorruption { block_index: u64 },
    /// Silent data corruption
    SilentCorruption,
    /// Metadata corruption
    MetadataCorruption,
    /// Cross-reference mismatch
    CrossReferenceMismatch,
    /// Other corruption type
    Other(String),
}

/// Byte range affected by corruption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteRange {
    /// Start offset
    pub start: u64,
    /// End offset
    pub end: u64,
    /// Size in bytes
    pub size: u64,
}

/// Detection method used
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Checksum verification
    ChecksumVerification(ChecksumAlgorithm),
    /// Parity check
    ParityCheck,
    /// Pattern analysis
    PatternAnalysis,
    /// Cross-reference validation
    CrossReference,
    /// Scrubbing operation
    Scrubbing,
    /// Real-time I/O monitoring
    RealtimeMonitoring,
    /// Manual inspection
    Manual,
}

/// Repair strategy for corruption
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Restore from backup
    RestoreFromBackup { backup_id: Uuid },
    /// Use redundant copy
    UseRedundantCopy { copy_location: String },
    /// Reconstruct from parity
    ReconstructFromParity,
    /// Apply forward error correction
    ForwardErrorCorrection,
    /// Attempt data recovery
    DataRecovery,
    /// Mark as bad and isolate
    IsolateCorruption,
    /// Manual intervention required
    ManualRepair,
    /// No repair possible
    NoRepair,
}

impl CorruptionDetection {
    /// Create new corruption detection record
    pub fn new(
        object_id: Uuid,
        corruption_type: CorruptionType,
        severity: CorruptionSeverity,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            object_id,
            detected_at: Utc::now(),
            corruption_type,
            severity,
            affected_ranges: Vec::new(),
            detection_method: DetectionMethod::Manual,
            expected_checksum: None,
            actual_checksum: None,
            error_details: String::new(),
            repair_strategies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add affected byte range
    pub fn add_affected_range(&mut self, start: u64, end: u64) {
        self.affected_ranges.push(ByteRange {
            start,
            end,
            size: end - start,
        });
    }

    /// Set checksums for comparison
    pub fn set_checksums(&mut self, expected: Vec<u8>, actual: Vec<u8>) {
        self.expected_checksum = Some(expected);
        self.actual_checksum = Some(actual);
    }

    /// Add repair strategy
    pub fn add_repair_strategy(&mut self, strategy: RepairStrategy) {
        if !self.repair_strategies.contains(&strategy) {
            self.repair_strategies.push(strategy);
        }
    }

    /// Get total corrupted bytes
    pub fn total_corrupted_bytes(&self) -> u64 {
        self.affected_ranges.iter().map(|r| r.size).sum()
    }
}