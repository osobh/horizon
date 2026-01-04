//! Integrity verification for audit logs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Integrity verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityVerificationResult {
    /// Verification timestamp
    pub timestamp: DateTime<Utc>,
    /// Overall integrity status
    pub is_valid: bool,
    /// Number of entries verified
    pub entries_verified: usize,
    /// Number of integrity errors found
    pub errors_found: usize,
    /// Detailed error information
    pub errors: Vec<IntegrityError>,
    /// Verification metadata
    pub metadata: HashMap<String, String>,
    /// Time taken for verification
    pub verification_duration_ms: u64,
    /// Cryptographic proof of verification
    pub verification_proof: Option<String>,
}

/// Integrity error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityError {
    /// Entry ID where error was found
    pub entry_id: Uuid,
    /// Error type
    pub error_type: IntegrityErrorType,
    /// Error description
    pub description: String,
    /// Expected value
    pub expected: Option<String>,
    /// Actual value
    pub actual: Option<String>,
    /// Error severity
    pub severity: IntegrityErrorSeverity,
}

/// Types of integrity errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrityErrorType {
    /// Hash mismatch
    HashMismatch,
    /// Broken chain link
    BrokenChain,
    /// Missing entry
    MissingEntry,
    /// Duplicate entry
    DuplicateEntry,
    /// Timestamp anomaly
    TimestampAnomaly,
    /// Signature verification failure
    SignatureFailure,
    /// Metadata corruption
    MetadataCorruption,
}

/// Severity of integrity errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IntegrityErrorSeverity {
    /// Minor issue that doesn't affect integrity
    Minor,
    /// Moderate issue that may affect integrity
    Moderate,
    /// Severe issue that compromises integrity
    Severe,
    /// Critical issue requiring immediate action
    Critical,
}

impl IntegrityVerificationResult {
    /// Create a new verification result
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            is_valid: true,
            entries_verified: 0,
            errors_found: 0,
            errors: Vec::new(),
            metadata: HashMap::new(),
            verification_duration_ms: 0,
            verification_proof: None,
        }
    }

    /// Add an integrity error
    pub fn add_error(&mut self, error: IntegrityError) {
        self.errors.push(error);
        self.errors_found += 1;
        self.is_valid = false;
    }

    /// Set verification statistics
    pub fn set_statistics(&mut self, entries_verified: usize, duration_ms: u64) {
        self.entries_verified = entries_verified;
        self.verification_duration_ms = duration_ms;
    }

    /// Check if verification found critical errors
    pub fn has_critical_errors(&self) -> bool {
        self.errors
            .iter()
            .any(|e| e.severity == IntegrityErrorSeverity::Critical)
    }

    /// Get errors by type
    pub fn errors_by_type(&self, error_type: IntegrityErrorType) -> Vec<&IntegrityError> {
        self.errors
            .iter()
            .filter(|e| e.error_type == error_type)
            .collect()
    }

    /// Get errors by severity
    pub fn errors_by_severity(&self, severity: IntegrityErrorSeverity) -> Vec<&IntegrityError> {
        self.errors
            .iter()
            .filter(|e| e.severity == severity)
            .collect()
    }

    /// Generate verification summary
    pub fn summary(&self) -> String {
        format!(
            "Integrity Verification: {} | Entries: {} | Errors: {} | Duration: {}ms",
            if self.is_valid { "VALID" } else { "INVALID" },
            self.entries_verified,
            self.errors_found,
            self.verification_duration_ms
        )
    }
}

impl IntegrityError {
    /// Create a new integrity error
    pub fn new(
        entry_id: Uuid,
        error_type: IntegrityErrorType,
        description: String,
        severity: IntegrityErrorSeverity,
    ) -> Self {
        Self {
            entry_id,
            error_type,
            description,
            expected: None,
            actual: None,
            severity,
        }
    }

    /// Set expected and actual values
    pub fn with_values(mut self, expected: String, actual: String) -> Self {
        self.expected = Some(expected);
        self.actual = Some(actual);
        self
    }
}

impl Default for IntegrityVerificationResult {
    fn default() -> Self {
        Self::new()
    }
}
