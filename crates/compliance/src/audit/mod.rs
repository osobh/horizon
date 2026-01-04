//! Immutable Audit Logging System
//!
//! Provides comprehensive audit logging capabilities including:
//! - Immutable audit trail creation
//! - Cryptographic integrity verification
//! - Tamper detection and prevention
//! - Compliance reporting and analysis
//! - Long-term audit record preservation

pub mod chain;
pub mod compliance;
pub mod engine;
pub mod entry;
pub mod integrity;
pub mod query;
pub mod types;

// Re-export main types
pub use chain::AuditChain;
pub use compliance::{
    AuditComplianceReport, ComplianceIssue, ComplianceIssueSeverity, ComplianceIssueType,
    CoverageAssessment,
};
pub use engine::AuditLogEngine;
pub use entry::AuditLogEntry;
pub use integrity::{
    IntegrityError, IntegrityErrorSeverity, IntegrityErrorType, IntegrityVerificationResult,
};
pub use query::AuditQuery;
pub use types::{AuditEventType, AuditOutcome, AuditSeverity};
