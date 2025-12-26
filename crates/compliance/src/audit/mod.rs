//! Immutable Audit Logging System
//!
//! Provides comprehensive audit logging capabilities including:
//! - Immutable audit trail creation
//! - Cryptographic integrity verification
//! - Tamper detection and prevention
//! - Compliance reporting and analysis
//! - Long-term audit record preservation

pub mod types;
pub mod entry;
pub mod chain;
pub mod integrity;
pub mod compliance;
pub mod engine;
pub mod query;

// Re-export main types
pub use types::{AuditEventType, AuditSeverity, AuditOutcome};
pub use entry::AuditLogEntry;
pub use chain::AuditChain;
pub use integrity::{IntegrityVerificationResult, IntegrityError, IntegrityErrorType, IntegrityErrorSeverity};
pub use compliance::{AuditComplianceReport, ComplianceIssue, ComplianceIssueType, ComplianceIssueSeverity, CoverageAssessment};
pub use engine::AuditLogEngine;
pub use query::AuditQuery;