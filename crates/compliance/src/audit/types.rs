//! Core types for audit logging

use serde::{Deserialize, Serialize};

/// Audit event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Data access event
    DataAccess,
    /// Data modification event
    DataModification,
    /// Data deletion event
    DataDeletion,
    /// Data creation event
    DataCreation,
    /// Authentication event
    Authentication,
    /// Authorization event
    Authorization,
    /// System configuration change
    ConfigurationChange,
    /// Policy violation
    PolicyViolation,
    /// Security incident
    SecurityIncident,
    /// Compliance check
    ComplianceCheck,
    /// Encryption/Decryption operation
    CryptographicOperation,
    /// Data export/transfer
    DataTransfer,
    /// User management
    UserManagement,
    /// System maintenance
    SystemMaintenance,
}

/// Audit severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum AuditSeverity {
    /// Informational event
    Info,
    /// Low importance event
    Low,
    /// Medium importance event
    Medium,
    /// High importance event
    High,
    /// Critical security event
    Critical,
}

/// Audit outcome status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditOutcome {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation was denied
    Denied,
    /// Operation resulted in error
    Error,
    /// Operation was cancelled
    Cancelled,
}

impl AuditEventType {
    /// Check if event requires immediate notification
    pub fn requires_notification(&self) -> bool {
        matches!(
            self,
            AuditEventType::PolicyViolation
                | AuditEventType::SecurityIncident
                | AuditEventType::DataDeletion
                | AuditEventType::ConfigurationChange
        )
    }

    /// Get the typical severity for this event type
    pub fn typical_severity(&self) -> AuditSeverity {
        match self {
            AuditEventType::SecurityIncident | AuditEventType::PolicyViolation => {
                AuditSeverity::Critical
            }
            AuditEventType::DataDeletion
            | AuditEventType::ConfigurationChange
            | AuditEventType::UserManagement => AuditSeverity::High,
            AuditEventType::DataModification
            | AuditEventType::DataTransfer
            | AuditEventType::CryptographicOperation => AuditSeverity::Medium,
            AuditEventType::DataAccess | AuditEventType::Authentication => AuditSeverity::Low,
            _ => AuditSeverity::Info,
        }
    }
}

impl AuditSeverity {
    /// Check if severity requires immediate action
    pub fn requires_immediate_action(&self) -> bool {
        matches!(self, AuditSeverity::Critical | AuditSeverity::High)
    }

    /// Get retention period in days for this severity
    pub fn retention_days(&self) -> u32 {
        match self {
            AuditSeverity::Critical => 2555, // 7 years
            AuditSeverity::High => 1825,     // 5 years
            AuditSeverity::Medium => 1095,   // 3 years
            AuditSeverity::Low => 365,       // 1 year
            AuditSeverity::Info => 90,       // 90 days
        }
    }
}
