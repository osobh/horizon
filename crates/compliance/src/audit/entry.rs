//! Audit log entry structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

use crate::audit::types::{AuditEventType, AuditSeverity, AuditOutcome};
use crate::data_classification::DataCategory;

/// Immutable audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique entry identifier
    pub id: Uuid,
    /// Timestamp when event occurred
    pub timestamp: DateTime<Utc>,
    /// Type of audit event
    pub event_type: AuditEventType,
    /// Severity level
    pub severity: AuditSeverity,
    /// Operation outcome
    pub outcome: AuditOutcome,
    /// User or system that initiated the event
    pub actor: String,
    /// Target resource or entity
    pub target: String,
    /// Event description
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Data categories involved (if applicable)
    pub data_categories: Vec<DataCategory>,
    /// IP address of the actor
    pub ip_address: Option<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Correlation ID for related events
    pub correlation_id: Option<Uuid>,
    /// Previous entry hash for chain integrity
    pub previous_hash: Option<String>,
    /// Cryptographic hash of this entry
    pub hash: String,
    /// Digital signature (optional)
    pub signature: Option<String>,
}

impl AuditLogEntry {
    /// Create a new audit log entry
    pub fn new(
        event_type: AuditEventType,
        severity: AuditSeverity,
        outcome: AuditOutcome,
        actor: String,
        target: String,
        description: String,
    ) -> Self {
        let mut entry = Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type,
            severity,
            outcome,
            actor,
            target,
            description,
            metadata: HashMap::new(),
            data_categories: Vec::new(),
            ip_address: None,
            session_id: None,
            correlation_id: None,
            previous_hash: None,
            hash: String::new(),
            signature: None,
        };

        // Calculate hash
        entry.hash = entry.calculate_hash();
        entry
    }

    /// Calculate the cryptographic hash of this entry
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Include all immutable fields in hash calculation
        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_rfc3339().as_bytes());
        hasher.update(format!("{:?}", self.event_type).as_bytes());
        hasher.update(format!("{:?}", self.severity).as_bytes());
        hasher.update(format!("{:?}", self.outcome).as_bytes());
        hasher.update(self.actor.as_bytes());
        hasher.update(self.target.as_bytes());
        hasher.update(self.description.as_bytes());

        // Include metadata
        let mut metadata_keys: Vec<_> = self.metadata.keys().collect();
        metadata_keys.sort();
        for key in metadata_keys {
            hasher.update(key.as_bytes());
            hasher.update(self.metadata[key].as_bytes());
        }

        // Include previous hash if present
        if let Some(ref prev_hash) = self.previous_hash {
            hasher.update(prev_hash.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Verify the integrity of this entry
    pub fn verify_integrity(&self) -> bool {
        self.hash == self.calculate_hash()
    }

    /// Add metadata to the entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        // Recalculate hash after modification
        self.hash = self.calculate_hash();
    }

    /// Set the previous hash for chain integrity
    pub fn set_previous_hash(&mut self, hash: String) {
        self.previous_hash = Some(hash);
        // Recalculate hash after modification
        self.hash = self.calculate_hash();
    }

    /// Check if this entry represents a security event
    pub fn is_security_event(&self) -> bool {
        matches!(
            self.event_type,
            AuditEventType::SecurityIncident
                | AuditEventType::PolicyViolation
                | AuditEventType::Authorization
                | AuditEventType::Authentication
        )
    }

    /// Check if this entry requires retention for compliance
    pub fn requires_compliance_retention(&self) -> bool {
        self.severity >= AuditSeverity::Medium
            || self.is_security_event()
            || self.data_categories.iter().any(|cat| {
                matches!(
                    cat,
                    DataCategory::PersonallyIdentifiable
                        | DataCategory::Sensitive
                        | DataCategory::HealthRecords
                        | DataCategory::FinancialRecords
                )
            })
    }
}