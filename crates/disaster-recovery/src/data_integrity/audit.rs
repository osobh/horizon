//! Audit trail and compliance tracking

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Audit entry for integrity operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: AuditEventType,
    /// Object ID
    pub object_id: Option<Uuid>,
    /// User or system that triggered the event
    pub actor: String,
    /// Event details
    pub details: String,
    /// Result of the operation
    pub result: AuditResult,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Audit event type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Integrity check performed
    IntegrityCheck,
    /// Corruption detected
    CorruptionDetected,
    /// Repair attempted
    RepairAttempted,
    /// Repair completed
    RepairCompleted,
    /// Checksum calculated
    ChecksumCalculated,
    /// Configuration changed
    ConfigurationChanged,
    /// Manual override
    ManualOverride,
    /// Alert triggered
    AlertTriggered,
    /// Report generated
    ReportGenerated,
}

/// Audit result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation partially succeeded
    Partial,
    /// Operation skipped
    Skipped,
    /// Information only
    Info,
}

impl AuditEntry {
    /// Create new audit entry
    pub fn new(
        event_type: AuditEventType,
        actor: String,
        details: String,
        result: AuditResult,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type,
            object_id: None,
            actor,
            details,
            result,
            metadata: HashMap::new(),
        }
    }

    /// Set object ID
    pub fn with_object_id(mut self, object_id: Uuid) -> Self {
        self.object_id = Some(object_id);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if audit entry represents a failure
    pub fn is_failure(&self) -> bool {
        matches!(self.result, AuditResult::Failure)
    }

    /// Check if audit entry represents success
    pub fn is_success(&self) -> bool {
        matches!(self.result, AuditResult::Success)
    }
}

/// Audit trail manager
#[derive(Debug, Clone)]
pub struct AuditTrail {
    entries: Vec<AuditEntry>,
    max_entries: usize,
}

impl AuditTrail {
    /// Create new audit trail
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Add audit entry
    pub fn add_entry(&mut self, entry: AuditEntry) {
        self.entries.push(entry);

        // Trim old entries if exceeding max
        if self.entries.len() > self.max_entries {
            let remove_count = self.entries.len() - self.max_entries;
            self.entries.drain(0..remove_count);
        }
    }

    /// Get entries for object
    pub fn get_entries_for_object(&self, object_id: Uuid) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.object_id == Some(object_id))
            .collect()
    }

    /// Get entries by event type
    pub fn get_entries_by_type(&self, event_type: &AuditEventType) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Get recent entries
    pub fn get_recent_entries(&self, count: usize) -> Vec<&AuditEntry> {
        let start = self.entries.len().saturating_sub(count);
        self.entries[start..].iter().collect()
    }

    /// Get failure entries
    pub fn get_failures(&self) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.is_failure()).collect()
    }

    /// Clear audit trail
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get total entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
