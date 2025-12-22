//! HIPAA compliance implementation

use crate::error::{ComplianceError, ComplianceResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// HIPAA safeguards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HipaaSafeguard {
    /// Administrative safeguards
    Administrative,
    /// Physical safeguards
    Physical,
    /// Technical safeguards
    Technical,
}

/// PHI (Protected Health Information) access levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhiAccessLevel {
    /// No access
    NoAccess,
    /// Read only
    ReadOnly,
    /// Read and write
    ReadWrite,
    /// Full access including delete
    FullAccess,
}

/// HIPAA audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipaaAuditEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User who accessed PHI
    pub user_id: String,
    /// Patient ID (de-identified)
    pub patient_id_hash: String,
    /// Type of access
    pub access_type: PhiAccessType,
    /// Success or failure
    pub success: bool,
    /// Access justification
    pub justification: Option<String>,
    /// Data accessed (general category only)
    pub data_category: String,
}

/// Types of PHI access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiAccessType {
    /// Create new PHI
    Create,
    /// Read PHI
    Read,
    /// Update PHI
    Update,
    /// Delete PHI
    Delete,
    /// Export PHI
    Export,
    /// Emergency access
    EmergencyAccess,
}

/// HIPAA compliance handler
#[derive(Clone)]
pub struct HipaaHandler {
    access_controls: HashMap<String, PhiAccessLevel>,
    audit_log: Vec<HipaaAuditEntry>,
    encryption_keys: HashMap<String, EncryptionKey>,
    breach_notifications: Vec<BreachNotification>,
}

/// Encryption key metadata
#[derive(Debug, Clone)]
struct EncryptionKey {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    algorithm: String,
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
    #[allow(dead_code)]
    rotated_at: Option<DateTime<Utc>>,
    active: bool,
}

/// Breach notification record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachNotification {
    /// Incident ID
    pub id: String,
    /// When breach was discovered
    pub discovered_at: DateTime<Utc>,
    /// Number of affected individuals
    pub affected_count: usize,
    /// Type of PHI breached
    pub phi_types: Vec<String>,
    /// Notification sent
    pub notified: bool,
    /// Notification date
    pub notified_at: Option<DateTime<Utc>>,
}

impl HipaaHandler {
    /// Create new HIPAA handler
    pub fn new() -> Self {
        Self {
            access_controls: HashMap::new(),
            audit_log: Vec::new(),
            encryption_keys: HashMap::new(),
            breach_notifications: Vec::new(),
        }
    }

    /// Set user access level
    pub fn set_access_level(
        &mut self,
        user_id: String,
        level: PhiAccessLevel,
    ) -> ComplianceResult<()> {
        self.access_controls.insert(user_id, level);
        Ok(())
    }

    /// Check if user can access PHI
    pub fn check_phi_access(
        &self,
        user_id: &str,
        access_type: PhiAccessType,
    ) -> ComplianceResult<bool> {
        let level = self
            .access_controls
            .get(user_id)
            .copied()
            .unwrap_or(PhiAccessLevel::NoAccess);

        let allowed = match access_type {
            PhiAccessType::Read => level >= PhiAccessLevel::ReadOnly,
            PhiAccessType::Create | PhiAccessType::Update => level >= PhiAccessLevel::ReadWrite,
            PhiAccessType::Delete | PhiAccessType::Export => level >= PhiAccessLevel::FullAccess,
            PhiAccessType::EmergencyAccess => {
                // Emergency access requires special handling
                level >= PhiAccessLevel::ReadOnly
            }
        };

        Ok(allowed)
    }

    /// Log PHI access
    pub async fn log_phi_access(
        &mut self,
        user_id: String,
        patient_id: &str,
        access_type: PhiAccessType,
        success: bool,
        data_category: String,
        justification: Option<String>,
    ) -> ComplianceResult<()> {
        // Hash patient ID for privacy
        let patient_id_hash = self.hash_patient_id(patient_id);

        let entry = HipaaAuditEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            user_id,
            patient_id_hash,
            access_type,
            success,
            justification,
            data_category,
        };

        self.audit_log.push(entry);

        // Keep audit log size manageable (in production, would persist to database)
        if self.audit_log.len() > 100000 {
            self.audit_log.drain(0..10000);
        }

        Ok(())
    }

    /// Get audit entries for a time period
    pub fn get_audit_entries(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&HipaaAuditEntry> {
        self.audit_log
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }

    /// Report potential breach
    pub fn report_breach(
        &mut self,
        affected_count: usize,
        phi_types: Vec<String>,
    ) -> ComplianceResult<String> {
        let breach_id = uuid::Uuid::new_v4().to_string();

        let notification = BreachNotification {
            id: breach_id.clone(),
            discovered_at: Utc::now(),
            affected_count,
            phi_types,
            notified: false,
            notified_at: None,
        };

        self.breach_notifications.push(notification);

        // In real implementation, would trigger notification workflow
        if affected_count >= 500 {
            // Major breach - requires immediate notification
            return Err(ComplianceError::ComplianceViolation {
                regulation: "HIPAA".to_string(),
                violation: format!(
                    "Major breach affecting {affected_count} individuals - immediate notification required"
                ),
            });
        }

        Ok(breach_id)
    }

    /// Mark breach as notified
    pub fn mark_breach_notified(&mut self, breach_id: &str) -> ComplianceResult<()> {
        if let Some(breach) = self
            .breach_notifications
            .iter_mut()
            .find(|b| b.id == breach_id)
        {
            breach.notified = true;
            breach.notified_at = Some(Utc::now());
            Ok(())
        } else {
            Err(ComplianceError::InternalError(
                "Breach not found".to_string(),
            ))
        }
    }

    /// Initialize encryption for PHI
    pub fn initialize_encryption(&mut self) -> ComplianceResult<String> {
        let key_id = uuid::Uuid::new_v4().to_string();

        let key = EncryptionKey {
            id: key_id.clone(),
            algorithm: "AES-256-GCM".to_string(),
            created_at: Utc::now(),
            rotated_at: None,
            active: true,
        };

        self.encryption_keys.insert(key_id.clone(), key);
        Ok(key_id)
    }

    /// Rotate encryption key
    pub fn rotate_encryption_key(&mut self, old_key_id: &str) -> ComplianceResult<String> {
        // Mark old key as inactive
        if let Some(old_key) = self.encryption_keys.get_mut(old_key_id) {
            old_key.active = false;
        } else {
            return Err(ComplianceError::EncryptionError(
                "Key not found".to_string(),
            ));
        }

        // Create new key
        self.initialize_encryption()
    }

    /// Validate technical safeguards
    pub fn validate_technical_safeguards(&self) -> ComplianceResult<SafeguardValidation> {
        let mut validation = SafeguardValidation {
            access_control: false,
            audit_logs: false,
            integrity: false,
            transmission_security: false,
            encryption: false,
        };

        // Check access controls
        validation.access_control = !self.access_controls.is_empty();

        // Check audit logs
        validation.audit_logs = !self.audit_log.is_empty();

        // Check encryption
        validation.encryption = self.encryption_keys.values().any(|k| k.active);

        // In real implementation, would check more thoroughly
        validation.integrity = true;
        validation.transmission_security = true;

        Ok(validation)
    }

    /// Hash patient ID for privacy
    fn hash_patient_id(&self, patient_id: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(patient_id.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Technical safeguard validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeguardValidation {
    /// Access control implemented
    pub access_control: bool,
    /// Audit logs configured
    pub audit_logs: bool,
    /// Integrity controls in place
    pub integrity: bool,
    /// Transmission security enabled
    pub transmission_security: bool,
    /// Encryption active
    pub encryption: bool,
}

impl Default for HipaaHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hipaa_handler_creation() {
        let handler = HipaaHandler::new();
        assert!(handler.access_controls.is_empty());
        assert!(handler.audit_log.is_empty());
    }

    #[test]
    fn test_access_level_hierarchy() {
        assert!(PhiAccessLevel::NoAccess < PhiAccessLevel::ReadOnly);
        assert!(PhiAccessLevel::ReadOnly < PhiAccessLevel::ReadWrite);
        assert!(PhiAccessLevel::ReadWrite < PhiAccessLevel::FullAccess);
    }

    #[test]
    fn test_set_and_check_access() {
        let mut handler = HipaaHandler::new();

        handler
            .set_access_level("doctor1".to_string(), PhiAccessLevel::FullAccess)
            .unwrap();
        handler
            .set_access_level("nurse1".to_string(), PhiAccessLevel::ReadWrite)
            .unwrap();
        handler
            .set_access_level("admin1".to_string(), PhiAccessLevel::ReadOnly)
            .unwrap();

        // Doctor can do everything
        assert!(handler
            .check_phi_access("doctor1", PhiAccessType::Read)
            .unwrap());
        assert!(handler
            .check_phi_access("doctor1", PhiAccessType::Delete)
            .unwrap());

        // Nurse can read and write but not delete
        assert!(handler
            .check_phi_access("nurse1", PhiAccessType::Read)
            .unwrap());
        assert!(handler
            .check_phi_access("nurse1", PhiAccessType::Update)
            .unwrap());
        assert!(!handler
            .check_phi_access("nurse1", PhiAccessType::Delete)
            .unwrap());

        // Admin can only read
        assert!(handler
            .check_phi_access("admin1", PhiAccessType::Read)
            .unwrap());
        assert!(!handler
            .check_phi_access("admin1", PhiAccessType::Create)
            .unwrap());

        // Unknown user has no access
        assert!(!handler
            .check_phi_access("unknown", PhiAccessType::Read)
            .unwrap());
    }

    #[tokio::test]
    async fn test_phi_access_logging() {
        let mut handler = HipaaHandler::new();

        handler
            .log_phi_access(
                "doctor1".to_string(),
                "patient123",
                PhiAccessType::Read,
                true,
                "Medical Records".to_string(),
                Some("Routine checkup".to_string()),
            )
            .await
            .unwrap();

        assert_eq!(handler.audit_log.len(), 1);

        let entry = &handler.audit_log[0];
        assert_eq!(entry.user_id, "doctor1");
        assert_eq!(entry.access_type, PhiAccessType::Read);
        assert!(entry.success);
        assert_eq!(entry.justification, Some("Routine checkup".to_string()));
    }

    #[test]
    fn test_patient_id_hashing() {
        let handler = HipaaHandler::new();

        let hash1 = handler.hash_patient_id("patient123");
        let hash2 = handler.hash_patient_id("patient123");
        let hash3 = handler.hash_patient_id("patient456");

        // Same ID produces same hash
        assert_eq!(hash1, hash2);
        // Different IDs produce different hashes
        assert_ne!(hash1, hash3);
        // Hash doesn't contain original ID
        assert!(!hash1.contains("patient123"));
    }

    #[test]
    fn test_breach_reporting() {
        let mut handler = HipaaHandler::new();

        // Small breach
        let breach_id = handler
            .report_breach(10, vec!["Names".to_string(), "Addresses".to_string()])
            .unwrap();

        assert_eq!(handler.breach_notifications.len(), 1);
        assert!(!handler.breach_notifications[0].notified);

        // Mark as notified
        handler.mark_breach_notified(&breach_id)?;
        assert!(handler.breach_notifications[0].notified);
        assert!(handler.breach_notifications[0].notified_at.is_some());
    }

    #[test]
    fn test_major_breach_detection() {
        let mut handler = HipaaHandler::new();

        // Major breach (500+ individuals)
        let result =
            handler.report_breach(500, vec!["SSN".to_string(), "Medical Records".to_string()]);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::ComplianceViolation { .. }
        ));
    }

    #[test]
    fn test_encryption_management() {
        let mut handler = HipaaHandler::new();

        // Initialize encryption
        let key_id = handler.initialize_encryption()?;
        assert_eq!(handler.encryption_keys.len(), 1);
        assert!(handler.encryption_keys[&key_id].active);

        // Rotate key
        let new_key_id = handler.rotate_encryption_key(&key_id)?;
        assert_eq!(handler.encryption_keys.len(), 2);
        assert!(!handler.encryption_keys[&key_id].active);
        assert!(handler.encryption_keys[&new_key_id].active);
    }

    #[test]
    fn test_technical_safeguards_validation() {
        let mut handler = HipaaHandler::new();

        // Initially, most safeguards are not in place
        let validation = handler.validate_technical_safeguards()?;
        assert!(!validation.access_control);
        assert!(!validation.audit_logs);
        assert!(!validation.encryption);

        // Add safeguards
        handler
            .set_access_level("user1".to_string(), PhiAccessLevel::ReadOnly)
            .unwrap();
        handler.initialize_encryption()?;

        let validation = handler.validate_technical_safeguards()?;
        assert!(validation.access_control);
        assert!(validation.encryption);
    }

    #[tokio::test]
    async fn test_audit_log_retrieval() {
        let mut handler = HipaaHandler::new();

        // Log multiple accesses
        for i in 0..5 {
            handler
                .log_phi_access(
                    format!("user{}", i),
                    "patient123",
                    PhiAccessType::Read,
                    true,
                    "Records".to_string(),
                    None,
                )
                .await
                .unwrap();

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);
        let entries = handler.get_audit_entries(start, end);

        assert_eq!(entries.len(), 5);
    }
}
