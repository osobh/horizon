//! Data Retention Policies and Automated Enforcement
//!
//! Provides comprehensive data retention management including:
//! - Automated retention policy enforcement
//! - Legal hold management
//! - Data lifecycle management
//! - Automated deletion and archival
//! - Compliance reporting

use crate::data_classification::{DataCategory, DataClassification, DataMetadata};
use crate::error::{ComplianceError, ComplianceResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Data classification this policy applies to
    pub classification: DataClassification,
    /// Data category this policy applies to
    pub category: DataCategory,
    /// Retention period
    pub retention_period: Duration,
    /// Grace period before deletion
    pub grace_period: Duration,
    /// Whether automatic deletion is enabled
    pub auto_delete: bool,
    /// Archive before deletion
    pub archive_before_delete: bool,
    /// Legal requirements this policy addresses
    pub legal_basis: Vec<String>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Policy owner
    pub owner: String,
    /// Whether policy is active
    pub active: bool,
}

/// Legal hold on data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHold {
    /// Hold identifier
    pub id: String,
    /// Hold name/case reference
    pub name: String,
    /// Reason for the hold
    pub reason: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Data categories under hold
    pub data_categories: Vec<DataCategory>,
    /// Hold created date
    pub created_at: DateTime<Utc>,
    /// Hold expiration date (if any)
    pub expires_at: Option<DateTime<Utc>>,
    /// Legal counsel or requestor
    pub requestor: String,
    /// Current status
    pub status: LegalHoldStatus,
    /// Associated case or matter ID
    pub case_id: Option<String>,
}

/// Legal hold status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegalHoldStatus {
    /// Hold is active
    Active,
    /// Hold is suspended
    Suspended,
    /// Hold has been released
    Released,
    /// Hold has expired
    Expired,
}

/// Data retention record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecord {
    /// Record identifier
    pub id: Uuid,
    /// Data subject identifier
    pub subject_id: String,
    /// Data category
    pub category: DataCategory,
    /// Data classification
    pub classification: DataClassification,
    /// Data created date
    pub data_created_at: DateTime<Utc>,
    /// Record created date
    pub record_created_at: DateTime<Utc>,
    /// Applicable retention policy
    pub policy_id: String,
    /// Planned deletion date
    pub planned_deletion_at: DateTime<Utc>,
    /// Actual deletion date
    pub deleted_at: Option<DateTime<Utc>>,
    /// Archive date
    pub archived_at: Option<DateTime<Utc>>,
    /// Current status
    pub status: RetentionStatus,
    /// Legal holds preventing deletion
    pub legal_holds: Vec<String>,
    /// Last assessment date
    pub last_assessed_at: DateTime<Utc>,
    /// Data location/system
    pub data_location: String,
    /// Data size in bytes
    pub data_size_bytes: Option<u64>,
}

/// Retention record status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetentionStatus {
    /// Data is within retention period
    Active,
    /// Data is eligible for deletion
    EligibleForDeletion,
    /// Data is under legal hold
    OnLegalHold,
    /// Data has been archived
    Archived,
    /// Data has been deleted
    Deleted,
    /// Manual review required
    RequiresReview,
}

/// Retention action result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionActionResult {
    /// Action performed
    pub action: RetentionAction,
    /// Number of records processed
    pub records_processed: usize,
    /// Number of successful actions
    pub successful_actions: usize,
    /// Errors encountered
    pub errors: Vec<String>,
    /// Action timestamp
    pub performed_at: DateTime<Utc>,
    /// Total data size processed (bytes)
    pub total_size_bytes: u64,
    /// Processing duration
    pub processing_duration_ms: u64,
}

/// Types of retention actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionAction {
    /// Assess records for retention status
    Assessment,
    /// Archive eligible data
    Archive,
    /// Delete eligible data
    Delete,
    /// Apply legal hold
    ApplyLegalHold,
    /// Release legal hold
    ReleaseLegalHold,
    /// Manual review flag
    FlagForReview,
}

/// Retention compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionComplianceReport {
    /// Report generation date
    pub generated_at: DateTime<Utc>,
    /// Reporting period start
    pub period_start: DateTime<Utc>,
    /// Reporting period end
    pub period_end: DateTime<Utc>,
    /// Total records tracked
    pub total_records: usize,
    /// Records by status
    pub records_by_status: HashMap<RetentionStatus, usize>,
    /// Records by category
    pub records_by_category: HashMap<DataCategory, usize>,
    /// Active legal holds
    pub active_legal_holds: usize,
    /// Records deleted in period
    pub deleted_in_period: usize,
    /// Records archived in period
    pub archived_in_period: usize,
    /// Policy compliance percentage
    pub compliance_percentage: f64,
    /// Identified issues
    pub issues: Vec<ComplianceIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Compliance issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Affected records count
    pub affected_records: usize,
    /// Recommended action
    pub recommended_action: String,
    /// Issue detected date
    pub detected_at: DateTime<Utc>,
}

/// Types of compliance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueType {
    /// Data held beyond retention period
    OverRetention,
    /// Missing retention policy
    MissingPolicy,
    /// Conflicting policies
    PolicyConflict,
    /// Expired legal hold
    ExpiredLegalHold,
    /// Missing deletion confirmations
    MissingDeletionConfirmation,
    /// Data location unknown
    UnknownDataLocation,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Low priority issue
    Low,
    /// Medium priority issue
    Medium,
    /// High priority issue requiring attention
    High,
    /// Critical issue requiring immediate action
    Critical,
}

/// Data retention engine
pub struct RetentionEngine {
    /// Retention policies by ID
    policies: HashMap<String, RetentionPolicy>,
    /// Legal holds by ID
    legal_holds: HashMap<String, LegalHold>,
    /// Retention records by ID
    records: HashMap<Uuid, RetentionRecord>,
    /// Legal holds by subject ID for quick lookup
    holds_by_subject: HashMap<String, Vec<String>>,
}

impl RetentionEngine {
    /// Create new retention engine
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            legal_holds: HashMap::new(),
            records: HashMap::new(),
            holds_by_subject: HashMap::new(),
        }
    }

    /// Add retention policy
    pub fn add_policy(&mut self, policy: RetentionPolicy) -> ComplianceResult<()> {
        // Validate policy
        if policy.retention_period.num_seconds() <= 0 {
            return Err(ComplianceError::RetentionError(
                "Retention period must be positive".to_string(),
            ));
        }

        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Get applicable retention policy for data
    pub fn get_applicable_policy(
        &self,
        classification: DataClassification,
        category: DataCategory,
    ) -> Option<&RetentionPolicy> {
        self.policies
            .values()
            .filter(|p| p.active && p.classification == classification && p.category == category)
            .max_by_key(|p| p.created_at)
    }

    /// Create retention record for data
    pub fn create_retention_record(
        &mut self,
        subject_id: String,
        metadata: &DataMetadata,
        data_location: String,
        data_size_bytes: Option<u64>,
    ) -> ComplianceResult<Uuid> {
        let policy = self
            .get_applicable_policy(metadata.classification, metadata.category)
            .ok_or_else(|| {
                ComplianceError::RetentionError(format!(
                    "No applicable retention policy for {:?}/{:?}",
                    metadata.classification, metadata.category
                ))
            })?;

        let record_id = Uuid::new_v4();
        let now = Utc::now();
        let planned_deletion = metadata.created_at + policy.retention_period + policy.grace_period;

        // Check for existing legal holds on this subject
        let legal_holds = self
            .holds_by_subject
            .get(&subject_id)
            .cloned()
            .unwrap_or_default();

        let status = if legal_holds.is_empty() {
            if planned_deletion <= now {
                RetentionStatus::EligibleForDeletion
            } else {
                RetentionStatus::Active
            }
        } else {
            RetentionStatus::OnLegalHold
        };

        let record = RetentionRecord {
            id: record_id,
            subject_id,
            category: metadata.category,
            classification: metadata.classification,
            data_created_at: metadata.created_at,
            record_created_at: now,
            policy_id: policy.id.clone(),
            planned_deletion_at: planned_deletion,
            deleted_at: None,
            archived_at: None,
            status,
            legal_holds,
            last_assessed_at: now,
            data_location,
            data_size_bytes,
        };

        self.records.insert(record_id, record);
        Ok(record_id)
    }

    /// Apply legal hold
    pub fn apply_legal_hold(&mut self, hold: LegalHold) -> ComplianceResult<()> {
        let hold_id = hold.id.clone();
        let subject_id = hold.subject_id.clone();

        // Add to legal holds
        self.legal_holds.insert(hold_id.clone(), hold);

        // Add to subject index
        self.holds_by_subject
            .entry(subject_id.clone())
            .or_default()
            .push(hold_id.clone());

        // Update affected retention records
        for record in self.records.values_mut() {
            if record.subject_id == subject_id {
                record.legal_holds.push(hold_id.clone());
                record.status = RetentionStatus::OnLegalHold;
                record.last_assessed_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Release legal hold
    pub fn release_legal_hold(&mut self, hold_id: &str) -> ComplianceResult<()> {
        let hold = self.legal_holds.get_mut(hold_id).ok_or_else(|| {
            ComplianceError::RetentionError(format!("Legal hold not found: {hold_id}"))
        })?;

        hold.status = LegalHoldStatus::Released;
        let subject_id = hold.subject_id.clone();

        // Remove from subject index
        if let Some(holds) = self.holds_by_subject.get_mut(&subject_id) {
            holds.retain(|id| id != hold_id);
        }

        // Update affected retention records
        let now = Utc::now();
        for record in self.records.values_mut() {
            if record.subject_id == subject_id {
                record.legal_holds.retain(|id| id != hold_id);

                // Update status if no more legal holds
                if record.legal_holds.is_empty() {
                    record.status = if record.planned_deletion_at <= now {
                        RetentionStatus::EligibleForDeletion
                    } else {
                        RetentionStatus::Active
                    };
                }
                record.last_assessed_at = now;
            }
        }

        Ok(())
    }

    /// Check if data is under legal hold
    pub fn is_under_legal_hold(&self, subject_id: &str, category: DataCategory) -> bool {
        if let Some(hold_ids) = self.holds_by_subject.get(subject_id) {
            for hold_id in hold_ids {
                if let Some(hold) = self.legal_holds.get(hold_id) {
                    if hold.status == LegalHoldStatus::Active
                        && hold.data_categories.contains(&category)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Assess all retention records and update statuses
    pub async fn assess_retention_records(&mut self) -> ComplianceResult<RetentionActionResult> {
        let start_time = std::time::Instant::now();
        let mut processed = 0;
        let mut successful = 0;
        let errors = Vec::new();
        let mut total_size = 0u64;

        let now = Utc::now();

        for record in self.records.values_mut() {
            processed += 1;

            // Add to total size
            if let Some(size) = record.data_size_bytes {
                total_size += size;
            }

            let old_status = record.status;

            // Update status based on current conditions
            let new_status = if !record.legal_holds.is_empty() {
                RetentionStatus::OnLegalHold
            } else if record.deleted_at.is_some() {
                RetentionStatus::Deleted
            } else if record.archived_at.is_some() {
                RetentionStatus::Archived
            } else if record.planned_deletion_at <= now {
                RetentionStatus::EligibleForDeletion
            } else {
                RetentionStatus::Active
            };

            if new_status != old_status {
                record.status = new_status;
                record.last_assessed_at = now;
                successful += 1;
            }
        }

        let duration = start_time.elapsed();

        Ok(RetentionActionResult {
            action: RetentionAction::Assessment,
            records_processed: processed,
            successful_actions: successful,
            errors,
            performed_at: now,
            total_size_bytes: total_size,
            processing_duration_ms: duration.as_millis() as u64,
        })
    }

    /// Archive eligible records
    pub async fn archive_eligible_records(&mut self) -> ComplianceResult<RetentionActionResult> {
        let start_time = std::time::Instant::now();
        let mut processed = 0;
        let mut successful = 0;
        let mut errors = Vec::new();
        let mut total_size = 0u64;

        let now = Utc::now();

        for record in self.records.values_mut() {
            if record.status == RetentionStatus::EligibleForDeletion && record.archived_at.is_none()
            {
                processed += 1;

                if let Some(size) = record.data_size_bytes {
                    total_size += size;
                }

                // Get policy to check if archival is required
                if let Some(policy) = self.policies.get(&record.policy_id) {
                    if policy.archive_before_delete {
                        // In a real implementation, this would perform actual archival
                        record.archived_at = Some(now);
                        record.status = RetentionStatus::Archived;
                        record.last_assessed_at = now;
                        successful += 1;
                    }
                } else {
                    errors.push(format!("Policy not found for record {}", record.id));
                }
            }
        }

        let duration = start_time.elapsed();

        Ok(RetentionActionResult {
            action: RetentionAction::Archive,
            records_processed: processed,
            successful_actions: successful,
            errors,
            performed_at: now,
            total_size_bytes: total_size,
            processing_duration_ms: duration.as_millis() as u64,
        })
    }

    /// Delete eligible records
    pub async fn delete_eligible_records(&mut self) -> ComplianceResult<RetentionActionResult> {
        let start_time = std::time::Instant::now();
        let mut processed = 0;
        let mut successful = 0;
        let mut errors = Vec::new();
        let mut total_size = 0u64;

        let now = Utc::now();

        for record in self.records.values_mut() {
            // Can delete if eligible and not under legal hold
            let can_delete = match record.status {
                RetentionStatus::EligibleForDeletion => record.legal_holds.is_empty(),
                RetentionStatus::Archived => true, // Can delete archived records
                _ => false,
            };

            if can_delete && record.deleted_at.is_none() {
                processed += 1;

                if let Some(size) = record.data_size_bytes {
                    total_size += size;
                }

                // Get policy to check if auto-delete is enabled
                if let Some(policy) = self.policies.get(&record.policy_id) {
                    if policy.auto_delete {
                        // In a real implementation, this would perform actual deletion
                        record.deleted_at = Some(now);
                        record.status = RetentionStatus::Deleted;
                        record.last_assessed_at = now;
                        successful += 1;
                    } else {
                        record.status = RetentionStatus::RequiresReview;
                        record.last_assessed_at = now;
                        successful += 1;
                    }
                } else {
                    errors.push(format!("Policy not found for record {}", record.id));
                }
            }
        }

        let duration = start_time.elapsed();

        Ok(RetentionActionResult {
            action: RetentionAction::Delete,
            records_processed: processed,
            successful_actions: successful,
            errors,
            performed_at: now,
            total_size_bytes: total_size,
            processing_duration_ms: duration.as_millis() as u64,
        })
    }

    /// Generate compliance report
    pub fn generate_compliance_report(
        &self,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> ComplianceResult<RetentionComplianceReport> {
        let mut records_by_status = HashMap::new();
        let mut records_by_category = HashMap::new();
        let mut deleted_in_period = 0;
        let mut archived_in_period = 0;
        let mut issues = Vec::new();

        // Analyze records
        for record in self.records.values() {
            // Count by status
            *records_by_status.entry(record.status).or_insert(0) += 1;

            // Count by category
            *records_by_category.entry(record.category).or_insert(0) += 1;

            // Count actions in period
            if let Some(deleted_at) = record.deleted_at {
                if deleted_at >= period_start && deleted_at <= period_end {
                    deleted_in_period += 1;
                }
            }

            if let Some(archived_at) = record.archived_at {
                if archived_at >= period_start && archived_at <= period_end {
                    archived_in_period += 1;
                }
            }

            // Identify issues
            let now = Utc::now();
            if record.status == RetentionStatus::Active && record.planned_deletion_at < now {
                issues.push(ComplianceIssue {
                    issue_type: IssueType::OverRetention,
                    severity: IssueSeverity::High,
                    description: format!("Record {} held beyond retention period", record.id),
                    affected_records: 1,
                    recommended_action: "Review for deletion or archival".to_string(),
                    detected_at: now,
                });
            }

            if !self.policies.contains_key(&record.policy_id) {
                issues.push(ComplianceIssue {
                    issue_type: IssueType::MissingPolicy,
                    severity: IssueSeverity::Medium,
                    description: format!("Record {} references missing policy", record.id),
                    affected_records: 1,
                    recommended_action: "Update policy reference or create missing policy"
                        .to_string(),
                    detected_at: now,
                });
            }
        }

        // Count active legal holds
        let active_legal_holds = self
            .legal_holds
            .values()
            .filter(|h| h.status == LegalHoldStatus::Active)
            .count();

        // Calculate compliance percentage
        let total_records = self.records.len();
        let compliant_records = total_records - issues.len();
        let compliance_percentage = if total_records > 0 {
            (compliant_records as f64 / total_records as f64) * 100.0
        } else {
            100.0
        };

        // Generate recommendations
        let mut recommendations = Vec::new();
        if compliance_percentage < 95.0 {
            recommendations.push("Review and address identified compliance issues".to_string());
        }
        if deleted_in_period == 0 && total_records > 0 {
            recommendations.push("Consider running retention cleanup process".to_string());
        }
        if active_legal_holds > 0 {
            recommendations.push("Review active legal holds for potential release".to_string());
        }

        Ok(RetentionComplianceReport {
            generated_at: Utc::now(),
            period_start,
            period_end,
            total_records,
            records_by_status,
            records_by_category,
            active_legal_holds,
            deleted_in_period,
            archived_in_period,
            compliance_percentage,
            issues,
            recommendations,
        })
    }

    /// Get retention record by ID
    pub fn get_record(&self, record_id: &Uuid) -> Option<&RetentionRecord> {
        self.records.get(record_id)
    }

    /// Get records by subject ID
    pub fn get_records_by_subject(&self, subject_id: &str) -> Vec<&RetentionRecord> {
        self.records
            .values()
            .filter(|r| r.subject_id == subject_id)
            .collect()
    }

    /// Get legal hold by ID
    pub fn get_legal_hold(&self, hold_id: &str) -> Option<&LegalHold> {
        self.legal_holds.get(hold_id)
    }

    /// Get retention policy by ID
    pub fn get_policy(&self, policy_id: &str) -> Option<&RetentionPolicy> {
        self.policies.get(policy_id)
    }
}

impl Default for RetentionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_classification::DataClassifier;

    #[test]
    fn test_retention_engine_creation() {
        let engine = RetentionEngine::new();
        assert!(engine.policies.is_empty());
        assert!(engine.legal_holds.is_empty());
        assert!(engine.records.is_empty());
    }

    #[test]
    fn test_add_retention_policy() {
        let mut engine = RetentionEngine::new();

        let policy = RetentionPolicy {
            id: "test_policy".to_string(),
            name: "Test Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365 * 7), // 7 years
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR Article 5".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };

        let result = engine.add_policy(policy);
        assert!(result.is_ok());
        assert_eq!(engine.policies.len(), 1);
    }

    #[test]
    fn test_add_invalid_retention_policy() {
        let mut engine = RetentionEngine::new();

        let invalid_policy = RetentionPolicy {
            id: "invalid_policy".to_string(),
            name: "Invalid Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::seconds(-1), // Invalid negative period
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec![],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };

        let result = engine.add_policy(invalid_policy);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::RetentionError(_)
        ));
    }

    #[test]
    fn test_get_applicable_policy() {
        let mut engine = RetentionEngine::new();

        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365 * 7),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };

        engine.add_policy(policy)?;

        let found =
            engine.get_applicable_policy(DataClassification::ConfidentialData, DataCategory::PII);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "pii_policy");

        let not_found = engine
            .get_applicable_policy(DataClassification::PublicData, DataCategory::BusinessData);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_create_retention_record() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Add policy first
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365 * 7),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        // Create metadata
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let result = engine.create_retention_record(
            "user123".to_string(),
            &metadata,
            "database_table_users".to_string(),
            Some(1024),
        );

        assert!(result.is_ok());
        let record_id = result?;

        let record = engine.get_record(&record_id)?;
        assert_eq!(record.subject_id, "user123");
        assert_eq!(record.category, DataCategory::PII);
        assert_eq!(record.data_location, "database_table_users");
        assert_eq!(record.data_size_bytes, Some(1024));
    }

    #[test]
    fn test_create_retention_record_no_policy() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let result = engine.create_retention_record(
            "user123".to_string(),
            &metadata,
            "database_table_users".to_string(),
            Some(1024),
        );

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::RetentionError(_)
        ));
    }

    #[test]
    fn test_apply_legal_hold() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy and record
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365 * 7),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let record_id = engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        // Apply legal hold
        let hold = LegalHold {
            id: "hold123".to_string(),
            name: "Litigation Hold Case XYZ".to_string(),
            reason: "Pending litigation".to_string(),
            subject_id: "user123".to_string(),
            data_categories: vec![DataCategory::PII],
            created_at: Utc::now(),
            expires_at: None,
            requestor: "Legal Counsel".to_string(),
            status: LegalHoldStatus::Active,
            case_id: Some("CASE-2024-001".to_string()),
        };

        let result = engine.apply_legal_hold(hold);
        assert!(result.is_ok());

        // Check record status updated
        let record = engine.get_record(&record_id)?;
        assert_eq!(record.status, RetentionStatus::OnLegalHold);
        assert!(!record.legal_holds.is_empty());

        // Check hold lookup works
        assert!(engine.is_under_legal_hold("user123", DataCategory::PII));
    }

    #[test]
    fn test_release_legal_hold() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy and record
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365 * 7),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let record_id = engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        // Apply and then release legal hold
        let hold = LegalHold {
            id: "hold123".to_string(),
            name: "Litigation Hold Case XYZ".to_string(),
            reason: "Pending litigation".to_string(),
            subject_id: "user123".to_string(),
            data_categories: vec![DataCategory::PII],
            created_at: Utc::now(),
            expires_at: None,
            requestor: "Legal Counsel".to_string(),
            status: LegalHoldStatus::Active,
            case_id: Some("CASE-2024-001".to_string()),
        };

        engine.apply_legal_hold(hold)?;

        let result = engine.release_legal_hold("hold123");
        assert!(result.is_ok());

        // Check record status updated
        let record = engine.get_record(&record_id)?;
        assert_ne!(record.status, RetentionStatus::OnLegalHold);
        assert!(record.legal_holds.is_empty());

        // Check hold lookup returns false
        assert!(!engine.is_under_legal_hold("user123", DataCategory::PII));

        // Check hold status updated
        let hold = engine.get_legal_hold("hold123")?;
        assert_eq!(hold.status, LegalHoldStatus::Released);
    }

    #[tokio::test]
    async fn test_assess_retention_records() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy with short retention period
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(1), // 1 day retention
            grace_period: Duration::days(1),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        // Create record that should be eligible for deletion
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let record_id = engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        // Manually set the record's planned deletion to the past to simulate expiry
        if let Some(record) = engine.records.get_mut(&record_id) {
            record.planned_deletion_at = Utc::now() - Duration::days(1);
        }

        let result = engine.assess_retention_records().await;
        assert!(result.is_ok());

        let action_result = result?;
        assert_eq!(action_result.action, RetentionAction::Assessment);
        assert!(action_result.records_processed > 0);
    }

    #[tokio::test]
    async fn test_archive_eligible_records() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy with archival enabled
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(1), // Short retention
            grace_period: Duration::days(1),
            auto_delete: true,
            archive_before_delete: true, // Enable archival
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        // Create and assess record
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let record_id = engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        // Manually set the record's planned deletion to the past to simulate expiry
        if let Some(record) = engine.records.get_mut(&record_id) {
            record.planned_deletion_at = Utc::now() - Duration::days(1);
        }

        // First assess to set status
        engine.assess_retention_records().await?;

        // Then archive
        let result = engine.archive_eligible_records().await;
        assert!(result.is_ok());

        let action_result = result?;
        assert_eq!(action_result.action, RetentionAction::Archive);

        // Check record was archived
        let record = engine.get_record(&record_id)?;
        assert_eq!(record.status, RetentionStatus::Archived);
        assert!(record.archived_at.is_some());
    }

    #[tokio::test]
    async fn test_delete_eligible_records() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy with auto-delete enabled
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(1), // Short retention
            grace_period: Duration::days(1),
            auto_delete: true, // Enable auto-delete
            archive_before_delete: false,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        // Create and assess record
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let record_id = engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        // Manually set the record's planned deletion to the past to simulate expiry
        if let Some(record) = engine.records.get_mut(&record_id) {
            record.planned_deletion_at = Utc::now() - Duration::days(1);
        }

        // First assess to set status
        engine.assess_retention_records().await?;

        // Then delete
        let result = engine.delete_eligible_records().await;
        assert!(result.is_ok());

        let action_result = result?;
        assert_eq!(action_result.action, RetentionAction::Delete);

        // Check record was deleted
        let record = engine.get_record(&record_id)?;
        assert_eq!(record.status, RetentionStatus::Deleted);
        assert!(record.deleted_at.is_some());
    }

    #[test]
    fn test_generate_compliance_report() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        // Create some records
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        for i in 0..5 {
            engine
                .create_retention_record(
                    format!("user{}", i),
                    &metadata,
                    "database_table_users".to_string(),
                    Some(1024 * (i + 1) as u64),
                )
                .unwrap();
        }

        let now = Utc::now();
        let report = engine.generate_compliance_report(now - Duration::days(30), now);

        assert!(report.is_ok());
        let compliance_report = report?;
        assert_eq!(compliance_report.total_records, 5);
        assert!(!compliance_report.records_by_status.is_empty());
        assert!(!compliance_report.records_by_category.is_empty());
        assert!(compliance_report.compliance_percentage >= 0.0);
        assert!(compliance_report.compliance_percentage <= 100.0);
    }

    #[test]
    fn test_get_records_by_subject() {
        let mut engine = RetentionEngine::new();
        let classifier = DataClassifier::new();

        // Set up policy
        let policy = RetentionPolicy {
            id: "pii_policy".to_string(),
            name: "PII Retention Policy".to_string(),
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: Duration::days(365),
            grace_period: Duration::days(30),
            auto_delete: true,
            archive_before_delete: true,
            legal_basis: vec!["GDPR".to_string()],
            created_at: Utc::now(),
            owner: "DPO".to_string(),
            active: true,
        };
        engine.add_policy(policy)?;

        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        // Create multiple records for same subject
        engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        engine
            .create_retention_record(
                "user123".to_string(),
                &metadata,
                "database_table_profiles".to_string(),
                Some(2048),
            )
            .unwrap();

        // Create record for different subject
        engine
            .create_retention_record(
                "user456".to_string(),
                &metadata,
                "database_table_users".to_string(),
                Some(1024),
            )
            .unwrap();

        let user123_records = engine.get_records_by_subject("user123");
        assert_eq!(user123_records.len(), 2);

        let user456_records = engine.get_records_by_subject("user456");
        assert_eq!(user456_records.len(), 1);

        let nonexistent_records = engine.get_records_by_subject("user999");
        assert_eq!(nonexistent_records.len(), 0);
    }
}
