//! GDPR compliance implementation

use crate::error::{ComplianceError, ComplianceResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GDPR rights that must be supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GdprRight {
    /// Right to access personal data
    Access,
    /// Right to rectification (correction)
    Rectification,
    /// Right to erasure (right to be forgotten)
    Erasure,
    /// Right to restrict processing
    Restriction,
    /// Right to data portability
    Portability,
    /// Right to object to processing
    Object,
    /// Right to not be subject to automated decision-making
    NoAutomatedDecision,
}

/// GDPR lawful basis for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LawfulBasis {
    /// Data subject has given consent
    Consent,
    /// Processing necessary for contract
    Contract,
    /// Processing necessary for legal obligation
    LegalObligation,
    /// Processing necessary to protect vital interests
    VitalInterests,
    /// Processing necessary for public task
    PublicTask,
    /// Processing necessary for legitimate interests
    LegitimateInterests,
}

/// GDPR data subject request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    /// Request ID
    pub id: String,
    /// Data subject identifier
    pub subject_id: String,
    /// Type of request
    pub request_type: GdprRight,
    /// Timestamp of request
    pub requested_at: DateTime<Utc>,
    /// Status of request
    pub status: RequestStatus,
    /// Deadline for response (typically 30 days)
    pub deadline: DateTime<Utc>,
    /// Processing notes
    pub notes: Vec<String>,
}

/// Request processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    /// Request received
    Received,
    /// Identity verification in progress
    VerifyingIdentity,
    /// Processing the request
    Processing,
    /// Request completed
    Completed,
    /// Request rejected
    Rejected,
}

/// GDPR compliance handler
pub struct GdprHandler {
    requests: HashMap<String, DataSubjectRequest>,
    processing_records: HashMap<String, ProcessingRecord>,
}

/// Record of processing activities
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessingRecord {
    purpose: String,
    lawful_basis: LawfulBasis,
    data_categories: Vec<String>,
    retention_period: chrono::Duration,
    recipients: Vec<String>,
    transfers: Vec<String>, // Third countries
    security_measures: Vec<String>,
}

impl GdprHandler {
    /// Create new GDPR handler
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            processing_records: HashMap::new(),
        }
    }

    /// Submit data subject request
    pub fn submit_request(
        &mut self,
        subject_id: String,
        request_type: GdprRight,
    ) -> ComplianceResult<String> {
        let request_id = format!("DSR-{}", uuid::Uuid::new_v4());
        let now = Utc::now();

        let request = DataSubjectRequest {
            id: request_id.clone(),
            subject_id,
            request_type,
            requested_at: now,
            status: RequestStatus::Received,
            deadline: now + chrono::Duration::days(30),
            notes: vec![],
        };

        self.requests.insert(request_id.clone(), request);
        Ok(request_id)
    }

    /// Process data access request
    pub async fn process_access_request(
        &mut self,
        request_id: &str,
    ) -> ComplianceResult<PersonalDataPackage> {
        let request = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| ComplianceError::InternalError("Request not found".to_string()))?;

        if request.request_type != GdprRight::Access {
            return Err(ComplianceError::ComplianceViolation {
                regulation: "GDPR".to_string(),
                violation: "Invalid request type for access processing".to_string(),
            });
        }

        request.status = RequestStatus::Processing;

        // In a real implementation, this would gather all personal data
        let package = PersonalDataPackage {
            subject_id: request.subject_id.clone(),
            generated_at: Utc::now(),
            data_categories: vec![
                (
                    "profile".to_string(),
                    serde_json::json!({
                        "name": "Example User",
                        "email": "user@example.com"
                    }),
                ),
                (
                    "activity".to_string(),
                    serde_json::json!({
                        "last_login": "2024-01-01T00:00:00Z"
                    }),
                ),
            ],
            processing_purposes: vec!["Service provision".to_string()],
            retention_periods: vec![("profile".to_string(), "7 years".to_string())],
        };

        request.status = RequestStatus::Completed;
        Ok(package)
    }

    /// Process erasure request (right to be forgotten)
    pub async fn process_erasure_request(
        &mut self,
        request_id: &str,
    ) -> ComplianceResult<ErasureReport> {
        // First, get the subject_id immutably
        let subject_id = {
            let request = self
                .requests
                .get(request_id)
                .ok_or_else(|| ComplianceError::InternalError("Request not found".to_string()))?;

            if request.request_type != GdprRight::Erasure {
                return Err(ComplianceError::ComplianceViolation {
                    regulation: "GDPR".to_string(),
                    violation: "Invalid request type for erasure processing".to_string(),
                });
            }

            request.subject_id.clone()
        };

        // Check if erasure can be performed
        let legal_hold = self.check_legal_hold(&subject_id)?;

        // Now get mutable reference to update the request
        let request = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| ComplianceError::InternalError("Request not found".to_string()))?;

        request.status = RequestStatus::Processing;
        if legal_hold {
            request.status = RequestStatus::Rejected;
            request
                .notes
                .push("Cannot erase due to legal hold".to_string());
            return Err(ComplianceError::ComplianceViolation {
                regulation: "GDPR".to_string(),
                violation: "Data under legal hold cannot be erased".to_string(),
            });
        }

        // In a real implementation, this would perform actual erasure
        let report = ErasureReport {
            subject_id: subject_id.clone(),
            erased_at: Utc::now(),
            data_categories_erased: vec![
                "profile".to_string(),
                "activity".to_string(),
                "preferences".to_string(),
            ],
            data_retained: vec![(
                "billing".to_string(),
                "Legal requirement - 7 years".to_string(),
            )],
            anonymized_data: vec!["analytics".to_string()],
        };

        request.status = RequestStatus::Completed;
        Ok(report)
    }

    /// Process portability request
    pub async fn process_portability_request(
        &mut self,
        request_id: &str,
    ) -> ComplianceResult<PortableDataPackage> {
        let request = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| ComplianceError::InternalError("Request not found".to_string()))?;

        if request.request_type != GdprRight::Portability {
            return Err(ComplianceError::ComplianceViolation {
                regulation: "GDPR".to_string(),
                violation: "Invalid request type for portability processing".to_string(),
            });
        }

        request.status = RequestStatus::Processing;

        // Create portable data package
        let package = PortableDataPackage {
            subject_id: request.subject_id.clone(),
            format: "JSON".to_string(),
            generated_at: Utc::now(),
            data: serde_json::json!({
                "profile": {
                    "name": "Example User",
                    "email": "user@example.com"
                },
                "preferences": {
                    "language": "en",
                    "timezone": "UTC"
                }
            }),
            schema_version: "1.0".to_string(),
        };

        request.status = RequestStatus::Completed;
        Ok(package)
    }

    /// Record processing activity
    pub fn record_processing_activity(
        &mut self,
        purpose: String,
        lawful_basis: LawfulBasis,
        data_categories: Vec<String>,
        retention_days: i64,
    ) -> ComplianceResult<()> {
        let record = ProcessingRecord {
            purpose: purpose.clone(),
            lawful_basis,
            data_categories,
            retention_period: chrono::Duration::days(retention_days),
            recipients: vec![],
            transfers: vec![],
            security_measures: vec![
                "Encryption at rest".to_string(),
                "Encryption in transit".to_string(),
                "Access controls".to_string(),
            ],
        };

        self.processing_records.insert(purpose, record);
        Ok(())
    }

    /// Check if data is under legal hold
    fn check_legal_hold(&self, _subject_id: &str) -> ComplianceResult<bool> {
        // In real implementation, check legal hold database
        Ok(false)
    }

    /// Get request status
    pub fn get_request_status(&self, request_id: &str) -> Option<RequestStatus> {
        self.requests.get(request_id).map(|r| r.status)
    }
}

/// Personal data package for access requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalDataPackage {
    /// Data subject ID
    pub subject_id: String,
    /// When the package was generated
    pub generated_at: DateTime<Utc>,
    /// Categories of data and their content
    pub data_categories: Vec<(String, serde_json::Value)>,
    /// Purposes of processing
    pub processing_purposes: Vec<String>,
    /// Retention periods
    pub retention_periods: Vec<(String, String)>,
}

/// Erasure report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureReport {
    /// Data subject ID
    pub subject_id: String,
    /// When erasure was performed
    pub erased_at: DateTime<Utc>,
    /// Categories of data erased
    pub data_categories_erased: Vec<String>,
    /// Data retained with justification
    pub data_retained: Vec<(String, String)>,
    /// Data that was anonymized instead of erased
    pub anonymized_data: Vec<String>,
}

/// Portable data package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortableDataPackage {
    /// Data subject ID
    pub subject_id: String,
    /// Data format
    pub format: String,
    /// When generated
    pub generated_at: DateTime<Utc>,
    /// The actual data
    pub data: serde_json::Value,
    /// Schema version
    pub schema_version: String,
}

impl Default for GdprHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdpr_handler_creation() {
        let handler = GdprHandler::new();
        assert!(handler.requests.is_empty());
        assert!(handler.processing_records.is_empty());
    }

    #[test]
    fn test_submit_access_request() {
        let mut handler = GdprHandler::new();
        let result = handler.submit_request("user123".to_string(), GdprRight::Access);
        assert!(result.is_ok());

        let request_id = result?;
        assert!(request_id.starts_with("DSR-"));

        let status = handler.get_request_status(&request_id);
        assert_eq!(status, Some(RequestStatus::Received));
    }

    #[tokio::test]
    async fn test_process_access_request() {
        let mut handler = GdprHandler::new();
        let request_id = handler
            .submit_request("user123".to_string(), GdprRight::Access)
            .unwrap();

        let result = handler.process_access_request(&request_id).await;
        assert!(result.is_ok());

        let package = result?;
        assert_eq!(package.subject_id, "user123");
        assert!(!package.data_categories.is_empty());

        let status = handler.get_request_status(&request_id);
        assert_eq!(status, Some(RequestStatus::Completed));
    }

    #[tokio::test]
    async fn test_process_erasure_request() {
        let mut handler = GdprHandler::new();
        let request_id = handler
            .submit_request("user123".to_string(), GdprRight::Erasure)
            .unwrap();

        let result = handler.process_erasure_request(&request_id).await;
        assert!(result.is_ok());

        let report = result?;
        assert_eq!(report.subject_id, "user123");
        assert!(!report.data_categories_erased.is_empty());
        assert!(!report.data_retained.is_empty()); // Some data retained for legal reasons
    }

    #[tokio::test]
    async fn test_process_portability_request() {
        let mut handler = GdprHandler::new();
        let request_id = handler
            .submit_request("user123".to_string(), GdprRight::Portability)
            .unwrap();

        let result = handler.process_portability_request(&request_id).await;
        assert!(result.is_ok());

        let package = result?;
        assert_eq!(package.subject_id, "user123");
        assert_eq!(package.format, "JSON");
        assert!(package.data.is_object());
    }

    #[test]
    fn test_record_processing_activity() {
        let mut handler = GdprHandler::new();

        let result = handler.record_processing_activity(
            "User profile management".to_string(),
            LawfulBasis::Consent,
            vec!["personal_data".to_string(), "preferences".to_string()],
            365 * 7, // 7 years
        );

        assert!(result.is_ok());
        assert_eq!(handler.processing_records.len(), 1);
    }

    #[tokio::test]
    async fn test_invalid_request_type() {
        let mut handler = GdprHandler::new();
        let request_id = handler
            .submit_request("user123".to_string(), GdprRight::Access)
            .unwrap();

        // Try to process as erasure request (wrong type)
        let result = handler.process_erasure_request(&request_id).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::ComplianceViolation { .. }
        ));
    }

    #[test]
    fn test_request_deadline() {
        let mut handler = GdprHandler::new();
        let request_id = handler
            .submit_request("user123".to_string(), GdprRight::Access)
            .unwrap();

        let request = &handler.requests[&request_id];
        let duration = request.deadline - request.requested_at;
        assert_eq!(duration.num_days(), 30);
    }

    #[test]
    fn test_lawful_basis_types() {
        assert_ne!(LawfulBasis::Consent, LawfulBasis::Contract);
        assert_ne!(
            LawfulBasis::LegalObligation,
            LawfulBasis::LegitimateInterests
        );
    }
}
