//! SOC2 compliance framework
//!
//! Provides SOC2 Type II compliance controls for:
//! - Security (Common Criteria)
//! - Availability
//! - Processing Integrity
//! - Confidentiality
//! - Privacy

use crate::data_classification::{DataCategory, DataClassification, DataMetadata};
use crate::error::{ComplianceError, ComplianceResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SOC2 Trust Service Criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrustServiceCriteria {
    /// Security - Protection against unauthorized access
    Security,
    /// Availability - System availability for operation and use
    Availability,
    /// Processing Integrity - System processing is complete, valid, accurate, timely, and authorized
    ProcessingIntegrity,
    /// Confidentiality - Information designated as confidential is protected
    Confidentiality,
    /// Privacy - Personal information is collected, used, retained, disclosed, and disposed of in conformity with commitments
    Privacy,
}

/// SOC2 Control Category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlCategory {
    /// Access Control
    AccessControl,
    /// Change Management
    ChangeManagement,
    /// System Operations
    SystemOperations,
    /// Risk Assessment
    RiskAssessment,
    /// Data Protection
    DataProtection,
    /// Incident Response
    IncidentResponse,
    /// Monitoring
    Monitoring,
    /// Vendor Management
    VendorManagement,
}

/// SOC2 Control Implementation Status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlStatus {
    /// Control is not implemented
    NotImplemented,
    /// Control is partially implemented
    PartiallyImplemented,
    /// Control is fully implemented
    FullyImplemented,
    /// Control implementation is being tested
    Testing,
    /// Control has failed testing
    Failed,
}

/// SOC2 Control Definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Soc2Control {
    /// Control identifier
    pub id: String,
    /// Control name
    pub name: String,
    /// Trust service criteria this control addresses
    pub criteria: Vec<TrustServiceCriteria>,
    /// Control category
    pub category: ControlCategory,
    /// Control description
    pub description: String,
    /// Implementation status
    pub status: ControlStatus,
    /// Control owner
    pub owner: String,
    /// Implementation date
    pub implementation_date: Option<DateTime<Utc>>,
    /// Last assessment date
    pub last_assessment: Option<DateTime<Utc>>,
    /// Evidence collection requirements
    pub evidence_requirements: Vec<String>,
    /// Automated control indicator
    pub is_automated: bool,
}

/// SOC2 Assessment Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResult {
    /// Control ID
    pub control_id: String,
    /// Assessment result
    pub result: ControlAssessmentResult,
    /// Assessment date
    pub assessment_date: DateTime<Utc>,
    /// Assessor
    pub assessor: String,
    /// Evidence collected
    pub evidence: Vec<String>,
    /// Findings and exceptions
    pub findings: Vec<String>,
    /// Remediation recommendations
    pub recommendations: Vec<String>,
}

/// Control Assessment Result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlAssessmentResult {
    /// Control is operating effectively
    Effective,
    /// Control has deficiencies but is substantially effective
    DeficiencyIdentified,
    /// Control is not operating effectively
    Ineffective,
    /// Control could not be tested
    NotTestable,
}

/// SOC2 Compliance Engine
pub struct Soc2Engine {
    controls: HashMap<String, Soc2Control>,
    assessments: HashMap<String, Vec<AssessmentResult>>,
}

impl Soc2Engine {
    /// Create new SOC2 compliance engine
    pub fn new() -> Self {
        let mut engine = Self {
            controls: HashMap::new(),
            assessments: HashMap::new(),
        };

        engine.initialize_standard_controls();
        engine
    }

    /// Initialize standard SOC2 controls
    fn initialize_standard_controls(&mut self) {
        // Security Controls
        self.add_control(Soc2Control {
            id: "CC6.1".to_string(),
            name: "Logical and Physical Access Controls".to_string(),
            criteria: vec![TrustServiceCriteria::Security],
            category: ControlCategory::AccessControl,
            description: "The entity implements logical and physical access controls to protect against threats from sources outside its system boundaries and restricts access to system components".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Security Team".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec![
                "Access control policies".to_string(),
                "User access reviews".to_string(),
                "Physical security assessments".to_string(),
            ],
            is_automated: false,
        });

        // Availability Controls
        self.add_control(Soc2Control {
            id: "A1.1".to_string(),
            name: "System Availability Management".to_string(),
            criteria: vec![TrustServiceCriteria::Availability],
            category: ControlCategory::SystemOperations,
            description: "The entity maintains, monitors, and evaluates current processing capacity and use of system components to manage capacity demand and to enable the implementation of additional capacity to help meet its objectives".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Operations Team".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec![
                "Capacity monitoring reports".to_string(),
                "Performance metrics".to_string(),
                "Incident response logs".to_string(),
            ],
            is_automated: true,
        });

        // Processing Integrity Controls
        self.add_control(Soc2Control {
            id: "PI1.1".to_string(),
            name: "Data Processing Integrity".to_string(),
            criteria: vec![TrustServiceCriteria::ProcessingIntegrity],
            category: ControlCategory::DataProtection,
            description: "The entity implements controls over inputs, processing, and outputs to meet the entity's objectives".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Data Team".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec![
                "Data validation controls".to_string(),
                "Processing logs".to_string(),
                "Error handling procedures".to_string(),
            ],
            is_automated: true,
        });

        // Confidentiality Controls
        self.add_control(Soc2Control {
            id: "C1.1".to_string(),
            name: "Confidential Information Protection".to_string(),
            criteria: vec![TrustServiceCriteria::Confidentiality],
            category: ControlCategory::DataProtection,
            description: "The entity identifies and maintains confidential information to meet the entity's objectives related to confidentiality".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Data Protection Officer".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec![
                "Data classification policies".to_string(),
                "Encryption implementation".to_string(),
                "Access control matrices".to_string(),
            ],
            is_automated: false,
        });

        // Privacy Controls
        self.add_control(Soc2Control {
            id: "P1.1".to_string(),
            name: "Privacy Notice and Consent".to_string(),
            criteria: vec![TrustServiceCriteria::Privacy],
            category: ControlCategory::DataProtection,
            description: "The entity provides notice to data subjects about its privacy practices and obtains consent for the collection or use of personal information".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Privacy Officer".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec![
                "Privacy notices".to_string(),
                "Consent records".to_string(),
                "Data subject communications".to_string(),
            ],
            is_automated: false,
        });
    }

    /// Add a control to the engine
    pub fn add_control(&mut self, control: Soc2Control) {
        self.controls.insert(control.id.clone(), control);
    }

    /// Get control by ID
    pub fn get_control(&self, control_id: &str) -> Option<&Soc2Control> {
        self.controls.get(control_id)
    }

    /// Get all controls for a trust service criteria
    pub fn get_controls_by_criteria(&self, criteria: TrustServiceCriteria) -> Vec<&Soc2Control> {
        self.controls
            .values()
            .filter(|control| control.criteria.contains(&criteria))
            .collect()
    }

    /// Update control status
    pub fn update_control_status(
        &mut self,
        control_id: &str,
        status: ControlStatus,
    ) -> ComplianceResult<()> {
        let control = self.controls.get_mut(control_id).ok_or_else(|| {
            ComplianceError::ConfigurationError(format!("Control not found: {control_id}"))
        })?;

        control.status = status;
        if status == ControlStatus::FullyImplemented && control.implementation_date.is_none() {
            control.implementation_date = Some(Utc::now());
        }

        Ok(())
    }

    /// Assess control effectiveness
    pub fn assess_control(
        &mut self,
        control_id: &str,
        result: ControlAssessmentResult,
        assessor: String,
        evidence: Vec<String>,
        findings: Vec<String>,
    ) -> ComplianceResult<()> {
        // Verify control exists
        if !self.controls.contains_key(control_id) {
            return Err(ComplianceError::ConfigurationError(format!(
                "Control not found: {control_id}"
            )));
        }

        let assessment = AssessmentResult {
            control_id: control_id.to_string(),
            result,
            assessment_date: Utc::now(),
            assessor,
            evidence,
            findings: findings.clone(),
            recommendations: self.generate_recommendations(result, &findings),
        };

        self.assessments
            .entry(control_id.to_string())
            .or_default()
            .push(assessment);

        // Update control last assessment date
        if let Some(control) = self.controls.get_mut(control_id) {
            control.last_assessment = Some(Utc::now());
        }

        Ok(())
    }

    /// Generate recommendations based on assessment result
    fn generate_recommendations(
        &self,
        result: ControlAssessmentResult,
        _findings: &[String],
    ) -> Vec<String> {
        match result {
            ControlAssessmentResult::Effective => vec![],
            ControlAssessmentResult::DeficiencyIdentified => {
                vec!["Address identified deficiencies within 30 days".to_string()]
            }
            ControlAssessmentResult::Ineffective => {
                vec![
                    "Immediate remediation required".to_string(),
                    "Reassess control within 7 days".to_string(),
                    "Consider compensating controls".to_string(),
                ]
            }
            ControlAssessmentResult::NotTestable => {
                vec!["Provide necessary evidence for testing".to_string()]
            }
        }
    }

    /// Validate data handling against SOC2 requirements
    pub fn validate_data_handling(
        &self,
        metadata: &DataMetadata,
        operation: &str,
    ) -> ComplianceResult<()> {
        // Check confidentiality requirements
        if (metadata.classification == DataClassification::RestrictedData
            || metadata.classification == DataClassification::ConfidentialData)
            && !metadata.encryption_required
        {
            return Err(ComplianceError::ComplianceViolation {
                regulation: "SOC2".to_string(),
                violation: format!(
                    "Confidential data requires encryption for operation: {operation}"
                ),
            });
        }

        // Check privacy requirements for PII
        if metadata.category == DataCategory::PII && !metadata.audit_required {
            return Err(ComplianceError::ComplianceViolation {
                regulation: "SOC2".to_string(),
                violation: "PII processing requires audit trail".to_string(),
            });
        }

        // Check processing integrity for sensitive operations
        if matches!(operation, "update" | "delete" | "transfer")
            && (metadata.category == DataCategory::Financial
                || metadata.category == DataCategory::PHI
                || metadata.category == DataCategory::PII)
        {
            // These operations require additional validation
            // This would typically integrate with actual processing controls
        }

        Ok(())
    }

    /// Get compliance status summary
    pub fn get_compliance_status(&self) -> HashMap<TrustServiceCriteria, ComplianceStatus> {
        let mut status_map = HashMap::new();

        for criteria in [
            TrustServiceCriteria::Security,
            TrustServiceCriteria::Availability,
            TrustServiceCriteria::ProcessingIntegrity,
            TrustServiceCriteria::Confidentiality,
            TrustServiceCriteria::Privacy,
        ] {
            let controls = self.get_controls_by_criteria(criteria);
            let total_controls = controls.len() as f64;

            if total_controls == 0.0 {
                status_map.insert(criteria, ComplianceStatus::NotAssessed);
                continue;
            }

            let implemented_count = controls
                .iter()
                .filter(|c| c.status == ControlStatus::FullyImplemented)
                .count() as f64;

            let compliance_percentage = (implemented_count / total_controls) * 100.0;

            let status = match compliance_percentage {
                p if p >= 95.0 => ComplianceStatus::Compliant,
                p if p >= 80.0 => ComplianceStatus::SubstantiallyCompliant,
                p if p >= 50.0 => ComplianceStatus::PartiallyCompliant,
                _ => ComplianceStatus::NonCompliant,
            };

            status_map.insert(criteria, status);
        }

        status_map
    }
}

/// Overall compliance status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStatus {
    /// Fully compliant (95%+ controls implemented)
    Compliant,
    /// Substantially compliant (80-94% controls implemented)
    SubstantiallyCompliant,
    /// Partially compliant (50-79% controls implemented)
    PartiallyCompliant,
    /// Non-compliant (<50% controls implemented)
    NonCompliant,
    /// Not assessed
    NotAssessed,
}

impl Default for Soc2Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_classification::DataClassifier;

    #[test]
    fn test_soc2_engine_creation() {
        let engine = Soc2Engine::new();
        assert_eq!(engine.controls.len(), 5); // Standard controls
    }

    #[test]
    fn test_control_retrieval_by_criteria() {
        let engine = Soc2Engine::new();

        let security_controls = engine.get_controls_by_criteria(TrustServiceCriteria::Security);
        assert!(!security_controls.is_empty());

        let privacy_controls = engine.get_controls_by_criteria(TrustServiceCriteria::Privacy);
        assert!(!privacy_controls.is_empty());
    }

    #[test]
    fn test_control_status_update() {
        let mut engine = Soc2Engine::new();

        let result = engine.update_control_status("CC6.1", ControlStatus::FullyImplemented);
        assert!(result.is_ok());

        let control = engine.get_control("CC6.1")?;
        assert_eq!(control.status, ControlStatus::FullyImplemented);
        assert!(control.implementation_date.is_some());
    }

    #[test]
    fn test_control_status_update_nonexistent() {
        let mut engine = Soc2Engine::new();

        let result = engine.update_control_status("INVALID", ControlStatus::FullyImplemented);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::ConfigurationError(_)
        ));
    }

    #[test]
    fn test_control_assessment() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "CC6.1",
            ControlAssessmentResult::Effective,
            "External Auditor".to_string(),
            vec!["Access logs reviewed".to_string()],
            vec![],
        );
        assert!(result.is_ok());

        let assessments = engine.assessments.get("CC6.1")?;
        assert_eq!(assessments.len(), 1);
        assert_eq!(assessments[0].result, ControlAssessmentResult::Effective);
    }

    #[test]
    fn test_control_assessment_with_findings() {
        let mut engine = Soc2Engine::new();

        let result = engine.assess_control(
            "CC6.1",
            ControlAssessmentResult::DeficiencyIdentified,
            "Internal Auditor".to_string(),
            vec!["Access review performed".to_string()],
            vec!["Some users have excessive privileges".to_string()],
        );
        assert!(result.is_ok());

        let assessments = engine.assessments.get("CC6.1")?;
        assert_eq!(assessments.len(), 1);
        assert!(!assessments[0].recommendations.is_empty());
    }

    #[test]
    fn test_data_handling_validation_confidential() {
        let engine = Soc2Engine::new();
        let classifier = DataClassifier::new();

        // Create confidential data with encryption
        let metadata = classifier
            .classify(
                DataCategory::PII,
                DataClassification::ConfidentialData,
                vec!["US".to_string()],
            )
            .unwrap();

        let result = engine.validate_data_handling(&metadata, "read");
        assert!(result.is_ok());
    }

    #[test]
    fn test_data_handling_validation_pii_requires_audit() {
        let engine = Soc2Engine::new();
        let metadata = DataMetadata {
            classification: DataClassification::ConfidentialData,
            category: DataCategory::PII,
            retention_period: None,
            encryption_required: true,
            allowed_regions: vec!["US".to_string()],
            audit_required: false, // This should cause failure
            owner: None,
            created_at: Utc::now(),
        };

        let result = engine.validate_data_handling(&metadata, "process");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ComplianceError::ComplianceViolation { .. }
        ));
    }

    #[test]
    fn test_compliance_status_calculation() {
        let mut engine = Soc2Engine::new();

        // Initially all controls are not implemented
        let status = engine.get_compliance_status();
        for (_, compliance_status) in status {
            assert_eq!(compliance_status, ComplianceStatus::NonCompliant);
        }

        // Implement all security controls
        let security_controls: Vec<String> = engine
            .get_controls_by_criteria(TrustServiceCriteria::Security)
            .iter()
            .map(|c| c.id.clone())
            .collect();

        for control_id in security_controls {
            engine
                .update_control_status(&control_id, ControlStatus::FullyImplemented)
                .unwrap();
        }

        let status = engine.get_compliance_status();
        assert_eq!(
            status[&TrustServiceCriteria::Security],
            ComplianceStatus::Compliant
        );
    }

    #[test]
    fn test_add_custom_control() {
        let mut engine = Soc2Engine::new();

        let custom_control = Soc2Control {
            id: "CUSTOM1".to_string(),
            name: "Custom Security Control".to_string(),
            criteria: vec![
                TrustServiceCriteria::Security,
                TrustServiceCriteria::Confidentiality,
            ],
            category: ControlCategory::AccessControl,
            description: "Custom control for additional security".to_string(),
            status: ControlStatus::NotImplemented,
            owner: "Security Team".to_string(),
            implementation_date: None,
            last_assessment: None,
            evidence_requirements: vec!["Custom evidence".to_string()],
            is_automated: true,
        };

        engine.add_control(custom_control);
        assert!(engine.get_control("CUSTOM1").is_some());

        // Should appear in both security and confidentiality criteria
        let security_controls = engine.get_controls_by_criteria(TrustServiceCriteria::Security);
        let confidentiality_controls =
            engine.get_controls_by_criteria(TrustServiceCriteria::Confidentiality);

        assert!(security_controls.iter().any(|c| c.id == "CUSTOM1"));
        assert!(confidentiality_controls.iter().any(|c| c.id == "CUSTOM1"));
    }

    #[test]
    fn test_recommendation_generation() {
        let engine = Soc2Engine::new();

        let effective_recs =
            engine.generate_recommendations(ControlAssessmentResult::Effective, &[]);
        assert!(effective_recs.is_empty());

        let ineffective_recs = engine.generate_recommendations(
            ControlAssessmentResult::Ineffective,
            &["Control not working".to_string()],
        );
        assert!(!ineffective_recs.is_empty());
        assert!(ineffective_recs
            .iter()
            .any(|r| r.contains("Immediate remediation")));
    }
}
