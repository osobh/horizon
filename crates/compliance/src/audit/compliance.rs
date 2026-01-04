//! Compliance reporting and assessment for audit logs

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::audit::types::{AuditEventType, AuditSeverity};

/// Audit compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditComplianceReport {
    /// Report identifier
    pub report_id: Uuid,
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Reporting period start
    pub period_start: DateTime<Utc>,
    /// Reporting period end
    pub period_end: DateTime<Utc>,
    /// Compliance framework (e.g., "GDPR", "HIPAA", "SOC2")
    pub framework: String,
    /// Overall compliance status
    pub is_compliant: bool,
    /// Compliance score (0-100)
    pub compliance_score: f32,
    /// Coverage assessment
    pub coverage: CoverageAssessment,
    /// Identified compliance issues
    pub issues: Vec<ComplianceIssue>,
    /// Compliance metrics
    pub metrics: HashMap<String, f64>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}

/// Compliance issue details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Issue identifier
    pub issue_id: Uuid,
    /// Issue type
    pub issue_type: ComplianceIssueType,
    /// Issue severity
    pub severity: ComplianceIssueSeverity,
    /// Issue description
    pub description: String,
    /// Affected requirement
    pub requirement: String,
    /// Number of occurrences
    pub occurrences: usize,
    /// Example entry IDs
    pub example_entries: Vec<Uuid>,
    /// Remediation guidance
    pub remediation: String,
}

/// Types of compliance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceIssueType {
    /// Missing required audit events
    MissingEvents,
    /// Insufficient retention period
    InsufficientRetention,
    /// Integrity verification failure
    IntegrityFailure,
    /// Missing required metadata
    MissingMetadata,
    /// Policy violation
    PolicyViolation,
    /// Access control issue
    AccessControlIssue,
    /// Encryption requirement not met
    EncryptionIssue,
}

/// Severity of compliance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplianceIssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Coverage assessment for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAssessment {
    /// Event types covered
    pub event_types_covered: Vec<AuditEventType>,
    /// Event types missing
    pub event_types_missing: Vec<AuditEventType>,
    /// Coverage percentage
    pub coverage_percentage: f32,
    /// Severity distribution
    pub severity_distribution: HashMap<AuditSeverity, usize>,
}

impl AuditComplianceReport {
    /// Create a new compliance report
    pub fn new(period_start: DateTime<Utc>, period_end: DateTime<Utc>, framework: String) -> Self {
        Self {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            period_start,
            period_end,
            framework,
            is_compliant: true,
            compliance_score: 100.0,
            coverage: CoverageAssessment::new(),
            issues: Vec::new(),
            metrics: HashMap::new(),
            recommendations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a compliance issue
    pub fn add_issue(&mut self, issue: ComplianceIssue) {
        // Update compliance status based on issue severity
        if issue.severity >= ComplianceIssueSeverity::High {
            self.is_compliant = false;
        }

        // Adjust compliance score
        let score_penalty = match issue.severity {
            ComplianceIssueSeverity::Critical => 25.0,
            ComplianceIssueSeverity::High => 15.0,
            ComplianceIssueSeverity::Medium => 10.0,
            ComplianceIssueSeverity::Low => 5.0,
        };

        self.compliance_score = (self.compliance_score - score_penalty).max(0.0);
        self.issues.push(issue);
    }

    /// Add a recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Set a metric value
    pub fn set_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
    }

    /// Get issues by type
    pub fn issues_by_type(&self, issue_type: ComplianceIssueType) -> Vec<&ComplianceIssue> {
        self.issues
            .iter()
            .filter(|i| i.issue_type == issue_type)
            .collect()
    }

    /// Get critical issues
    pub fn critical_issues(&self) -> Vec<&ComplianceIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == ComplianceIssueSeverity::Critical)
            .collect()
    }
}

impl CoverageAssessment {
    /// Create a new coverage assessment
    pub fn new() -> Self {
        Self {
            event_types_covered: Vec::new(),
            event_types_missing: Vec::new(),
            coverage_percentage: 0.0,
            severity_distribution: HashMap::new(),
        }
    }

    /// Calculate coverage percentage
    pub fn calculate_coverage(&mut self) {
        let total_types = self.event_types_covered.len() + self.event_types_missing.len();
        if total_types > 0 {
            self.coverage_percentage =
                (self.event_types_covered.len() as f32 / total_types as f32) * 100.0;
        }
    }
}

impl Default for CoverageAssessment {
    fn default() -> Self {
        Self::new()
    }
}
