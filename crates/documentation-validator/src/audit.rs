//! Documentation audit functionality
//!
//! This module provides functionality to audit documentation files for
//! performance claims and categorize them by validation status.

use crate::{DocumentationValidator, PerformanceClaim, ValidationConfig};
use anyhow::{Context, Result};
use std::path::PathBuf;

/// Audit results for documentation analysis
#[derive(Debug, Clone, Default)]
pub struct AuditResults {
    /// Total files scanned
    pub files_scanned: usize,
    /// Total claims found
    pub total_claims: usize,
    /// Claims by status
    pub validated_claims: usize,
    pub in_progress_claims: usize,
    pub planned_claims: usize,
    pub unknown_claims: usize,
    /// All performance claims found
    pub claims: Vec<PerformanceClaim>,
}

impl AuditResults {
    /// Create new empty audit results
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a claim to the results and update counters
    pub fn add_claim(&mut self, claim: PerformanceClaim) {
        use crate::ClaimStatus;

        match claim.status {
            ClaimStatus::Validated => self.validated_claims += 1,
            ClaimStatus::InProgress => self.in_progress_claims += 1,
            ClaimStatus::Planned => self.planned_claims += 1,
            ClaimStatus::Unknown => self.unknown_claims += 1,
        }

        self.claims.push(claim);
        self.total_claims += 1;
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "Documentation Audit Summary:\n\
             Files scanned: {}\n\
             Total claims: {}\n\
             ✅ Validated: {} ({:.1}%)\n\
             🔄 In Progress: {} ({:.1}%)\n\
             📋 Planned: {} ({:.1}%)\n\
             ❓ Unknown: {} ({:.1}%)",
            self.files_scanned,
            self.total_claims,
            self.validated_claims,
            (self.validated_claims as f64 / self.total_claims as f64) * 100.0,
            self.in_progress_claims,
            (self.in_progress_claims as f64 / self.total_claims as f64) * 100.0,
            self.planned_claims,
            (self.planned_claims as f64 / self.total_claims as f64) * 100.0,
            self.unknown_claims,
            (self.unknown_claims as f64 / self.total_claims as f64) * 100.0
        )
    }
}

/// Documentation auditor
pub struct DocumentationAuditor {
    validator: DocumentationValidator,
}

impl DocumentationAuditor {
    /// Create new auditor with configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            validator: DocumentationValidator::new(config),
        }
    }

    /// Create auditor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Run full documentation audit
    pub fn audit_documentation(&self) -> Result<AuditResults> {
        let mut results = AuditResults::new();

        // Find all documentation files
        let files = self
            .validator
            .find_documentation_files()
            .context("Failed to find documentation files")?;

        results.files_scanned = files.len();

        // Extract claims from each file
        for file_path in &files {
            let claims = self.validator.extract_claims(file_path).with_context(|| {
                format!("Failed to extract claims from {}", file_path.display())
            })?;

            for claim in claims {
                results.add_claim(claim);
            }
        }

        Ok(results)
    }

    /// Audit specific files
    pub fn audit_files(&self, file_paths: &[PathBuf]) -> Result<AuditResults> {
        let mut results = AuditResults::new();
        results.files_scanned = file_paths.len();

        for file_path in file_paths {
            let claims = self.validator.extract_claims(file_path).with_context(|| {
                format!("Failed to extract claims from {}", file_path.display())
            })?;

            for claim in claims {
                results.add_claim(claim);
            }
        }

        Ok(results)
    }
}

impl Default for DocumentationAuditor {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClaimStatus;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_audit_results_creation() {
        let results = AuditResults::new();
        assert_eq!(results.files_scanned, 0);
        assert_eq!(results.total_claims, 0);
        assert_eq!(results.validated_claims, 0);
        assert!(results.claims.is_empty());
    }

    #[test]
    fn test_audit_results_add_claim() {
        let mut results = AuditResults::new();

        let claim = PerformanceClaim {
            text: "Test claim".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 1,
            status: ClaimStatus::Validated,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        results.add_claim(claim);

        assert_eq!(results.total_claims, 1);
        assert_eq!(results.validated_claims, 1);
        assert_eq!(results.unknown_claims, 0);
    }

    #[test]
    fn test_audit_results_summary() {
        let mut results = AuditResults::new();
        results.files_scanned = 5;
        results.total_claims = 10;
        results.validated_claims = 6;
        results.unknown_claims = 4;

        let summary = results.summary();
        assert!(summary.contains("Files scanned: 5"));
        assert!(summary.contains("Total claims: 10"));
        assert!(summary.contains("✅ Validated: 6 (60.0%)"));
        assert!(summary.contains("❓ Unknown: 4 (40.0%)"));
    }

    #[test]
    fn test_documentation_auditor_creation() {
        let auditor = DocumentationAuditor::default();
        // Should create successfully
        assert!(!auditor.validator.config.include_patterns.is_empty());
    }

    #[test]
    fn test_audit_files() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.md");

        let content = r#"
# Performance Test

- Consensus latency: 73.89μs (validated)
- CPU utilization: 49.2% average
- Throughput: 2.6B ops/sec
"#;

        fs::write(&file_path, content)?;

        let auditor = DocumentationAuditor::default();
        let results = auditor.audit_files(&[file_path])?;

        assert_eq!(results.files_scanned, 1);
        assert!(results.total_claims >= 3);

        // Check that claims were extracted
        assert!(!results.claims.is_empty());
        assert!(results.claims.iter().any(|c| c.text.contains("73.89μs")));

        Ok(())
    }
}
