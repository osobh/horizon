//! Comprehensive crate validation runner
//! TDD GREEN Phase - Make tests pass with implementation

use crate::documentation_validator::{DocumentationValidator, ValidationReport};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Complete validation results for all crates
#[derive(Debug, Serialize, Deserialize)]
pub struct CrateValidationSummary {
    pub total_crates: usize,
    pub crates_with_issues: usize,
    pub total_files_over_limit: usize,
    pub average_accuracy_score: f64,
    pub crate_reports: Vec<ValidationReport>,
    pub critical_issues: Vec<CriticalIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub crate_name: String,
    pub file_path: PathBuf,
    pub issue: String,
    pub line_count: Option<usize>,
}

/// Validates all crates in the workspace
pub struct WorkspaceValidator {
    workspace_root: PathBuf,
}

impl WorkspaceValidator {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root }
    }

    /// Validate all crates in the workspace
    pub fn validate_all_crates(&self) -> Result<CrateValidationSummary, String> {
        let crates_dir = self.workspace_root.join("crates");
        let mut crate_reports = Vec::new();
        let mut critical_issues = Vec::new();
        let mut total_files_over_limit = 0;

        // Find all crate directories
        for entry in fs::read_dir(&crates_dir)
            .map_err(|e| format!("Failed to read crates directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() && path.join("Cargo.toml").exists() {
                let validator = DocumentationValidator::new(path.clone());
                match validator.validate() {
                    Ok(report) => {
                        // Collect critical issues (files over 750 lines)
                        for file_path in &report.file_counts.files_over_750_lines {
                            if let Ok(content) = fs::read_to_string(file_path) {
                                let line_count = content.lines().count();
                                critical_issues.push(CriticalIssue {
                                    crate_name: report.crate_name.clone(),
                                    file_path: file_path.clone(),
                                    issue: format!(
                                        "File exceeds 750 line limit: {} lines",
                                        line_count
                                    ),
                                    line_count: Some(line_count),
                                });
                            }
                        }

                        total_files_over_limit += report.file_counts.files_over_750_lines.len();
                        crate_reports.push(report);
                    }
                    Err(e) => {
                        eprintln!("Failed to validate crate {:?}: {}", path, e);
                    }
                }
            }
        }

        let total_crates = crate_reports.len();
        let crates_with_issues = crate_reports
            .iter()
            .filter(|r| !r.issues.is_empty())
            .count();

        let average_accuracy_score = if total_crates > 0 {
            crate_reports.iter().map(|r| r.accuracy_score).sum::<f64>() / total_crates as f64
        } else {
            0.0
        };

        // Sort critical issues by line count (descending)
        critical_issues.sort_by(|a, b| b.line_count.unwrap_or(0).cmp(&a.line_count.unwrap_or(0)));

        Ok(CrateValidationSummary {
            total_crates,
            crates_with_issues,
            total_files_over_limit,
            average_accuracy_score,
            crate_reports,
            critical_issues,
        })
    }

    /// Generate a markdown report of validation results
    pub fn generate_markdown_report(&self, summary: &CrateValidationSummary) -> String {
        let mut report = String::new();

        report.push_str("# StratoSwarm Documentation Validation Report\n\n");
        report.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));

        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Total Crates**: {}\n", summary.total_crates));
        report.push_str(&format!(
            "- **Crates with Issues**: {}\n",
            summary.crates_with_issues
        ));
        report.push_str(&format!(
            "- **Files Over 750 Lines**: {}\n",
            summary.total_files_over_limit
        ));
        report.push_str(&format!(
            "- **Average Accuracy Score**: {:.1}%\n\n",
            summary.average_accuracy_score
        ));

        if !summary.critical_issues.is_empty() {
            report.push_str("## Critical Issues (Files Over 750 Lines)\n\n");
            report.push_str("| Crate | File | Lines | Action Required |\n");
            report.push_str("|-------|------|-------|----------------|\n");

            for issue in &summary.critical_issues[..20.min(summary.critical_issues.len())] {
                let file_name = issue
                    .file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                report.push_str(&format!(
                    "| {} | {} | {} | Split into modules |\n",
                    issue.crate_name,
                    file_name,
                    issue.line_count.unwrap_or(0)
                ));
            }

            if summary.critical_issues.len() > 20 {
                report.push_str(&format!(
                    "\n... and {} more files\n",
                    summary.critical_issues.len() - 20
                ));
            }
            report.push('\n');
        }

        report.push_str("## Crate Details\n\n");

        // Sort crates by accuracy score (ascending, so worst first)
        let mut sorted_reports = summary.crate_reports.clone();
        sorted_reports.sort_by(|a, b| a.accuracy_score.partial_cmp(&b.accuracy_score).unwrap());

        for crate_report in &sorted_reports[..10.min(sorted_reports.len())] {
            report.push_str(&format!(
                "### {} (Score: {:.1}%)\n\n",
                crate_report.crate_name, crate_report.accuracy_score
            ));

            if !crate_report.issues.is_empty() {
                report.push_str("**Issues:**\n");
                for issue in &crate_report.issues[..5.min(crate_report.issues.len())] {
                    report.push_str(&format!("- {:?}: {}\n", issue.severity, issue.description));
                }
                if crate_report.issues.len() > 5 {
                    report.push_str(&format!(
                        "- ... and {} more issues\n",
                        crate_report.issues.len() - 5
                    ));
                }
                report.push('\n');
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_workspace_validation() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path().to_path_buf();
        let crates_dir = workspace_root.join("crates");
        fs::create_dir(&crates_dir).unwrap();

        // Create test crate 1 (good)
        let crate1_dir = crates_dir.join("good-crate");
        fs::create_dir(&crate1_dir).unwrap();
        fs::write(
            crate1_dir.join("Cargo.toml"),
            r#"
[package]
name = "good-crate"
version = "0.1.0"
"#,
        )
        .unwrap();
        fs::write(
            crate1_dir.join("README.md"),
            r#"
# Good Crate

## Overview
This is a good crate.

## Features
- Feature 1

## Usage
Use it well.
"#,
        )
        .unwrap();

        // Create test crate 2 (bad - large file)
        let crate2_dir = crates_dir.join("bad-crate");
        fs::create_dir(&crate2_dir).unwrap();
        fs::create_dir(crate2_dir.join("src")).unwrap();
        fs::write(
            crate2_dir.join("Cargo.toml"),
            r#"
[package]
name = "bad-crate"
version = "0.1.0"
"#,
        )
        .unwrap();

        let mut large_file = String::new();
        for i in 0..800 {
            large_file.push_str(&format!("// Line {}\n", i));
        }
        fs::write(crate2_dir.join("src/lib.rs"), large_file).unwrap();

        let validator = WorkspaceValidator::new(workspace_root);
        let summary = validator.validate_all_crates().unwrap();

        assert_eq!(summary.total_crates, 2);
        assert!(summary.total_files_over_limit > 0);
        assert!(summary.average_accuracy_score > 0.0);
        assert!(!summary.critical_issues.is_empty());
    }

    #[test]
    fn test_markdown_report_generation() {
        let summary = CrateValidationSummary {
            total_crates: 2,
            crates_with_issues: 1,
            total_files_over_limit: 1,
            average_accuracy_score: 75.0,
            crate_reports: vec![],
            critical_issues: vec![CriticalIssue {
                crate_name: "test-crate".to_string(),
                file_path: PathBuf::from("src/lib.rs"),
                issue: "File exceeds 750 line limit: 800 lines".to_string(),
                line_count: Some(800),
            }],
        };

        let validator = WorkspaceValidator::new(PathBuf::new());
        let report = validator.generate_markdown_report(&summary);

        assert!(report.contains("# StratoSwarm Documentation Validation Report"));
        assert!(report.contains("**Total Crates**: 2"));
        assert!(report.contains("**Files Over 750 Lines**: 1"));
        assert!(report.contains("test-crate"));
    }
}
