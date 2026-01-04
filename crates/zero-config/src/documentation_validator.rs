//! Documentation Validator Module
//! Ensures documentation accurately reflects code reality
//! TDD Implementation: RED Phase - Create failing tests

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Documentation validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub crate_name: String,
    pub issues: Vec<ValidationIssue>,
    pub accuracy_score: f64,
    pub file_counts: FileCountStats,
    pub dependency_accuracy: DependencyAccuracy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub description: String,
    pub file_path: Option<PathBuf>,
    pub line_number: Option<usize>,
    pub severity: Severity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IssueType {
    MissingReadme,
    OutdatedExample,
    IncorrectDependency,
    MissingFeature,
    IncorrectFileCount,
    BrokenLink,
    OutdatedVersion,
    MissingTestCoverage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCountStats {
    pub documented_count: usize,
    pub actual_count: usize,
    pub files_over_750_lines: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAccuracy {
    pub documented_deps: HashSet<String>,
    pub actual_deps: HashSet<String>,
    pub missing_deps: HashSet<String>,
    pub extra_deps: HashSet<String>,
}

/// Documentation validator for ensuring accuracy
pub struct DocumentationValidator {
    crate_path: PathBuf,
    max_line_limit: usize,
}

impl DocumentationValidator {
    pub fn new(crate_path: PathBuf) -> Self {
        Self {
            crate_path,
            max_line_limit: 750,
        }
    }

    /// Validate all documentation in a crate
    pub fn validate(&self) -> Result<ValidationReport, String> {
        let crate_name = self.get_crate_name()?;
        let mut issues = Vec::new();

        // Check for README
        let readme_path = self.crate_path.join("README.md");
        if !readme_path.exists() {
            issues.push(ValidationIssue {
                issue_type: IssueType::MissingReadme,
                description: "No README.md found in crate root".to_string(),
                file_path: Some(readme_path),
                line_number: None,
                severity: Severity::Error,
            });
        } else {
            // Validate README content
            self.validate_readme(&readme_path, &mut issues)?;
        }

        // Check file counts
        let file_counts = self.check_file_counts(&mut issues)?;

        // Check dependencies
        let dependency_accuracy = self.check_dependencies(&mut issues)?;

        // Calculate accuracy score
        let accuracy_score = self.calculate_accuracy_score(&issues, &file_counts);

        Ok(ValidationReport {
            crate_name,
            issues,
            accuracy_score,
            file_counts,
            dependency_accuracy,
        })
    }

    fn get_crate_name(&self) -> Result<String, String> {
        let cargo_toml_path = self.crate_path.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)
            .map_err(|e| format!("Failed to read Cargo.toml: {}", e))?;

        // Simple regex to extract crate name
        let name_regex = Regex::new(r#"name\s*=\s*"([^"]+)""#).unwrap();
        if let Some(captures) = name_regex.captures(&content) {
            Ok(captures[1].to_string())
        } else {
            Err("Failed to extract crate name from Cargo.toml".to_string())
        }
    }

    fn validate_readme(
        &self,
        readme_path: &Path,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), String> {
        let content =
            fs::read_to_string(readme_path).map_err(|e| format!("Failed to read README: {}", e))?;

        // Check for common documentation patterns
        let required_sections = vec![
            ("## Overview", "Overview section"),
            ("## Features", "Features section"),
            ("## Usage", "Usage section"),
        ];

        for (section, description) in required_sections {
            if !content.contains(section) {
                issues.push(ValidationIssue {
                    issue_type: IssueType::MissingFeature,
                    description: format!("Missing {} in README", description),
                    file_path: Some(readme_path.to_path_buf()),
                    line_number: None,
                    severity: Severity::Warning,
                });
            }
        }

        // Check for broken links
        self.check_broken_links(&content, readme_path, issues);

        Ok(())
    }

    fn check_broken_links(
        &self,
        content: &str,
        file_path: &Path,
        issues: &mut Vec<ValidationIssue>,
    ) {
        let link_regex = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();

        for (line_num, line) in content.lines().enumerate() {
            for captures in link_regex.captures_iter(line) {
                let link = &captures[2];

                // Check relative file links
                if !link.starts_with("http") && !link.starts_with("#") {
                    let link_path = file_path.parent().unwrap().join(link);
                    if !link_path.exists() {
                        issues.push(ValidationIssue {
                            issue_type: IssueType::BrokenLink,
                            description: format!("Broken link: {}", link),
                            file_path: Some(file_path.to_path_buf()),
                            line_number: Some(line_num + 1),
                            severity: Severity::Error,
                        });
                    }
                }
            }
        }
    }

    fn check_file_counts(
        &self,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<FileCountStats, String> {
        let mut actual_count = 0;
        let mut files_over_750_lines = Vec::new();

        // Count Rust source files
        for entry in walkdir::WalkDir::new(&self.crate_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            .filter(|e| !e.path().to_string_lossy().contains("target"))
        {
            actual_count += 1;

            // Check line count
            if let Ok(content) = fs::read_to_string(entry.path()) {
                let line_count = content.lines().count();
                if line_count > self.max_line_limit {
                    files_over_750_lines.push(entry.path().to_path_buf());
                    issues.push(ValidationIssue {
                        issue_type: IssueType::IncorrectFileCount,
                        description: format!(
                            "File exceeds {} line limit: {} lines",
                            self.max_line_limit, line_count
                        ),
                        file_path: Some(entry.path().to_path_buf()),
                        line_number: None,
                        severity: Severity::Error,
                    });
                }
            }
        }

        // TODO: Compare with documented count in README
        let documented_count = actual_count; // Placeholder

        Ok(FileCountStats {
            documented_count,
            actual_count,
            files_over_750_lines,
        })
    }

    fn check_dependencies(
        &self,
        _issues: &mut Vec<ValidationIssue>,
    ) -> Result<DependencyAccuracy, String> {
        let cargo_toml_path = self.crate_path.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml_path)
            .map_err(|e| format!("Failed to read Cargo.toml: {}", e))?;

        // Extract dependencies using regex
        let mut actual_deps = HashSet::new();
        let dep_regex = Regex::new(r#"^(\w[\w-]*)\s*="#).unwrap();

        let mut in_dependencies = false;
        for line in content.lines() {
            if line.starts_with("[dependencies]") {
                in_dependencies = true;
                continue;
            }
            if line.starts_with("[") && in_dependencies {
                break;
            }
            if in_dependencies {
                if let Some(captures) = dep_regex.captures(line.trim()) {
                    actual_deps.insert(captures[1].to_string());
                }
            }
        }

        // TODO: Compare with documented dependencies in README
        let documented_deps = actual_deps.clone(); // Placeholder
        let missing_deps = HashSet::new();
        let extra_deps = HashSet::new();

        Ok(DependencyAccuracy {
            documented_deps,
            actual_deps,
            missing_deps,
            extra_deps,
        })
    }

    fn calculate_accuracy_score(
        &self,
        issues: &[ValidationIssue],
        file_counts: &FileCountStats,
    ) -> f64 {
        let total_possible_issues = 10.0; // Base score
        let mut deductions = 0.0;

        for issue in issues {
            match issue.severity {
                Severity::Error => deductions += 2.0,
                Severity::Warning => deductions += 1.0,
                Severity::Info => deductions += 0.5,
            }
        }

        // File count accuracy
        if file_counts.documented_count != file_counts.actual_count {
            deductions += 1.0;
        }

        // Files over limit penalty
        deductions += file_counts.files_over_750_lines.len() as f64 * 0.5;

        ((total_possible_issues - deductions) / total_possible_issues * 100.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_missing_readme_detection() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().to_path_buf();

        // Create Cargo.toml
        let cargo_content = r#"
[package]
name = "test-crate"
version = "0.1.0"
"#;
        fs::write(crate_path.join("Cargo.toml"), cargo_content).unwrap();

        let validator = DocumentationValidator::new(crate_path);
        let report = validator.validate().unwrap();

        assert!(report
            .issues
            .iter()
            .any(|issue| issue.issue_type == IssueType::MissingReadme));
        assert!(report.accuracy_score < 100.0);
    }

    #[test]
    fn test_file_over_limit_detection() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().to_path_buf();

        // Create Cargo.toml
        let cargo_content = r#"
[package]
name = "test-crate"
version = "0.1.0"
"#;
        fs::write(crate_path.join("Cargo.toml"), cargo_content).unwrap();

        // Create src directory
        fs::create_dir(crate_path.join("src")).unwrap();

        // Create a file with >750 lines
        let mut large_content = String::new();
        for i in 0..800 {
            large_content.push_str(&format!("// Line {}\n", i));
        }
        fs::write(crate_path.join("src/large.rs"), large_content).unwrap();

        let validator = DocumentationValidator::new(crate_path);
        let report = validator.validate().unwrap();

        assert!(!report.file_counts.files_over_750_lines.is_empty());
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.issue_type == IssueType::IncorrectFileCount));
    }

    #[test]
    fn test_broken_link_detection() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().to_path_buf();

        // Create Cargo.toml
        let cargo_content = r#"
[package]
name = "test-crate"
version = "0.1.0"
"#;
        fs::write(crate_path.join("Cargo.toml"), cargo_content).unwrap();

        // Create README with broken link
        let readme_content = r#"
# Test Crate

## Overview
This is a test.

See [broken link](./non-existent-file.md) for more info.
"#;
        fs::write(crate_path.join("README.md"), readme_content).unwrap();

        let validator = DocumentationValidator::new(crate_path);
        let report = validator.validate().unwrap();

        assert!(report
            .issues
            .iter()
            .any(|issue| issue.issue_type == IssueType::BrokenLink));
    }

    #[test]
    fn test_complete_valid_documentation() {
        let temp_dir = TempDir::new().unwrap();
        let crate_path = temp_dir.path().to_path_buf();

        // Create Cargo.toml
        let cargo_content = r#"
[package]
name = "test-crate"
version = "0.1.0"

[dependencies]
serde = "1.0"
"#;
        fs::write(crate_path.join("Cargo.toml"), cargo_content).unwrap();

        // Create complete README
        let readme_content = r#"
# Test Crate

## Overview
This is a complete test crate with proper documentation.

## Features
- Feature 1
- Feature 2

## Usage
```rust
use test_crate;
```
"#;
        fs::write(crate_path.join("README.md"), readme_content).unwrap();

        // Create src directory with valid file
        fs::create_dir(crate_path.join("src")).unwrap();
        fs::write(crate_path.join("src/lib.rs"), "// Small file\n").unwrap();

        let validator = DocumentationValidator::new(crate_path);
        let report = validator.validate().unwrap();

        // Should have high accuracy score
        assert!(report.accuracy_score > 80.0);
        assert!(report.file_counts.files_over_750_lines.is_empty());
    }
}
