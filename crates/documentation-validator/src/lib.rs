//! Documentation Reality Alignment Tool
//!
//! This crate provides functionality to audit and align documentation claims
//! with validated performance metrics, ensuring all statements have proper
//! status indicators and evidence backing.

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub mod alignment;
pub mod audit;
pub mod status_indicators;
pub mod validation;

/// Status of a performance claim or feature
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimStatus {
    /// Feature is validated with evidence
    Validated,
    /// Feature is in progress with partial validation
    InProgress,
    /// Feature is planned but not yet implemented
    Planned,
    /// Status is unknown or needs review
    Unknown,
}

impl ClaimStatus {
    /// Get the emoji indicator for this status
    pub fn indicator(&self) -> &'static str {
        match self {
            ClaimStatus::Validated => "✅",
            ClaimStatus::InProgress => "🔄",
            ClaimStatus::Planned => "📋",
            ClaimStatus::Unknown => "❓",
        }
    }

    /// Get the text representation
    pub fn as_text(&self) -> &'static str {
        match self {
            ClaimStatus::Validated => "VALIDATED",
            ClaimStatus::InProgress => "IN PROGRESS",
            ClaimStatus::Planned => "PLANNED",
            ClaimStatus::Unknown => "UNKNOWN",
        }
    }
}

/// A performance claim found in documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceClaim {
    /// The claim text
    pub text: String,
    /// File where claim was found
    pub file_path: PathBuf,
    /// Line number in file
    pub line_number: usize,
    /// Current status of the claim
    pub status: ClaimStatus,
    /// Evidence supporting this claim (if any)
    pub evidence: Option<String>,
    /// Validated metric value (if available)
    pub validated_value: Option<String>,
    /// Target or claimed value
    pub claimed_value: Option<String>,
}

/// Configuration for documentation validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Base directory to search for documentation
    pub base_dir: PathBuf,
    /// File patterns to include (glob patterns)
    pub include_patterns: Vec<String>,
    /// File patterns to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,
    /// Known validated performance metrics
    pub validated_metrics: HashMap<String, String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("."),
            include_patterns: vec![
                "*.md".to_string(),
                "README.md".to_string(),
                "docs/**/*.md".to_string(),
                "memory-bank/**/*.md".to_string(),
            ],
            exclude_patterns: vec!["target/**".to_string(), ".git/**".to_string()],
            validated_metrics: HashMap::new(),
        }
    }
}

/// Main documentation validator
#[derive(Debug)]
pub struct DocumentationValidator {
    config: ValidationConfig,
}

impl DocumentationValidator {
    /// Create new validator with configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Create validator with default configuration
    pub fn with_default_config() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Find all markdown files matching patterns
    pub fn find_documentation_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in WalkDir::new(&self.config.base_dir) {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.is_file() && self.should_include_file(path) {
                files.push(path.to_path_buf());
            }
        }

        Ok(files)
    }

    /// Check if file should be included based on patterns
    fn should_include_file(&self, path: &Path) -> bool {
        // Simple pattern matching for now - would expand with glob crate
        let path_str = path.to_string_lossy();

        // Check include patterns
        let included = self.config.include_patterns.iter().any(|pattern| {
            if pattern.ends_with("*.md") {
                path_str.ends_with(".md")
            } else if pattern.contains("**") {
                // Simple recursive pattern matching
                let base = pattern.replace("**/*.md", "");
                path_str.contains(&base) && path_str.ends_with(".md")
            } else {
                path_str.contains(pattern)
            }
        });

        if !included {
            return false;
        }

        // Check exclude patterns
        let excluded = self.config.exclude_patterns.iter().any(|pattern| {
            if pattern.ends_with("**") {
                let base = pattern.trim_end_matches("**").trim_end_matches("/");
                path_str.contains(&format!("/{base}/")) || path_str.starts_with(&format!("{base}/"))
            } else {
                path_str.contains(pattern)
            }
        });

        !excluded
    }

    /// Extract performance claims from a file
    pub fn extract_claims(&self, file_path: &Path) -> Result<Vec<PerformanceClaim>> {
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let claims = self.find_performance_claims(&content, file_path)?;
        Ok(claims)
    }

    /// Find performance claims in text content
    fn find_performance_claims(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Result<Vec<PerformanceClaim>> {
        let mut claims = Vec::new();

        // Performance patterns to look for
        let patterns = [
            r"\d+(?:\.\d+)?μs",                   // Microseconds
            r"\d+(?:\.\d+)?ms",                   // Milliseconds
            r"\d+(?:\.\d+)?%",                    // Percentages
            r"\d+(?:\.\d+)?\s*(?:M|B|K|G)\s*ops", // Operations per second
            r"\d+(?:M|K)\+?\s*agents",            // Agent counts
            r"\d+(?:K)?\+?\s*nodes",              // Node counts
        ];

        for (line_num, line) in content.lines().enumerate() {
            for pattern in &patterns {
                let regex = Regex::new(pattern).context("Invalid regex pattern")?;

                if regex.is_match(line) {
                    // Extract the claim
                    let claim = PerformanceClaim {
                        text: line.trim().to_string(),
                        file_path: file_path.to_path_buf(),
                        line_number: line_num + 1,
                        status: ClaimStatus::Unknown, // Will be determined by validation
                        evidence: None,
                        validated_value: None,
                        claimed_value: regex.find(line).map(|m| m.as_str().to_string()),
                    };
                    claims.push(claim);
                }
            }
        }

        Ok(claims)
    }
}

impl Default for DocumentationValidator {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_claim_status_indicators() {
        assert_eq!(ClaimStatus::Validated.indicator(), "✅");
        assert_eq!(ClaimStatus::InProgress.indicator(), "🔄");
        assert_eq!(ClaimStatus::Planned.indicator(), "📋");
        assert_eq!(ClaimStatus::Unknown.indicator(), "❓");
    }

    #[test]
    fn test_claim_status_text() {
        assert_eq!(ClaimStatus::Validated.as_text(), "VALIDATED");
        assert_eq!(ClaimStatus::InProgress.as_text(), "IN PROGRESS");
        assert_eq!(ClaimStatus::Planned.as_text(), "PLANNED");
        assert_eq!(ClaimStatus::Unknown.as_text(), "UNKNOWN");
    }

    #[test]
    fn test_documentation_validator_creation() {
        let validator = DocumentationValidator::default();
        assert_eq!(validator.config.base_dir, PathBuf::from("."));
        assert!(validator
            .config
            .include_patterns
            .contains(&"*.md".to_string()));
    }

    #[test]
    fn test_should_include_file() {
        let validator = DocumentationValidator::default();

        // Should include .md files
        assert!(validator.should_include_file(Path::new("README.md")));
        assert!(validator.should_include_file(Path::new("docs/api.md")));

        // Should exclude target directory
        assert!(!validator.should_include_file(Path::new("target/debug/file.md")));
        assert!(!validator.should_include_file(Path::new(".git/config")));

        // Should exclude non-markdown files
        assert!(!validator.should_include_file(Path::new("src/main.rs")));
    }

    #[test]
    fn test_extract_performance_claims() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.md");

        let content = r#"
# Test Document

Performance metrics:
- Consensus latency: 73.89μs (excellent performance)
- CPU utilization: 49.2% average 
- GPU utilization: 73.7% sustained
- Processing: 2.6B ops/sec achieved
- Scale: 1000+ nodes supported
- Agents: 10M+ agents capability

Some text without metrics.
"#;

        fs::write(&file_path, content)?;

        let validator = DocumentationValidator::default();
        let claims = validator.extract_claims(&file_path)?;

        // Should find multiple performance claims
        assert!(claims.len() >= 5);

        // Check specific claims
        let latency_claim = claims
            .iter()
            .find(|c| c.text.contains("73.89μs"))
            .expect("Should find latency claim");
        assert_eq!(latency_claim.claimed_value, Some("73.89μs".to_string()));

        let cpu_claim = claims
            .iter()
            .find(|c| c.text.contains("49.2%"))
            .expect("Should find CPU claim");
        assert_eq!(cpu_claim.claimed_value, Some("49.2%".to_string()));

        Ok(())
    }

    #[test]
    fn test_find_documentation_files() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Create test file structure
        fs::create_dir_all(temp_dir.path().join("docs"))?;
        fs::create_dir_all(temp_dir.path().join("src"))?;
        fs::create_dir_all(temp_dir.path().join("target"))?;

        fs::write(temp_dir.path().join("README.md"), "# Test")?;
        fs::write(temp_dir.path().join("docs/api.md"), "# API")?;
        fs::write(temp_dir.path().join("src/main.rs"), "fn main() {}")?;
        fs::write(temp_dir.path().join("target/debug.md"), "# Debug")?;

        let config = ValidationConfig {
            base_dir: temp_dir.path().to_path_buf(),
            include_patterns: vec!["*.md".to_string()],
            exclude_patterns: vec!["target/**".to_string()],
            validated_metrics: HashMap::new(),
        };

        let validator = DocumentationValidator::new(config);
        let files = validator.find_documentation_files()?;

        // Should find README.md and docs/api.md but not target/debug.md
        assert_eq!(files.len(), 2);
        assert!(files
            .iter()
            .any(|f| f.file_name() == Some(std::ffi::OsStr::new("README.md"))));
        assert!(files
            .iter()
            .any(|f| f.file_name() == Some(std::ffi::OsStr::new("api.md"))));
        assert!(!files.iter().any(|f| f.to_string_lossy().contains("target")));

        Ok(())
    }

    #[test]
    fn test_performance_claim_serialization() -> Result<()> {
        let claim = PerformanceClaim {
            text: "Consensus latency: 73.89μs".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 5,
            status: ClaimStatus::Validated,
            evidence: Some("scale_test_results.json".to_string()),
            validated_value: Some("73.89μs".to_string()),
            claimed_value: Some("73.89μs".to_string()),
        };

        let json = serde_json::to_string(&claim)?;
        let deserialized: PerformanceClaim = serde_json::from_str(&json)?;

        assert_eq!(claim.text, deserialized.text);
        assert_eq!(claim.status, deserialized.status);
        assert_eq!(claim.validated_value, deserialized.validated_value);

        Ok(())
    }
}
