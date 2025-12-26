//! Documentation alignment functionality
//!
//! This module provides functionality to update documentation files with
//! validated status indicators and replace aspirational claims with
//! validated performance data.

use crate::{ClaimStatus, PerformanceClaim};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Documentation alignment engine
#[derive(Debug)]
pub struct DocumentationAligner {
    /// Cache of file contents for batch updates
    file_cache: HashMap<PathBuf, String>,
    /// Replacements to make in each file
    pending_replacements: HashMap<PathBuf, Vec<TextReplacement>>,
}

/// A text replacement to be made in a file
#[derive(Debug, Clone)]
pub struct TextReplacement {
    /// Original text to replace
    pub original: String,
    /// New text with status indicators
    pub replacement: String,
    /// Line number where replacement occurs
    pub line_number: usize,
}

impl DocumentationAligner {
    /// Create new aligner
    pub fn new() -> Self {
        Self {
            file_cache: HashMap::new(),
            pending_replacements: HashMap::new(),
        }
    }

    /// Align claims in documentation with validated status
    pub fn align_claims(&mut self, claims: &[PerformanceClaim]) -> Result<()> {
        // Group claims by file
        let mut claims_by_file: HashMap<PathBuf, Vec<&PerformanceClaim>> = HashMap::new();

        for claim in claims {
            claims_by_file
                .entry(claim.file_path.clone())
                .or_default()
                .push(claim);
        }

        // Process each file
        for (file_path, file_claims) in claims_by_file {
            self.align_file_claims(&file_path, &file_claims)?;
        }

        Ok(())
    }

    /// Align claims in a specific file
    fn align_file_claims(
        &mut self,
        file_path: &PathBuf,
        claims: &[&PerformanceClaim],
    ) -> Result<()> {
        // Load file content
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        self.file_cache.insert(file_path.clone(), content.clone());

        let mut replacements = Vec::new();

        for claim in claims {
            let aligned_text = self.create_aligned_text(claim)?;

            if aligned_text != claim.text {
                replacements.push(TextReplacement {
                    original: claim.text.clone(),
                    replacement: aligned_text,
                    line_number: claim.line_number,
                });
            }
        }

        if !replacements.is_empty() {
            self.pending_replacements
                .insert(file_path.clone(), replacements);
        }

        Ok(())
    }

    /// Create aligned text with proper status indicators
    fn create_aligned_text(&self, claim: &PerformanceClaim) -> Result<String> {
        let status_indicator = claim.status.indicator();

        // Check if this is a list item
        let is_list_item = claim.text.trim_start().starts_with('-');

        let aligned = if is_list_item {
            // For list items, replace the leading dash with indicator + dash
            let trimmed = claim.text.trim_start();
            let after_dash = trimmed.strip_prefix('-').unwrap_or(trimmed).trim_start();

            match &claim.status {
                ClaimStatus::Validated => {
                    let mut aligned = format!("{status_indicator} - {after_dash}");
                    if let Some(evidence) = &claim.evidence {
                        aligned.push_str(&format!(" (validated: {evidence})"));
                    }
                    aligned
                }
                ClaimStatus::InProgress => {
                    format!("{status_indicator} - {after_dash} (implementation in progress)")
                }
                ClaimStatus::Planned => {
                    format!("{status_indicator} - {after_dash} (planned feature)")
                }
                ClaimStatus::Unknown => {
                    format!("{status_indicator} - {after_dash} (needs validation)")
                }
            }
        } else {
            // For non-list items, prepend the indicator
            match &claim.status {
                ClaimStatus::Validated => {
                    let mut aligned = format!("{} {}", status_indicator, claim.text);
                    if let Some(evidence) = &claim.evidence {
                        aligned.push_str(&format!(" (validated: {evidence})"));
                    }
                    aligned
                }
                ClaimStatus::InProgress => {
                    format!(
                        "{} {} (implementation in progress)",
                        status_indicator, claim.text
                    )
                }
                ClaimStatus::Planned => {
                    format!("{} {} (planned feature)", status_indicator, claim.text)
                }
                ClaimStatus::Unknown => {
                    format!("{} {} (needs validation)", status_indicator, claim.text)
                }
            }
        };

        Ok(aligned)
    }

    /// Apply all pending replacements to files
    pub fn apply_replacements(&mut self) -> Result<Vec<PathBuf>> {
        let mut updated_files = Vec::new();

        for (file_path, replacements) in &self.pending_replacements {
            if let Some(content) = self.file_cache.get(file_path) {
                let updated_content = self.apply_file_replacements(content, replacements)?;

                fs::write(file_path, updated_content).with_context(|| {
                    format!("Failed to write updated file: {}", file_path.display())
                })?;

                updated_files.push(file_path.clone());
            }
        }

        // Clear caches after applying
        self.file_cache.clear();
        self.pending_replacements.clear();

        Ok(updated_files)
    }

    /// Apply replacements to file content
    fn apply_file_replacements(
        &self,
        content: &str,
        replacements: &[TextReplacement],
    ) -> Result<String> {
        let mut updated_content = content.to_string();

        // Sort replacements by line number in descending order to avoid offset issues
        let mut sorted_replacements = replacements.to_vec();
        sorted_replacements.sort_by(|a, b| b.line_number.cmp(&a.line_number));

        for replacement in sorted_replacements {
            updated_content =
                updated_content.replace(&replacement.original, &replacement.replacement);
        }

        Ok(updated_content)
    }

    /// Preview changes without applying them
    pub fn preview_changes(&self) -> Result<String> {
        let mut preview = String::new();

        preview.push_str("Documentation Alignment Preview:\n");
        preview.push_str("=====================================\n\n");

        for (file_path, replacements) in &self.pending_replacements {
            preview.push_str(&format!("File: {}\n", file_path.display()));
            preview.push_str(&format!("{}\n", "=".repeat(60)));

            for replacement in replacements {
                preview.push_str(&format!("Line {}: \n", replacement.line_number));
                preview.push_str(&format!("- {}\n", replacement.original));
                preview.push_str(&format!("+ {}\n\n", replacement.replacement));
            }
        }

        Ok(preview)
    }

    /// Create comprehensive performance documentation
    pub fn create_performance_documentation(&self, claims: &[PerformanceClaim]) -> Result<String> {
        let mut doc = String::new();

        doc.push_str("# StratoSwarm Performance Documentation\n\n");
        doc.push_str("## Validated Performance Metrics\n\n");

        // Group claims by status
        let validated: Vec<_> = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Validated)
            .collect();
        let in_progress: Vec<_> = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::InProgress)
            .collect();
        let planned: Vec<_> = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Planned)
            .collect();

        if !validated.is_empty() {
            doc.push_str("### ✅ Validated Performance\n\n");
            for claim in validated {
                // Clean up claim text - remove list prefix if present
                let text = claim.text.trim_start_matches('-').trim();

                // Try to extract the metric name and value
                if text.to_lowercase().contains("consensus") && text.contains("73.89μs") {
                    doc.push_str("- **Consensus Latency: 73.89μs**\n");
                } else if text.to_lowercase().contains("gpu") && text.contains("73.7%") {
                    doc.push_str("- **GPU Utilization: 73.7%**\n");
                } else if text.contains("1000+ nodes") {
                    doc.push_str("- **Scale: 1000+ nodes**\n");
                } else if text.contains("10M+ agents") {
                    doc.push_str("- **Agent Capacity: 10M+ agents**\n");
                } else {
                    doc.push_str(&format!("- **{text}**\n"));
                }

                if let Some(evidence) = &claim.evidence {
                    doc.push_str(&format!("  - Evidence: {evidence}\n"));
                }
                if let Some(value) = &claim.validated_value {
                    doc.push_str(&format!("  - Validated Value: {value}\n"));
                }
                doc.push('\n');
            }
        }

        if !in_progress.is_empty() {
            doc.push_str("### 🔄 In Progress\n\n");
            for claim in in_progress {
                doc.push_str(&format!("- {}\n", claim.text));
            }
            doc.push('\n');
        }

        if !planned.is_empty() {
            doc.push_str("### 📋 Planned Features\n\n");
            for claim in planned {
                doc.push_str(&format!("- {}\n", claim.text));
            }
            doc.push('\n');
        }

        doc.push_str("---\n");
        doc.push_str(&format!(
            "*Generated on: {}*\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        Ok(doc)
    }
}

impl Default for DocumentationAligner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_documentation_aligner_creation() {
        let aligner = DocumentationAligner::new();
        assert!(aligner.file_cache.is_empty());
        assert!(aligner.pending_replacements.is_empty());
    }

    #[test]
    fn test_create_aligned_text_validated() -> Result<()> {
        let aligner = DocumentationAligner::new();

        let claim = PerformanceClaim {
            text: "Consensus latency: 73.89μs".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 5,
            status: ClaimStatus::Validated,
            evidence: Some("scale_test_results.json".to_string()),
            validated_value: Some("73.89μs".to_string()),
            claimed_value: Some("73.89μs".to_string()),
        };

        let aligned = aligner.create_aligned_text(&claim)?;

        assert!(aligned.starts_with("✅"));
        assert!(aligned.contains("validated: scale_test_results.json"));

        Ok(())
    }

    #[test]
    fn test_create_aligned_text_planned() -> Result<()> {
        let aligner = DocumentationAligner::new();

        let claim = PerformanceClaim {
            text: "Target latency: 1μs optimization goal".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 10,
            status: ClaimStatus::Planned,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        let aligned = aligner.create_aligned_text(&claim)?;

        assert!(aligned.starts_with("📋"));
        assert!(aligned.contains("planned feature"));

        Ok(())
    }

    #[test]
    fn test_apply_file_replacements() -> Result<()> {
        let aligner = DocumentationAligner::new();

        let content = "Line 1\nPerformance: 100μs\nLine 3\n";
        let replacements = vec![TextReplacement {
            original: "Performance: 100μs".to_string(),
            replacement: "✅ Performance: 100μs (validated: test.json)".to_string(),
            line_number: 2,
        }];

        let updated = aligner.apply_file_replacements(content, &replacements)?;

        assert!(updated.contains("✅ Performance: 100μs (validated: test.json)"));
        assert!(!updated.contains("Performance: 100μs\n"));

        Ok(())
    }

    #[test]
    fn test_preview_changes() -> Result<()> {
        let mut aligner = DocumentationAligner::new();

        let replacement = TextReplacement {
            original: "Old text".to_string(),
            replacement: "✅ New validated text".to_string(),
            line_number: 5,
        };

        aligner
            .pending_replacements
            .insert(PathBuf::from("test.md"), vec![replacement]);

        let preview = aligner.preview_changes()?;

        assert!(preview.contains("Documentation Alignment Preview"));
        assert!(preview.contains("test.md"));
        assert!(preview.contains("Line 5"));
        assert!(preview.contains("- Old text"));
        assert!(preview.contains("+ ✅ New validated text"));

        Ok(())
    }

    #[test]
    fn test_create_performance_documentation() -> Result<()> {
        let aligner = DocumentationAligner::new();

        let claims = vec![
            PerformanceClaim {
                text: "Validated claim".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 1,
                status: ClaimStatus::Validated,
                evidence: Some("test.json".to_string()),
                validated_value: Some("100ms".to_string()),
                claimed_value: Some("100ms".to_string()),
            },
            PerformanceClaim {
                text: "Planned feature".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 2,
                status: ClaimStatus::Planned,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
        ];

        let doc = aligner.create_performance_documentation(&claims)?;

        assert!(doc.contains("# StratoSwarm Performance Documentation"));
        assert!(doc.contains("✅ Validated Performance"));
        assert!(doc.contains("📋 Planned Features"));
        assert!(doc.contains("Evidence: test.json"));
        assert!(doc.contains("Validated Value: 100ms"));

        Ok(())
    }

    #[test]
    fn test_align_claims_integration() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.md");

        let content = "# Test\n\nPerformance: 73.89μs achieved\nOther content\n";
        fs::write(&file_path, content)?;

        let claims = vec![PerformanceClaim {
            text: "Performance: 73.89μs achieved".to_string(),
            file_path: file_path.clone(),
            line_number: 3,
            status: ClaimStatus::Validated,
            evidence: Some("test.json".to_string()),
            validated_value: Some("73.89μs".to_string()),
            claimed_value: Some("73.89μs".to_string()),
        }];

        let mut aligner = DocumentationAligner::new();
        aligner.align_claims(&claims)?;

        // Check that replacement was prepared
        assert!(aligner.pending_replacements.contains_key(&file_path));

        let updated_files = aligner.apply_replacements()?;
        assert_eq!(updated_files.len(), 1);
        assert_eq!(updated_files[0], file_path);

        // Verify file was updated
        let updated_content = fs::read_to_string(&file_path)?;
        assert!(updated_content.contains("✅"));
        assert!(updated_content.contains("validated: test.json"));

        Ok(())
    }
}
