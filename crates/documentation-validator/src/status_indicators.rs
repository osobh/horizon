//! Status indicator utilities
//!
//! This module provides utilities for working with status indicators
//! in documentation, including parsing existing indicators and
//! formatting new ones.

use crate::ClaimStatus;

/// Status indicator parser and formatter
pub struct StatusIndicators;

impl StatusIndicators {
    /// Parse status from text that may already have indicators
    pub fn parse_existing_status(text: &str) -> ClaimStatus {
        let trimmed = text.trim();

        if trimmed.starts_with("✅") {
            ClaimStatus::Validated
        } else if trimmed.starts_with("🔄") {
            ClaimStatus::InProgress
        } else if trimmed.starts_with("📋") {
            ClaimStatus::Planned
        } else if trimmed.starts_with("❓") {
            ClaimStatus::Unknown
        } else {
            // Check for text-based status indicators
            let lower = trimmed.to_lowercase();
            if lower.contains("validated") || lower.contains("✅") {
                ClaimStatus::Validated
            } else if lower.contains("in progress") || lower.contains("implementing") {
                ClaimStatus::InProgress
            } else if lower.contains("planned") || lower.contains("roadmap") {
                ClaimStatus::Planned
            } else {
                ClaimStatus::Unknown
            }
        }
    }

    /// Format text with appropriate status indicator
    pub fn format_with_status(text: &str, status: &ClaimStatus) -> String {
        // Remove existing status indicators first
        let clean_text = Self::remove_existing_indicators(text);

        format!("{} {}", status.indicator(), clean_text)
    }

    /// Remove existing status indicators from text
    pub fn remove_existing_indicators(text: &str) -> String {
        let mut clean = text.trim().to_string();

        // Remove emoji indicators
        let indicators = ["✅", "🔄", "📋", "❓"];
        for indicator in &indicators {
            if clean.starts_with(indicator) {
                clean = clean[indicator.len()..].trim_start().to_string();
                break;
            }
        }

        // Remove validation notes
        let validation_patterns = [
            r"\(validated: [^)]+\)",
            r"\(implementation in progress\)",
            r"\(planned feature\)",
            r"\(needs validation\)",
        ];

        for pattern in &validation_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                clean = regex.replace(&clean, "").trim().to_string();
            }
        }

        clean
    }

    /// Generate status summary for multiple claims
    pub fn generate_status_summary(claims: &[crate::PerformanceClaim]) -> String {
        let total = claims.len();
        let validated = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Validated)
            .count();
        let in_progress = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::InProgress)
            .count();
        let planned = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Planned)
            .count();
        let unknown = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Unknown)
            .count();

        format!(
            "Status Summary: {total} claims total\n\
             ✅ Validated: {validated} ({:.1}%)\n\
             🔄 In Progress: {in_progress} ({:.1}%)\n\
             📋 Planned: {planned} ({:.1}%)\n\
             ❓ Unknown: {unknown} ({:.1}%)",
            (validated as f64 / total as f64) * 100.0,
            (in_progress as f64 / total as f64) * 100.0,
            (planned as f64 / total as f64) * 100.0,
            (unknown as f64 / total as f64) * 100.0
        )
    }

    /// Create a legend for status indicators
    pub fn create_legend() -> String {
        "## Status Legend\n\n\
         ✅ **Validated** - Feature is implemented and tested with evidence\n\
         🔄 **In Progress** - Feature is currently being developed\n\
         📋 **Planned** - Feature is planned for future development\n\
         ❓ **Unknown** - Status needs to be reviewed and determined\n"
            .to_string()
    }

    /// Extract performance claims that need attention
    pub fn claims_needing_attention(
        claims: &[crate::PerformanceClaim],
    ) -> Vec<&crate::PerformanceClaim> {
        claims
            .iter()
            .filter(|claim| matches!(claim.status, ClaimStatus::Unknown | ClaimStatus::InProgress))
            .collect()
    }

    /// Generate validation report
    pub fn generate_validation_report(claims: &[crate::PerformanceClaim]) -> String {
        let mut report = String::new();

        report.push_str("# Documentation Validation Report\n\n");
        report.push_str(&Self::generate_status_summary(claims));
        report.push_str("\n\n");
        report.push_str(&Self::create_legend());
        report.push('\n');

        // Claims needing attention
        let attention_claims = Self::claims_needing_attention(claims);
        if !attention_claims.is_empty() {
            report.push_str("## Claims Requiring Attention\n\n");

            for claim in attention_claims {
                report.push_str(&format!(
                    "- **{}** `{}:{}`\n  {}\n\n",
                    claim.status.indicator(),
                    claim.file_path.display(),
                    claim.line_number,
                    claim.text
                ));
            }
        }

        // Validated claims
        let validated_claims: Vec<_> = claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Validated)
            .collect();

        if !validated_claims.is_empty() {
            report.push_str("## Validated Claims\n\n");

            for claim in validated_claims {
                report.push_str(&format!(
                    "- ✅ **{}**\n  - File: `{}:{}`\n",
                    claim.text,
                    claim.file_path.display(),
                    claim.line_number
                ));

                if let Some(evidence) = &claim.evidence {
                    report.push_str(&format!("  - Evidence: {evidence}\n"));
                }

                if let Some(value) = &claim.validated_value {
                    report.push_str(&format!("  - Value: {value}\n"));
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
    use std::path::PathBuf;

    #[test]
    fn test_parse_existing_status() {
        assert_eq!(
            StatusIndicators::parse_existing_status("✅ Validated claim"),
            ClaimStatus::Validated
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("🔄 In progress claim"),
            ClaimStatus::InProgress
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("📋 Planned claim"),
            ClaimStatus::Planned
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("❓ Unknown claim"),
            ClaimStatus::Unknown
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("Plain text"),
            ClaimStatus::Unknown
        );
    }

    #[test]
    fn test_parse_text_based_status() {
        assert_eq!(
            StatusIndicators::parse_existing_status("This is validated performance"),
            ClaimStatus::Validated
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("Currently implementing feature"),
            ClaimStatus::InProgress
        );
        assert_eq!(
            StatusIndicators::parse_existing_status("Planned for next release"),
            ClaimStatus::Planned
        );
    }

    #[test]
    fn test_format_with_status() {
        let text = "Performance metric: 100ms";
        let formatted = StatusIndicators::format_with_status(text, &ClaimStatus::Validated);
        assert_eq!(formatted, "✅ Performance metric: 100ms");
    }

    #[test]
    fn test_format_with_status_removes_existing() {
        let text = "🔄 Performance metric: 100ms (in progress)";
        let formatted = StatusIndicators::format_with_status(text, &ClaimStatus::Validated);
        assert!(formatted.starts_with("✅"));
        assert!(formatted.contains("Performance metric: 100ms"));
        assert!(!formatted.contains("🔄"));
    }

    #[test]
    fn test_remove_existing_indicators() {
        assert_eq!(
            StatusIndicators::remove_existing_indicators("✅ Clean text"),
            "Clean text"
        );
        assert_eq!(
            StatusIndicators::remove_existing_indicators("🔄 Text (implementation in progress)"),
            "Text"
        );
        assert_eq!(
            StatusIndicators::remove_existing_indicators("Plain text"),
            "Plain text"
        );
    }

    #[test]
    fn test_generate_status_summary() {
        let claims = vec![
            crate::PerformanceClaim {
                text: "Claim 1".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 1,
                status: ClaimStatus::Validated,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
            crate::PerformanceClaim {
                text: "Claim 2".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 2,
                status: ClaimStatus::Planned,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
        ];

        let summary = StatusIndicators::generate_status_summary(&claims);
        assert!(summary.contains("2 claims total"));
        assert!(summary.contains("✅ Validated: 1 (50.0%)"));
        assert!(summary.contains("📋 Planned: 1 (50.0%)"));
    }

    #[test]
    fn test_create_legend() {
        let legend = StatusIndicators::create_legend();
        assert!(legend.contains("## Status Legend"));
        assert!(legend.contains("✅ **Validated**"));
        assert!(legend.contains("🔄 **In Progress**"));
        assert!(legend.contains("📋 **Planned**"));
        assert!(legend.contains("❓ **Unknown**"));
    }

    #[test]
    fn test_claims_needing_attention() {
        let claims = vec![
            crate::PerformanceClaim {
                text: "Validated claim".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 1,
                status: ClaimStatus::Validated,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
            crate::PerformanceClaim {
                text: "Unknown claim".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 2,
                status: ClaimStatus::Unknown,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
            crate::PerformanceClaim {
                text: "In progress claim".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 3,
                status: ClaimStatus::InProgress,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
        ];

        let attention_claims = StatusIndicators::claims_needing_attention(&claims);
        assert_eq!(attention_claims.len(), 2);
        assert!(attention_claims.iter().any(|c| c.text == "Unknown claim"));
        assert!(attention_claims
            .iter()
            .any(|c| c.text == "In progress claim"));
    }

    #[test]
    fn test_generate_validation_report() {
        let claims = vec![crate::PerformanceClaim {
            text: "Validated claim".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 1,
            status: ClaimStatus::Validated,
            evidence: Some("test.json".to_string()),
            validated_value: Some("100ms".to_string()),
            claimed_value: None,
        }];

        let report = StatusIndicators::generate_validation_report(&claims);
        assert!(report.contains("# Documentation Validation Report"));
        assert!(report.contains("## Status Legend"));
        assert!(report.contains("## Validated Claims"));
        assert!(report.contains("Evidence: test.json"));
    }
}
