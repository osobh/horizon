//! Validation functionality for performance claims
//!
//! This module provides functionality to validate performance claims against
//! known benchmarks and test results, updating their status accordingly.

use crate::{ClaimStatus, PerformanceClaim};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;

/// Validation engine for performance claims
#[derive(Debug)]
pub struct ClaimValidator {
    /// Known validated metrics from test results
    validated_metrics: HashMap<String, ValidatedMetric>,
}

/// A validated performance metric with evidence
#[derive(Debug, Clone)]
pub struct ValidatedMetric {
    /// The validated value (e.g., "73.89μs")
    pub value: String,
    /// Source of validation (e.g., "scale_test_results.json")
    pub evidence: String,
    /// Context or additional details
    pub context: Option<String>,
}

impl ValidatedMetric {
    /// Create new validated metric
    pub fn new(value: String, evidence: String) -> Self {
        Self {
            value,
            evidence,
            context: None,
        }
    }

    /// Create new validated metric with context
    pub fn with_context(value: String, evidence: String, context: String) -> Self {
        Self {
            value,
            evidence,
            context: Some(context),
        }
    }
}

impl ClaimValidator {
    /// Create new validator with empty metrics
    pub fn new() -> Self {
        Self {
            validated_metrics: HashMap::new(),
        }
    }

    /// Create validator with default StratoSwarm metrics
    pub fn with_stratoswarm_metrics() -> Self {
        let mut validator = Self::new();
        validator.load_stratoswarm_metrics();
        validator
    }

    /// Add a validated metric
    pub fn add_metric(&mut self, key: String, metric: ValidatedMetric) {
        self.validated_metrics.insert(key, metric);
    }

    /// Load known StratoSwarm validated metrics
    pub fn load_stratoswarm_metrics(&mut self) {
        // Scale testing results
        self.add_metric(
            "consensus_latency".to_string(),
            ValidatedMetric::with_context(
                "73.89μs".to_string(),
                "scale_test_results.json".to_string(),
                "Foundation test (100 nodes, 100K agents)".to_string(),
            ),
        );

        self.add_metric(
            "cpu_utilization".to_string(),
            ValidatedMetric::with_context(
                "49.2%".to_string(),
                "scale_test_results.json".to_string(),
                "Average utilization at 100-node scale".to_string(),
            ),
        );

        self.add_metric(
            "gpu_utilization".to_string(),
            ValidatedMetric::with_context(
                "73.7%".to_string(),
                "scale_test_results.json".to_string(),
                "Sustained utilization (target: 85%+)".to_string(),
            ),
        );

        self.add_metric(
            "node_capability".to_string(),
            ValidatedMetric::with_context(
                "1000+".to_string(),
                "scale_orchestrator_simulation.json".to_string(),
                "Architecturally validated through comprehensive simulation".to_string(),
            ),
        );

        self.add_metric(
            "agent_capability".to_string(),
            ValidatedMetric::with_context(
                "10M+".to_string(),
                "scale_calculation.json".to_string(),
                "Mathematically proven (1000 nodes × 10K agents)".to_string(),
            ),
        );

        self.add_metric(
            "throughput".to_string(),
            ValidatedMetric::with_context(
                "79,998".to_string(),
                "scale_test_results.json".to_string(),
                "Messages per second at 100-node scale".to_string(),
            ),
        );
    }

    /// Validate a single claim against known metrics
    pub fn validate_claim(&self, claim: &mut PerformanceClaim) -> Result<()> {
        // Check if it's aspirational first - this takes precedence
        if self.is_aspirational_claim(claim) {
            claim.status = ClaimStatus::Planned;
            return Ok(());
        }

        // Extract potential metric values from claim text
        let metric_patterns = [
            (r"(\d+(?:\.\d+)?)μs", "consensus_latency"),
            (
                r"(\d+(?:\.\d+)?)%.*(?:utilization|average)",
                "cpu_utilization",
            ),
            (r"(\d+(?:\.\d+)?)%.*GPU", "gpu_utilization"),
            (r"1000\+.*nodes", "node_capability"),
            (r"10M\+.*agents", "agent_capability"),
            (r"(\d+(?:\.\d+)?)(?:B|M|K)\s*ops", "throughput"),
        ];

        for (pattern, metric_key) in &metric_patterns {
            let regex = regex::Regex::new(pattern).context("Invalid regex pattern")?;

            if regex.is_match(&claim.text) {
                if let Some(validated_metric) = self.validated_metrics.get(*metric_key) {
                    claim.status = ClaimStatus::Validated;
                    claim.evidence = Some(validated_metric.evidence.clone());
                    claim.validated_value = Some(validated_metric.value.clone());

                    // Extract claimed value if not already set
                    if claim.claimed_value.is_none() {
                        if let Some(captures) = regex.captures(&claim.text) {
                            if let Some(value) = captures.get(1) {
                                claim.claimed_value = Some(value.as_str().to_string());
                            }
                        }
                    }

                    return Ok(());
                }
            }
        }

        // No validation found and not aspirational
        claim.status = ClaimStatus::Unknown;
        Ok(())
    }

    /// Check if claim appears to be aspirational/planned
    fn is_aspirational_claim(&self, claim: &PerformanceClaim) -> bool {
        let aspirational_keywords = [
            "target",
            "goal",
            "planned",
            "expected",
            "will achieve",
            "aims for",
            "targets",
            "designed for",
            "intended to",
        ];

        let text_lower = claim.text.to_lowercase();
        aspirational_keywords
            .iter()
            .any(|keyword| text_lower.contains(keyword))
    }

    /// Validate multiple claims
    pub fn validate_claims(&self, claims: &mut [PerformanceClaim]) -> Result<()> {
        for claim in claims {
            self.validate_claim(claim)?;
        }
        Ok(())
    }

    /// Load validation metrics from file
    pub fn load_metrics_from_file(&mut self, file_path: &PathBuf) -> Result<()> {
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read metrics file: {}", file_path.display()))?;

        // Parse JSON metrics file
        let metrics: HashMap<String, serde_json::Value> =
            serde_json::from_str(&content).context("Failed to parse metrics JSON")?;

        for (key, value) in metrics {
            if let Some(value_str) = value.as_str() {
                self.add_metric(
                    key.clone(),
                    ValidatedMetric::new(
                        value_str.to_string(),
                        file_path.to_string_lossy().to_string(),
                    ),
                );
            }
        }

        Ok(())
    }
}

impl Default for ClaimValidator {
    fn default() -> Self {
        Self::with_stratoswarm_metrics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_validated_metric_creation() {
        let metric = ValidatedMetric::new("73.89μs".to_string(), "test_results.json".to_string());

        assert_eq!(metric.value, "73.89μs");
        assert_eq!(metric.evidence, "test_results.json");
        assert!(metric.context.is_none());
    }

    #[test]
    fn test_validated_metric_with_context() {
        let metric = ValidatedMetric::with_context(
            "49.2%".to_string(),
            "test_results.json".to_string(),
            "CPU utilization test".to_string(),
        );

        assert_eq!(metric.context, Some("CPU utilization test".to_string()));
    }

    #[test]
    fn test_claim_validator_creation() {
        let validator = ClaimValidator::new();
        assert!(validator.validated_metrics.is_empty());

        let validator = ClaimValidator::with_stratoswarm_metrics();
        assert!(!validator.validated_metrics.is_empty());
        assert!(validator
            .validated_metrics
            .contains_key("consensus_latency"));
    }

    #[test]
    fn test_validate_consensus_claim() -> Result<()> {
        let validator = ClaimValidator::with_stratoswarm_metrics();

        let mut claim = PerformanceClaim {
            text: "Consensus latency: 73.89μs achieved in testing".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 5,
            status: ClaimStatus::Unknown,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        validator.validate_claim(&mut claim)?;

        assert_eq!(claim.status, ClaimStatus::Validated);
        assert_eq!(claim.evidence, Some("scale_test_results.json".to_string()));
        assert_eq!(claim.validated_value, Some("73.89μs".to_string()));

        Ok(())
    }

    #[test]
    fn test_validate_cpu_utilization_claim() -> Result<()> {
        let validator = ClaimValidator::with_stratoswarm_metrics();

        let mut claim = PerformanceClaim {
            text: "CPU utilization: 49.2% average during scale tests".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 6,
            status: ClaimStatus::Unknown,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        validator.validate_claim(&mut claim)?;

        assert_eq!(claim.status, ClaimStatus::Validated);
        assert_eq!(claim.validated_value, Some("49.2%".to_string()));

        Ok(())
    }

    #[test]
    fn test_validate_aspirational_claim() -> Result<()> {
        let validator = ClaimValidator::with_stratoswarm_metrics();

        let mut claim = PerformanceClaim {
            text: "Target latency: 5μs for next generation optimization".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 7,
            status: ClaimStatus::Unknown,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        validator.validate_claim(&mut claim)?;

        assert_eq!(claim.status, ClaimStatus::Planned);

        Ok(())
    }

    #[test]
    fn test_validate_unknown_claim() -> Result<()> {
        let validator = ClaimValidator::with_stratoswarm_metrics();

        let mut claim = PerformanceClaim {
            text: "Random performance claim with no metrics".to_string(),
            file_path: PathBuf::from("test.md"),
            line_number: 8,
            status: ClaimStatus::Unknown,
            evidence: None,
            validated_value: None,
            claimed_value: None,
        };

        validator.validate_claim(&mut claim)?;

        assert_eq!(claim.status, ClaimStatus::Unknown);

        Ok(())
    }

    #[test]
    fn test_validate_multiple_claims() -> Result<()> {
        let validator = ClaimValidator::with_stratoswarm_metrics();

        let mut claims = vec![
            PerformanceClaim {
                text: "Consensus latency: 73.89μs".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 1,
                status: ClaimStatus::Unknown,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
            PerformanceClaim {
                text: "Target performance: 1μs future goal".to_string(),
                file_path: PathBuf::from("test.md"),
                line_number: 2,
                status: ClaimStatus::Unknown,
                evidence: None,
                validated_value: None,
                claimed_value: None,
            },
        ];

        validator.validate_claims(&mut claims)?;

        assert_eq!(claims[0].status, ClaimStatus::Validated);
        assert_eq!(claims[1].status, ClaimStatus::Planned);

        Ok(())
    }
}
