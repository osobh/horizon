//! Evidence types and analysis for causal relationships

use chrono::Duration as ChronoDuration;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::EffectDirection;

/// Evidence supporting causal relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEvidence {
    /// Statistical evidence measures
    pub statistical_metrics: StatisticalMetrics,
    /// Temporal evidence
    pub temporal_evidence: TemporalEvidence,
    /// Experimental evidence (if available)
    pub experimental_evidence: Option<ExperimentalEvidence>,
    /// Observational studies supporting the relationship
    pub observational_studies: Vec<ObservationalStudy>,
}

/// Statistical metrics for causal inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetrics {
    /// Correlation coefficient
    pub correlation: f64,
    /// Granger causality test statistic
    pub granger_causality: f64,
    /// Transfer entropy
    pub transfer_entropy: f64,
    /// Partial correlation controlling for confounders
    pub partial_correlation: f64,
    /// P-value for statistical significance
    pub p_value: f64,
}

/// Temporal evidence for causation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvidence {
    /// Temporal precedence (cause before effect)
    pub temporal_precedence: bool,
    /// Consistent temporal ordering across observations
    pub temporal_consistency: f64,
    /// Lag time distribution
    pub lag_distribution: Vec<ChronoDuration>,
}

/// Experimental evidence from controlled studies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalEvidence {
    /// Controlled experiment identifier
    pub experiment_id: String,
    /// Randomized treatment assignment
    pub randomized: bool,
    /// Effect size observed
    pub effect_size: f64,
    /// Confidence interval for effect
    pub confidence_interval: (f64, f64),
}

/// Observational study data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationalStudy {
    /// Study identifier
    pub study_id: String,
    /// Sample size
    pub sample_size: usize,
    /// Observed effect direction
    pub effect_direction: EffectDirection,
    /// Confounders controlled for
    pub controlled_confounders: Vec<String>,
}

/// Counterfactual analysis results
#[derive(Debug, Clone)]
pub struct CounterfactualAnalysis {
    /// Original scenario outcome
    pub factual_outcome: String,
    /// Counterfactual scenario descriptions
    pub counterfactuals: Vec<CounterfactualScenario>,
    /// Causal effect estimates
    pub causal_effects: Vec<CausalEffect>,
}

/// Counterfactual scenario
#[derive(Debug, Clone)]
pub struct CounterfactualScenario {
    /// Description of counterfactual intervention
    pub intervention: String,
    /// Predicted outcome under intervention
    pub predicted_outcome: String,
    /// Confidence in prediction
    pub confidence: f64,
    /// Changed variables and their values
    pub changed_variables: HashMap<String, serde_json::Value>,
}

/// Causal effect measurement
#[derive(Debug, Clone)]
pub struct CausalEffect {
    /// Treatment variable
    pub treatment: String,
    /// Outcome variable
    pub outcome: String,
    /// Average treatment effect
    pub average_effect: f64,
    /// Effect heterogeneity across subgroups
    pub heterogeneity: Vec<SubgroupEffect>,
}

/// Effect within specific subgroups
#[derive(Debug, Clone)]
pub struct SubgroupEffect {
    pub subgroup_characteristics: HashMap<String, serde_json::Value>,
    pub effect_size: f64,
    pub confidence_interval: (f64, f64),
}
