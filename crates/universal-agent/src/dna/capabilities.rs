//! Capabilities and fitness tracking for service discovery

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A capability that can be advertised for service discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Unique capability identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this capability does
    pub description: String,
    /// Required skills to provide this capability
    pub required_skills: Vec<String>,
    /// Input schema (JSON Schema)
    pub input_schema: serde_json::Value,
    /// Output schema (JSON Schema)
    pub output_schema: serde_json::Value,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: u64,
    /// Tags for search
    pub tags: Vec<String>,
}

impl Capability {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            required_skills: Vec::new(),
            input_schema: serde_json::Value::Null,
            output_schema: serde_json::Value::Null,
            quality_score: 0.5,
            avg_latency_ms: 0,
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn with_required_skills(mut self, skills: Vec<String>) -> Self {
        self.required_skills = skills;
        self
    }

    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_avg_latency(mut self, latency_ms: u64) -> Self {
        self.avg_latency_ms = latency_ms;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
}

/// Multi-dimensional fitness profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessProfile {
    /// Overall fitness score (0.0 - 1.0)
    pub overall: f64,
    /// Fitness by dimension
    pub dimensions: HashMap<FitnessDimension, f64>,
    /// Historical fitness progression
    pub history: Vec<(DateTime<Utc>, f64)>,
}

impl Default for FitnessProfile {
    fn default() -> Self {
        Self {
            overall: 0.5,
            dimensions: HashMap::new(),
            history: Vec::new(),
        }
    }
}

impl FitnessProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_overall(mut self, score: f64) -> Self {
        self.overall = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_dimension(mut self, dimension: FitnessDimension, score: f64) -> Self {
        self.dimensions.insert(dimension, score.clamp(0.0, 1.0));
        self
    }

    /// Update a fitness dimension
    pub fn update_dimension(&mut self, dimension: FitnessDimension, score: f64) {
        self.dimensions.insert(dimension, score.clamp(0.0, 1.0));
        self.recalculate_overall();
    }

    /// Recalculate overall fitness from dimensions
    pub fn recalculate_overall(&mut self) {
        if self.dimensions.is_empty() {
            return;
        }
        let sum: f64 = self.dimensions.values().sum();
        self.overall = sum / self.dimensions.len() as f64;
        self.record_history();
    }

    /// Record current fitness in history
    fn record_history(&mut self) {
        self.history.push((Utc::now(), self.overall));
        // Keep only last 1000 entries
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Get fitness trend (positive = improving, negative = declining)
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let recent: Vec<f64> = self.history.iter().rev().take(20).map(|(_, s)| *s).collect();
        if recent.len() < 2 {
            return 0.0;
        }
        let first_half: f64 = recent[recent.len() / 2..].iter().sum::<f64>() / (recent.len() / 2) as f64;
        let second_half: f64 = recent[..recent.len() / 2].iter().sum::<f64>() / (recent.len() / 2) as f64;
        second_half - first_half
    }

    /// Get fitness at a specific dimension
    pub fn get_dimension(&self, dimension: &FitnessDimension) -> f64 {
        *self.dimensions.get(dimension).unwrap_or(&0.0)
    }
}

/// Dimensions of fitness measurement
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FitnessDimension {
    TaskCompletion,
    Efficiency,
    Accuracy,
    Latency,
    ResourceUsage,
    Reliability,
    Adaptability,
    Custom(String),
}

impl FitnessDimension {
    pub fn as_str(&self) -> &str {
        match self {
            Self::TaskCompletion => "task_completion",
            Self::Efficiency => "efficiency",
            Self::Accuracy => "accuracy",
            Self::Latency => "latency",
            Self::ResourceUsage => "resource_usage",
            Self::Reliability => "reliability",
            Self::Adaptability => "adaptability",
            Self::Custom(s) => s,
        }
    }
}

/// Benchmark evidence for DNA validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkEvidence {
    /// Latest benchmark results
    pub latest: Option<BenchmarkResult>,
    /// Historical benchmarks
    pub history: Vec<BenchmarkResult>,
    /// Comparison with baseline
    pub baseline_comparison: Option<BaselineComparison>,
}

impl BenchmarkEvidence {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new benchmark result
    pub fn record(&mut self, result: BenchmarkResult) {
        if let Some(old_latest) = self.latest.take() {
            self.history.push(old_latest);
        }
        self.latest = Some(result);
        // Keep only last 100 historical benchmarks
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Check if the agent has passing benchmarks
    pub fn is_passing(&self) -> bool {
        self.latest.as_ref().map(|b| b.passed).unwrap_or(false)
    }

    /// Get the latest benchmark score
    pub fn latest_score(&self) -> Option<f64> {
        self.latest.as_ref().map(|b| b.total_score)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub benchmark_suite: String,
    pub scores: HashMap<String, f64>,
    pub total_score: f64,
    pub passed: bool,
}

impl BenchmarkResult {
    pub fn new(benchmark_suite: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            benchmark_suite: benchmark_suite.into(),
            scores: HashMap::new(),
            total_score: 0.0,
            passed: false,
        }
    }

    pub fn with_score(mut self, category: impl Into<String>, score: f64) -> Self {
        self.scores.insert(category.into(), score);
        self.recalculate_total();
        self
    }

    pub fn with_passed(mut self, passed: bool) -> Self {
        self.passed = passed;
        self
    }

    fn recalculate_total(&mut self) {
        if self.scores.is_empty() {
            self.total_score = 0.0;
        } else {
            self.total_score = self.scores.values().sum::<f64>() / self.scores.len() as f64;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_dna_id: crate::dna::DNAId,
    pub baseline_version: crate::dna::DNAVersion,
    pub improvement_delta: f64,
    pub dimension_deltas: HashMap<FitnessDimension, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_creation() {
        let cap = Capability::new("efficiency_analysis", "Efficiency Analysis")
            .with_description("Analyze system efficiency")
            .with_quality_score(0.85)
            .with_tags(vec!["efficiency".to_string(), "analysis".to_string()]);

        assert_eq!(cap.id, "efficiency_analysis");
        assert_eq!(cap.quality_score, 0.85);
        assert_eq!(cap.tags.len(), 2);
    }

    #[test]
    fn test_fitness_profile() {
        let mut fitness = FitnessProfile::new()
            .with_dimension(FitnessDimension::TaskCompletion, 0.9)
            .with_dimension(FitnessDimension::Efficiency, 0.8)
            .with_dimension(FitnessDimension::Accuracy, 0.85);

        fitness.recalculate_overall();
        assert!((fitness.overall - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_evidence() {
        let mut evidence = BenchmarkEvidence::new();

        let result = BenchmarkResult::new("standard_suite")
            .with_score("speed", 0.9)
            .with_score("accuracy", 0.85)
            .with_passed(true);

        evidence.record(result);

        assert!(evidence.is_passing());
        assert!(evidence.latest_score().unwrap() > 0.8);
    }
}
