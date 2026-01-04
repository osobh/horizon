//! Core types for the evolution marketplace

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Evolution algorithm package for cross-cluster sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPackage {
    pub id: Uuid,
    pub algorithm_name: String,
    pub version: u64,
    pub cluster_id: String,
    pub author_node: String,
    pub creation_timestamp: u64,
    pub performance_score: f64,
    pub complexity_score: f64,
    pub gpu_architecture: String,
    pub cuda_code: String,
    pub optimization_parameters: HashMap<String, f64>,
    pub benchmark_results: BenchmarkResults,
    pub compatibility_matrix: CompatibilityMatrix,
    pub usage_statistics: UsageStatistics,
    pub reputation_score: f64,
    pub security_hash: String,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub throughput_ops_per_sec: f64,
    pub memory_efficiency: f64,
    pub power_efficiency: f64,
    pub latency_ms: f64,
    pub accuracy_score: f64,
    pub stability_score: f64,
    pub test_dataset_size: usize,
    pub test_duration_seconds: u64,
    pub energy_efficiency_score: f64,
    pub scalability_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMatrix {
    pub cuda_architectures: Vec<String>,
    pub min_memory_gb: f64,
    pub min_compute_capability: f64,
    pub required_features: Vec<String>,
    pub performance_by_arch: HashMap<String, f64>,
    pub supported_data_types: Vec<String>,
    pub memory_access_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageStatistics {
    pub download_count: u64,
    pub successful_deployments: u64,
    pub average_rating: f64,
    pub total_runtime_hours: f64,
    pub reported_issues: u32,
    pub performance_improvements: Vec<f64>,
    pub user_feedback_scores: Vec<f64>,
    pub adoption_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub validator_cluster: String,
    pub package_id: Uuid,
    pub performance_score: f64,
    pub security_score: f64,
    pub compatibility_score: f64,
    pub code_quality_score: f64,
    pub timestamp: u64,
    pub signature: String,
    pub validation_metadata: ValidationMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub validator_reputation: f64,
    pub validation_duration_ms: u64,
    pub gpu_architecture_tested: String,
    pub test_cases_passed: u32,
    pub test_cases_total: u32,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub max_memory_mb: f64,
    pub avg_gpu_utilization: f64,
    pub peak_power_watts: f64,
    pub compilation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionParameters {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub elite_percentage: f64,
    pub max_generations: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidatorInfo {
    pub cluster_id: String,
    pub reputation_score: f64,
    pub total_validations: u64,
    pub successful_validations: u64,
    pub last_seen: u64,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ConsensusRound {
    pub package_id: Uuid,
    pub round_id: Uuid,
    pub start_time: u64,
    pub required_validators: u32,
    pub received_validations: u32,
    pub status: ConsensusStatus,
}

#[derive(Debug, Clone)]
pub(crate) enum ConsensusStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Timeout,
}
