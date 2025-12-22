//! Evolution streaming pipeline for high-throughput agent processing
//!
//! This crate provides specialized streaming components for agent evolution,
//! including mutation streams, GPU evaluation pipelines, and archive updates.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;
use uuid::Uuid;

pub mod archive;
pub mod evaluation;
pub mod mutation;
pub mod pipeline;

// Re-export main types
pub use pipeline::*;

/// Evolution streaming errors
#[derive(Error, Debug)]
pub enum EvolutionStreamingError {
    #[error("Agent processing failed: {reason}")]
    AgentProcessingFailed { reason: String },

    #[error("Mutation failed: {0}")]
    MutationFailed(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Archive operation failed: {0}")]
    ArchiveFailed(String),

    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Resource exhausted: {reason}")]
    ResourceExhausted { reason: String },

    #[error("Safety violation: {reason}")]
    SafetyViolation { reason: String },
}

/// Agent identifier
pub type AgentId = Uuid;

/// Agent genome representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenome {
    pub id: AgentId,
    pub code: String,
    pub parameters: Vec<f32>,
    pub metadata: GenomeMetadata,
}

/// Metadata associated with agent genome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeMetadata {
    pub generation: u64,
    pub parent_id: Option<AgentId>,
    pub mutation_count: u32,
    pub creation_time: u64,
    pub lineage_depth: u32,
}

/// Mutated agent containing original and mutated genomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutatedAgent {
    pub original: AgentGenome,
    pub mutated: AgentGenome,
    pub mutations: Vec<MutationInfo>,
    pub mutation_time: u64,
}

/// Information about applied mutations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationInfo {
    pub mutation_type: MutationType,
    pub location: CodeLocation,
    pub impact_score: f32,
}

/// Types of mutations that can be applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    /// Parameter adjustment
    ParameterTweak { parameter_id: usize, delta: f32 },
    /// Code substitution
    CodeSubstitution {
        old_pattern: String,
        new_pattern: String,
    },
    /// Structure modification
    StructureChange { change_type: String },
    /// Random noise injection
    NoiseInjection { intensity: f32 },
}

/// Location in code where mutation occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub line: u32,
    pub column: u32,
    pub length: u32,
}

/// Agent evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub agent_id: AgentId,
    pub fitness: f64,
    pub metrics: EvaluationMetrics,
    pub execution_time: u64, // nanoseconds
    pub memory_usage: u64,   // bytes
}

/// Detailed evaluation metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub performance_score: f64,
    pub correctness_score: f64,
    pub efficiency_score: f64,
    pub safety_score: f64,
    pub benchmark_results: Vec<BenchmarkResult>,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub score: f64,
    pub execution_time: u64,
    pub passed: bool,
}

/// Evolution event in the streaming pipeline
#[derive(Debug, Clone)]
pub enum EvolutionEvent {
    /// Request to mutate an agent
    MutationRequest {
        agent: AgentGenome,
        mutation_count: u32,
    },
    /// Mutation completed
    MutationComplete { mutated_agent: MutatedAgent },
    /// Evaluation request
    EvaluationRequest { agent: AgentGenome },
    /// Evaluation completed
    EvaluationComplete { result: EvaluationResult },
    /// Archive update request
    ArchiveUpdate { agent: AgentGenome, fitness: f64 },
    /// Pipeline statistics
    PipelineStats {
        throughput: f64,
        latency: u64,
        queue_depth: usize,
    },
}

/// Selection strategy for choosing agents
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Tournament selection
    Tournament { size: usize },
    /// Roulette wheel selection
    RouletteWheel,
    /// Rank-based selection
    RankBased,
    /// Elite selection (top performers)
    Elite { count: usize },
    /// Random selection
    Random,
}

/// Archive update operations
#[derive(Debug, Clone)]
pub enum ArchiveUpdate {
    /// Add new agent to archive
    AddAgent { agent: AgentGenome, fitness: f64 },
    /// Update existing agent fitness
    UpdateFitness { agent_id: AgentId, new_fitness: f64 },
    /// Remove agent from archive
    RemoveAgent { agent_id: AgentId },
    /// Compact archive storage
    CompactArchive,
}

impl AgentGenome {
    /// Create a new agent genome
    pub fn new(code: String, parameters: Vec<f32>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: Uuid::new_v4(),
            code,
            parameters,
            metadata: GenomeMetadata {
                generation: 0,
                parent_id: None,
                mutation_count: 0,
                creation_time: timestamp,
                lineage_depth: 0,
            },
        }
    }

    /// Create child genome from parent
    pub fn from_parent(parent: &AgentGenome, code: String, parameters: Vec<f32>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: Uuid::new_v4(),
            code,
            parameters,
            metadata: GenomeMetadata {
                generation: parent.metadata.generation + 1,
                parent_id: Some(parent.id),
                mutation_count: parent.metadata.mutation_count + 1,
                creation_time: timestamp,
                lineage_depth: parent.metadata.lineage_depth + 1,
            },
        }
    }

    /// Get the size of the genome in bytes
    pub fn size(&self) -> usize {
        self.code.len() + (self.parameters.len() * 4) + 64 // rough estimate
    }

    /// Calculate similarity to another genome
    pub fn similarity(&self, other: &AgentGenome) -> f32 {
        // Simple similarity based on code length and parameter differences
        let code_similarity = if self.code == other.code {
            1.0
        } else {
            let max_len = self.code.len().max(other.code.len()) as f32;
            let common_chars = self
                .code
                .chars()
                .zip(other.code.chars())
                .filter(|(a, b)| a == b)
                .count() as f32;
            common_chars / max_len
        };

        let param_similarity = if self.parameters.len() == other.parameters.len() {
            let diff_sum: f32 = self
                .parameters
                .iter()
                .zip(other.parameters.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            1.0 - (diff_sum / self.parameters.len() as f32).min(1.0)
        } else {
            0.0
        };

        (code_similarity + param_similarity) / 2.0
    }
}

impl fmt::Display for AgentGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Agent(id={}, gen={}, params={})",
            self.id,
            self.metadata.generation,
            self.parameters.len()
        )
    }
}

impl EvaluationResult {
    /// Create a new evaluation result
    pub fn new(agent_id: AgentId, fitness: f64) -> Self {
        Self {
            agent_id,
            fitness,
            metrics: EvaluationMetrics::default(),
            execution_time: 0,
            memory_usage: 0,
        }
    }

    /// Check if evaluation passed all benchmarks
    pub fn passed_all_benchmarks(&self) -> bool {
        !self.metrics.benchmark_results.is_empty()
            && self.metrics.benchmark_results.iter().all(|r| r.passed)
    }

    /// Get average benchmark score
    pub fn average_benchmark_score(&self) -> f64 {
        if self.metrics.benchmark_results.is_empty() {
            0.0
        } else {
            self.metrics
                .benchmark_results
                .iter()
                .map(|r| r.score)
                .sum::<f64>()
                / self.metrics.benchmark_results.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_genome_creation() {
        let code = "fn test() { return 42; }".to_string();
        let params = vec![1.0, 2.0, 3.0];
        let genome = AgentGenome::new(code.clone(), params.clone());

        assert_eq!(genome.code, code);
        assert_eq!(genome.parameters, params);
        assert_eq!(genome.metadata.generation, 0);
        assert!(genome.metadata.parent_id.is_none());
        assert_eq!(genome.metadata.mutation_count, 0);
        assert!(genome.metadata.creation_time > 0);
    }

    #[test]
    fn test_agent_genome_from_parent() {
        let parent_code = "fn parent() { return 1; }".to_string();
        let parent_params = vec![1.0];
        let parent = AgentGenome::new(parent_code, parent_params);

        let child_code = "fn child() { return 2; }".to_string();
        let child_params = vec![2.0];
        let child = AgentGenome::from_parent(&parent, child_code.clone(), child_params.clone());

        assert_eq!(child.code, child_code);
        assert_eq!(child.parameters, child_params);
        assert_eq!(child.metadata.generation, 1);
        assert_eq!(child.metadata.parent_id, Some(parent.id));
        assert_eq!(child.metadata.mutation_count, 1);
        assert_eq!(child.metadata.lineage_depth, 1);
    }

    #[test]
    fn test_genome_similarity() {
        let genome1 = AgentGenome::new("test".to_string(), vec![1.0, 2.0]);
        let genome2 = AgentGenome::new("test".to_string(), vec![1.0, 2.0]);
        let genome3 = AgentGenome::new("different".to_string(), vec![3.0, 4.0]);

        assert_eq!(genome1.similarity(&genome2), 1.0);
        assert!(genome1.similarity(&genome3) < 1.0);
    }

    #[test]
    fn test_evaluation_result() {
        let agent_id = Uuid::new_v4();
        let mut result = EvaluationResult::new(agent_id, 0.8);

        result.metrics.benchmark_results = vec![
            BenchmarkResult {
                name: "test1".to_string(),
                score: 0.9,
                execution_time: 1000,
                passed: true,
            },
            BenchmarkResult {
                name: "test2".to_string(),
                score: 0.7,
                execution_time: 2000,
                passed: true,
            },
        ];

        assert!(result.passed_all_benchmarks());
        assert_eq!(result.average_benchmark_score(), 0.8);
    }

    #[test]
    fn test_genome_display() {
        let genome = AgentGenome::new("test".to_string(), vec![1.0, 2.0, 3.0]);
        let display = format!("{genome}");

        assert!(display.contains(&genome.id.to_string()));
        assert!(display.contains("gen=0"));
        assert!(display.contains("params=3"));
    }

    #[test]
    fn test_mutation_type_serialization() {
        let mutations = vec![
            MutationType::ParameterTweak {
                parameter_id: 0,
                delta: 0.5,
            },
            MutationType::CodeSubstitution {
                old_pattern: "old".to_string(),
                new_pattern: "new".to_string(),
            },
            MutationType::StructureChange {
                change_type: "refactor".to_string(),
            },
            MutationType::NoiseInjection { intensity: 0.1 },
        ];

        for mutation in mutations {
            let json = serde_json::to_string(&mutation).unwrap();
            let parsed: MutationType = serde_json::from_str(&json).unwrap();
            match (mutation, parsed) {
                (
                    MutationType::ParameterTweak {
                        parameter_id: id1,
                        delta: d1,
                    },
                    MutationType::ParameterTweak {
                        parameter_id: id2,
                        delta: d2,
                    },
                ) => {
                    assert_eq!(id1, id2);
                    assert_eq!(d1, d2);
                }
                _ => {} // Other variants checked by successful deserialization
            }
        }
    }

    #[test]
    fn test_code_location() {
        let loc = CodeLocation {
            line: 42,
            column: 10,
            length: 5,
        };

        let json = serde_json::to_string(&loc).unwrap();
        let parsed: CodeLocation = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.line, 42);
        assert_eq!(parsed.column, 10);
        assert_eq!(parsed.length, 5);
    }

    #[test]
    fn test_genome_metadata() {
        let metadata = GenomeMetadata {
            generation: 10,
            parent_id: Some(Uuid::new_v4()),
            mutation_count: 5,
            creation_time: 1000,
            lineage_depth: 3,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: GenomeMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.generation, metadata.generation);
        assert_eq!(parsed.parent_id, metadata.parent_id);
        assert_eq!(parsed.mutation_count, metadata.mutation_count);
        assert_eq!(parsed.creation_time, metadata.creation_time);
        assert_eq!(parsed.lineage_depth, metadata.lineage_depth);
    }

    #[test]
    fn test_mutated_agent() {
        let original = AgentGenome::new("original".to_string(), vec![1.0]);
        let mutated = AgentGenome::from_parent(&original, "mutated".to_string(), vec![1.5]);

        let mutated_agent = MutatedAgent {
            original: original.clone(),
            mutated: mutated.clone(),
            mutations: vec![MutationInfo {
                mutation_type: MutationType::ParameterTweak {
                    parameter_id: 0,
                    delta: 0.5,
                },
                location: CodeLocation {
                    line: 1,
                    column: 1,
                    length: 10,
                },
                impact_score: 0.3,
            }],
            mutation_time: 2000,
        };

        assert_eq!(mutated_agent.original.id, original.id);
        assert_eq!(mutated_agent.mutated.id, mutated.id);
        assert_eq!(mutated_agent.mutations.len(), 1);
        assert_eq!(mutated_agent.mutation_time, 2000);
    }

    #[test]
    fn test_evaluation_metrics_default() {
        let metrics = EvaluationMetrics::default();

        assert_eq!(metrics.performance_score, 0.0);
        assert_eq!(metrics.correctness_score, 0.0);
        assert_eq!(metrics.efficiency_score, 0.0);
        assert_eq!(metrics.safety_score, 0.0);
        assert!(metrics.benchmark_results.is_empty());
    }

    #[test]
    fn test_benchmark_result() {
        let benchmark = BenchmarkResult {
            name: "matrix_multiply".to_string(),
            score: 0.95,
            execution_time: 1500,
            passed: true,
        };

        let json = serde_json::to_string(&benchmark).unwrap();
        let parsed: BenchmarkResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, benchmark.name);
        assert_eq!(parsed.score, benchmark.score);
        assert_eq!(parsed.execution_time, benchmark.execution_time);
        assert_eq!(parsed.passed, benchmark.passed);
    }

    #[test]
    fn test_genome_size_calculation() {
        let genome = AgentGenome::new("a".repeat(100), vec![1.0; 10]);
        let size = genome.size();

        // Code: 100 bytes, Parameters: 10 * 4 = 40 bytes, Metadata: ~64 bytes
        assert!(size >= 100 + 40 + 64);
    }

    #[test]
    fn test_similarity_identical_genomes() {
        let genome1 = AgentGenome::new("identical".to_string(), vec![1.0, 2.0, 3.0]);
        let genome2 = AgentGenome::new("identical".to_string(), vec![1.0, 2.0, 3.0]);

        assert_eq!(genome1.similarity(&genome2), 1.0);
    }

    #[test]
    fn test_similarity_different_params() {
        let genome1 = AgentGenome::new("same".to_string(), vec![1.0, 2.0]);
        let genome2 = AgentGenome::new("same".to_string(), vec![3.0, 4.0]);

        let similarity = genome1.similarity(&genome2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_similarity_different_code() {
        let genome1 = AgentGenome::new("code1".to_string(), vec![1.0]);
        let genome2 = AgentGenome::new("code2".to_string(), vec![1.0]);

        let similarity = genome1.similarity(&genome2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_similarity_different_param_lengths() {
        let genome1 = AgentGenome::new("code".to_string(), vec![1.0, 2.0]);
        let genome2 = AgentGenome::new("code".to_string(), vec![1.0, 2.0, 3.0]);

        let similarity = genome1.similarity(&genome2);
        assert_eq!(similarity, 0.5); // Code similarity = 1.0, param similarity = 0.0
    }

    #[test]
    fn test_evaluation_result_no_benchmarks() {
        let result = EvaluationResult::new(Uuid::new_v4(), 0.5);

        assert!(!result.passed_all_benchmarks());
        assert_eq!(result.average_benchmark_score(), 0.0);
    }

    #[test]
    fn test_evaluation_result_failed_benchmark() {
        let agent_id = Uuid::new_v4();
        let mut result = EvaluationResult::new(agent_id, 0.8);

        result.metrics.benchmark_results = vec![
            BenchmarkResult {
                name: "test1".to_string(),
                score: 0.9,
                execution_time: 1000,
                passed: true,
            },
            BenchmarkResult {
                name: "test2".to_string(),
                score: 0.3,
                execution_time: 2000,
                passed: false,
            },
        ];

        assert!(!result.passed_all_benchmarks());
        assert_eq!(result.average_benchmark_score(), 0.6);
    }

    #[test]
    fn test_evolution_streaming_error_messages() {
        let errors = vec![
            EvolutionStreamingError::AgentProcessingFailed {
                reason: "Out of memory".to_string(),
            },
            EvolutionStreamingError::MutationFailed("Invalid mutation".to_string()),
            EvolutionStreamingError::EvaluationFailed("GPU error".to_string()),
            EvolutionStreamingError::ArchiveFailed("Storage full".to_string()),
            EvolutionStreamingError::PipelineError("Queue overflow".to_string()),
            EvolutionStreamingError::ResourceExhausted {
                reason: "GPU memory full".to_string(),
            },
            EvolutionStreamingError::SafetyViolation {
                reason: "Unsafe operation detected".to_string(),
            },
        ];

        for error in errors {
            let message = error.to_string();
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn test_mutation_info_impact_score() {
        let info = MutationInfo {
            mutation_type: MutationType::NoiseInjection { intensity: 0.5 },
            location: CodeLocation {
                line: 10,
                column: 5,
                length: 20,
            },
            impact_score: 0.75,
        };

        assert_eq!(info.impact_score, 0.75);
        assert_eq!(info.location.line, 10);
    }

    #[test]
    fn test_genome_lineage_depth() {
        let mut genome = AgentGenome::new("root".to_string(), vec![]);

        for i in 1..=10 {
            genome = AgentGenome::from_parent(&genome, format!("gen_{i}"), vec![i as f32]);
            assert_eq!(genome.metadata.generation, i as u64);
            assert_eq!(genome.metadata.lineage_depth, i as u32);
        }
    }

    #[test]
    fn test_evaluation_metrics_serialization() {
        let metrics = EvaluationMetrics {
            performance_score: 0.9,
            correctness_score: 1.0,
            efficiency_score: 0.8,
            safety_score: 0.95,
            benchmark_results: vec![BenchmarkResult {
                name: "test".to_string(),
                score: 0.85,
                execution_time: 100,
                passed: true,
            }],
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: EvaluationMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.performance_score, metrics.performance_score);
        assert_eq!(parsed.correctness_score, metrics.correctness_score);
        assert_eq!(parsed.efficiency_score, metrics.efficiency_score);
        assert_eq!(parsed.safety_score, metrics.safety_score);
        assert_eq!(parsed.benchmark_results.len(), 1);
    }

    #[test]
    fn test_genome_creation_timestamp() {
        let genome1 = AgentGenome::new("test1".to_string(), vec![]);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let genome2 = AgentGenome::new("test2".to_string(), vec![]);

        assert!(genome2.metadata.creation_time > genome1.metadata.creation_time);
    }

    #[test]
    fn test_mutated_agent_serialization() {
        let original = AgentGenome::new("original".to_string(), vec![1.0]);
        let mutated = AgentGenome::from_parent(&original, "mutated".to_string(), vec![2.0]);

        let mutated_agent = MutatedAgent {
            original: original.clone(),
            mutated: mutated.clone(),
            mutations: vec![],
            mutation_time: 3000,
        };

        let json = serde_json::to_string(&mutated_agent).unwrap();
        let parsed: MutatedAgent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.original.id, original.id);
        assert_eq!(parsed.mutated.id, mutated.id);
        assert_eq!(parsed.mutation_time, 3000);
    }

    #[test]
    fn test_evaluation_result_with_metrics() {
        let agent_id = Uuid::new_v4();
        let mut result = EvaluationResult::new(agent_id, 0.85);

        result.execution_time = 1_000_000; // 1ms in nanoseconds
        result.memory_usage = 1024 * 1024; // 1MB
        result.metrics.performance_score = 0.9;
        result.metrics.correctness_score = 0.8;
        result.metrics.efficiency_score = 0.85;
        result.metrics.safety_score = 1.0;

        assert_eq!(result.fitness, 0.85);
        assert_eq!(result.execution_time, 1_000_000);
        assert_eq!(result.memory_usage, 1024 * 1024);
    }

    #[test]
    fn test_genome_empty_code() {
        let genome = AgentGenome::new(String::new(), vec![1.0, 2.0]);
        assert!(genome.code.is_empty());
        assert_eq!(genome.parameters.len(), 2);
    }

    #[test]
    fn test_genome_empty_parameters() {
        let genome = AgentGenome::new("code".to_string(), vec![]);
        assert_eq!(genome.code, "code");
        assert!(genome.parameters.is_empty());
    }

    #[test]
    fn test_similarity_empty_genomes() {
        let genome1 = AgentGenome::new(String::new(), vec![]);
        let genome2 = AgentGenome::new(String::new(), vec![]);

        assert_eq!(genome1.similarity(&genome2), 1.0);
    }

    #[test]
    fn test_multiple_mutations() {
        let original = AgentGenome::new("original".to_string(), vec![1.0, 2.0, 3.0]);
        let mutated =
            AgentGenome::from_parent(&original, "mutated".to_string(), vec![1.1, 2.2, 3.3]);

        let mutations = vec![
            MutationInfo {
                mutation_type: MutationType::ParameterTweak {
                    parameter_id: 0,
                    delta: 0.1,
                },
                location: CodeLocation {
                    line: 1,
                    column: 1,
                    length: 5,
                },
                impact_score: 0.1,
            },
            MutationInfo {
                mutation_type: MutationType::ParameterTweak {
                    parameter_id: 1,
                    delta: 0.2,
                },
                location: CodeLocation {
                    line: 2,
                    column: 1,
                    length: 5,
                },
                impact_score: 0.2,
            },
            MutationInfo {
                mutation_type: MutationType::ParameterTweak {
                    parameter_id: 2,
                    delta: 0.3,
                },
                location: CodeLocation {
                    line: 3,
                    column: 1,
                    length: 5,
                },
                impact_score: 0.3,
            },
        ];

        let mutated_agent = MutatedAgent {
            original,
            mutated,
            mutations,
            mutation_time: 4000,
        };

        assert_eq!(mutated_agent.mutations.len(), 3);
        assert_eq!(mutated_agent.mutations[0].impact_score, 0.1);
        assert_eq!(mutated_agent.mutations[1].impact_score, 0.2);
        assert_eq!(mutated_agent.mutations[2].impact_score, 0.3);
    }

    #[test]
    fn test_code_substitution_mutation() {
        let mutation = MutationType::CodeSubstitution {
            old_pattern: "for i in 0..10".to_string(),
            new_pattern: "for i in 0..20".to_string(),
        };

        if let MutationType::CodeSubstitution {
            old_pattern,
            new_pattern,
        } = mutation
        {
            assert_eq!(old_pattern, "for i in 0..10");
            assert_eq!(new_pattern, "for i in 0..20");
        } else {
            panic!("Wrong mutation type");
        }
    }

    #[test]
    fn test_structure_change_mutation() {
        let mutation = MutationType::StructureChange {
            change_type: "add_loop_unrolling".to_string(),
        };

        if let MutationType::StructureChange { change_type } = mutation {
            assert_eq!(change_type, "add_loop_unrolling");
        } else {
            panic!("Wrong mutation type");
        }
    }

    #[test]
    fn test_genome_id_uniqueness() {
        let genome1 = AgentGenome::new("test".to_string(), vec![]);
        let genome2 = AgentGenome::new("test".to_string(), vec![]);

        assert_ne!(genome1.id, genome2.id);
    }

    #[test]
    fn test_parent_child_relationship() {
        let parent = AgentGenome::new("parent".to_string(), vec![1.0]);
        let child1 = AgentGenome::from_parent(&parent, "child1".to_string(), vec![1.1]);
        let child2 = AgentGenome::from_parent(&parent, "child2".to_string(), vec![1.2]);

        assert_eq!(child1.metadata.parent_id, Some(parent.id));
        assert_eq!(child2.metadata.parent_id, Some(parent.id));
        assert_ne!(child1.id, child2.id);
        assert_eq!(child1.metadata.generation, child2.metadata.generation);
    }

    #[test]
    fn test_fitness_edge_cases() {
        let agent_id = Uuid::new_v4();

        // Test with negative fitness
        let result1 = EvaluationResult::new(agent_id, -1.0);
        assert_eq!(result1.fitness, -1.0);

        // Test with very large fitness
        let result2 = EvaluationResult::new(agent_id, f64::MAX);
        assert_eq!(result2.fitness, f64::MAX);

        // Test with NaN fitness
        let result3 = EvaluationResult::new(agent_id, f64::NAN);
        assert!(result3.fitness.is_nan());
    }

    #[test]
    fn test_benchmark_execution_time() {
        let mut benchmarks = vec![];

        for i in 0..5 {
            benchmarks.push(BenchmarkResult {
                name: format!("bench_{i}"),
                score: 0.8 + (i as f64 * 0.02),
                execution_time: 100 * (i + 1) as u64,
                passed: true,
            });
        }

        let total_time: u64 = benchmarks.iter().map(|b| b.execution_time).sum();
        assert_eq!(total_time, 100 + 200 + 300 + 400 + 500); // 1500
    }

    #[test]
    fn test_mutation_count_accumulation() {
        let mut genome = AgentGenome::new("root".to_string(), vec![0.0]);
        let initial_count = genome.metadata.mutation_count;

        for i in 1..=5 {
            genome = AgentGenome::from_parent(&genome, format!("gen_{i}"), vec![i as f32]);
            assert_eq!(genome.metadata.mutation_count, initial_count + i as u32);
        }
    }

    #[test]
    fn test_similarity_partial_code_match() {
        let genome1 = AgentGenome::new("abcdefghij".to_string(), vec![1.0]);
        let genome2 = AgentGenome::new("abcde12345".to_string(), vec![1.0]);

        let similarity = genome1.similarity(&genome2);
        // 5 matching chars out of 10, params identical
        assert!(similarity > 0.5 && similarity < 1.0);
    }

    #[test]
    fn test_genome_size_large() {
        let large_code = "x".repeat(10000);
        let large_params = vec![1.0; 1000];
        let genome = AgentGenome::new(large_code, large_params);

        let size = genome.size();
        assert!(size >= 10000 + 4000 + 64); // code + params + metadata
    }

    #[test]
    fn test_evaluation_metrics_all_perfect() {
        let metrics = EvaluationMetrics {
            performance_score: 1.0,
            correctness_score: 1.0,
            efficiency_score: 1.0,
            safety_score: 1.0,
            benchmark_results: vec![],
        };

        assert_eq!(metrics.performance_score, 1.0);
        assert_eq!(metrics.correctness_score, 1.0);
        assert_eq!(metrics.efficiency_score, 1.0);
        assert_eq!(metrics.safety_score, 1.0);
    }

    #[test]
    fn test_evaluation_metrics_all_zero() {
        let metrics = EvaluationMetrics {
            performance_score: 0.0,
            correctness_score: 0.0,
            efficiency_score: 0.0,
            safety_score: 0.0,
            benchmark_results: vec![],
        };

        assert_eq!(metrics.performance_score, 0.0);
        assert_eq!(metrics.correctness_score, 0.0);
        assert_eq!(metrics.efficiency_score, 0.0);
        assert_eq!(metrics.safety_score, 0.0);
    }
}
