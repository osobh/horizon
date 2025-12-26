//! Mutation engines for agent evolution

use crate::{AgentGenome, MutationType};
use async_trait::async_trait;
use std::collections::HashMap;
use thiserror::Error;

/// Mutation engine errors
#[derive(Error, Debug)]
pub enum MutationError {
    #[error("Invalid mutation type: {0}")]
    InvalidMutationType(String),

    #[error("Code parsing failed: {0}")]
    CodeParsingFailed(String),

    #[error("Parameter out of range: {0}")]
    ParameterOutOfRange(String),

    #[error("Mutation failed: {0}")]
    MutationFailed(String),
}

/// Trait for mutation engines
#[async_trait]
pub trait MutationEngine: Send + Sync {
    /// Apply mutation to an agent genome
    async fn mutate(
        &self,
        genome: &AgentGenome,
        mutation_type: &MutationType,
    ) -> Result<AgentGenome, MutationError>;

    /// Select appropriate mutation type for genome
    fn select_mutation_type(&self) -> MutationType;

    /// Clone the mutation engine
    fn clone_engine(&self) -> Box<dyn MutationEngine>;

    /// Get engine name
    fn name(&self) -> &str;
}

/// Basic mutation engine implementation
#[derive(Debug, Clone)]
pub struct BasicMutationEngine {
    name: String,
    parameter_mutation_rate: f32,
    code_mutation_rate: f32,
    max_parameter_delta: f32,
    code_patterns: HashMap<String, Vec<String>>,
}

impl BasicMutationEngine {
    /// Create a new basic mutation engine
    pub fn new() -> Self {
        let mut code_patterns = HashMap::new();

        // Simple code substitution patterns
        code_patterns.insert(
            "return".to_string(),
            vec!["return".to_string(), "yield".to_string()],
        );

        code_patterns.insert(
            "fn".to_string(),
            vec!["fn".to_string(), "function".to_string()],
        );

        Self {
            name: "BasicMutationEngine".to_string(),
            parameter_mutation_rate: 0.1,
            code_mutation_rate: 0.05,
            max_parameter_delta: 0.5,
            code_patterns,
        }
    }

    /// Configure parameter mutation rate
    pub fn with_parameter_mutation_rate(mut self, rate: f32) -> Self {
        self.parameter_mutation_rate = rate;
        self
    }

    /// Configure code mutation rate
    pub fn with_code_mutation_rate(mut self, rate: f32) -> Self {
        self.code_mutation_rate = rate;
        self
    }

    /// Apply parameter mutation
    fn mutate_parameters(&self, parameters: &[f32]) -> Vec<f32> {
        parameters
            .iter()
            .map(|&param| {
                if fastrand::f32() < self.parameter_mutation_rate {
                    let delta = (fastrand::f32() - 0.5) * 2.0 * self.max_parameter_delta;
                    param + delta
                } else {
                    param
                }
            })
            .collect()
    }

    /// Apply code mutation
    fn mutate_code(&self, code: &str) -> String {
        if fastrand::f32() > self.code_mutation_rate {
            return code.to_string();
        }

        let mut mutated_code = code.to_string();

        // Simple pattern replacement
        for (pattern, replacements) in &self.code_patterns {
            if mutated_code.contains(pattern) && !replacements.is_empty() {
                let replacement_idx = fastrand::usize(..replacements.len());
                let replacement = &replacements[replacement_idx];
                mutated_code = mutated_code.replace(pattern, replacement);
                break;
            }
        }

        mutated_code
    }
}

impl Default for BasicMutationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MutationEngine for BasicMutationEngine {
    async fn mutate(
        &self,
        genome: &AgentGenome,
        mutation_type: &MutationType,
    ) -> Result<AgentGenome, MutationError> {
        match mutation_type {
            MutationType::ParameterTweak {
                parameter_id,
                delta,
            } => {
                if *parameter_id >= genome.parameters.len() {
                    return Err(MutationError::ParameterOutOfRange(format!(
                        "Parameter {} not found (genome has {} parameters)",
                        parameter_id,
                        genome.parameters.len()
                    )));
                }

                let mut new_parameters = genome.parameters.clone();
                new_parameters[*parameter_id] += delta;

                Ok(AgentGenome::from_parent(
                    genome,
                    genome.code.clone(),
                    new_parameters,
                ))
            }

            MutationType::CodeSubstitution {
                old_pattern,
                new_pattern,
            } => {
                let new_code = genome.code.replace(old_pattern, new_pattern);
                Ok(AgentGenome::from_parent(
                    genome,
                    new_code,
                    genome.parameters.clone(),
                ))
            }

            MutationType::StructureChange { change_type: _ } => {
                // For now, just apply random code and parameter mutations
                let new_code = self.mutate_code(&genome.code);
                let new_parameters = self.mutate_parameters(&genome.parameters);
                Ok(AgentGenome::from_parent(genome, new_code, new_parameters))
            }

            MutationType::NoiseInjection { intensity } => {
                let new_parameters = genome
                    .parameters
                    .iter()
                    .map(|&param| {
                        let noise = (fastrand::f32() - 0.5) * 2.0 * intensity;
                        param + noise
                    })
                    .collect();

                Ok(AgentGenome::from_parent(
                    genome,
                    genome.code.clone(),
                    new_parameters,
                ))
            }
        }
    }

    fn select_mutation_type(&self) -> MutationType {
        let mutation_types = [
            MutationType::ParameterTweak {
                parameter_id: fastrand::usize(..10),
                delta: (fastrand::f32() - 0.5) * self.max_parameter_delta,
            },
            MutationType::CodeSubstitution {
                old_pattern: "return".to_string(),
                new_pattern: "yield".to_string(),
            },
            MutationType::StructureChange {
                change_type: "random".to_string(),
            },
            MutationType::NoiseInjection {
                intensity: fastrand::f32() * 0.1,
            },
        ];

        let idx = fastrand::usize(..mutation_types.len());
        mutation_types[idx].clone()
    }

    fn clone_engine(&self) -> Box<dyn MutationEngine> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Advanced mutation engine with more sophisticated mutations
#[derive(Debug, Clone)]
pub struct AdvancedMutationEngine {
    name: String,
    basic_engine: BasicMutationEngine,
    crossover_rate: f32,
}

impl AdvancedMutationEngine {
    /// Create a new advanced mutation engine
    pub fn new() -> Self {
        Self {
            name: "AdvancedMutationEngine".to_string(),
            basic_engine: BasicMutationEngine::new(),
            crossover_rate: 0.1,
        }
    }

    /// Configure crossover rate
    pub fn with_crossover_rate(mut self, rate: f32) -> Self {
        self.crossover_rate = rate;
        self
    }
}

#[async_trait]
impl MutationEngine for AdvancedMutationEngine {
    async fn mutate(
        &self,
        genome: &AgentGenome,
        mutation_type: &MutationType,
    ) -> Result<AgentGenome, MutationError> {
        // For now, delegate to basic engine
        // In a real implementation, this would include more advanced mutations
        self.basic_engine.mutate(genome, mutation_type).await
    }

    fn select_mutation_type(&self) -> MutationType {
        // More sophisticated mutation type selection
        if fastrand::f32() < 0.3 {
            MutationType::ParameterTweak {
                parameter_id: fastrand::usize(..20),
                delta: fastrand::f32() * 0.2 - 0.1,
            }
        } else if fastrand::f32() < 0.6 {
            MutationType::NoiseInjection {
                intensity: fastrand::f32() * 0.05,
            }
        } else if fastrand::f32() < 0.9 {
            MutationType::StructureChange {
                change_type: "advanced".to_string(),
            }
        } else {
            MutationType::CodeSubstitution {
                old_pattern: "fn".to_string(),
                new_pattern: "function".to_string(),
            }
        }
    }

    fn clone_engine(&self) -> Box<dyn MutationEngine> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_mutation_engine() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![1.0, 2.0, 3.0]);

        // Test parameter tweak
        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: 0.5,
        };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.parameters[0], 1.5);
        assert_eq!(mutated.metadata.generation, genome.metadata.generation + 1);
    }

    #[tokio::test]
    async fn test_parameter_out_of_range() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0]);

        let mutation = MutationType::ParameterTweak {
            parameter_id: 5,
            delta: 0.1,
        };
        let result = engine.mutate(&genome, &mutation).await;
        assert!(matches!(result, Err(MutationError::ParameterOutOfRange(_))));
    }

    #[tokio::test]
    async fn test_code_substitution() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![]);

        let mutation = MutationType::CodeSubstitution {
            old_pattern: "return".to_string(),
            new_pattern: "yield".to_string(),
        };

        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert!(mutated.code.contains("yield"));
        assert!(!mutated.code.contains("return"));
    }

    #[tokio::test]
    async fn test_noise_injection() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0, 2.0]);

        let mutation = MutationType::NoiseInjection { intensity: 0.1 };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();

        // Parameters should be different but close
        assert_ne!(mutated.parameters[0], genome.parameters[0]);
        assert!((mutated.parameters[0] - genome.parameters[0]).abs() <= 0.1);
    }

    #[tokio::test]
    async fn test_mutation_type_selection() {
        let engine = BasicMutationEngine::new();

        // Test that selection produces valid mutation types
        for _ in 0..10 {
            let mutation_type = engine.select_mutation_type();
            match mutation_type {
                MutationType::ParameterTweak {
                    parameter_id,
                    delta: _,
                } => {
                    assert!(parameter_id < 10);
                }
                MutationType::CodeSubstitution {
                    old_pattern,
                    new_pattern,
                } => {
                    assert!(!old_pattern.is_empty());
                    assert!(!new_pattern.is_empty());
                }
                MutationType::StructureChange { change_type } => {
                    assert!(!change_type.is_empty());
                }
                MutationType::NoiseInjection { intensity } => {
                    assert!(intensity >= 0.0 && intensity <= 0.1);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_advanced_mutation_engine() {
        let engine = AdvancedMutationEngine::new();
        let genome = AgentGenome::new(
            "test return value".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        );

        // Try multiple mutations since some might fail due to random selection
        let mut successful_mutation = None;
        for _ in 0..20 {
            let mutation = engine.select_mutation_type();
            if let Ok(mutated) = engine.mutate(&genome, &mutation).await {
                successful_mutation = Some(mutated);
                break;
            }
        }

        let mutated = successful_mutation.expect("At least one mutation should succeed");
        assert_eq!(mutated.metadata.generation, genome.metadata.generation + 1);
        assert_eq!(mutated.metadata.parent_id, Some(genome.id));
    }

    #[test]
    fn test_engine_configuration() {
        let engine = BasicMutationEngine::new()
            .with_parameter_mutation_rate(0.2)
            .with_code_mutation_rate(0.1);

        assert_eq!(engine.parameter_mutation_rate, 0.2);
        assert_eq!(engine.code_mutation_rate, 0.1);
    }

    #[test]
    fn test_engine_clone() {
        let engine = BasicMutationEngine::new();
        let cloned = engine.clone_engine();

        assert_eq!(cloned.name(), engine.name());
    }

    #[tokio::test]
    async fn test_structure_change_mutation() {
        let engine = BasicMutationEngine::new()
            .with_parameter_mutation_rate(1.0) // Always mutate
            .with_code_mutation_rate(1.0); // Always mutate

        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![1.0, 2.0]);

        let mutation = MutationType::StructureChange {
            change_type: "random".to_string(),
        };

        let mutated = engine.mutate(&genome, &mutation).await.unwrap();

        // With 100% mutation rates, something should change
        let params_changed = mutated.parameters != genome.parameters;
        let code_changed = mutated.code != genome.code;

        assert!(
            params_changed || code_changed,
            "Structure change should modify genome"
        );
    }

    #[test]
    fn test_basic_engine_default() {
        let engine1 = BasicMutationEngine::new();
        let engine2 = BasicMutationEngine::default();

        assert_eq!(engine1.name(), engine2.name());
        assert_eq!(
            engine1.parameter_mutation_rate,
            engine2.parameter_mutation_rate
        );
        assert_eq!(engine1.code_mutation_rate, engine2.code_mutation_rate);
        assert_eq!(engine1.max_parameter_delta, engine2.max_parameter_delta);
    }

    #[test]
    fn test_advanced_engine_configuration() {
        let engine = AdvancedMutationEngine::new().with_crossover_rate(0.5);

        assert_eq!(engine.crossover_rate, 0.5);
        assert_eq!(engine.name(), "AdvancedMutationEngine");
    }

    #[tokio::test]
    async fn test_advanced_engine_mutation_delegation() {
        let engine = AdvancedMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0]);

        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: 0.1,
        };

        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.parameters[0], 1.1);
        assert_eq!(mutated.metadata.parent_id, Some(genome.id));
    }

    #[test]
    fn test_advanced_engine_mutation_type_distribution() {
        let engine = AdvancedMutationEngine::new();
        let mut mutation_counts = HashMap::new();

        // Generate many mutation types to check distribution
        for _ in 0..1000 {
            let mutation = engine.select_mutation_type();
            let key = match mutation {
                MutationType::ParameterTweak { .. } => "ParameterTweak",
                MutationType::NoiseInjection { .. } => "NoiseInjection",
                MutationType::StructureChange { .. } => "StructureChange",
                MutationType::CodeSubstitution { .. } => "CodeSubstitution",
            };
            *mutation_counts.entry(key).or_insert(0) += 1;
        }

        // Should have all types represented
        assert!(mutation_counts.contains_key("ParameterTweak"));
        assert!(mutation_counts.contains_key("NoiseInjection"));
        assert!(mutation_counts.contains_key("StructureChange"));
        assert!(mutation_counts.contains_key("CodeSubstitution"));
    }

    #[test]
    fn test_mutation_error_display() {
        let errors = vec![
            MutationError::InvalidMutationType("test".to_string()),
            MutationError::CodeParsingFailed("syntax error".to_string()),
            MutationError::ParameterOutOfRange("index 5".to_string()),
            MutationError::MutationFailed("unknown error".to_string()),
        ];

        for error in errors {
            let message = error.to_string();
            assert!(!message.is_empty());
            assert!(
                message.contains("test")
                    || message.contains("syntax")
                    || message.contains("index")
                    || message.contains("unknown")
            );
        }
    }

    #[tokio::test]
    async fn test_zero_intensity_noise_injection() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![1.0, 2.0, 3.0]);

        let mutation = MutationType::NoiseInjection { intensity: 0.0 };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();

        // With zero intensity, parameters might still change slightly due to random noise
        // but should be very close to original
        for (orig, new) in genome.parameters.iter().zip(mutated.parameters.iter()) {
            assert!(
                (orig - new).abs() <= 0.01,
                "Zero intensity should produce minimal change"
            );
        }
    }

    #[tokio::test]
    async fn test_maximum_intensity_noise_injection() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![0.0]);

        let mutation = MutationType::NoiseInjection { intensity: 1.0 };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();

        // With high intensity, parameter should change significantly
        assert!(
            (mutated.parameters[0]).abs() <= 1.0,
            "High intensity noise should be bounded"
        );
    }

    #[tokio::test]
    async fn test_parameter_tweak_edge_cases() {
        let engine = BasicMutationEngine::new();

        // Test empty parameters
        let empty_genome = AgentGenome::new("test".to_string(), vec![]);
        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: 0.1,
        };
        let result = engine.mutate(&empty_genome, &mutation).await;
        assert!(result.is_err());

        // Test very large parameter index
        let genome = AgentGenome::new("test".to_string(), vec![1.0]);
        let mutation = MutationType::ParameterTweak {
            parameter_id: 1000,
            delta: 0.1,
        };
        let result = engine.mutate(&genome, &mutation).await;
        assert!(result.is_err());

        // Test negative delta
        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: -0.5,
        };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.parameters[0], 0.5);
    }

    #[tokio::test]
    async fn test_code_substitution_edge_cases() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("simple code".to_string(), vec![]);

        // Test substitution with pattern not found
        let mutation = MutationType::CodeSubstitution {
            old_pattern: "nonexistent".to_string(),
            new_pattern: "replacement".to_string(),
        };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.code, genome.code); // Should remain unchanged

        // Test empty pattern replacement
        let mutation = MutationType::CodeSubstitution {
            old_pattern: "code".to_string(),
            new_pattern: "".to_string(),
        };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.code, "simple ");

        // Test replacing entire code
        let mutation = MutationType::CodeSubstitution {
            old_pattern: "simple code".to_string(),
            new_pattern: "complex algorithm".to_string(),
        };
        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.code, "complex algorithm");
    }

    #[test]
    fn test_basic_engine_code_patterns() {
        let engine = BasicMutationEngine::new();

        assert!(engine.code_patterns.contains_key("return"));
        assert!(engine.code_patterns.contains_key("fn"));

        let return_patterns = engine.code_patterns.get("return").unwrap();
        assert!(return_patterns.contains(&"return".to_string()));
        assert!(return_patterns.contains(&"yield".to_string()));
    }

    #[test]
    fn test_parameter_mutations_with_rates() {
        let engine = BasicMutationEngine::new().with_parameter_mutation_rate(0.0); // Never mutate

        let original_params = vec![1.0, 2.0, 3.0];
        let mutated_params = engine.mutate_parameters(&original_params);
        assert_eq!(mutated_params, original_params);

        let engine_high_rate = BasicMutationEngine::new().with_parameter_mutation_rate(1.0); // Always mutate

        let mutated_params_high = engine_high_rate.mutate_parameters(&original_params);
        // Should be different from original (though randomness could theoretically make them same)
        let different = mutated_params_high
            .iter()
            .zip(original_params.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(different, "High mutation rate should change parameters");
    }

    #[test]
    fn test_code_mutations_with_rates() {
        let engine = BasicMutationEngine::new().with_code_mutation_rate(0.0); // Never mutate

        let original_code = "fn test() { return 42; }";
        let mutated_code = engine.mutate_code(original_code);
        assert_eq!(mutated_code, original_code);
    }

    #[tokio::test]
    async fn test_mutation_preserves_genome_metadata() {
        let engine = BasicMutationEngine::new();
        let original_genome = AgentGenome::new("test".to_string(), vec![1.0]);

        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: 0.1,
        };

        let mutated = engine.mutate(&original_genome, &mutation).await.unwrap();

        assert_eq!(
            mutated.metadata.generation,
            original_genome.metadata.generation + 1
        );
        assert_eq!(mutated.metadata.parent_id, Some(original_genome.id));
        assert_eq!(
            mutated.metadata.mutation_count,
            original_genome.metadata.mutation_count + 1
        );
        assert_eq!(
            mutated.metadata.lineage_depth,
            original_genome.metadata.lineage_depth + 1
        );
        assert!(mutated.metadata.creation_time >= original_genome.metadata.creation_time);
    }

    #[tokio::test]
    async fn test_multiple_mutations_on_same_genome() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("fn test() { return 42; }".to_string(), vec![1.0, 2.0]);

        let mutations = vec![
            MutationType::ParameterTweak {
                parameter_id: 0,
                delta: 0.1,
            },
            MutationType::ParameterTweak {
                parameter_id: 1,
                delta: -0.2,
            },
            MutationType::CodeSubstitution {
                old_pattern: "return".to_string(),
                new_pattern: "yield".to_string(),
            },
            MutationType::NoiseInjection { intensity: 0.05 },
        ];

        let original_id = genome.id;
        let mut current_genome = genome;
        for mutation in mutations {
            current_genome = engine.mutate(&current_genome, &mutation).await.unwrap();
        }

        // Final genome should be quite different from original
        assert_ne!(current_genome.id, original_id);
        assert_eq!(current_genome.metadata.generation, 4); // 4 mutations deep
        assert!(current_genome.code.contains("yield"));
    }

    #[test]
    fn test_engine_names() {
        let basic = BasicMutationEngine::new();
        let advanced = AdvancedMutationEngine::new();

        assert_eq!(basic.name(), "BasicMutationEngine");
        assert_eq!(advanced.name(), "AdvancedMutationEngine");
    }

    #[test]
    fn test_advanced_engine_clone() {
        let engine = AdvancedMutationEngine::new().with_crossover_rate(0.8);
        let cloned = engine.clone_engine();

        assert_eq!(cloned.name(), engine.name());
    }

    #[tokio::test]
    async fn test_concurrent_mutations() {
        use std::sync::Arc;
        use tokio::task;

        let engine = Arc::new(BasicMutationEngine::new());
        let genome = Arc::new(AgentGenome::new("test".to_string(), vec![1.0, 2.0, 3.0]));

        let mut handles = vec![];

        for i in 0..10 {
            let engine_clone = engine.clone();
            let genome_clone = genome.clone();

            let handle = task::spawn(async move {
                let mutation = MutationType::ParameterTweak {
                    parameter_id: i % 3, // Cycle through parameter indices
                    delta: 0.1,
                };
                engine_clone.mutate(&genome_clone, &mutation).await
            });
            handles.push(handle);
        }

        // All mutations should succeed
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            let mutated = result.unwrap();
            assert_ne!(mutated.id, genome.id);
        }
    }

    #[test]
    fn test_mutation_type_cloning() {
        let mutation = MutationType::ParameterTweak {
            parameter_id: 5,
            delta: 0.25,
        };

        let cloned = mutation.clone();

        match (mutation, cloned) {
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
            _ => panic!("Clone should preserve mutation type"),
        }
    }

    #[tokio::test]
    async fn test_large_parameter_arrays() {
        let engine = BasicMutationEngine::new();
        let large_params: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let genome = AgentGenome::new("test".to_string(), large_params);

        let mutation = MutationType::ParameterTweak {
            parameter_id: 500,
            delta: 10.0,
        };

        let mutated = engine.mutate(&genome, &mutation).await.unwrap();
        assert_eq!(mutated.parameters[500], 510.0);
        assert_eq!(mutated.parameters.len(), 1000);
    }

    #[test]
    fn test_max_parameter_delta_configuration() {
        let engine = BasicMutationEngine::new()
            .with_parameter_mutation_rate(1.0) // Always mutate
            .with_code_mutation_rate(0.0); // Never mutate code

        // Test that parameter mutations respect max_parameter_delta
        let original_max_delta = engine.max_parameter_delta;
        assert_eq!(original_max_delta, 0.5);

        // Test selection respects delta bounds
        for _ in 0..100 {
            let mutation = engine.select_mutation_type();
            if let MutationType::ParameterTweak { delta, .. } = mutation {
                assert!(delta.abs() <= original_max_delta);
            }
        }
    }

    #[tokio::test]
    async fn test_mutation_error_propagation() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new("test".to_string(), vec![]);

        // This should fail and return a proper error
        let mutation = MutationType::ParameterTweak {
            parameter_id: 0,
            delta: 0.1,
        };

        match engine.mutate(&genome, &mutation).await {
            Err(MutationError::ParameterOutOfRange(msg)) => {
                assert!(msg.contains("Parameter 0 not found"));
                assert!(msg.contains("genome has 0 parameters"));
            }
            _ => panic!("Expected ParameterOutOfRange error"),
        }
    }

    #[test]
    fn test_code_pattern_mutation_deterministic() {
        // Test code mutation with specific patterns
        let mut code_patterns = HashMap::new();
        code_patterns.insert("test".to_string(), vec!["TEST".to_string()]);

        let engine = BasicMutationEngine {
            name: "TestEngine".to_string(),
            parameter_mutation_rate: 0.0,
            code_mutation_rate: 1.0, // Always mutate
            max_parameter_delta: 0.1,
            code_patterns,
        };

        // Should replace "test" with "TEST"
        let code = "this is a test function";
        let mutated = engine.mutate_code(code);
        assert!(
            mutated.contains("TEST"),
            "Expected 'test' to be replaced with 'TEST'"
        );
    }

    #[tokio::test]
    async fn test_all_mutation_types_with_complex_genome() {
        let engine = BasicMutationEngine::new();
        let genome = AgentGenome::new(
            "fn complex_function() { return calculate_value(); }".to_string(),
            vec![1.0, -2.5, 3.14159, 0.0, 100.0],
        );

        let mutations = vec![
            MutationType::ParameterTweak {
                parameter_id: 0,
                delta: 0.1,
            },
            MutationType::ParameterTweak {
                parameter_id: 4,
                delta: -10.0,
            },
            MutationType::CodeSubstitution {
                old_pattern: "return".to_string(),
                new_pattern: "yield".to_string(),
            },
            MutationType::CodeSubstitution {
                old_pattern: "fn".to_string(),
                new_pattern: "function".to_string(),
            },
            MutationType::StructureChange {
                change_type: "comprehensive".to_string(),
            },
            MutationType::NoiseInjection { intensity: 0.2 },
        ];

        for (i, mutation) in mutations.iter().enumerate() {
            let result = engine.mutate(&genome, mutation).await;
            assert!(
                result.is_ok(),
                "Mutation {} should succeed: {:?}",
                i,
                mutation
            );

            let mutated = result.unwrap();
            assert_ne!(mutated.id, genome.id);
            assert_eq!(mutated.metadata.generation, 1);
            assert_eq!(mutated.metadata.parent_id, Some(genome.id));
        }
    }
}
