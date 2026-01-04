//! Agent DNA system for bootstrap agents
//!
//! Each agent carries DNA that defines its core capabilities, variable traits,
//! experience memory, and replication history.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent DNA containing all heritable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDNA {
    /// Unique DNA identifier
    pub id: Uuid,
    /// Essential capabilities that persist through evolution
    pub core_traits: CoreTraits,
    /// Traits subject to mutation and evolution
    pub variable_traits: VariableTraits,
    /// Learned patterns and experience
    pub experience_memory: ExperienceMemory,
    /// Lineage and replication history
    pub lineage: LineageHistory,
    /// Generation number (0 for template agents)
    pub generation: u32,
    /// Fitness score history
    pub fitness_history: Vec<f32>,
}

/// Core traits that define essential agent capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreTraits {
    /// Basic agent type
    pub agent_type: AgentType,
    /// Can this agent create other agents?
    pub can_replicate: bool,
    /// Can this agent synthesize kernels?
    pub can_synthesize: bool,
    /// Can this agent evaluate fitness?
    pub can_evaluate: bool,
    /// Can this agent evolve itself?
    pub can_evolve: bool,
    /// Basic memory capacity (MB)
    pub base_memory: usize,
    /// Basic processing capacity (abstract units)
    pub base_processing: u32,
}

/// Variable traits subject to mutation and evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableTraits {
    /// Exploration vs exploitation balance (0.0 = exploit, 1.0 = explore)
    pub exploration_rate: f32,
    /// Learning rate for adaptation (0.0 = no learning, 1.0 = max learning)
    pub learning_rate: f32,
    /// Risk tolerance (0.0 = risk averse, 1.0 = risk taking)
    pub risk_tolerance: f32,
    /// Cooperation tendency (0.0 = individualistic, 1.0 = cooperative)
    pub cooperation_rate: f32,
    /// Resource efficiency modifier (0.5 = inefficient, 2.0 = very efficient)
    pub efficiency_modifier: f32,
    /// Mutation resistance (0.0 = highly mutable, 1.0 = mutation resistant)
    pub mutation_resistance: f32,
    /// Specialization depth (0.0 = generalist, 1.0 = specialist)
    pub specialization: f32,
    /// Energy conservation tendency (0.0 = energy hungry, 1.0 = conservative)
    pub energy_conservation: f32,
}

/// Experience memory storing learned patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceMemory {
    /// Successful goal patterns
    pub successful_goals: HashMap<String, f32>,
    /// Failed goal patterns (to avoid)
    pub failed_goals: HashMap<String, f32>,
    /// Kernel synthesis patterns that worked
    pub synthesis_patterns: HashMap<String, f32>,
    /// Resource usage patterns
    pub resource_patterns: HashMap<String, f32>,
    /// Interaction patterns with other agents
    pub interaction_patterns: HashMap<Uuid, f32>,
}

/// Lineage and replication history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageHistory {
    /// Parent agent DNA ID (None for template agents)
    pub parent_id: Option<Uuid>,
    /// List of child agent DNA IDs
    pub children: Vec<Uuid>,
    /// Mutations applied to create this agent
    pub mutations_applied: Vec<MutationRecord>,
    /// Birth timestamp
    pub birth_time: u64,
    /// Death timestamp (None if still alive)
    pub death_time: Option<u64>,
    /// Survival duration in seconds
    pub survival_duration: Option<u64>,
}

/// Record of a mutation applied during replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    /// Type of mutation
    pub mutation_type: MutationType,
    /// Trait that was mutated
    pub trait_name: String,
    /// Original value
    pub original_value: f32,
    /// New value after mutation
    pub mutated_value: f32,
    /// Timestamp of mutation
    pub timestamp: u64,
}

/// Types of mutations that can be applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    /// Small random adjustment
    PointMutation,
    /// Large random change
    DramaticMutation,
    /// Copy value from another agent
    CrossoverMutation,
    /// Reset to default value
    ResetMutation,
}

/// Basic agent types for bootstrap
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    /// Most basic self-replicating agent
    Prime,
    /// Specialized in creating new agents
    Replicator,
    /// Specialized in evolution and fitness evaluation
    Evolution,
    /// Specialized variant that emerged naturally
    Specialized,
}

impl AgentDNA {
    /// Create template agent DNA
    pub fn create_template(agent_type: AgentType) -> Self {
        let core_traits = match agent_type {
            AgentType::Prime => CoreTraits {
                agent_type,
                can_replicate: true,
                can_synthesize: true,
                can_evaluate: false,
                can_evolve: true,
                base_memory: 32,
                base_processing: 100,
            },
            AgentType::Replicator => CoreTraits {
                agent_type,
                can_replicate: true,
                can_synthesize: true,
                can_evaluate: true,
                can_evolve: false,
                base_memory: 64,
                base_processing: 80,
            },
            AgentType::Evolution => CoreTraits {
                agent_type,
                can_replicate: false,
                can_synthesize: false,
                can_evaluate: true,
                can_evolve: true,
                base_memory: 48,
                base_processing: 120,
            },
            AgentType::Specialized => CoreTraits {
                agent_type,
                can_replicate: true,
                can_synthesize: true,
                can_evaluate: true,
                can_evolve: true,
                base_memory: 64,
                base_processing: 100,
            },
        };

        Self {
            id: Uuid::new_v4(),
            core_traits,
            variable_traits: VariableTraits::default_for_type(agent_type),
            experience_memory: ExperienceMemory::default(),
            lineage: LineageHistory {
                parent_id: None,
                children: Vec::new(),
                mutations_applied: Vec::new(),
                birth_time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                death_time: None,
                survival_duration: None,
            },
            generation: 0,
            fitness_history: Vec::new(),
        }
    }

    /// Create offspring DNA through mutation and crossover
    pub fn reproduce(
        &self,
        partner: Option<&AgentDNA>,
        mutation_rate: f32,
    ) -> anyhow::Result<AgentDNA> {
        let mut offspring = self.clone();
        offspring.id = Uuid::new_v4();
        offspring.generation = self.generation + 1;
        offspring.fitness_history.clear();
        offspring.lineage.parent_id = Some(self.id);
        offspring.lineage.children.clear();
        offspring.lineage.mutations_applied.clear();
        offspring.lineage.birth_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        // Apply mutations based on mutation rate
        if rand::random::<f32>() < mutation_rate {
            offspring.apply_mutation()?;
        }

        // Apply crossover if partner available
        if let Some(partner) = partner {
            if rand::random::<f32>() < 0.3 {
                // 30% chance of crossover
                offspring.apply_crossover(partner)?;
            }
        }

        Ok(offspring)
    }

    /// Apply a random mutation to variable traits
    fn apply_mutation(&mut self) -> anyhow::Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Select random trait to mutate
        let trait_choice = rng.gen_range(0..8);
        let mutation_type = if rng.gen::<f32>() < 0.8 {
            MutationType::PointMutation
        } else {
            MutationType::DramaticMutation
        };

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match trait_choice {
            0 => {
                let original = self.variable_traits.exploration_rate;
                self.variable_traits.exploration_rate = match mutation_type {
                    MutationType::PointMutation => {
                        (original + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
                    }
                    MutationType::DramaticMutation => rng.gen_range(0.0..1.0),
                    _ => original,
                };
                self.lineage.mutations_applied.push(MutationRecord {
                    mutation_type,
                    trait_name: "exploration_rate".to_string(),
                    original_value: original,
                    mutated_value: self.variable_traits.exploration_rate,
                    timestamp: current_time,
                });
            }
            1 => {
                let original = self.variable_traits.learning_rate;
                self.variable_traits.learning_rate = match mutation_type {
                    MutationType::PointMutation => {
                        (original + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
                    }
                    MutationType::DramaticMutation => rng.gen_range(0.0..1.0),
                    _ => original,
                };
                self.lineage.mutations_applied.push(MutationRecord {
                    mutation_type,
                    trait_name: "learning_rate".to_string(),
                    original_value: original,
                    mutated_value: self.variable_traits.learning_rate,
                    timestamp: current_time,
                });
            }
            2 => {
                let original = self.variable_traits.risk_tolerance;
                self.variable_traits.risk_tolerance = match mutation_type {
                    MutationType::PointMutation => {
                        (original + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
                    }
                    MutationType::DramaticMutation => rng.gen_range(0.0..1.0),
                    _ => original,
                };
                self.lineage.mutations_applied.push(MutationRecord {
                    mutation_type,
                    trait_name: "risk_tolerance".to_string(),
                    original_value: original,
                    mutated_value: self.variable_traits.risk_tolerance,
                    timestamp: current_time,
                });
            }
            // Continue for other traits...
            _ => {}
        }

        Ok(())
    }

    /// Apply crossover with partner DNA
    fn apply_crossover(&mut self, partner: &AgentDNA) -> anyhow::Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Randomly inherit traits from partner
        if rng.gen_bool(0.5) {
            self.variable_traits.cooperation_rate = partner.variable_traits.cooperation_rate;
        }
        if rng.gen_bool(0.5) {
            self.variable_traits.efficiency_modifier = partner.variable_traits.efficiency_modifier;
        }
        if rng.gen_bool(0.5) {
            self.variable_traits.specialization = partner.variable_traits.specialization;
        }

        // Record crossover mutations
        self.lineage.mutations_applied.push(MutationRecord {
            mutation_type: MutationType::CrossoverMutation,
            trait_name: "crossover_applied".to_string(),
            original_value: 0.0,
            mutated_value: 1.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });

        Ok(())
    }

    /// Calculate fitness score based on performance metrics
    pub fn calculate_fitness(&self, metrics: &PerformanceMetrics) -> f32 {
        let mut fitness = 0.0;

        // Base fitness from survival
        fitness += metrics.survival_time as f32 * 0.1;

        // Reward successful goals
        fitness += metrics.goals_completed as f32 * 10.0;

        // Reward successful replications
        fitness += metrics.offspring_created as f32 * 5.0;

        // Reward resource efficiency
        let efficiency = metrics.resources_used as f32 / metrics.resources_allocated as f32;
        fitness += (2.0 - efficiency) * 10.0; // Higher fitness for lower resource usage

        // Penalty for failures
        fitness -= metrics.goals_failed as f32 * 2.0;

        // Apply trait modifiers
        fitness *= self.variable_traits.efficiency_modifier;

        fitness.max(0.0)
    }

    /// Update experience based on outcomes
    pub fn update_experience(&mut self, goal: &str, success: bool, resource_usage: f32) {
        let score = if success { 1.0 } else { -0.5 };

        if success {
            *self
                .experience_memory
                .successful_goals
                .entry(goal.to_string())
                .or_insert(0.0) += score;
        } else {
            *self
                .experience_memory
                .failed_goals
                .entry(goal.to_string())
                .or_insert(0.0) += score.abs();
        }

        // Update resource usage patterns
        *self
            .experience_memory
            .resource_patterns
            .entry("average_usage".to_string())
            .or_insert(0.0) = (self
            .experience_memory
            .resource_patterns
            .get("average_usage")
            .unwrap_or(&0.0)
            + resource_usage)
            / 2.0;
    }
}

impl VariableTraits {
    /// Create default traits for agent type
    pub fn default_for_type(agent_type: AgentType) -> Self {
        match agent_type {
            AgentType::Prime => Self {
                exploration_rate: 0.7,
                learning_rate: 0.6,
                risk_tolerance: 0.5,
                cooperation_rate: 0.4,
                efficiency_modifier: 1.0,
                mutation_resistance: 0.3,
                specialization: 0.2,
                energy_conservation: 0.5,
            },
            AgentType::Replicator => Self {
                exploration_rate: 0.3,
                learning_rate: 0.8,
                risk_tolerance: 0.2,
                cooperation_rate: 0.8,
                efficiency_modifier: 1.2,
                mutation_resistance: 0.5,
                specialization: 0.7,
                energy_conservation: 0.7,
            },
            AgentType::Evolution => Self {
                exploration_rate: 0.9,
                learning_rate: 0.9,
                risk_tolerance: 0.8,
                cooperation_rate: 0.5,
                efficiency_modifier: 0.8,
                mutation_resistance: 0.1,
                specialization: 0.9,
                energy_conservation: 0.3,
            },
            AgentType::Specialized => Self {
                exploration_rate: 0.5,
                learning_rate: 0.7,
                risk_tolerance: 0.6,
                cooperation_rate: 0.6,
                efficiency_modifier: 1.1,
                mutation_resistance: 0.4,
                specialization: 0.5,
                energy_conservation: 0.5,
            },
        }
    }
}

impl Default for ExperienceMemory {
    fn default() -> Self {
        Self {
            successful_goals: HashMap::new(),
            failed_goals: HashMap::new(),
            synthesis_patterns: HashMap::new(),
            resource_patterns: HashMap::new(),
            interaction_patterns: HashMap::new(),
        }
    }
}

/// Performance metrics for fitness calculation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub survival_time: u64,
    pub goals_completed: u32,
    pub goals_failed: u32,
    pub offspring_created: u32,
    pub resources_used: u64,
    pub resources_allocated: u64,
    pub kernels_synthesized: u32,
    pub successful_interactions: u32,
}
