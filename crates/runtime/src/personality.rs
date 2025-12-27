//! Agent Personality System
//!
//! This module implements personality traits that affect agent behavior and decision-making.
//! Personalities can evolve based on success/failure outcomes.

use serde::{Deserialize, Serialize};

/// Agent personality traits that influence behavior
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentPersonality {
    /// Risk tolerance (0.0-1.0): Affects resource allocation decisions
    pub risk_tolerance: f32,
    /// Cooperation (0.0-1.0): Influences work sharing behavior
    pub cooperation: f32,
    /// Exploration (0.0-1.0): Drives trying new strategies
    pub exploration: f32,
    /// Efficiency focus (0.0-1.0): Optimizes for speed vs resources
    pub efficiency_focus: f32,
    /// Stability preference (0.0-1.0): Affects update frequency
    pub stability_preference: f32,
    /// Generation number for evolution tracking
    pub generation: u64,
    /// Current fitness score
    pub fitness: f32,
    /// Mutation rate for evolution
    pub mutation_rate: f32,
}

impl Default for AgentPersonality {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            cooperation: 0.7,
            exploration: 0.3,
            efficiency_focus: 0.6,
            stability_preference: 0.8,
            generation: 0,
            fitness: 0.0,
            mutation_rate: 0.1,
        }
    }
}

impl AgentPersonality {
    /// Create a new personality with specified traits
    pub fn new(
        risk_tolerance: f32,
        cooperation: f32,
        exploration: f32,
        efficiency_focus: f32,
        stability_preference: f32,
    ) -> Result<Self, PersonalityError> {
        // Validate trait ranges
        if !Self::is_valid_trait(risk_tolerance)
            || !Self::is_valid_trait(cooperation)
            || !Self::is_valid_trait(exploration)
            || !Self::is_valid_trait(efficiency_focus)
            || !Self::is_valid_trait(stability_preference)
        {
            return Err(PersonalityError::InvalidTraitValue);
        }

        Ok(Self {
            risk_tolerance,
            cooperation,
            exploration,
            efficiency_focus,
            stability_preference,
            generation: 0,
            fitness: 0.0,
            mutation_rate: 0.1,
        })
    }

    /// Validate trait value is in valid range [0.0, 1.0]
    fn is_valid_trait(value: f32) -> bool {
        (0.0..=1.0).contains(&value) && !value.is_nan()
    }

    /// Create a personality from a predefined type
    pub fn from_type(personality_type: PersonalityType) -> Self {
        match personality_type {
            PersonalityType::Conservative => Self {
                risk_tolerance: 0.2,
                cooperation: 0.8,
                exploration: 0.1,
                efficiency_focus: 0.7,
                stability_preference: 0.9,
                ..Default::default()
            },
            PersonalityType::Aggressive => Self {
                risk_tolerance: 0.9,
                cooperation: 0.3,
                exploration: 0.8,
                efficiency_focus: 0.9,
                stability_preference: 0.2,
                ..Default::default()
            },
            PersonalityType::Balanced => Self {
                risk_tolerance: 0.5,
                cooperation: 0.6,
                exploration: 0.4,
                efficiency_focus: 0.6,
                stability_preference: 0.5,
                ..Default::default()
            },
            PersonalityType::Explorer => Self {
                risk_tolerance: 0.7,
                cooperation: 0.5,
                exploration: 0.9,
                efficiency_focus: 0.4,
                stability_preference: 0.3,
                ..Default::default()
            },
            PersonalityType::Cooperative => Self {
                risk_tolerance: 0.4,
                cooperation: 0.9,
                exploration: 0.5,
                efficiency_focus: 0.5,
                stability_preference: 0.7,
                ..Default::default()
            },
        }
    }

    /// Update fitness based on outcome
    pub fn update_fitness(&mut self, outcome: &Outcome) {
        let old_fitness = self.fitness;

        // Calculate new fitness based on outcome
        self.fitness = match outcome {
            Outcome::Success(score) => (old_fitness + score) / 2.0,
            Outcome::Failure(penalty) => (old_fitness - penalty).max(0.0),
            Outcome::Neutral => old_fitness * 0.99, // Slight decay for neutral outcomes
        };

        // Clamp fitness to valid range
        self.fitness = self.fitness.clamp(0.0, 1.0);
    }

    /// Mutate personality traits for evolution
    pub fn mutate(&mut self) -> Result<(), PersonalityError> {
        let mutation_strength = self.mutation_rate;

        // Apply random mutations within bounds
        self.risk_tolerance = Self::mutate_trait(self.risk_tolerance, mutation_strength);
        self.cooperation = Self::mutate_trait(self.cooperation, mutation_strength);
        self.exploration = Self::mutate_trait(self.exploration, mutation_strength);
        self.efficiency_focus = Self::mutate_trait(self.efficiency_focus, mutation_strength);
        self.stability_preference =
            Self::mutate_trait(self.stability_preference, mutation_strength);

        self.generation += 1;

        Ok(())
    }

    /// Mutate a single trait value
    fn mutate_trait(current: f32, mutation_rate: f32) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let change = rng.gen_range(-mutation_rate..=mutation_rate);
        (current + change).clamp(0.0, 1.0)
    }

    /// Cross-breed with another personality to create offspring
    pub fn crossover(
        &self,
        other: &AgentPersonality,
    ) -> Result<AgentPersonality, PersonalityError> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Randomly select traits from parents
        let offspring = AgentPersonality {
            risk_tolerance: if rng.gen_bool(0.5) {
                self.risk_tolerance
            } else {
                other.risk_tolerance
            },
            cooperation: if rng.gen_bool(0.5) {
                self.cooperation
            } else {
                other.cooperation
            },
            exploration: if rng.gen_bool(0.5) {
                self.exploration
            } else {
                other.exploration
            },
            efficiency_focus: if rng.gen_bool(0.5) {
                self.efficiency_focus
            } else {
                other.efficiency_focus
            },
            stability_preference: if rng.gen_bool(0.5) {
                self.stability_preference
            } else {
                other.stability_preference
            },
            generation: self.generation.max(other.generation) + 1,
            fitness: 0.0, // Reset fitness for new offspring
            mutation_rate: (self.mutation_rate + other.mutation_rate) / 2.0,
        };

        Ok(offspring)
    }
}

/// Predefined personality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalityType {
    Conservative,
    Aggressive,
    Balanced,
    Explorer,
    Cooperative,
}

/// Outcome of an action for fitness evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Success(f32), // Positive score
    Failure(f32), // Penalty amount
    Neutral,      // No change
}

/// Decision that can be influenced by personality
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub resource_allocation: ResourceAllocation,
    pub communication_pattern: CommunicationPattern,
    pub algorithm_choice: AlgorithmChoice,
    pub optimization_target: OptimizationTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceAllocation {
    pub memory_request: usize,
    pub cpu_cores: usize,
    pub gpu_compute_units: usize,
    pub priority: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationPattern {
    Broadcast,
    Targeted,
    Minimal,
    Cooperative,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmChoice {
    Conservative,
    Experimental,
    Hybrid,
    Adaptive,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationTarget {
    Speed,
    Resources,
    Reliability,
    Balanced,
}

/// Trait for personality influence on decisions
pub trait PersonalityInfluence {
    /// Influence a decision based on personality traits
    fn influence_decision(&self, decision: &mut Decision);

    /// Update personality from outcome
    fn update_from_outcome(&mut self, outcome: &Outcome);

    /// Calculate trait fitness
    fn calculate_trait_fitness(&self) -> f32;
}

impl PersonalityInfluence for AgentPersonality {
    fn influence_decision(&self, decision: &mut Decision) {
        // Influence resource allocation based on risk tolerance
        if self.risk_tolerance > 0.7 {
            decision.resource_allocation.memory_request =
                (decision.resource_allocation.memory_request as f32 * 1.5) as usize;
            decision.resource_allocation.gpu_compute_units = decision
                .resource_allocation
                .gpu_compute_units
                .saturating_add(1);
        } else if self.risk_tolerance < 0.3 {
            decision.resource_allocation.memory_request =
                (decision.resource_allocation.memory_request as f32 * 0.7) as usize;
        }

        // Influence communication pattern based on cooperation
        decision.communication_pattern = if self.cooperation > 0.8 {
            CommunicationPattern::Cooperative
        } else if self.cooperation > 0.5 {
            CommunicationPattern::Targeted
        } else {
            CommunicationPattern::Minimal
        };

        // Influence algorithm choice based on exploration
        decision.algorithm_choice = if self.exploration > 0.7 {
            AlgorithmChoice::Experimental
        } else if self.exploration > 0.4 {
            AlgorithmChoice::Adaptive
        } else {
            AlgorithmChoice::Conservative
        };

        // Influence optimization target based on efficiency focus
        decision.optimization_target = if self.efficiency_focus > 0.8 {
            OptimizationTarget::Speed
        } else if self.efficiency_focus > 0.6 {
            OptimizationTarget::Balanced
        } else {
            OptimizationTarget::Reliability
        };
    }

    fn update_from_outcome(&mut self, outcome: &Outcome) {
        self.update_fitness(outcome);
    }

    fn calculate_trait_fitness(&self) -> f32 {
        // Simple fitness calculation based on trait balance
        let trait_sum = self.risk_tolerance
            + self.cooperation
            + self.exploration
            + self.efficiency_focus
            + self.stability_preference;
        let trait_balance = 1.0 - (trait_sum - 2.5).abs() / 2.5;

        // If fitness has been updated, combine with trait balance, otherwise use trait balance only
        if self.fitness > 0.0 {
            (self.fitness + trait_balance) / 2.0
        } else {
            trait_balance
        }
    }
}

/// Personality system errors
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PersonalityError {
    #[error("Invalid trait value: must be between 0.0 and 1.0")]
    InvalidTraitValue,
    #[error("Crossover failed: incompatible personalities")]
    CrossoverFailed,
    #[error("Mutation failed: {reason}")]
    MutationFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_personality() {
        let personality = AgentPersonality::default();

        assert_eq!(personality.risk_tolerance, 0.5);
        assert_eq!(personality.cooperation, 0.7);
        assert_eq!(personality.exploration, 0.3);
        assert_eq!(personality.efficiency_focus, 0.6);
        assert_eq!(personality.stability_preference, 0.8);
        assert_eq!(personality.generation, 0);
        assert_eq!(personality.fitness, 0.0);
        assert_eq!(personality.mutation_rate, 0.1);
    }

    #[test]
    fn test_personality_creation() {
        let personality = AgentPersonality::new(0.8, 0.6, 0.4, 0.7, 0.5).unwrap();

        assert_eq!(personality.risk_tolerance, 0.8);
        assert_eq!(personality.cooperation, 0.6);
        assert_eq!(personality.exploration, 0.4);
        assert_eq!(personality.efficiency_focus, 0.7);
        assert_eq!(personality.stability_preference, 0.5);
    }

    #[test]
    fn test_invalid_trait_values() {
        // Test out of range values
        assert!(AgentPersonality::new(-0.1, 0.5, 0.5, 0.5, 0.5).is_err());
        assert!(AgentPersonality::new(1.1, 0.5, 0.5, 0.5, 0.5).is_err());
        assert!(AgentPersonality::new(0.5, f32::NAN, 0.5, 0.5, 0.5).is_err());
    }

    #[test]
    fn test_personality_types() {
        let conservative = AgentPersonality::from_type(PersonalityType::Conservative);
        assert_eq!(conservative.risk_tolerance, 0.2);
        assert_eq!(conservative.cooperation, 0.8);

        let aggressive = AgentPersonality::from_type(PersonalityType::Aggressive);
        assert_eq!(aggressive.risk_tolerance, 0.9);
        assert_eq!(aggressive.exploration, 0.8);

        let explorer = AgentPersonality::from_type(PersonalityType::Explorer);
        assert_eq!(explorer.exploration, 0.9);
        assert_eq!(explorer.stability_preference, 0.3);
    }

    #[test]
    fn test_fitness_update() {
        let mut personality = AgentPersonality::default();

        // Test success outcome
        personality.update_fitness(&Outcome::Success(0.8));
        assert!(personality.fitness > 0.0);

        let old_fitness = personality.fitness;

        // Test failure outcome
        personality.update_fitness(&Outcome::Failure(0.3));
        assert!(personality.fitness < old_fitness);

        // Test neutral outcome
        let fitness_before_neutral = personality.fitness;
        personality.update_fitness(&Outcome::Neutral);
        assert!(personality.fitness < fitness_before_neutral);
    }

    #[test]
    fn test_mutation() {
        let mut personality = AgentPersonality::default();
        let original = personality.clone();

        personality.mutate().unwrap();

        // Check that generation increased
        assert_eq!(personality.generation, original.generation + 1);

        // Check that at least some traits might have changed
        // (Note: due to randomness, this test might occasionally fail)
        // We'll just verify the values are still in valid range
        assert!(personality.risk_tolerance >= 0.0 && personality.risk_tolerance <= 1.0);
        assert!(personality.cooperation >= 0.0 && personality.cooperation <= 1.0);
        assert!(personality.exploration >= 0.0 && personality.exploration <= 1.0);
        assert!(personality.efficiency_focus >= 0.0 && personality.efficiency_focus <= 1.0);
        assert!(personality.stability_preference >= 0.0 && personality.stability_preference <= 1.0);
    }

    #[test]
    fn test_crossover() {
        let parent1 = AgentPersonality::from_type(PersonalityType::Conservative);
        let parent2 = AgentPersonality::from_type(PersonalityType::Aggressive);

        let offspring = parent1.crossover(&parent2).unwrap();

        // Check that offspring has valid traits
        assert!(offspring.risk_tolerance >= 0.0 && offspring.risk_tolerance <= 1.0);
        assert!(offspring.cooperation >= 0.0 && offspring.cooperation <= 1.0);
        assert!(offspring.exploration >= 0.0 && offspring.exploration <= 1.0);

        // Check that generation increased
        assert_eq!(
            offspring.generation,
            parent1.generation.max(parent2.generation) + 1
        );

        // Check that fitness is reset
        assert_eq!(offspring.fitness, 0.0);
    }

    #[test]
    fn test_decision_influence() {
        let personality = AgentPersonality::from_type(PersonalityType::Aggressive);

        let mut decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024,
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Broadcast,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        personality.influence_decision(&mut decision);

        // Aggressive personality should request more resources
        assert!(decision.resource_allocation.memory_request > 1024);
        assert_eq!(decision.resource_allocation.gpu_compute_units, 2);

        // Should prefer experimental algorithms
        assert_eq!(decision.algorithm_choice, AlgorithmChoice::Experimental);

        // Should prefer speed optimization
        assert_eq!(decision.optimization_target, OptimizationTarget::Speed);
    }

    #[test]
    fn test_personality_serialization() {
        let personality = AgentPersonality::from_type(PersonalityType::Explorer);

        let json = serde_json::to_string(&personality).unwrap();
        let deserialized: AgentPersonality = serde_json::from_str(&json).unwrap();

        assert_eq!(personality, deserialized);
    }

    #[test]
    fn test_trait_fitness_calculation() {
        let personality = AgentPersonality::new(0.5, 0.5, 0.5, 0.5, 0.5).unwrap();
        let fitness = personality.calculate_trait_fitness();

        // Balanced personality should have good trait fitness
        assert!(fitness > 0.5);

        let unbalanced = AgentPersonality::new(1.0, 0.0, 1.0, 0.0, 1.0).unwrap();
        let unbalanced_fitness = unbalanced.calculate_trait_fitness();

        // Unbalanced personality should have lower trait fitness
        assert!(unbalanced_fitness < fitness);
    }

    #[test]
    fn test_cooperative_communication() {
        let cooperative = AgentPersonality::from_type(PersonalityType::Cooperative);

        let mut decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024,
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Minimal,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        cooperative.influence_decision(&mut decision);

        // Cooperative personality should prefer cooperative communication
        assert_eq!(
            decision.communication_pattern,
            CommunicationPattern::Cooperative
        );
    }

    #[test]
    fn test_explorer_algorithm_choice() {
        let explorer = AgentPersonality::from_type(PersonalityType::Explorer);

        let mut decision = Decision {
            resource_allocation: ResourceAllocation {
                memory_request: 1024,
                cpu_cores: 1,
                gpu_compute_units: 1,
                priority: 5,
            },
            communication_pattern: CommunicationPattern::Broadcast,
            algorithm_choice: AlgorithmChoice::Conservative,
            optimization_target: OptimizationTarget::Balanced,
        };

        explorer.influence_decision(&mut decision);

        // Explorer personality should prefer experimental algorithms
        assert_eq!(decision.algorithm_choice, AlgorithmChoice::Experimental);
    }

    #[test]
    fn test_personality_evolution_cycle() {
        let mut personality = AgentPersonality::default();

        // Simulate evolution cycle
        for _ in 0..5 {
            // Update fitness based on random outcomes
            personality.update_fitness(&Outcome::Success(0.7));
            personality.mutate().unwrap();
        }

        assert_eq!(personality.generation, 5);
        assert!(personality.fitness > 0.0);
    }

    #[test]
    fn test_fitness_bounds() {
        let mut personality = AgentPersonality::default();

        // Test fitness doesn't go below 0
        personality.update_fitness(&Outcome::Failure(2.0));
        assert_eq!(personality.fitness, 0.0);

        // Test fitness doesn't go above 1
        for _ in 0..10 {
            personality.update_fitness(&Outcome::Success(1.0));
        }
        assert!(personality.fitness <= 1.0);
    }

    #[test]
    fn test_multiple_crossovers() {
        let parent1 = AgentPersonality::from_type(PersonalityType::Conservative);
        let parent2 = AgentPersonality::from_type(PersonalityType::Aggressive);

        // Create multiple offspring
        let mut offspring = Vec::new();
        for _ in 0..10 {
            offspring.push(parent1.crossover(&parent2).unwrap());
        }

        // Check that all offspring are valid
        for child in offspring {
            assert!(child.risk_tolerance >= 0.0 && child.risk_tolerance <= 1.0);
            assert!(child.cooperation >= 0.0 && child.cooperation <= 1.0);
            assert_eq!(child.generation, 1);
            assert_eq!(child.fitness, 0.0);
        }
    }
}
