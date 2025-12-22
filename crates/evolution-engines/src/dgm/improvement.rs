//! Self-improvement logic for DGM engine

use crate::traits::AgentGenome;
use serde::{Deserialize, Serialize};

/// Growth pattern record
#[derive(Debug, Clone)]
pub struct GrowthPattern {
    /// Pattern identifier
    pub id: String,
    /// Source genome
    pub source: AgentGenome,
    /// Target genome
    pub target: AgentGenome,
    /// Fitness improvement
    pub fitness_delta: f64,
    /// Times successfully applied
    pub success_count: u32,
    /// Times failed
    pub failure_count: u32,
    /// Last generation used
    pub last_used: u32,
}

impl GrowthPattern {
    /// Create a new growth pattern
    pub fn new(
        id: String,
        source: AgentGenome,
        target: AgentGenome,
        fitness_delta: f64,
        generation: u32,
    ) -> Self {
        Self {
            id,
            source,
            target,
            fitness_delta,
            success_count: 1,
            failure_count: 0,
            last_used: generation,
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Check if pattern is still relevant
    pub fn is_relevant(&self, current_generation: u32, max_age: u32) -> bool {
        current_generation - self.last_used <= max_age
    }
}

/// Type of discovered pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Architectural improvement
    Architectural,
    /// Behavioral improvement
    Behavioral,
    /// Resource optimization
    ResourceOptimization,
    /// Learning enhancement
    LearningEnhancement,
    /// Mixed improvement
    Mixed,
}

/// Discovered pattern for tracking improvements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPattern {
    /// Pattern ID
    pub id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Fitness improvement
    pub fitness_delta: f64,
    /// Success rate
    pub success_rate: f64,
    /// Discovery generation
    pub discovered_at: u32,
    /// Times applied
    pub application_count: u32,
}

impl DiscoveredPattern {
    /// Create a new discovered pattern
    pub fn new(id: String, pattern_type: PatternType, fitness_delta: f64, generation: u32) -> Self {
        Self {
            id,
            pattern_type,
            fitness_delta,
            success_rate: 1.0,
            discovered_at: generation,
            application_count: 1,
        }
    }

    /// Update pattern statistics
    pub fn update(&mut self, success: bool, fitness_delta: f64) {
        self.application_count += 1;

        // Update success rate with exponential moving average
        let alpha = 0.1; // Learning rate for EMA
        let success_value = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * success_value + (1.0 - alpha) * self.success_rate;

        // Update fitness delta with EMA
        if success {
            self.fitness_delta = alpha * fitness_delta + (1.0 - alpha) * self.fitness_delta;
        }
    }

    /// Calculate pattern value based on success rate and fitness improvement
    pub fn value(&self) -> f64 {
        self.success_rate * self.fitness_delta
    }
}

/// Growth history for tracking improvement trends
#[derive(Debug, Clone, Default)]
pub struct GrowthHistory {
    /// Pattern application history
    pub pattern_applications: Vec<(String, f64, u32)>, // (pattern_id, fitness_delta, generation)
    /// Fitness progression
    pub fitness_history: Vec<(u32, f64)>, // (generation, fitness)
    /// Discovery events
    pub discoveries: Vec<(u32, String)>, // (generation, pattern_id)
}

impl GrowthHistory {
    /// Add pattern application
    pub fn add_application(&mut self, pattern_id: String, fitness_delta: f64, generation: u32) {
        self.pattern_applications
            .push((pattern_id, fitness_delta, generation));
    }

    /// Add fitness measurement
    pub fn add_fitness(&mut self, generation: u32, fitness: f64) {
        self.fitness_history.push((generation, fitness));
    }

    /// Add discovery event
    pub fn add_discovery(&mut self, generation: u32, pattern_id: String) {
        self.discoveries.push((generation, pattern_id));
    }

    /// Calculate improvement velocity over recent generations
    pub fn improvement_velocity(&self, window: usize) -> f64 {
        if self.fitness_history.len() < 2 {
            return 0.0;
        }

        let start_idx = self.fitness_history.len().saturating_sub(window);
        let recent_history = &self.fitness_history[start_idx..];

        if recent_history.len() < 2 {
            return 0.0;
        }

        let (start_gen, start_fitness) = recent_history.first()?;
        let (end_gen, end_fitness) = recent_history.last()?;

        if end_gen == start_gen {
            return 0.0;
        }

        (end_fitness - start_fitness) / (end_gen - start_gen) as f64
    }

    /// Get discovery rate over recent generations
    pub fn discovery_rate(&self, window: u32) -> f64 {
        let current_gen = self.fitness_history.last().map(|(g, _)| *g).unwrap_or(0);
        let start_gen = current_gen.saturating_sub(window);

        let recent_discoveries = self
            .discoveries
            .iter()
            .filter(|(gen, _)| *gen >= start_gen)
            .count();

        recent_discoveries as f64 / window.max(1) as f64
    }
}
