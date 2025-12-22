//! Pattern discovery and application for DGM engine

use crate::{
    error::EvolutionEngineResult,
    population::Population,
    traits::{AgentGenome, EvolvableAgent},
};
use std::collections::HashMap;

/// Pattern discovery system
pub struct PatternDiscovery {
    /// Similarity threshold for pattern matching
    similarity_threshold: f64,
    /// Maximum patterns to track
    max_patterns: usize,
    /// Pattern consolidation interval
    consolidation_interval: u32,
}

impl PatternDiscovery {
    /// Create new pattern discovery system
    pub fn new(
        similarity_threshold: f64,
        max_patterns: usize,
        consolidation_interval: u32,
    ) -> Self {
        Self {
            similarity_threshold,
            max_patterns,
            consolidation_interval,
        }
    }

    /// Discover patterns from population improvements
    pub fn discover_patterns(
        &self,
        population: &Population<EvolvableAgent>,
        previous_fitnesses: &HashMap<String, f64>,
    ) -> Vec<(AgentGenome, AgentGenome, f64)> {
        let mut patterns = Vec::new();

        for individual in &population.individuals {
            let current_fitness = individual.fitness.unwrap_or(0.0);
            let genome_id = format!("{:?}", individual.entity.genome);

            if let Some(prev_fitness) = previous_fitnesses.get(&genome_id) {
                let fitness_delta = current_fitness - prev_fitness;

                // Only track significant improvements
                if fitness_delta > 0.05 {
                    // Store pattern as (before_genome, after_genome, improvement)
                    patterns.push((
                        individual.entity.genome.clone(),
                        individual.entity.genome.clone(),
                        fitness_delta,
                    ));
                }
            }
        }

        patterns
    }

    /// Check if two genomes are similar
    pub fn genomes_similar(&self, genome1: &AgentGenome, genome2: &AgentGenome) -> bool {
        // Compare architecture similarity
        let arch_similarity =
            self.architecture_similarity(&genome1.architecture, &genome2.architecture);

        // Compare behavior similarity
        let behavior_similarity = self.behavior_similarity(&genome1.behavior, &genome2.behavior);

        // Combined similarity
        let combined_similarity = (arch_similarity + behavior_similarity) / 2.0;
        combined_similarity >= self.similarity_threshold
    }

    /// Calculate architecture similarity
    fn architecture_similarity(
        &self,
        arch1: &crate::traits::ArchitectureGenes,
        arch2: &crate::traits::ArchitectureGenes,
    ) -> f64 {
        let mut similarity_score = 0.0_f64;
        let mut total_comparisons = 0.0_f64;

        // Compare memory capacity
        let memory_diff = (arch1.memory_capacity as f64 - arch2.memory_capacity as f64).abs();
        let max_memory = (arch1.memory_capacity.max(arch2.memory_capacity)) as f64;
        if max_memory > 0.0 {
            similarity_score += 1.0 - (memory_diff / max_memory);
        }
        total_comparisons += 1.0;

        // Compare processing units
        let units_diff = (arch1.processing_units as f64 - arch2.processing_units as f64).abs();
        let max_units = (arch1.processing_units.max(arch2.processing_units)) as f64;
        if max_units > 0.0 {
            similarity_score += 1.0 - (units_diff / max_units);
        }
        total_comparisons += 1.0;

        // Compare network topology
        if arch1.network_topology.len() == arch2.network_topology.len() {
            similarity_score += 0.5;
            total_comparisons += 0.5;

            // Compare individual topology values
            let min_len = arch1
                .network_topology
                .len()
                .min(arch2.network_topology.len());
            for i in 0..min_len {
                let topo_diff =
                    (arch1.network_topology[i] as f64 - arch2.network_topology[i] as f64).abs();
                let max_topo = (arch1.network_topology[i].max(arch2.network_topology[i])) as f64;
                if max_topo > 0.0 {
                    similarity_score += 0.5 * (1.0 - (topo_diff / max_topo));
                }
                total_comparisons += 0.5;
            }
        } else {
            total_comparisons += 1.0;
        }

        similarity_score / total_comparisons.max(1.0_f64)
    }

    /// Calculate behavior similarity
    fn behavior_similarity(
        &self,
        behavior1: &crate::traits::BehaviorGenes,
        behavior2: &crate::traits::BehaviorGenes,
    ) -> f64 {
        let mut similarity_score = 0.0_f64;
        let mut total_comparisons = 0.0_f64;

        // Compare exploration rate
        let exploration_diff = (behavior1.exploration_rate - behavior2.exploration_rate).abs();
        similarity_score += 1.0 - exploration_diff;
        total_comparisons += 1.0;

        // Compare learning rate
        let learning_diff = (behavior1.learning_rate - behavior2.learning_rate).abs();
        similarity_score += 1.0 - learning_diff;
        total_comparisons += 1.0;

        // Compare risk tolerance
        let risk_diff = (behavior1.risk_tolerance - behavior2.risk_tolerance).abs();
        similarity_score += 1.0 - risk_diff;
        total_comparisons += 1.0;

        similarity_score / total_comparisons
    }

    /// Apply a pattern to a genome
    pub fn apply_pattern(
        &self,
        source_genome: &AgentGenome,
        pattern_source: &AgentGenome,
        pattern_target: &AgentGenome,
    ) -> EvolutionEngineResult<AgentGenome> {
        // Create a new genome based on the pattern transformation
        let mut new_genome = source_genome.clone();

        // Apply architectural changes
        if self.should_apply_architecture_change(pattern_source, pattern_target) {
            self.apply_architecture_pattern(
                &mut new_genome.architecture,
                &pattern_source.architecture,
                &pattern_target.architecture,
            );
        }

        // Apply behavioral changes
        if self.should_apply_behavior_change(pattern_source, pattern_target) {
            self.apply_behavior_pattern(
                &mut new_genome.behavior,
                &pattern_source.behavior,
                &pattern_target.behavior,
            );
        }

        Ok(new_genome)
    }

    /// Check if architecture change should be applied
    fn should_apply_architecture_change(&self, source: &AgentGenome, target: &AgentGenome) -> bool {
        // Apply if architecture changed significantly
        self.architecture_similarity(&source.architecture, &target.architecture) < 0.95
    }

    /// Check if behavior change should be applied
    fn should_apply_behavior_change(&self, source: &AgentGenome, target: &AgentGenome) -> bool {
        // Apply if behavior changed significantly
        self.behavior_similarity(&source.behavior, &target.behavior) < 0.95
    }

    /// Apply architecture pattern transformation
    fn apply_architecture_pattern(
        &self,
        target: &mut crate::traits::ArchitectureGenes,
        pattern_source: &crate::traits::ArchitectureGenes,
        pattern_target: &crate::traits::ArchitectureGenes,
    ) {
        // Apply memory capacity change
        let memory_ratio =
            pattern_target.memory_capacity as f64 / pattern_source.memory_capacity.max(1) as f64;
        target.memory_capacity = (target.memory_capacity as f64 * memory_ratio) as usize;

        // Apply processing units change
        let units_ratio =
            pattern_target.processing_units as f64 / pattern_source.processing_units.max(1) as f64;
        target.processing_units = (target.processing_units as f64 * units_ratio) as u32;

        // Apply network topology changes if compatible
        if pattern_source.network_topology.len() == pattern_target.network_topology.len()
            && target.network_topology.len() == pattern_source.network_topology.len()
        {
            for i in 0..target.network_topology.len() {
                let topo_ratio = pattern_target.network_topology[i] as f64
                    / pattern_source.network_topology[i].max(1) as f64;
                target.network_topology[i] =
                    (target.network_topology[i] as f64 * topo_ratio) as u32;
            }
        }
    }

    /// Apply behavior pattern transformation
    fn apply_behavior_pattern(
        &self,
        target: &mut crate::traits::BehaviorGenes,
        pattern_source: &crate::traits::BehaviorGenes,
        pattern_target: &crate::traits::BehaviorGenes,
    ) {
        // Apply behavior changes with momentum
        let momentum = 0.3; // How much of the pattern change to apply

        // Exploration rate
        let exploration_delta = pattern_target.exploration_rate - pattern_source.exploration_rate;
        target.exploration_rate =
            (target.exploration_rate + momentum * exploration_delta).clamp(0.0, 1.0);

        // Learning rate
        let learning_delta = pattern_target.learning_rate - pattern_source.learning_rate;
        target.learning_rate = (target.learning_rate + momentum * learning_delta).clamp(0.0, 1.0);

        // Risk tolerance
        let risk_delta = pattern_target.risk_tolerance - pattern_source.risk_tolerance;
        target.risk_tolerance = (target.risk_tolerance + momentum * risk_delta).clamp(0.0, 1.0);
    }

    /// Consolidate similar patterns
    pub fn consolidate_patterns(
        &self,
        patterns: Vec<(String, AgentGenome, AgentGenome, f64)>,
    ) -> Vec<(String, AgentGenome, AgentGenome, f64)> {
        if patterns.is_empty() {
            return patterns;
        }

        let mut consolidated = Vec::new();
        let mut processed = vec![false; patterns.len()];

        for i in 0..patterns.len() {
            if processed[i] {
                continue;
            }

            let mut group = vec![i];
            let (id, source, target, fitness) = &patterns[i];

            // Find similar patterns
            for j in (i + 1)..patterns.len() {
                if processed[j] {
                    continue;
                }

                let (_, other_source, other_target, _) = &patterns[j];

                if self.genomes_similar(source, other_source)
                    && self.genomes_similar(target, other_target)
                {
                    group.push(j);
                    processed[j] = true;
                }
            }

            // Average the fitness improvements for consolidated pattern
            let avg_fitness =
                group.iter().map(|&idx| patterns[idx].3).sum::<f64>() / group.len() as f64;

            consolidated.push((id.clone(), source.clone(), target.clone(), avg_fitness));

            processed[i] = true;
        }

        // Keep only the best patterns if we exceed the limit
        if consolidated.len() > self.max_patterns {
            consolidated.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
            consolidated.truncate(self.max_patterns);
        }

        consolidated
    }
}
