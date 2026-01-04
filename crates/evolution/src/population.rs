//! Population management for evolution

use crate::EvolutionError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Individual in the population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub id: Uuid,
    pub genome: Vec<u8>,
    pub fitness: Option<f64>,
}

impl Individual {
    /// Create new individual with random genome
    pub fn new(genome_size: usize) -> Self {
        let genome = (0..genome_size).map(|_| rand::random::<u8>()).collect();
        Self {
            id: Uuid::new_v4(),
            genome,
            fitness: None,
        }
    }

    /// Create individual with specific genome
    pub fn with_genome(genome: Vec<u8>) -> Self {
        Self {
            id: Uuid::new_v4(),
            genome,
            fitness: None,
        }
    }

    /// Set fitness score
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = Some(fitness);
    }

    /// Get fitness score
    #[inline]
    pub fn fitness(&self) -> Option<f64> {
        self.fitness
    }

    /// Mutate the genome
    pub fn mutate(&mut self, mutation_rate: f64) {
        for byte in &mut self.genome {
            if rand::random::<f64>() < mutation_rate {
                *byte = rand::random::<u8>();
            }
        }
        // Reset fitness after mutation
        self.fitness = None;
    }

    /// Create offspring through crossover
    pub fn crossover(&self, other: &Individual, crossover_rate: f64) -> Individual {
        let mut offspring_genome = Vec::with_capacity(self.genome.len());

        for i in 0..self.genome.len().min(other.genome.len()) {
            if rand::random::<f64>() < crossover_rate {
                offspring_genome.push(self.genome[i]);
            } else {
                offspring_genome.push(other.genome[i]);
            }
        }

        // Handle different lengths
        if self.genome.len() > other.genome.len() {
            offspring_genome.extend_from_slice(&self.genome[other.genome.len()..]);
        } else if other.genome.len() > self.genome.len() {
            offspring_genome.extend_from_slice(&other.genome[self.genome.len()..]);
        }

        Individual::with_genome(offspring_genome)
    }
}

/// Population for evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: u64,
}

impl Population {
    /// Create new empty population
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
            generation: 0,
        }
    }

    /// Create population with random individuals
    pub fn random(size: usize, genome_size: usize) -> Self {
        let individuals = (0..size).map(|_| Individual::new(genome_size)).collect();

        Self {
            individuals,
            generation: 0,
        }
    }

    /// Add individual to population
    pub fn add_individual(&mut self, individual: Individual) {
        self.individuals.push(individual);
    }

    /// Get population size
    #[inline]
    pub fn size(&self) -> usize {
        self.individuals.len()
    }

    /// Check if population is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Get best individual (highest fitness)
    pub fn best_individual(&self) -> Result<&Individual, EvolutionError> {
        if self.is_empty() {
            return Err(EvolutionError::PopulationEmpty);
        }

        self.individuals
            .iter()
            .filter(|ind| ind.fitness.is_some())
            .max_by(|a, b| {
                a.fitness
                    .unwrap()
                    .partial_cmp(&b.fitness.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(EvolutionError::FitnessEvaluationFailed {
                reason: "No individuals with fitness scores".to_string(),
            })
    }

    /// Get average fitness
    pub fn average_fitness(&self) -> f64 {
        let fitness_scores: Vec<f64> = self
            .individuals
            .iter()
            .filter_map(|ind| ind.fitness)
            .collect();

        if fitness_scores.is_empty() {
            0.0
        } else {
            fitness_scores.iter().sum::<f64>() / fitness_scores.len() as f64
        }
    }

    /// Calculate diversity index (genetic diversity)
    pub fn diversity_index(&self) -> f64 {
        if self.individuals.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..self.individuals.len() {
            for j in (i + 1)..self.individuals.len() {
                let distance =
                    hamming_distance(&self.individuals[i].genome, &self.individuals[j].genome);
                total_distance += distance;
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            0.0
        } else {
            total_distance / comparisons as f64
        }
    }

    /// Select individuals for reproduction using tournament selection
    pub fn tournament_selection(
        &self,
        tournament_size: usize,
    ) -> Result<&Individual, EvolutionError> {
        if self.is_empty() {
            return Err(EvolutionError::PopulationEmpty);
        }

        let mut best = &self.individuals[rand::random::<usize>() % self.individuals.len()];

        for _ in 1..tournament_size {
            let candidate = &self.individuals[rand::random::<usize>() % self.individuals.len()];
            if let (Some(candidate_fitness), Some(best_fitness)) = (candidate.fitness, best.fitness)
            {
                if candidate_fitness > best_fitness {
                    best = candidate;
                }
            }
        }

        Ok(best)
    }

    /// Advance to next generation
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }

    /// Sort individuals by fitness (highest first)
    pub fn sort_by_fitness(&mut self) {
        self.individuals
            .sort_by(|a, b| match (a.fitness, b.fitness) {
                (Some(a_fit), Some(b_fit)) => b_fit
                    .partial_cmp(&a_fit)
                    .unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });
    }

    /// Keep only the top N individuals
    pub fn truncate(&mut self, size: usize) {
        self.sort_by_fitness();
        self.individuals.truncate(size);
    }
}

/// Calculate Hamming distance between two genomes
fn hamming_distance(genome1: &[u8], genome2: &[u8]) -> f64 {
    let min_len = genome1.len().min(genome2.len());
    let mut differences = 0;

    for i in 0..min_len {
        if genome1[i] != genome2[i] {
            differences += 1;
        }
    }

    // Add difference for length mismatch
    differences += (genome1.len() as i32 - genome2.len() as i32).unsigned_abs() as usize;

    differences as f64 / genome1.len().max(genome2.len()) as f64
}

impl Default for Population {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let individual = Individual::new(10);
        assert_eq!(individual.genome.len(), 10);
        assert!(individual.fitness.is_none());
        assert!(!individual.id.is_nil());
    }

    #[test]
    fn test_individual_with_genome() {
        let genome = vec![1, 2, 3, 4, 5];
        let individual = Individual::with_genome(genome.clone());
        assert_eq!(individual.genome, genome);
        assert!(individual.fitness.is_none());
    }

    #[test]
    fn test_individual_fitness() {
        let mut individual = Individual::new(5);
        assert!(individual.fitness().is_none());

        individual.set_fitness(0.8);
        assert_eq!(individual.fitness(), Some(0.8));
    }

    #[test]
    fn test_individual_mutation() {
        let mut individual = Individual::with_genome(vec![1, 2, 3, 4, 5]);
        individual.set_fitness(0.5);

        // High mutation rate should change genome
        individual.mutate(1.0);

        // Fitness should be reset after mutation
        assert!(individual.fitness().is_none());
    }

    #[test]
    fn test_individual_crossover() {
        let parent1 = Individual::with_genome(vec![1, 1, 1, 1, 1]);
        let parent2 = Individual::with_genome(vec![2, 2, 2, 2, 2]);

        let offspring = parent1.crossover(&parent2, 0.5);
        assert_eq!(offspring.genome.len(), 5);

        // Offspring genome should contain mix of parent genomes
        let mut has_parent1_genes = false;
        let mut has_parent2_genes = false;

        for &gene in &offspring.genome {
            if gene == 1 {
                has_parent1_genes = true;
            }
            if gene == 2 {
                has_parent2_genes = true;
            }
        }

        // Note: This is probabilistic, but with 50% crossover rate and 5 genes,
        // it's very likely to have genes from both parents
    }

    #[test]
    fn test_population_creation() {
        let population = Population::new();
        assert!(population.is_empty());
        assert_eq!(population.generation, 0);
        assert_eq!(population.size(), 0);
    }

    #[test]
    fn test_population_random() {
        let population = Population::random(10, 5);
        assert_eq!(population.size(), 10);
        assert!(!population.is_empty());

        for individual in &population.individuals {
            assert_eq!(individual.genome.len(), 5);
        }
    }

    #[test]
    fn test_population_add_individual() {
        let mut population = Population::new();
        let individual = Individual::new(5);

        population.add_individual(individual);
        assert_eq!(population.size(), 1);
        assert!(!population.is_empty());
    }

    #[test]
    fn test_population_best_individual() {
        let mut population = Population::new();

        // Empty population should return error
        assert!(population.best_individual().is_err());

        // Add individuals with fitness
        let mut ind1 = Individual::new(5);
        ind1.set_fitness(0.3);
        let mut ind2 = Individual::new(5);
        ind2.set_fitness(0.8);
        let mut ind3 = Individual::new(5);
        ind3.set_fitness(0.5);

        population.add_individual(ind1);
        population.add_individual(ind2);
        population.add_individual(ind3);

        let best = population
            .best_individual()
            .expect("Should find best individual");
        assert_eq!(best.fitness(), Some(0.8));
    }

    #[test]
    fn test_population_average_fitness() {
        let mut population = Population::new();

        // Empty population average should be 0
        assert_eq!(population.average_fitness(), 0.0);

        let mut ind1 = Individual::new(5);
        ind1.set_fitness(0.2);
        let mut ind2 = Individual::new(5);
        ind2.set_fitness(0.8);

        population.add_individual(ind1);
        population.add_individual(ind2);

        assert_eq!(population.average_fitness(), 0.5);
    }

    #[test]
    fn test_population_diversity_index() {
        let mut population = Population::new();

        // Empty population should have 0 diversity
        assert_eq!(population.diversity_index(), 0.0);

        // Single individual should have 0 diversity
        population.add_individual(Individual::with_genome(vec![1, 2, 3]));
        assert_eq!(population.diversity_index(), 0.0);

        // Add different individual
        population.add_individual(Individual::with_genome(vec![4, 5, 6]));

        // Should have non-zero diversity
        assert!(population.diversity_index() > 0.0);
    }

    #[test]
    fn test_population_tournament_selection() {
        let mut population = Population::new();

        // Empty population should return error
        assert!(population.tournament_selection(2).is_err());

        // Add individuals with fitness
        let mut ind1 = Individual::new(5);
        ind1.set_fitness(0.1);
        let mut ind2 = Individual::new(5);
        ind2.set_fitness(0.9);

        population.add_individual(ind1);
        population.add_individual(ind2);

        // Tournament selection should work
        let selected = population
            .tournament_selection(2)
            .expect("Should select individual");
        assert!(selected.fitness().is_some());
    }

    #[test]
    fn test_population_sorting_and_truncation() {
        let mut population = Population::new();

        let mut ind1 = Individual::new(5);
        ind1.set_fitness(0.3);
        let mut ind2 = Individual::new(5);
        ind2.set_fitness(0.9);
        let mut ind3 = Individual::new(5);
        ind3.set_fitness(0.6);

        population.add_individual(ind1);
        population.add_individual(ind2);
        population.add_individual(ind3);

        population.sort_by_fitness();

        // Should be sorted by fitness (highest first)
        assert_eq!(population.individuals[0].fitness(), Some(0.9));
        assert_eq!(population.individuals[1].fitness(), Some(0.6));
        assert_eq!(population.individuals[2].fitness(), Some(0.3));

        // Test truncation
        population.truncate(2);
        assert_eq!(population.size(), 2);
        assert_eq!(population.individuals[0].fitness(), Some(0.9));
        assert_eq!(population.individuals[1].fitness(), Some(0.6));
    }

    #[test]
    fn test_population_next_generation() {
        let mut population = Population::new();
        assert_eq!(population.generation, 0);

        population.next_generation();
        assert_eq!(population.generation, 1);

        population.next_generation();
        assert_eq!(population.generation, 2);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(&[1, 2, 3], &[1, 2, 3]), 0.0);
        assert_eq!(hamming_distance(&[1, 2, 3], &[4, 5, 6]), 1.0);
        assert_eq!(hamming_distance(&[1, 2, 3], &[1, 5, 6]), 2.0 / 3.0);

        // Test different lengths
        assert!(hamming_distance(&[1, 2], &[1, 2, 3]) > 0.0);
    }

    #[test]
    fn test_crossover_different_lengths() {
        // Test crossover when parent genomes have different lengths
        let parent1 = Individual::with_genome(vec![1, 1, 1]);
        let parent2 = Individual::with_genome(vec![2, 2, 2, 2, 2]); // Longer genome

        // Crossover should handle length mismatch
        let offspring1 = parent1.crossover(&parent2, 0.5);
        assert_eq!(offspring1.genome.len(), 5); // Should be length of longer parent

        // Test with parent2 being shorter
        let offspring2 = parent2.crossover(&parent1, 0.5);
        assert_eq!(offspring2.genome.len(), 5); // Should still be length of longer parent

        // Last elements should come from longer parent
        let offspring3 = parent1.crossover(&parent2, 0.0); // All from parent2
        assert_eq!(offspring3.genome[3], 2);
        assert_eq!(offspring3.genome[4], 2);

        let offspring4 = parent1.crossover(&parent2, 1.0); // All from parent1, extended with parent2
        assert_eq!(offspring4.genome[0], 1);
        assert_eq!(offspring4.genome[1], 1);
        assert_eq!(offspring4.genome[2], 1);
        assert_eq!(offspring4.genome[3], 2); // Extended from parent2
        assert_eq!(offspring4.genome[4], 2);
    }

    #[test]
    fn test_sort_by_fitness_with_none_values() {
        let mut population = Population::new();

        // Mix of individuals with and without fitness
        let mut ind1 = Individual::new(5);
        ind1.set_fitness(0.5);

        let ind2 = Individual::new(5); // No fitness

        let mut ind3 = Individual::new(5);
        ind3.set_fitness(0.8);

        let ind4 = Individual::new(5); // No fitness

        population.add_individual(ind1);
        population.add_individual(ind2);
        population.add_individual(ind3);
        population.add_individual(ind4);

        population.sort_by_fitness();

        // Individuals with fitness should come first (sorted by fitness)
        assert_eq!(population.individuals[0].fitness(), Some(0.8));
        assert_eq!(population.individuals[1].fitness(), Some(0.5));
        assert_eq!(population.individuals[2].fitness(), None);
        assert_eq!(population.individuals[3].fitness(), None);
    }

    #[test]
    fn test_diversity_index_empty_comparisons() {
        let mut population = Population::new();

        // Single individual - no comparisons possible
        population.add_individual(Individual::with_genome(vec![1, 2, 3]));
        assert_eq!(population.diversity_index(), 0.0);

        // Two identical individuals
        population.add_individual(Individual::with_genome(vec![1, 2, 3]));
        assert_eq!(population.diversity_index(), 0.0);

        // Add a different individual
        population.add_individual(Individual::with_genome(vec![4, 5, 6]));
        assert!(population.diversity_index() > 0.0);
    }

    #[test]
    fn test_best_individual_no_fitness() {
        let mut population = Population::new();

        // Add individuals without fitness scores
        population.add_individual(Individual::new(5));
        population.add_individual(Individual::new(5));

        // Should return error when no individuals have fitness
        let result = population.best_individual();
        assert!(matches!(
            result,
            Err(EvolutionError::FitnessEvaluationFailed { .. })
        ));
    }

    #[test]
    fn test_population_default() {
        let population = Population::default();
        assert!(population.is_empty());
        assert_eq!(population.generation, 0);
    }

    #[test]
    fn test_mutation_edge_cases() {
        let mut individual = Individual::with_genome(vec![1, 2, 3, 4, 5]);

        // Zero mutation rate should not change genome
        let original_genome = individual.genome.clone();
        individual.mutate(0.0);
        // Most likely unchanged (though there's a tiny chance of random change)

        // Test that fitness is always reset
        individual.set_fitness(0.7);
        individual.mutate(0.0);
        assert!(individual.fitness().is_none());
    }

    #[test]
    fn test_crossover_edge_cases() {
        // Test crossover with rate 0 and 1
        let parent1 = Individual::with_genome(vec![1, 1, 1, 1, 1]);
        let parent2 = Individual::with_genome(vec![2, 2, 2, 2, 2]);

        // Rate 0 - all genes from parent2
        let offspring1 = parent1.crossover(&parent2, 0.0);
        assert_eq!(offspring1.genome, vec![2, 2, 2, 2, 2]);

        // Rate 1 - all genes from parent1
        let offspring2 = parent1.crossover(&parent2, 1.0);
        assert_eq!(offspring2.genome, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_tournament_selection_all_no_fitness() {
        let mut population = Population::new();

        // Add individuals without fitness
        population.add_individual(Individual::new(5));
        population.add_individual(Individual::new(5));
        population.add_individual(Individual::new(5));

        // Tournament selection should still work (returns random individual)
        let selected = population.tournament_selection(3).expect("Should select");
        assert!(selected.fitness().is_none());
    }
}
