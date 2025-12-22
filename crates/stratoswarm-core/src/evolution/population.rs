//! Population management with Structure of Arrays (SoA) layout.
//!
//! This module provides efficient population storage optimized for GPU operations
//! using a Structure of Arrays layout for better memory access patterns.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Population identifier.
pub type PopulationId = Uuid;

/// An individual in the population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Individual {
    /// Unique identifier
    pub id: Uuid,
    /// Genome (parameter vector)
    pub genome: Vec<f64>,
    /// Fitness value (None if not yet evaluated)
    pub fitness: Option<f64>,
}

impl Individual {
    /// Create a new individual with a random ID.
    #[must_use]
    pub fn new(genome: Vec<f64>) -> Self {
        Self {
            id: Uuid::new_v4(),
            genome,
            fitness: None,
        }
    }

    /// Create an individual with a specific ID (for testing).
    #[must_use]
    pub fn with_id(id: Uuid, genome: Vec<f64>) -> Self {
        Self {
            id,
            genome,
            fitness: None,
        }
    }

    /// Set the fitness value.
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = Some(fitness);
    }
}

/// Population statistics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PopulationStats {
    /// Population size
    pub size: usize,
    /// Best fitness value
    pub best_fitness: f64,
    /// Worst fitness value
    pub worst_fitness: f64,
    /// Average fitness
    pub average_fitness: f64,
    /// Fitness standard deviation
    pub std_dev: f64,
    /// Genome diversity (average pairwise distance)
    pub diversity: f64,
}

/// Population using Structure of Arrays layout for GPU efficiency.
///
/// This layout stores genomes contiguously in memory, enabling efficient
/// GPU transfers and vectorized operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    /// Population ID
    pub id: PopulationId,
    /// Individual IDs (SoA)
    ids: Vec<Uuid>,
    /// Genomes stored contiguously (SoA): [genome0_param0, genome0_param1, ..., genome1_param0, ...]
    genomes: Vec<f64>,
    /// Fitness values (SoA)
    fitnesses: Vec<Option<f64>>,
    /// Number of parameters per genome
    genome_size: usize,
}

impl Population {
    /// Create a new population with the given individuals.
    ///
    /// # Panics
    ///
    /// Panics if individuals have different genome sizes.
    #[must_use]
    pub fn new(individuals: Vec<Individual>) -> Self {
        let id = Uuid::new_v4();
        Self::with_id(id, individuals)
    }

    /// Create a population with a specific ID.
    ///
    /// # Panics
    ///
    /// Panics if individuals have different genome sizes or if the population is empty.
    #[must_use]
    pub fn with_id(id: PopulationId, individuals: Vec<Individual>) -> Self {
        assert!(!individuals.is_empty(), "Population cannot be empty");

        let genome_size = individuals[0].genome.len();
        assert!(
            individuals.iter().all(|ind| ind.genome.len() == genome_size),
            "All individuals must have the same genome size"
        );

        let mut ids = Vec::with_capacity(individuals.len());
        let mut genomes = Vec::with_capacity(individuals.len() * genome_size);
        let mut fitnesses = Vec::with_capacity(individuals.len());

        for individual in individuals {
            ids.push(individual.id);
            genomes.extend_from_slice(&individual.genome);
            fitnesses.push(individual.fitness);
        }

        Self {
            id,
            ids,
            genomes,
            fitnesses,
            genome_size,
        }
    }

    /// Get the population size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.ids.len()
    }

    /// Get the genome size.
    #[must_use]
    pub fn genome_size(&self) -> usize {
        self.genome_size
    }

    /// Get an individual by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Individual> {
        if index >= self.size() {
            return None;
        }

        let start = index * self.genome_size;
        let end = start + self.genome_size;
        let genome = self.genomes[start..end].to_vec();

        Some(Individual {
            id: self.ids[index],
            genome,
            fitness: self.fitnesses[index],
        })
    }

    /// Get all individuals.
    #[must_use]
    pub fn individuals(&self) -> Vec<Individual> {
        (0..self.size()).filter_map(|i| self.get(i)).collect()
    }

    /// Get a reference to the genome data (SoA).
    #[must_use]
    pub fn genome_data(&self) -> &[f64] {
        &self.genomes
    }

    /// Get a mutable reference to the genome data (SoA).
    pub fn genome_data_mut(&mut self) -> &mut [f64] {
        &mut self.genomes
    }

    /// Get a reference to the fitness data (SoA).
    #[must_use]
    pub fn fitness_data(&self) -> &[Option<f64>] {
        &self.fitnesses
    }

    /// Set the fitness value for an individual by index.
    pub fn set_fitness(&mut self, index: usize, fitness: f64) {
        if index < self.size() {
            self.fitnesses[index] = Some(fitness);
        }
    }

    /// Set all fitness values.
    pub fn set_all_fitnesses(&mut self, fitnesses: Vec<f64>) {
        assert_eq!(fitnesses.len(), self.size(), "Fitness count must match population size");
        self.fitnesses = fitnesses.into_iter().map(Some).collect();
    }

    /// Calculate population statistics.
    ///
    /// # Panics
    ///
    /// Panics if any individual has not been evaluated (fitness is None).
    #[must_use]
    pub fn stats(&self) -> PopulationStats {
        let evaluated: Vec<f64> = self
            .fitnesses
            .iter()
            .map(|f| f.expect("All individuals must be evaluated"))
            .collect();

        let size = evaluated.len();
        let best_fitness = evaluated.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let worst_fitness = evaluated.iter().copied().fold(f64::INFINITY, f64::min);
        let sum: f64 = evaluated.iter().sum();
        let average_fitness = sum / size as f64;

        let variance: f64 = evaluated
            .iter()
            .map(|f| {
                let diff = f - average_fitness;
                diff * diff
            })
            .sum::<f64>()
            / size as f64;
        let std_dev = variance.sqrt();

        // Calculate diversity as average pairwise Euclidean distance
        let mut total_distance = 0.0;
        let mut pair_count = 0;

        for i in 0..size {
            for j in (i + 1)..size {
                let dist = self.euclidean_distance(i, j);
                total_distance += dist;
                pair_count += 1;
            }
        }

        let diversity = if pair_count > 0 {
            total_distance / pair_count as f64
        } else {
            0.0
        };

        PopulationStats {
            size,
            best_fitness,
            worst_fitness,
            average_fitness,
            std_dev,
            diversity,
        }
    }

    /// Calculate Euclidean distance between two individuals.
    fn euclidean_distance(&self, idx1: usize, idx2: usize) -> f64 {
        let start1 = idx1 * self.genome_size;
        let end1 = start1 + self.genome_size;
        let start2 = idx2 * self.genome_size;
        let end2 = start2 + self.genome_size;

        let genome1 = &self.genomes[start1..end1];
        let genome2 = &self.genomes[start2..end2];

        genome1
            .iter()
            .zip(genome2.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Get the best individual.
    ///
    /// # Panics
    ///
    /// Panics if any individual has not been evaluated.
    #[must_use]
    pub fn best(&self) -> Individual {
        let best_idx = self
            .fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_val = a.expect("All individuals must be evaluated");
                let b_val = b.expect("All individuals must be evaluated");
                a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .expect("Population must not be empty");

        self.get(best_idx).expect("Best index must be valid")
    }

    /// Get the N best individuals.
    ///
    /// # Panics
    ///
    /// Panics if any individual has not been evaluated.
    #[must_use]
    pub fn best_n(&self, n: usize) -> Vec<Individual> {
        let mut indexed: Vec<(usize, f64)> = self
            .fitnesses
            .iter()
            .enumerate()
            .map(|(idx, f)| (idx, f.expect("All individuals must be evaluated")))
            .collect();

        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        indexed
            .iter()
            .take(n)
            .filter_map(|(idx, _)| self.get(*idx))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let genome = vec![1.0, 2.0, 3.0];
        let ind = Individual::new(genome.clone());

        assert_eq!(ind.genome, genome);
        assert!(ind.fitness.is_none());
    }

    #[test]
    fn test_individual_set_fitness() {
        let mut ind = Individual::new(vec![1.0, 2.0]);
        ind.set_fitness(42.0);

        assert_eq!(ind.fitness, Some(42.0));
    }

    #[test]
    fn test_population_creation() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
            Individual::new(vec![5.0, 6.0]),
        ];

        let pop = Population::new(individuals.clone());

        assert_eq!(pop.size(), 3);
        assert_eq!(pop.genome_size(), 2);

        // Verify SoA layout
        assert_eq!(pop.genome_data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Population cannot be empty")]
    fn test_population_empty_panic() {
        Population::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "All individuals must have the same genome size")]
    fn test_population_mismatched_genome_sizes() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0, 5.0]), // Different size
        ];

        Population::new(individuals);
    }

    #[test]
    fn test_population_get() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        let pop = Population::new(individuals);

        let ind0 = pop.get(0).unwrap();
        assert_eq!(ind0.genome, vec![1.0, 2.0]);

        let ind1 = pop.get(1).unwrap();
        assert_eq!(ind1.genome, vec![3.0, 4.0]);

        assert!(pop.get(2).is_none());
    }

    #[test]
    fn test_population_set_fitness() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        let mut pop = Population::new(individuals);

        pop.set_fitness(0, 10.0);
        pop.set_fitness(1, 20.0);

        assert_eq!(pop.fitness_data(), &[Some(10.0), Some(20.0)]);
    }

    #[test]
    fn test_population_set_all_fitnesses() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
            Individual::new(vec![5.0, 6.0]),
        ];

        let mut pop = Population::new(individuals);
        pop.set_all_fitnesses(vec![10.0, 20.0, 15.0]);

        assert_eq!(pop.fitness_data(), &[Some(10.0), Some(20.0), Some(15.0)]);
    }

    #[test]
    fn test_population_stats() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
            Individual::new(vec![5.0, 6.0]),
        ];

        let mut pop = Population::new(individuals);
        pop.set_all_fitnesses(vec![10.0, 20.0, 15.0]);

        let stats = pop.stats();

        assert_eq!(stats.size, 3);
        assert_eq!(stats.best_fitness, 20.0);
        assert_eq!(stats.worst_fitness, 10.0);
        assert!((stats.average_fitness - 15.0).abs() < 1e-10);

        // Standard deviation should be sqrt(((10-15)^2 + (20-15)^2 + (15-15)^2) / 3)
        // = sqrt((25 + 25 + 0) / 3) = sqrt(50/3) ≈ 4.08
        assert!((stats.std_dev - 4.082).abs() < 0.01);

        // Diversity should be positive
        assert!(stats.diversity > 0.0);
    }

    #[test]
    fn test_population_best() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
            Individual::new(vec![5.0, 6.0]),
        ];

        let mut pop = Population::new(individuals);
        pop.set_all_fitnesses(vec![10.0, 30.0, 20.0]);

        let best = pop.best();
        assert_eq!(best.genome, vec![3.0, 4.0]);
        assert_eq!(best.fitness, Some(30.0));
    }

    #[test]
    fn test_population_best_n() {
        let individuals = vec![
            Individual::new(vec![1.0]),
            Individual::new(vec![2.0]),
            Individual::new(vec![3.0]),
            Individual::new(vec![4.0]),
        ];

        let mut pop = Population::new(individuals);
        pop.set_all_fitnesses(vec![40.0, 10.0, 30.0, 20.0]);

        let best_2 = pop.best_n(2);
        assert_eq!(best_2.len(), 2);
        assert_eq!(best_2[0].genome, vec![1.0]); // Fitness 40.0
        assert_eq!(best_2[1].genome, vec![3.0]); // Fitness 30.0
    }

    #[test]
    fn test_population_genome_data_mut() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        let mut pop = Population::new(individuals);

        // Modify genome data directly
        let data = pop.genome_data_mut();
        data[0] = 10.0;
        data[3] = 40.0;

        // Verify changes
        let ind0 = pop.get(0).unwrap();
        assert_eq!(ind0.genome, vec![10.0, 2.0]);

        let ind1 = pop.get(1).unwrap();
        assert_eq!(ind1.genome, vec![3.0, 40.0]);
    }

    #[test]
    fn test_population_individuals() {
        let individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        let pop = Population::new(individuals);
        let retrieved = pop.individuals();

        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].genome, vec![1.0, 2.0]);
        assert_eq!(retrieved[1].genome, vec![3.0, 4.0]);
    }
}
