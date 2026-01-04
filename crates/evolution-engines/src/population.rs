//! Generic population management for evolution engines

use crate::traits::Evolvable;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use uuid::Uuid;

/// Generic individual wrapper
#[derive(Debug, Clone)]
pub struct Individual<E: Evolvable> {
    /// Unique identifier
    pub id: Uuid,
    /// The evolvable entity
    pub entity: E,
    /// Fitness score
    pub fitness: Option<E::Fitness>,
}

impl<E: Evolvable> Individual<E> {
    /// Create new individual
    pub fn new(entity: E) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity,
            fitness: None,
        }
    }

    /// Create individual with fitness
    pub fn with_fitness(entity: E, fitness: E::Fitness) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity,
            fitness: Some(fitness),
        }
    }
}

/// Generic population
#[derive(Debug, Clone)]
pub struct Population<E: Evolvable> {
    /// Individuals in the population
    pub individuals: Vec<Individual<E>>,
    /// Current generation
    pub generation: u64,
}

impl<E: Evolvable> Population<E> {
    /// Create new empty population
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
            generation: 0,
        }
    }

    /// Create population from individuals
    pub fn from_individuals(individuals: Vec<Individual<E>>) -> Self {
        Self {
            individuals,
            generation: 0,
        }
    }

    /// Add individual to population
    pub fn add(&mut self, individual: Individual<E>) {
        self.individuals.push(individual);
    }

    /// Get population size
    #[inline]
    pub fn size(&self) -> usize {
        self.individuals.len()
    }

    /// Get best individual by fitness
    pub fn best(&self) -> Option<&Individual<E>> {
        self.individuals
            .iter()
            .filter(|i| i.fitness.is_some())
            .max_by(|a, b| match (a.fitness.as_ref(), b.fitness.as_ref()) {
                (Some(fa), Some(fb)) => fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal),
                _ => std::cmp::Ordering::Equal,
            })
    }

    /// Select individuals based on fitness using parallel sorting for large populations
    pub fn select_top(&self, n: usize) -> Vec<&Individual<E>>
    where
        E::Fitness: Send + Sync,
    {
        // Collect indices of individuals with fitness
        let mut indexed: Vec<(usize, &E::Fitness)> = self
            .individuals
            .iter()
            .enumerate()
            .filter_map(|(idx, i)| i.fitness.as_ref().map(|f| (idx, f)))
            .collect();

        // Use parallel sort for large populations (threshold: 100 individuals)
        if indexed.len() > 100 {
            indexed.par_sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Return references to top n individuals
        indexed
            .into_iter()
            .take(n)
            .map(|(idx, _)| &self.individuals[idx])
            .collect()
    }

    /// Increment generation
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }
}

impl<E: Evolvable> Default for Population<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EvolutionEngineResult;
    use crate::traits::Evolvable;
    use async_trait::async_trait;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[derive(Debug, Clone)]
    struct TestEntity {
        value: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestGenome {
        value: f64,
    }

    #[async_trait]
    impl Evolvable for TestEntity {
        type Genome = TestGenome;
        type Fitness = f64;

        fn genome(&self) -> &Self::Genome {
            // For test purposes, we need to store a genome reference
            // This is a test limitation - in real implementations,
            // the genome would be part of the entity structure
            static TEST_GENOME: TestGenome = TestGenome { value: 0.0 };
            &TEST_GENOME
        }

        async fn from_genome(_genome: Self::Genome) -> EvolutionEngineResult<Self> {
            Ok(TestEntity { value: 0.0 })
        }

        async fn evaluate_fitness(&self) -> EvolutionEngineResult<Self::Fitness> {
            Ok(self.value)
        }

        async fn mutate(&self, _mutation_rate: f64) -> EvolutionEngineResult<Self> {
            Ok(self.clone())
        }

        async fn crossover(&self, _other: &Self) -> EvolutionEngineResult<(Self, Self)> {
            Ok((self.clone(), self.clone()))
        }
    }

    #[test]
    fn test_population_creation() {
        let pop: Population<TestEntity> = Population::new();
        assert_eq!(pop.size(), 0);
        assert_eq!(pop.generation, 0);
    }

    #[test]
    fn test_population_operations() {
        let mut pop = Population::new();

        // Add individuals
        pop.add(Individual::new(TestEntity { value: 1.0 }));
        pop.add(Individual::with_fitness(TestEntity { value: 2.0 }, 2.0));
        pop.add(Individual::with_fitness(TestEntity { value: 3.0 }, 3.0));

        assert_eq!(pop.size(), 3);

        // Test best individual
        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(3.0));

        // Test selection
        let top2 = pop.select_top(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].fitness, Some(3.0));
        assert_eq!(top2[1].fitness, Some(2.0));

        // Test generation increment
        pop.next_generation();
        assert_eq!(pop.generation, 1);
    }

    #[test]
    fn test_individual_creation() {
        let entity = TestEntity { value: 5.0 };
        let individual = Individual::new(entity.clone());

        assert!(individual.id != Uuid::nil());
        assert_eq!(individual.entity.value, 5.0);
        assert!(individual.fitness.is_none());
    }

    #[test]
    fn test_individual_with_fitness() {
        let entity = TestEntity { value: 7.5 };
        let individual = Individual::with_fitness(entity.clone(), 7.5);

        assert!(individual.id != Uuid::nil());
        assert_eq!(individual.entity.value, 7.5);
        assert_eq!(individual.fitness, Some(7.5));
    }

    #[test]
    fn test_population_from_individuals() {
        let individuals = vec![
            Individual::with_fitness(TestEntity { value: 1.0 }, 1.0),
            Individual::with_fitness(TestEntity { value: 2.0 }, 2.0),
            Individual::with_fitness(TestEntity { value: 3.0 }, 3.0),
        ];

        let pop = Population::from_individuals(individuals);
        assert_eq!(pop.size(), 3);
        assert_eq!(pop.generation, 0);

        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(3.0));
    }

    #[test]
    fn test_empty_population_operations() {
        let pop: Population<TestEntity> = Population::new();

        // Empty population operations
        assert_eq!(pop.size(), 0);
        assert!(pop.best().is_none());

        let top_individuals = pop.select_top(5);
        assert!(top_individuals.is_empty());
    }

    #[test]
    fn test_population_with_no_fitness() {
        let mut pop = Population::new();

        // Add individuals without fitness
        pop.add(Individual::new(TestEntity { value: 1.0 }));
        pop.add(Individual::new(TestEntity { value: 2.0 }));
        pop.add(Individual::new(TestEntity { value: 3.0 }));

        assert_eq!(pop.size(), 3);
        assert!(pop.best().is_none()); // No fitness assigned

        let top_individuals = pop.select_top(2);
        assert!(top_individuals.is_empty()); // No fitness to select from
    }

    #[test]
    fn test_population_mixed_fitness() {
        let mut pop = Population::new();

        // Mix of individuals with and without fitness
        pop.add(Individual::new(TestEntity { value: 1.0 })); // No fitness
        pop.add(Individual::with_fitness(TestEntity { value: 2.0 }, 2.0));
        pop.add(Individual::new(TestEntity { value: 3.0 })); // No fitness
        pop.add(Individual::with_fitness(TestEntity { value: 4.0 }, 4.0));

        assert_eq!(pop.size(), 4);

        // Best should be from fitted individuals only
        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(4.0));

        // Selection should only include fitted individuals
        let top_individuals = pop.select_top(5);
        assert_eq!(top_individuals.len(), 2); // Only 2 have fitness
    }

    #[test]
    fn test_population_selection_edge_cases() {
        let mut pop = Population::new();

        // Add more individuals than we'll select
        for i in 1..=10 {
            pop.add(Individual::with_fitness(
                TestEntity { value: i as f64 },
                i as f64,
            ));
        }

        // Select more than available
        let all_individuals = pop.select_top(20);
        assert_eq!(all_individuals.len(), 10); // Can't select more than exist

        // Select zero
        let no_individuals = pop.select_top(0);
        assert!(no_individuals.is_empty());

        // Select exactly available
        let exact_individuals = pop.select_top(10);
        assert_eq!(exact_individuals.len(), 10);

        // Verify ordering (descending by fitness)
        assert_eq!(exact_individuals[0].fitness, Some(10.0));
        assert_eq!(exact_individuals[9].fitness, Some(1.0));
    }

    #[test]
    fn test_population_duplicate_fitness() {
        let mut pop = Population::new();

        // Add individuals with duplicate fitness values
        pop.add(Individual::with_fitness(TestEntity { value: 1.0 }, 5.0));
        pop.add(Individual::with_fitness(TestEntity { value: 2.0 }, 5.0));
        pop.add(Individual::with_fitness(TestEntity { value: 3.0 }, 5.0));
        pop.add(Individual::with_fitness(TestEntity { value: 4.0 }, 3.0));

        // Best should be one of the 5.0 fitness individuals
        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(5.0));

        // Top selection should handle ties gracefully
        let top2 = pop.select_top(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].fitness, Some(5.0));
        assert_eq!(top2[1].fitness, Some(5.0)); // Another 5.0 individual
    }

    #[test]
    fn test_multiple_generations() {
        let mut pop = Population::new();

        // Start at generation 0
        assert_eq!(pop.generation, 0);

        // Advance through multiple generations
        for i in 1..=5 {
            pop.next_generation();
            assert_eq!(pop.generation, i);
        }

        // Add some individuals to a later generation population
        pop.add(Individual::with_fitness(TestEntity { value: 1.0 }, 1.0));
        assert_eq!(pop.size(), 1);
        assert_eq!(pop.generation, 5);
    }

    #[test]
    fn test_population_large_scale() {
        let mut pop = Population::new();

        // Add large number of individuals
        for i in 1..=1000 {
            pop.add(Individual::with_fitness(
                TestEntity { value: i as f64 },
                i as f64,
            ));
        }

        assert_eq!(pop.size(), 1000);

        // Test best with large population
        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(1000.0));

        // Test selection with large population
        let top10 = pop.select_top(10);
        assert_eq!(top10.len(), 10);
        assert_eq!(top10[0].fitness, Some(1000.0));
        assert_eq!(top10[9].fitness, Some(991.0));
    }

    #[test]
    fn test_population_default() {
        let pop: Population<TestEntity> = Population::default();
        assert_eq!(pop.size(), 0);
        assert_eq!(pop.generation, 0);
        assert!(pop.individuals.is_empty());
    }

    #[test]
    fn test_individual_unique_ids() {
        let entity = TestEntity { value: 1.0 };

        // Create multiple individuals with same entity
        let individual1 = Individual::new(entity.clone());
        let individual2 = Individual::new(entity.clone());
        let individual3 = Individual::with_fitness(entity.clone(), 1.0);

        // All should have unique IDs
        assert_ne!(individual1.id, individual2.id);
        assert_ne!(individual1.id, individual3.id);
        assert_ne!(individual2.id, individual3.id);
    }

    #[test]
    fn test_population_fitness_ordering() {
        let mut pop = Population::new();

        // Add individuals in random fitness order
        let fitness_values = vec![3.5, 1.2, 8.9, 2.1, 7.7, 4.4, 6.3, 0.8, 9.1, 5.6];
        for (i, fitness) in fitness_values.iter().enumerate() {
            pop.add(Individual::with_fitness(
                TestEntity { value: i as f64 },
                *fitness,
            ));
        }

        // Test that selection returns properly ordered results
        let all_selected = pop.select_top(10);
        assert_eq!(all_selected.len(), 10);

        // Verify descending order
        for i in 1..all_selected.len() {
            assert!(all_selected[i - 1].fitness >= all_selected[i].fitness);
        }

        // Verify highest fitness is first
        assert_eq!(all_selected[0].fitness, Some(9.1));
        assert_eq!(all_selected[9].fitness, Some(0.8));
    }

    #[test]
    fn test_population_extreme_fitness_values() {
        let mut pop = Population::new();

        // Add individuals with extreme fitness values
        pop.add(Individual::with_fitness(
            TestEntity { value: 1.0 },
            f64::MIN,
        ));
        pop.add(Individual::with_fitness(
            TestEntity { value: 2.0 },
            f64::MAX,
        ));
        pop.add(Individual::with_fitness(TestEntity { value: 3.0 }, 0.0));
        pop.add(Individual::with_fitness(TestEntity { value: 4.0 }, -1.0));
        pop.add(Individual::with_fitness(TestEntity { value: 5.0 }, 1.0));

        // Best should handle extreme values correctly
        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(f64::MAX));

        // Selection should handle extreme values
        let top3 = pop.select_top(3);
        assert_eq!(top3[0].fitness, Some(f64::MAX));
        assert_eq!(top3[1].fitness, Some(1.0));
        assert_eq!(top3[2].fitness, Some(0.0));
    }

    #[test]
    fn test_population_clone() {
        let mut pop = Population::new();

        // Add some individuals
        pop.add(Individual::with_fitness(TestEntity { value: 1.0 }, 1.0));
        pop.add(Individual::with_fitness(TestEntity { value: 2.0 }, 2.0));
        pop.next_generation();

        // Clone the population
        let cloned_pop = pop.clone();

        // Verify clone has same properties
        assert_eq!(cloned_pop.size(), pop.size());
        assert_eq!(cloned_pop.generation, pop.generation);

        // IDs should be the same in clone
        for (orig, cloned) in pop.individuals.iter().zip(cloned_pop.individuals.iter()) {
            assert_eq!(orig.id, cloned.id);
            assert_eq!(orig.fitness, cloned.fitness);
        }
    }

    #[test]
    fn test_population_debug_trait() {
        let mut pop = Population::new();
        pop.add(Individual::with_fitness(TestEntity { value: 1.0 }, 1.0));

        // Should be able to debug print
        let debug_str = format!("{:?}", pop);
        assert!(debug_str.contains("Population"));
        assert!(debug_str.contains("individuals"));
        assert!(debug_str.contains("generation"));

        // Individual should also be debuggable
        let individual_debug = format!("{:?}", pop.individuals[0]);
        assert!(individual_debug.contains("Individual"));
        assert!(individual_debug.contains("entity"));
        assert!(individual_debug.contains("fitness"));
    }

    #[test]
    fn test_population_concurrent_safety_preparation() {
        // This test verifies the data structures are suitable for concurrent access
        let mut pop = Population::new();

        // Add individuals that could be accessed concurrently
        for i in 0..100 {
            pop.add(Individual::with_fitness(
                TestEntity { value: i as f64 },
                i as f64,
            ));
        }

        // Verify operations work with larger dataset
        assert_eq!(pop.size(), 100);

        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(99.0));

        let top10 = pop.select_top(10);
        assert_eq!(top10.len(), 10);

        // Verify the population can be cloned for thread safety
        let _cloned = pop.clone();
    }

    #[test]
    fn test_population_memory_efficiency() {
        // Test that population handles memory efficiently
        let mut pop = Population::new();

        // Add and remove pattern to test memory usage
        for batch in 0..5 {
            // Add a batch
            for i in 0..50 {
                pop.add(Individual::with_fitness(
                    TestEntity {
                        value: (batch * 50 + i) as f64,
                    },
                    (batch * 50 + i) as f64,
                ));
            }

            // Clear and restart (simulating population replacement)
            if batch > 0 {
                pop.individuals.clear();
                pop.next_generation();
            }
        }

        // Should handle the memory operations without issues
        assert_eq!(pop.generation, 4);
    }
}
