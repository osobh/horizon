//! Selection strategies for evolutionary algorithms.
//!
//! This module provides various selection strategies used to choose
//! individuals for reproduction based on their fitness.

use crate::evolution::population::Individual;
use rand::prelude::*;
use rand::seq::SliceRandom;

/// Selection strategy trait.
pub trait SelectionStrategy: Send + Sync {
    /// Select N individuals from the population.
    ///
    /// # Panics
    ///
    /// Panics if population is empty or individuals have no fitness values.
    fn select(&self, population: &[Individual], count: usize) -> Vec<Individual>;

    /// Get the name of this selection strategy.
    fn name(&self) -> &str;
}

/// Tournament selection strategy.
///
/// Randomly selects K individuals and chooses the best one.
/// Repeats until N individuals are selected.
#[derive(Debug, Clone)]
pub struct TournamentSelection {
    /// Tournament size
    pub tournament_size: usize,
}

impl TournamentSelection {
    /// Create a new tournament selection with the given size.
    #[must_use]
    pub fn new(tournament_size: usize) -> Self {
        assert!(
            tournament_size > 0,
            "Tournament size must be greater than 0"
        );
        Self { tournament_size }
    }
}

impl SelectionStrategy for TournamentSelection {
    fn select(&self, population: &[Individual], count: usize) -> Vec<Individual> {
        assert!(!population.is_empty(), "Population cannot be empty");

        let mut rng = thread_rng();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let tournament: Vec<&Individual> = population
                .choose_multiple(&mut rng, self.tournament_size.min(population.len()))
                .collect();

            let winner = tournament
                .iter()
                .max_by(|a, b| {
                    let a_fit = a.fitness.expect("Individual must have fitness");
                    let b_fit = b.fitness.expect("Individual must have fitness");
                    a_fit
                        .partial_cmp(&b_fit)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("Tournament must have at least one individual");

            selected.push((*winner).clone());
        }

        selected
    }

    fn name(&self) -> &'static str {
        "Tournament"
    }
}

/// Roulette wheel selection strategy.
///
/// Selects individuals with probability proportional to their fitness.
/// Fitness values are shifted to ensure all are positive.
#[derive(Debug, Clone)]
pub struct RouletteSelection;

impl RouletteSelection {
    /// Create a new roulette selection.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for RouletteSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for RouletteSelection {
    fn select(&self, population: &[Individual], count: usize) -> Vec<Individual> {
        assert!(!population.is_empty(), "Population cannot be empty");

        let fitnesses: Vec<f64> = population
            .iter()
            .map(|ind| ind.fitness.expect("Individual must have fitness"))
            .collect();

        // Shift fitnesses to be positive
        let min_fitness = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
        let shift = if min_fitness < 0.0 {
            -min_fitness + 1.0
        } else {
            0.0
        };
        let adjusted: Vec<f64> = fitnesses.iter().map(|f| f + shift).collect();

        let total: f64 = adjusted.iter().sum();
        let mut rng = thread_rng();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let mut spin = rng.r#gen::<f64>() * total;
            let mut idx = 0;

            for (i, &fitness) in adjusted.iter().enumerate() {
                spin -= fitness;
                if spin <= 0.0 {
                    idx = i;
                    break;
                }
            }

            selected.push(population[idx].clone());
        }

        selected
    }

    fn name(&self) -> &'static str {
        "Roulette"
    }
}

/// Rank-based selection strategy.
///
/// Assigns selection probability based on rank rather than raw fitness.
/// Better individuals get higher rank (and thus higher selection probability).
#[derive(Debug, Clone)]
pub struct RankSelection;

impl RankSelection {
    /// Create a new rank selection.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for RankSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for RankSelection {
    fn select(&self, population: &[Individual], count: usize) -> Vec<Individual> {
        assert!(!population.is_empty(), "Population cannot be empty");

        // Create indexed population and sort by fitness
        let mut indexed: Vec<(usize, f64)> = population
            .iter()
            .enumerate()
            .map(|(i, ind)| (i, ind.fitness.expect("Individual must have fitness")))
            .collect();

        indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks (1 to N)
        let ranks: Vec<usize> = (1..=indexed.len()).collect();
        let total_rank: usize = ranks.iter().sum();

        let mut rng = thread_rng();
        let mut selected = Vec::with_capacity(count);

        for _ in 0..count {
            let mut spin = rng.gen_range(0..total_rank);
            let mut selected_idx = 0;

            for (rank_idx, &rank) in ranks.iter().enumerate() {
                if spin < rank {
                    selected_idx = indexed[rank_idx].0;
                    break;
                }
                spin -= rank;
            }

            selected.push(population[selected_idx].clone());
        }

        selected
    }

    fn name(&self) -> &'static str {
        "Rank"
    }
}

/// Elitist selection strategy.
///
/// Simply selects the top N individuals by fitness.
#[derive(Debug, Clone)]
pub struct EliteSelection;

impl EliteSelection {
    /// Create a new elite selection.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for EliteSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionStrategy for EliteSelection {
    fn select(&self, population: &[Individual], count: usize) -> Vec<Individual> {
        assert!(!population.is_empty(), "Population cannot be empty");

        let mut sorted = population.to_vec();
        sorted.sort_by(|a, b| {
            let a_fit = a.fitness.expect("Individual must have fitness");
            let b_fit = b.fitness.expect("Individual must have fitness");
            b_fit
                .partial_cmp(&a_fit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted.into_iter().take(count).collect()
    }

    fn name(&self) -> &'static str {
        "Elite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_population() -> Vec<Individual> {
        vec![
            {
                let mut ind = Individual::new(vec![1.0]);
                ind.set_fitness(10.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![2.0]);
                ind.set_fitness(20.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![3.0]);
                ind.set_fitness(30.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![4.0]);
                ind.set_fitness(40.0);
                ind
            },
        ]
    }

    #[test]
    fn test_tournament_selection() {
        let population = create_test_population();
        let tournament = TournamentSelection::new(2);

        let selected = tournament.select(&population, 2);
        assert_eq!(selected.len(), 2);

        // All selected individuals should have fitness values
        for ind in &selected {
            assert!(ind.fitness.is_some());
        }

        assert_eq!(tournament.name(), "Tournament");
    }

    #[test]
    #[should_panic(expected = "Tournament size must be greater than 0")]
    fn test_tournament_selection_invalid_size() {
        TournamentSelection::new(0);
    }

    #[test]
    #[should_panic(expected = "Population cannot be empty")]
    fn test_tournament_selection_empty_population() {
        let tournament = TournamentSelection::new(2);
        tournament.select(&[], 1);
    }

    #[test]
    fn test_tournament_selection_small_tournament() {
        let population = create_test_population();
        let tournament = TournamentSelection::new(1);

        // With tournament size 1, we're just randomly sampling
        let selected = tournament.select(&population, 10);
        assert_eq!(selected.len(), 10);
    }

    #[test]
    fn test_roulette_selection() {
        let population = create_test_population();
        let roulette = RouletteSelection::new();

        let selected = roulette.select(&population, 2);
        assert_eq!(selected.len(), 2);

        // All selected individuals should have fitness values
        for ind in &selected {
            assert!(ind.fitness.is_some());
        }

        assert_eq!(roulette.name(), "Roulette");
    }

    #[test]
    fn test_roulette_selection_negative_fitness() {
        let population = vec![
            {
                let mut ind = Individual::new(vec![1.0]);
                ind.set_fitness(-10.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![2.0]);
                ind.set_fitness(-5.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![3.0]);
                ind.set_fitness(-1.0);
                ind
            },
        ];

        let roulette = RouletteSelection::new();
        let selected = roulette.select(&population, 2);
        assert_eq!(selected.len(), 2);

        // Should handle negative fitness by shifting
        for ind in &selected {
            assert!(ind.fitness.is_some());
        }
    }

    #[test]
    fn test_rank_selection() {
        let population = create_test_population();
        let rank = RankSelection::new();

        let selected = rank.select(&population, 2);
        assert_eq!(selected.len(), 2);

        // All selected individuals should have fitness values
        for ind in &selected {
            assert!(ind.fitness.is_some());
        }

        assert_eq!(rank.name(), "Rank");
    }

    #[test]
    fn test_rank_selection_equal_fitness() {
        let population = vec![
            {
                let mut ind = Individual::new(vec![1.0]);
                ind.set_fitness(10.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![2.0]);
                ind.set_fitness(10.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![3.0]);
                ind.set_fitness(10.0);
                ind
            },
        ];

        let rank = RankSelection::new();
        let selected = rank.select(&population, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_elite_selection() {
        let population = create_test_population();
        let elite = EliteSelection::new();

        let selected = elite.select(&population, 2);
        assert_eq!(selected.len(), 2);

        // Should select the top 2
        assert_eq!(selected[0].fitness, Some(40.0));
        assert_eq!(selected[1].fitness, Some(30.0));

        assert_eq!(elite.name(), "Elite");
    }

    #[test]
    fn test_elite_selection_all() {
        let population = create_test_population();
        let elite = EliteSelection::new();

        let selected = elite.select(&population, 10);

        // Should return all 4 individuals, sorted by fitness
        assert_eq!(selected.len(), 4);
        assert_eq!(selected[0].fitness, Some(40.0));
        assert_eq!(selected[1].fitness, Some(30.0));
        assert_eq!(selected[2].fitness, Some(20.0));
        assert_eq!(selected[3].fitness, Some(10.0));
    }

    #[test]
    fn test_elite_selection_maintains_order() {
        let population = vec![
            {
                let mut ind = Individual::new(vec![1.0]);
                ind.set_fitness(5.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![2.0]);
                ind.set_fitness(50.0);
                ind
            },
            {
                let mut ind = Individual::new(vec![3.0]);
                ind.set_fitness(25.0);
                ind
            },
        ];

        let elite = EliteSelection::new();
        let selected = elite.select(&population, 2);

        assert_eq!(selected[0].genome, vec![2.0]); // Fitness 50
        assert_eq!(selected[1].genome, vec![3.0]); // Fitness 25
    }

    #[test]
    fn test_all_strategies_preserve_ids() {
        let population = create_test_population();
        let original_ids: Vec<_> = population.iter().map(|i| i.id).collect();

        let strategies: Vec<Box<dyn SelectionStrategy>> = vec![
            Box::new(TournamentSelection::new(2)),
            Box::new(RouletteSelection::new()),
            Box::new(RankSelection::new()),
            Box::new(EliteSelection::new()),
        ];

        for strategy in strategies {
            let selected = strategy.select(&population, 2);

            // All selected IDs should be from the original population
            for ind in &selected {
                assert!(original_ids.contains(&ind.id));
            }
        }
    }
}
