//! Mutation operators for evolutionary algorithms.
//!
//! This module provides various mutation strategies to introduce
//! variation into the population.

use crate::evolution::population::Individual;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Mutation operator trait.
pub trait MutationOperator: Send + Sync {
    /// Apply mutation to an individual's genome.
    ///
    /// Returns a new mutated genome.
    fn mutate(&self, genome: &[f64], rate: f64) -> Vec<f64>;

    /// Get the name of this mutation operator.
    fn name(&self) -> &str;
}

/// Gaussian mutation operator.
///
/// Adds Gaussian noise to genes with given probability.
#[derive(Debug, Clone)]
pub struct GaussianMutation {
    /// Standard deviation of Gaussian noise
    pub std_dev: f64,
}

impl GaussianMutation {
    /// Create a new Gaussian mutation operator.
    #[must_use]
    pub fn new(std_dev: f64) -> Self {
        assert!(std_dev > 0.0, "Standard deviation must be positive");
        Self { std_dev }
    }
}

impl MutationOperator for GaussianMutation {
    fn mutate(&self, genome: &[f64], rate: f64) -> Vec<f64> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.std_dev).expect("Valid normal distribution");

        genome
            .iter()
            .map(|&gene| {
                if rng.r#gen::<f64>() < rate {
                    gene + normal.sample(&mut rng)
                } else {
                    gene
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

/// Uniform mutation operator.
///
/// Replaces genes with uniform random values in a range.
#[derive(Debug, Clone)]
pub struct UniformMutation {
    /// Range for uniform mutation (gene ± range)
    pub range: f64,
}

impl UniformMutation {
    /// Create a new uniform mutation operator.
    #[must_use]
    pub fn new(range: f64) -> Self {
        assert!(range > 0.0, "Range must be positive");
        Self { range }
    }
}

impl MutationOperator for UniformMutation {
    fn mutate(&self, genome: &[f64], rate: f64) -> Vec<f64> {
        let mut rng = thread_rng();

        genome
            .iter()
            .map(|&gene| {
                if rng.r#gen::<f64>() < rate {
                    gene + rng.gen_range(-self.range..=self.range)
                } else {
                    gene
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Uniform"
    }
}

/// Polynomial mutation operator.
///
/// Uses polynomial probability distribution for mutation.
/// Common in real-coded genetic algorithms.
#[derive(Debug, Clone)]
pub struct PolynomialMutation {
    /// Distribution index (eta)
    pub eta: f64,
}

impl PolynomialMutation {
    /// Create a new polynomial mutation operator.
    #[must_use]
    pub fn new(eta: f64) -> Self {
        assert!(eta >= 0.0, "Eta must be non-negative");
        Self { eta }
    }
}

impl MutationOperator for PolynomialMutation {
    fn mutate(&self, genome: &[f64], rate: f64) -> Vec<f64> {
        let mut rng = thread_rng();

        genome
            .iter()
            .map(|&gene| {
                if rng.r#gen::<f64>() < rate {
                    let u = rng.r#gen::<f64>();
                    let delta = if u < 0.5 {
                        (2.0 * u).powf(1.0 / (self.eta + 1.0)) - 1.0
                    } else {
                        1.0 - (2.0 * (1.0 - u)).powf(1.0 / (self.eta + 1.0))
                    };
                    gene + delta
                } else {
                    gene
                }
            })
            .collect()
    }

    fn name(&self) -> &'static str {
        "Polynomial"
    }
}

/// Apply mutation to multiple individuals.
pub fn mutate_population<M: MutationOperator>(
    individuals: &mut [Individual],
    operator: &M,
    rate: f64,
) {
    for individual in individuals {
        let mutated_genome = operator.mutate(&individual.genome, rate);
        individual.genome = mutated_genome;
        individual.fitness = None; // Reset fitness after mutation
    }
}

/// Adaptive mutation rate calculator.
///
/// Adjusts mutation rate based on population diversity.
#[derive(Debug, Clone)]
pub struct AdaptiveMutationRate {
    /// Minimum mutation rate
    pub min_rate: f64,
    /// Maximum mutation rate
    pub max_rate: f64,
}

impl AdaptiveMutationRate {
    /// Create a new adaptive mutation rate calculator.
    #[must_use]
    pub fn new(min_rate: f64, max_rate: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&min_rate),
            "Min rate must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&max_rate),
            "Max rate must be in [0, 1]"
        );
        assert!(min_rate <= max_rate, "Min rate must be <= max rate");

        Self { min_rate, max_rate }
    }

    /// Calculate mutation rate based on diversity.
    ///
    /// Higher diversity -> lower mutation rate
    /// Lower diversity -> higher mutation rate
    #[must_use]
    pub fn calculate(&self, diversity: f64, max_diversity: f64) -> f64 {
        if max_diversity == 0.0 {
            return self.max_rate;
        }

        let normalized_diversity = (diversity / max_diversity).clamp(0.0, 1.0);

        // Inverse relationship: low diversity -> high mutation
        self.max_rate - normalized_diversity * (self.max_rate - self.min_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mutation() {
        let gaussian = GaussianMutation::new(0.1);
        let genome = vec![1.0, 2.0, 3.0];

        // With rate 1.0, all genes should be mutated
        let mutated = gaussian.mutate(&genome, 1.0);
        assert_eq!(mutated.len(), genome.len());

        // Genes should be different (with very high probability)
        assert_ne!(mutated, genome);

        assert_eq!(gaussian.name(), "Gaussian");
    }

    #[test]
    fn test_gaussian_mutation_zero_rate() {
        let gaussian = GaussianMutation::new(0.1);
        let genome = vec![1.0, 2.0, 3.0];

        // With rate 0.0, no genes should be mutated
        let mutated = gaussian.mutate(&genome, 0.0);
        assert_eq!(mutated, genome);
    }

    #[test]
    #[should_panic(expected = "Standard deviation must be positive")]
    fn test_gaussian_mutation_invalid_std_dev() {
        GaussianMutation::new(0.0);
    }

    #[test]
    fn test_uniform_mutation() {
        let uniform = UniformMutation::new(0.5);
        let genome = vec![1.0, 2.0, 3.0];

        // With rate 1.0, all genes should be mutated
        let mutated = uniform.mutate(&genome, 1.0);
        assert_eq!(mutated.len(), genome.len());

        // Genes should be different
        assert_ne!(mutated, genome);

        // All mutated genes should be within range
        for (original, &mutated_val) in genome.iter().zip(mutated.iter()) {
            let diff = (mutated_val - original).abs();
            assert!(diff <= 0.5);
        }

        assert_eq!(uniform.name(), "Uniform");
    }

    #[test]
    fn test_uniform_mutation_zero_rate() {
        let uniform = UniformMutation::new(0.5);
        let genome = vec![1.0, 2.0, 3.0];

        let mutated = uniform.mutate(&genome, 0.0);
        assert_eq!(mutated, genome);
    }

    #[test]
    #[should_panic(expected = "Range must be positive")]
    fn test_uniform_mutation_invalid_range() {
        UniformMutation::new(0.0);
    }

    #[test]
    fn test_polynomial_mutation() {
        let polynomial = PolynomialMutation::new(20.0);
        let genome = vec![1.0, 2.0, 3.0];

        // With rate 1.0, all genes should be mutated
        let mutated = polynomial.mutate(&genome, 1.0);
        assert_eq!(mutated.len(), genome.len());

        // Genes should be different
        assert_ne!(mutated, genome);

        assert_eq!(polynomial.name(), "Polynomial");
    }

    #[test]
    fn test_polynomial_mutation_zero_rate() {
        let polynomial = PolynomialMutation::new(20.0);
        let genome = vec![1.0, 2.0, 3.0];

        let mutated = polynomial.mutate(&genome, 0.0);
        assert_eq!(mutated, genome);
    }

    #[test]
    #[should_panic(expected = "Eta must be non-negative")]
    fn test_polynomial_mutation_invalid_eta() {
        PolynomialMutation::new(-1.0);
    }

    #[test]
    fn test_mutate_population() {
        let gaussian = GaussianMutation::new(0.1);
        let mut individuals = vec![
            Individual::new(vec![1.0, 2.0]),
            Individual::new(vec![3.0, 4.0]),
        ];

        // Set fitness values
        individuals[0].set_fitness(10.0);
        individuals[1].set_fitness(20.0);

        mutate_population(&mut individuals, &gaussian, 1.0);

        // All individuals should have been mutated
        assert_eq!(individuals.len(), 2);

        // Fitness should be reset
        assert!(individuals[0].fitness.is_none());
        assert!(individuals[1].fitness.is_none());
    }

    #[test]
    fn test_adaptive_mutation_rate() {
        let adaptive = AdaptiveMutationRate::new(0.01, 0.1);

        // High diversity -> low mutation rate
        let rate_high_diversity = adaptive.calculate(1.0, 1.0);
        assert!((rate_high_diversity - 0.01).abs() < 1e-10);

        // Low diversity -> high mutation rate
        let rate_low_diversity = adaptive.calculate(0.0, 1.0);
        assert!((rate_low_diversity - 0.1).abs() < 1e-10);

        // Medium diversity -> medium mutation rate
        let rate_medium_diversity = adaptive.calculate(0.5, 1.0);
        assert!(rate_medium_diversity > 0.01 && rate_medium_diversity < 0.1);
    }

    #[test]
    fn test_adaptive_mutation_rate_zero_max_diversity() {
        let adaptive = AdaptiveMutationRate::new(0.01, 0.1);

        // When max diversity is zero, should return max rate
        let rate = adaptive.calculate(0.0, 0.0);
        assert_eq!(rate, 0.1);
    }

    #[test]
    #[should_panic(expected = "Min rate must be in [0, 1]")]
    fn test_adaptive_mutation_rate_invalid_min() {
        AdaptiveMutationRate::new(-0.1, 0.5);
    }

    #[test]
    #[should_panic(expected = "Max rate must be in [0, 1]")]
    fn test_adaptive_mutation_rate_invalid_max() {
        AdaptiveMutationRate::new(0.1, 1.5);
    }

    #[test]
    #[should_panic(expected = "Min rate must be <= max rate")]
    fn test_adaptive_mutation_rate_min_greater_than_max() {
        AdaptiveMutationRate::new(0.5, 0.1);
    }

    #[test]
    fn test_all_operators_preserve_length() {
        let genome = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let operators: Vec<Box<dyn MutationOperator>> = vec![
            Box::new(GaussianMutation::new(0.1)),
            Box::new(UniformMutation::new(0.5)),
            Box::new(PolynomialMutation::new(20.0)),
        ];

        for operator in operators {
            let mutated = operator.mutate(&genome, 1.0);
            assert_eq!(mutated.len(), genome.len());
        }
    }

    #[test]
    fn test_mutation_respects_rate() {
        let gaussian = GaussianMutation::new(1.0); // Large std_dev for clear changes
        let genome = vec![0.0; 100]; // Large genome for statistical test

        // With very low rate, most genes should remain unchanged
        let mutated = gaussian.mutate(&genome, 0.01);
        let unchanged = mutated.iter().filter(|&&x| x == 0.0).count();

        // Expect most genes to be unchanged (allow for some randomness)
        assert!(unchanged > 90);
    }
}
