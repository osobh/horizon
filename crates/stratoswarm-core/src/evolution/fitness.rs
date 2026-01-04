//! Fitness evaluation functions and parallel evaluation support.
//!
//! This module provides the `FitnessFunction` trait and built-in fitness
//! functions commonly used in evolutionary algorithms.

use crate::evolution::population::Individual;
use std::sync::Arc;

/// Trait for fitness evaluation functions.
pub trait FitnessFunction: Send + Sync {
    /// Evaluate the fitness of an individual.
    ///
    /// Higher fitness values are better (maximization).
    fn evaluate(&self, genome: &[f64]) -> f64;

    /// Get the name of this fitness function.
    fn name(&self) -> &str;

    /// Get the optimal value (if known).
    fn optimal(&self) -> Option<f64> {
        None
    }
}

/// Sphere function: f(x) = -`sum(x_i^2)`
///
/// Minimization problem (negated for maximization).
/// Optimal: f(0, 0, ..., 0) = 0
#[derive(Debug, Clone)]
pub struct SphereFunction;

impl FitnessFunction for SphereFunction {
    fn evaluate(&self, genome: &[f64]) -> f64 {
        -genome.iter().map(|x| x * x).sum::<f64>()
    }

    fn name(&self) -> &'static str {
        "Sphere"
    }

    fn optimal(&self) -> Option<f64> {
        Some(0.0)
    }
}

/// Rosenbrock function: f(x) = -sum(100*(x_{i+1} - `x_i^2)^2` + (1 - `x_i)^2`)
///
/// Minimization problem (negated for maximization).
/// Optimal: f(1, 1, ..., 1) = 0
#[derive(Debug, Clone)]
pub struct RosenbrockFunction;

impl FitnessFunction for RosenbrockFunction {
    fn evaluate(&self, genome: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..genome.len().saturating_sub(1) {
            let xi = genome[i];
            let xi_next = genome[i + 1];
            sum += 100.0 * (xi_next - xi * xi).powi(2) + (1.0 - xi).powi(2);
        }
        -sum
    }

    fn name(&self) -> &'static str {
        "Rosenbrock"
    }

    fn optimal(&self) -> Option<f64> {
        Some(0.0)
    }
}

/// Rastrigin function: f(x) = -(10n + `sum(x_i^2` - 10*cos(2*pi*`x_i`)))
///
/// Minimization problem (negated for maximization).
/// Optimal: f(0, 0, ..., 0) = 0
#[derive(Debug, Clone)]
pub struct RastriginFunction;

impl FitnessFunction for RastriginFunction {
    fn evaluate(&self, genome: &[f64]) -> f64 {
        let n = genome.len() as f64;
        let sum: f64 = genome
            .iter()
            .map(|x| x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos())
            .sum();
        -(10.0 * n + sum)
    }

    fn name(&self) -> &'static str {
        "Rastrigin"
    }

    fn optimal(&self) -> Option<f64> {
        Some(0.0)
    }
}

/// Evaluate fitness for multiple individuals in parallel.
pub async fn evaluate_parallel<F>(individuals: &mut [Individual], fitness_fn: Arc<F>) -> Vec<f64>
where
    F: FitnessFunction + 'static,
{
    let fitnesses: Vec<f64> = individuals
        .iter()
        .map(|ind| fitness_fn.evaluate(&ind.genome))
        .collect();

    // Update individuals with fitness values
    for (ind, &fitness) in individuals.iter_mut().zip(fitnesses.iter()) {
        ind.set_fitness(fitness);
    }

    fitnesses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_function() {
        let sphere = SphereFunction;

        // Test at origin (optimal)
        assert_eq!(sphere.evaluate(&[0.0, 0.0, 0.0]), 0.0);

        // Test at other points
        assert_eq!(sphere.evaluate(&[1.0, 0.0, 0.0]), -1.0);
        assert_eq!(sphere.evaluate(&[1.0, 1.0, 0.0]), -2.0);
        assert_eq!(sphere.evaluate(&[1.0, 1.0, 1.0]), -3.0);

        // Test name
        assert_eq!(sphere.name(), "Sphere");

        // Test optimal
        assert_eq!(sphere.optimal(), Some(0.0));
    }

    #[test]
    fn test_rosenbrock_function() {
        let rosenbrock = RosenbrockFunction;

        // Test at optimal (1, 1, ..., 1)
        let optimal = vec![1.0, 1.0, 1.0];
        assert_eq!(rosenbrock.evaluate(&optimal), 0.0);

        // Test at other points (should be worse)
        assert!(rosenbrock.evaluate(&[0.0, 0.0]) < 0.0);
        assert!(rosenbrock.evaluate(&[2.0, 2.0]) < 0.0);

        // Test name
        assert_eq!(rosenbrock.name(), "Rosenbrock");

        // Test optimal
        assert_eq!(rosenbrock.optimal(), Some(0.0));
    }

    #[test]
    fn test_rastrigin_function() {
        let rastrigin = RastriginFunction;

        // Test at origin (optimal)
        let fitness = rastrigin.evaluate(&[0.0, 0.0, 0.0]);
        assert!((fitness - 0.0).abs() < 1e-10);

        // Test at other points (should be worse)
        assert!(rastrigin.evaluate(&[1.0, 0.0, 0.0]) < -1.0);
        assert!(rastrigin.evaluate(&[1.0, 1.0, 1.0]) < -1.0);

        // Test name
        assert_eq!(rastrigin.name(), "Rastrigin");

        // Test optimal
        assert_eq!(rastrigin.optimal(), Some(0.0));
    }

    #[tokio::test]
    async fn test_evaluate_parallel() {
        let sphere = Arc::new(SphereFunction);

        let mut individuals = vec![
            Individual::new(vec![0.0, 0.0]),
            Individual::new(vec![1.0, 0.0]),
            Individual::new(vec![1.0, 1.0]),
        ];

        let fitnesses = evaluate_parallel(&mut individuals, sphere).await;

        assert_eq!(fitnesses, vec![0.0, -1.0, -2.0]);

        // Verify individuals were updated
        assert_eq!(individuals[0].fitness, Some(0.0));
        assert_eq!(individuals[1].fitness, Some(-1.0));
        assert_eq!(individuals[2].fitness, Some(-2.0));
    }

    #[tokio::test]
    async fn test_evaluate_parallel_preserves_order() {
        let sphere = Arc::new(SphereFunction);

        let mut individuals = vec![
            Individual::new(vec![2.0, 0.0]),
            Individual::new(vec![0.0, 0.0]),
            Individual::new(vec![1.0, 1.0]),
        ];

        let original_ids: Vec<_> = individuals.iter().map(|i| i.id).collect();

        let _fitnesses = evaluate_parallel(&mut individuals, sphere).await;

        // Verify order is preserved
        let new_ids: Vec<_> = individuals.iter().map(|i| i.id).collect();
        assert_eq!(original_ids, new_ids);
    }
}
