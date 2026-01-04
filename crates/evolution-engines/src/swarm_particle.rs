//! Particle implementation for SwarmAgentic optimization
//!
//! This module contains the particle structure and related operations for
//! particle swarm optimization algorithms.

use crate::traits::Evolvable;
use serde::{Deserialize, Serialize};
use wide::f64x4;

/// Compute mean absolute value of a slice using SIMD
///
/// This is useful for computing velocity magnitudes and similar metrics.
#[inline]
pub fn simd_mean_abs(values: &[f64]) -> f64 {
    let len = values.len();
    if len == 0 {
        return 0.0;
    }

    let chunks = len / 4;
    let remainder = len % 4;

    // SIMD accumulator for sum of absolute values
    let mut sum_simd = f64x4::ZERO;

    // Process 4 elements at a time with SIMD
    for chunk in 0..chunks {
        let i = chunk * 4;
        let val = f64x4::new([values[i], values[i + 1], values[i + 2], values[i + 3]]);
        sum_simd += val.abs();
    }

    // Horizontal sum of SIMD lanes
    let sum_arr: [f64; 4] = sum_simd.into();
    let mut total = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

    // Handle remainder with scalar operations
    let base = chunks * 4;
    for i in 0..remainder {
        total += values[base + i].abs();
    }

    total / len as f64
}

/// Particle parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleParameters {
    /// Velocity clamping factor
    pub velocity_clamp: f64,
    /// Position bounds
    pub position_bounds: Option<(f64, f64)>,
    /// Velocity initialization range
    pub velocity_init_range: f64,
}

impl Default for ParticleParameters {
    fn default() -> Self {
        Self {
            velocity_clamp: 1.0,
            position_bounds: Some((-10.0, 10.0)),
            velocity_init_range: 0.1,
        }
    }
}

/// Particle in the swarm
#[derive(Debug, Clone)]
pub struct Particle<E: Evolvable> {
    /// Current position (entity)
    pub position: E,
    /// Velocity vector
    pub velocity: Vec<f64>,
    /// Personal best position
    pub personal_best: E,
    /// Personal best fitness
    pub personal_best_fitness: E::Fitness,
}

impl<E: Evolvable> Particle<E> {
    /// Create new particle
    pub fn new(
        position: E,
        velocity: Vec<f64>,
        personal_best: E,
        personal_best_fitness: E::Fitness,
    ) -> Self {
        Self {
            position,
            velocity,
            personal_best,
            personal_best_fitness,
        }
    }

    /// Update personal best if current fitness is better
    pub fn update_personal_best(&mut self, fitness: E::Fitness)
    where
        E::Fitness: PartialOrd + Copy,
    {
        if fitness > self.personal_best_fitness {
            self.personal_best = self.position.clone();
            self.personal_best_fitness = fitness;
        }
    }

    /// Apply velocity constraints
    #[inline]
    pub fn clamp_velocity(&mut self, clamp_value: f64) {
        for velocity in &mut self.velocity {
            *velocity = velocity.clamp(-clamp_value, clamp_value);
        }
    }

    /// Calculate velocity magnitude using SIMD for improved performance
    #[inline]
    pub fn velocity_magnitude(&self) -> f64 {
        simd_mean_abs(&self.velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================================
    // Tests for simd_mean_abs - SIMD-accelerated mean absolute value
    // =====================================================================

    #[test]
    fn test_simd_mean_abs_empty() {
        assert_eq!(simd_mean_abs(&[]), 0.0);
    }

    #[test]
    fn test_simd_mean_abs_single_element() {
        assert_eq!(simd_mean_abs(&[5.0]), 5.0);
        assert_eq!(simd_mean_abs(&[-5.0]), 5.0);
    }

    #[test]
    fn test_simd_mean_abs_less_than_simd_width() {
        // 3 elements - scalar path only
        let values = [1.0, -2.0, 3.0];
        let expected = (1.0 + 2.0 + 3.0) / 3.0;
        assert_eq!(simd_mean_abs(&values), expected);
    }

    #[test]
    fn test_simd_mean_abs_exact_simd_width() {
        // 4 elements - exactly one SIMD chunk
        let values = [1.0, -2.0, 3.0, -4.0];
        let expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert_eq!(simd_mean_abs(&values), expected);
    }

    #[test]
    fn test_simd_mean_abs_multiple_simd_chunks() {
        // 8 elements - two SIMD chunks
        let values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
        assert_eq!(simd_mean_abs(&values), expected);
    }

    #[test]
    fn test_simd_mean_abs_with_remainder() {
        // 7 elements - one SIMD chunk + 3 remainder
        let values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];
        let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0) / 7.0;
        assert_eq!(simd_mean_abs(&values), expected);
    }

    #[test]
    fn test_simd_mean_abs_all_zeros() {
        let values = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(simd_mean_abs(&values), 0.0);
    }

    #[test]
    fn test_simd_mean_abs_all_negative() {
        let values = [-1.0, -2.0, -3.0, -4.0, -5.0];
        let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0;
        assert_eq!(simd_mean_abs(&values), expected);
    }

    #[test]
    fn test_simd_mean_abs_large_vector() {
        // Test with a larger vector to stress SIMD path
        let values: Vec<f64> = (1..=100)
            .map(|i| if i % 2 == 0 { -(i as f64) } else { i as f64 })
            .collect();
        let expected: f64 = (1..=100).map(|i| i as f64).sum::<f64>() / 100.0;
        let result = simd_mean_abs(&values);
        assert!((result - expected).abs() < 1e-10);
    }

    // =====================================================================
    // Tests for ParticleParameters
    // =====================================================================

    #[test]
    fn test_particle_parameters_default() {
        let params = ParticleParameters::default();
        assert_eq!(params.velocity_clamp, 1.0);
        assert_eq!(params.position_bounds, Some((-10.0, 10.0)));
        assert_eq!(params.velocity_init_range, 0.1);
    }

    #[test]
    fn test_particle_parameters_serialization() {
        let params = ParticleParameters::default();
        let json = serde_json::to_string(&params).expect("serialization should work");
        let deserialized: ParticleParameters =
            serde_json::from_str(&json).expect("deserialization should work");
        assert_eq!(params.velocity_clamp, deserialized.velocity_clamp);
        assert_eq!(params.position_bounds, deserialized.position_bounds);
    }
}
