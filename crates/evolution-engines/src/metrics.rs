//! Evolution metrics tracking

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Evolution metrics for tracking engine performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Current generation number
    pub generation: u32,
    /// Total evaluations performed
    pub total_evaluations: u64,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Average fitness of current population
    pub average_fitness: f64,
    /// Population diversity score
    pub diversity_score: f64,
    /// Time elapsed
    pub elapsed_time: Duration,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Performance history
    pub history: PerformanceHistory,
}

impl Default for EvolutionMetrics {
    fn default() -> Self {
        Self {
            generation: 0,
            total_evaluations: 0,
            best_fitness: 0.0,
            average_fitness: 0.0,
            diversity_score: 1.0,
            elapsed_time: Duration::from_secs(0),
            convergence_rate: 0.0,
            custom_metrics: HashMap::new(),
            history: PerformanceHistory::default(),
        }
    }
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceHistory {
    /// Best fitness per generation
    pub best_fitness_history: Vec<f64>,
    /// Average fitness per generation
    pub average_fitness_history: Vec<f64>,
    /// Diversity per generation
    pub diversity_history: Vec<f64>,
    /// Generation times
    pub generation_times: Vec<Duration>,
}

impl PerformanceHistory {
    /// Add generation data
    pub fn add_generation(
        &mut self,
        best_fitness: f64,
        average_fitness: f64,
        diversity: f64,
        generation_time: Duration,
    ) {
        self.best_fitness_history.push(best_fitness);
        self.average_fitness_history.push(average_fitness);
        self.diversity_history.push(diversity);
        self.generation_times.push(generation_time);
    }

    /// Get improvement rate over last N generations
    pub fn improvement_rate(&self, window: usize) -> f64 {
        if self.best_fitness_history.len() < window + 1 {
            return 0.0;
        }

        let recent = &self.best_fitness_history[self.best_fitness_history.len() - window..];
        let first = recent.first()?;
        let last = recent.last()?;

        if *first == 0.0 {
            return 0.0;
        }

        (last - first) / first
    }
}

/// Metrics collector for evolution engines
pub struct MetricsCollector {
    start_time: Instant,
    generation_start: Instant,
    metrics: EvolutionMetrics,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            generation_start: Instant::now(),
            metrics: EvolutionMetrics::default(),
        }
    }

    /// Start new generation
    pub fn start_generation(&mut self) {
        self.generation_start = Instant::now();
        self.metrics.generation += 1;
    }

    /// End generation and update metrics
    pub fn end_generation(
        &mut self,
        best_fitness: f64,
        average_fitness: f64,
        diversity: f64,
        evaluations: u64,
    ) {
        let generation_time = self.generation_start.elapsed();

        self.metrics.best_fitness = best_fitness;
        self.metrics.average_fitness = average_fitness;
        self.metrics.diversity_score = diversity;
        self.metrics.total_evaluations += evaluations;
        self.metrics.elapsed_time = self.start_time.elapsed();

        // Calculate convergence rate
        let improvement = self.metrics.history.improvement_rate(5);
        self.metrics.convergence_rate = improvement.abs();

        // Update history
        self.metrics.history.add_generation(
            best_fitness,
            average_fitness,
            diversity,
            generation_time,
        );
    }

    /// Get current metrics
    pub fn metrics(&self) -> &EvolutionMetrics {
        &self.metrics
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.metrics.custom_metrics.insert(name, value);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_metrics_default() {
        let metrics = EvolutionMetrics::default();
        assert_eq!(metrics.generation, 0);
        assert_eq!(metrics.total_evaluations, 0);
        assert_eq!(metrics.diversity_score, 1.0);
    }

    #[test]
    fn test_performance_history() {
        let mut history = PerformanceHistory::default();

        for i in 0..5 {
            history.add_generation(
                i as f64 * 10.0,
                i as f64 * 5.0,
                1.0 - i as f64 * 0.1,
                Duration::from_millis(100),
            );
        }

        assert_eq!(history.best_fitness_history.len(), 5);
        assert_eq!(history.best_fitness_history[4], 40.0);

        let improvement = history.improvement_rate(3);
        assert_eq!(improvement, 1.0); // (40 - 20) / 20 = 1.0
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        collector.start_generation();
        sleep(Duration::from_millis(10));
        collector.end_generation(100.0, 50.0, 0.8, 1000);

        let metrics = collector.metrics();
        assert_eq!(metrics.generation, 1);
        assert_eq!(metrics.best_fitness, 100.0);
        assert_eq!(metrics.average_fitness, 50.0);
        assert_eq!(metrics.diversity_score, 0.8);
        assert_eq!(metrics.total_evaluations, 1000);

        collector.add_custom_metric("test_metric".to_string(), 42.0);
        let metrics_after = collector.metrics();
        assert_eq!(metrics_after.custom_metrics.get("test_metric"), Some(&42.0));
    }

    #[test]
    fn test_convergence_tracking() {
        let mut collector = MetricsCollector::new();

        // Simulate improving generations
        for i in 0..10 {
            collector.start_generation();
            collector.end_generation((i + 1) as f64 * 10.0, (i + 1) as f64 * 5.0, 0.9, 100);
        }

        let metrics = collector.metrics();
        assert_eq!(metrics.generation, 10);
        assert!(metrics.convergence_rate > 0.0);
        assert_eq!(metrics.history.best_fitness_history.len(), 10);
    }

    #[test]
    fn test_evolution_metrics_creation() {
        let metrics = EvolutionMetrics {
            generation: 5,
            total_evaluations: 5000,
            best_fitness: 95.5,
            average_fitness: 75.2,
            diversity_score: 0.85,
            elapsed_time: Duration::from_secs(60),
            convergence_rate: 0.15,
            custom_metrics: HashMap::new(),
            history: PerformanceHistory::default(),
        };

        assert_eq!(metrics.generation, 5);
        assert_eq!(metrics.total_evaluations, 5000);
        assert_eq!(metrics.best_fitness, 95.5);
        assert_eq!(metrics.average_fitness, 75.2);
        assert_eq!(metrics.diversity_score, 0.85);
        assert_eq!(metrics.convergence_rate, 0.15);
    }

    #[test]
    fn test_performance_history_empty() {
        let history = PerformanceHistory::default();

        assert!(history.best_fitness_history.is_empty());
        assert!(history.average_fitness_history.is_empty());
        assert!(history.diversity_history.is_empty());
        assert!(history.generation_times.is_empty());

        // Improvement rate should be 0 for empty history
        assert_eq!(history.improvement_rate(5), 0.0);
    }

    #[test]
    fn test_performance_history_insufficient_data() {
        let mut history = PerformanceHistory::default();

        // Add only 2 generations, request improvement over 5
        history.add_generation(10.0, 5.0, 0.9, Duration::from_millis(100));
        history.add_generation(15.0, 7.5, 0.85, Duration::from_millis(120));

        let improvement = history.improvement_rate(5);
        assert_eq!(improvement, 0.0); // Insufficient data
    }

    #[test]
    fn test_performance_history_zero_first_fitness() {
        let mut history = PerformanceHistory::default();

        // First fitness is 0.0 - should handle division by zero
        history.add_generation(0.0, 0.0, 1.0, Duration::from_millis(100));
        history.add_generation(10.0, 5.0, 0.9, Duration::from_millis(110));
        history.add_generation(20.0, 10.0, 0.8, Duration::from_millis(120));

        let improvement = history.improvement_rate(2);
        assert_eq!(improvement, 0.0); // Should handle zero division
    }

    #[test]
    fn test_performance_history_negative_improvement() {
        let mut history = PerformanceHistory::default();

        // Fitness getting worse
        history.add_generation(100.0, 50.0, 1.0, Duration::from_millis(100));
        history.add_generation(80.0, 40.0, 0.9, Duration::from_millis(110));
        history.add_generation(60.0, 30.0, 0.8, Duration::from_millis(120));

        let improvement = history.improvement_rate(3);
        assert_eq!(improvement, -0.4); // (60 - 100) / 100 = -0.4
    }

    #[test]
    fn test_metrics_collector_multiple_generations() {
        let mut collector = MetricsCollector::new();

        // Simulate 5 generations with varying performance
        let fitness_values = vec![10.0, 25.0, 40.0, 35.0, 50.0];
        let avg_values = vec![5.0, 12.5, 20.0, 17.5, 25.0];
        let diversity_values = vec![1.0, 0.9, 0.8, 0.85, 0.7];

        for (i, ((best, avg), div)) in fitness_values
            .iter()
            .zip(avg_values.iter())
            .zip(diversity_values.iter())
            .enumerate()
        {
            collector.start_generation();
            sleep(Duration::from_millis(5)); // Small delay
            collector.end_generation(*best, *avg, *div, 100 + i as u64 * 10);
        }

        let metrics = collector.metrics();
        assert_eq!(metrics.generation, 5);
        assert_eq!(metrics.best_fitness, 50.0); // Last best fitness
        assert_eq!(metrics.average_fitness, 25.0); // Last average fitness
        assert_eq!(metrics.total_evaluations, 100 + 110 + 120 + 130 + 140); // Cumulative
        assert_eq!(metrics.history.best_fitness_history.len(), 5);
    }

    #[test]
    fn test_custom_metrics_operations() {
        let mut collector = MetricsCollector::new();

        // Add various custom metrics
        collector.add_custom_metric("mutation_rate".to_string(), 0.01);
        collector.add_custom_metric("crossover_rate".to_string(), 0.8);
        collector.add_custom_metric("population_size".to_string(), 100.0);
        collector.add_custom_metric("elite_percentage".to_string(), 0.1);

        let metrics = collector.metrics();
        assert_eq!(metrics.custom_metrics.len(), 4);
        assert_eq!(metrics.custom_metrics.get("mutation_rate"), Some(&0.01));
        assert_eq!(metrics.custom_metrics.get("crossover_rate"), Some(&0.8));
        assert_eq!(metrics.custom_metrics.get("population_size"), Some(&100.0));
        assert_eq!(metrics.custom_metrics.get("elite_percentage"), Some(&0.1));

        // Overwrite existing metric
        collector.add_custom_metric("mutation_rate".to_string(), 0.02);
        let updated_metrics = collector.metrics();
        assert_eq!(
            updated_metrics.custom_metrics.get("mutation_rate"),
            Some(&0.02)
        );
    }

    #[test]
    fn test_metrics_serialization() {
        let mut metrics = EvolutionMetrics::default();
        metrics.generation = 10;
        metrics.best_fitness = 95.5;
        metrics.custom_metrics.insert("test".to_string(), 42.0);

        // Test JSON serialization
        let json = serde_json::to_string(&metrics)?;
        let deserialized: EvolutionMetrics = serde_json::from_str(&json)?;

        assert_eq!(deserialized.generation, 10);
        assert_eq!(deserialized.best_fitness, 95.5);
        assert_eq!(deserialized.custom_metrics.get("test"), Some(&42.0));
    }

    #[test]
    fn test_performance_history_serialization() {
        let mut history = PerformanceHistory::default();
        history.add_generation(10.0, 5.0, 0.9, Duration::from_millis(100));
        history.add_generation(20.0, 10.0, 0.8, Duration::from_millis(110));

        // Test JSON serialization
        let json = serde_json::to_string(&history)?;
        let deserialized: PerformanceHistory = serde_json::from_str(&json)?;

        assert_eq!(deserialized.best_fitness_history.len(), 2);
        assert_eq!(deserialized.best_fitness_history[0], 10.0);
        assert_eq!(deserialized.best_fitness_history[1], 20.0);
        assert_eq!(deserialized.generation_times.len(), 2);
    }

    #[test]
    fn test_metrics_collector_timing() {
        let mut collector = MetricsCollector::new();

        collector.start_generation();
        sleep(Duration::from_millis(50));
        collector.end_generation(100.0, 50.0, 0.8, 1000);

        let metrics = collector.metrics();

        // Elapsed time should be at least 50ms
        assert!(metrics.elapsed_time >= Duration::from_millis(50));

        // Generation time should be recorded in history
        assert_eq!(metrics.history.generation_times.len(), 1);
        assert!(metrics.history.generation_times[0] >= Duration::from_millis(50));
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let mut collector = MetricsCollector::new();

        // Simulate steady improvement
        for i in 0..10 {
            collector.start_generation();
            collector.end_generation((i + 1) as f64 * 10.0, (i + 1) as f64 * 5.0, 0.9, 100);
        }

        let initial_convergence_rate = collector.metrics().convergence_rate;
        assert!(initial_convergence_rate > 0.0);

        // Now simulate plateau (no improvement)
        for _ in 0..5 {
            collector.start_generation();
            collector.end_generation(100.0, 50.0, 0.9, 100); // Same fitness
        }

        let plateau_convergence_rate = collector.metrics().convergence_rate;
        assert!(plateau_convergence_rate < initial_convergence_rate);
    }

    #[test]
    fn test_large_scale_metrics() {
        let mut collector = MetricsCollector::new();

        // Simulate large-scale evolution
        for generation in 0..1000 {
            collector.start_generation();

            // Simulate realistic fitness progression
            let best_fitness = 100.0 * (1.0 - (-generation as f64 / 100.0).exp());
            let avg_fitness = best_fitness * 0.8;
            let diversity = 1.0 / (1.0 + generation as f64 / 100.0);

            collector.end_generation(best_fitness, avg_fitness, diversity, 50);
        }

        let metrics = collector.metrics();
        assert_eq!(metrics.generation, 1000);
        assert_eq!(metrics.total_evaluations, 50000); // 1000 generations * 50 evaluations
        assert!(metrics.best_fitness > 60.0); // Should have converged significantly
        assert_eq!(metrics.history.best_fitness_history.len(), 1000);
    }

    #[test]
    fn test_metrics_edge_cases() {
        let mut collector = MetricsCollector::new();

        // Test with extreme values
        collector.start_generation();
        collector.end_generation(f64::MAX, f64::MIN, f64::NAN, u64::MAX);

        let metrics = collector.metrics();
        assert_eq!(metrics.best_fitness, f64::MAX);
        assert_eq!(metrics.average_fitness, f64::MIN);
        assert!(metrics.diversity_score.is_nan());
        assert_eq!(metrics.total_evaluations, u64::MAX);

        // Test with zero values
        collector.start_generation();
        collector.end_generation(0.0, 0.0, 0.0, 0);

        let zero_metrics = collector.metrics();
        assert_eq!(zero_metrics.best_fitness, 0.0);
        assert_eq!(zero_metrics.average_fitness, 0.0);
        assert_eq!(zero_metrics.diversity_score, 0.0);
    }

    #[test]
    fn test_performance_history_window_sizes() {
        let mut history = PerformanceHistory::default();

        // Add 20 generations
        for i in 0..20 {
            history.add_generation(
                i as f64 * 5.0,
                i as f64 * 2.5,
                1.0 - i as f64 * 0.01,
                Duration::from_millis(100 + i * 10),
            );
        }

        // Test different window sizes
        let improvement_1 = history.improvement_rate(1);
        let improvement_5 = history.improvement_rate(5);
        let improvement_10 = history.improvement_rate(10);
        let improvement_20 = history.improvement_rate(20);

        // All should be positive (improving)
        assert!(improvement_1 > 0.0);
        assert!(improvement_5 > 0.0);
        assert!(improvement_10 > 0.0);
        assert!(improvement_20 > 0.0);

        // Smaller windows should show higher improvement rates
        assert!(improvement_1 > improvement_20);
    }

    #[test]
    fn test_metrics_clone_and_debug() {
        let mut metrics = EvolutionMetrics::default();
        metrics.generation = 5;
        metrics.best_fitness = 85.5;
        metrics
            .custom_metrics
            .insert("test_metric".to_string(), 123.45);

        // Test cloning
        let cloned_metrics = metrics.clone();
        assert_eq!(cloned_metrics.generation, metrics.generation);
        assert_eq!(cloned_metrics.best_fitness, metrics.best_fitness);
        assert_eq!(
            cloned_metrics.custom_metrics.len(),
            metrics.custom_metrics.len()
        );

        // Test debug formatting
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("EvolutionMetrics"));
        assert!(debug_str.contains("generation"));
        assert!(debug_str.contains("best_fitness"));
    }

    #[test]
    fn test_concurrent_metrics_collection() {
        // Test that metrics structure supports concurrent access patterns
        let mut collector = MetricsCollector::new();

        // Simulate rapid generation cycles
        for i in 0..100 {
            collector.start_generation();

            // Simulate minimal processing time
            if i % 10 == 0 {
                sleep(Duration::from_millis(1));
            }

            collector.end_generation(
                i as f64 + 10.0,
                i as f64 / 2.0 + 5.0,
                0.9 - i as f64 * 0.001,
                50,
            );

            // Access metrics frequently
            let _current_metrics = collector.metrics();
        }

        let final_metrics = collector.metrics();
        assert_eq!(final_metrics.generation, 100);
        assert_eq!(final_metrics.history.best_fitness_history.len(), 100);
    }

    #[test]
    fn test_metrics_with_no_generations() {
        let collector = MetricsCollector::new();
        let metrics = collector.metrics();

        // Should have default values
        assert_eq!(metrics.generation, 0);
        assert_eq!(metrics.total_evaluations, 0);
        assert_eq!(metrics.best_fitness, 0.0);
        assert_eq!(metrics.convergence_rate, 0.0);
        assert!(metrics.custom_metrics.is_empty());
        assert!(metrics.history.best_fitness_history.is_empty());
    }

    #[test]
    fn test_custom_metrics_edge_cases() {
        let mut collector = MetricsCollector::new();

        // Test with special float values
        collector.add_custom_metric("infinity".to_string(), f64::INFINITY);
        collector.add_custom_metric("neg_infinity".to_string(), f64::NEG_INFINITY);
        collector.add_custom_metric("nan".to_string(), f64::NAN);
        collector.add_custom_metric("zero".to_string(), 0.0);
        collector.add_custom_metric("negative".to_string(), -42.5);

        let metrics = collector.metrics();
        assert_eq!(metrics.custom_metrics.get("infinity"), Some(&f64::INFINITY));
        assert_eq!(
            metrics.custom_metrics.get("neg_infinity"),
            Some(&f64::NEG_INFINITY)
        );
        assert!(metrics.custom_metrics.get("nan").unwrap().is_nan());
        assert_eq!(metrics.custom_metrics.get("zero"), Some(&0.0));
        assert_eq!(metrics.custom_metrics.get("negative"), Some(&-42.5));
    }

    #[test]
    fn test_performance_history_consistency() {
        let mut history = PerformanceHistory::default();

        // Add generations and verify all vectors stay in sync
        for i in 0..50 {
            history.add_generation(
                i as f64,
                i as f64 / 2.0,
                1.0 - i as f64 / 100.0,
                Duration::from_millis(100 + i),
            );

            // All vectors should have same length
            assert_eq!(history.best_fitness_history.len(), (i + 1) as usize);
            assert_eq!(history.average_fitness_history.len(), (i + 1) as usize);
            assert_eq!(history.diversity_history.len(), (i + 1) as usize);
            assert_eq!(history.generation_times.len(), (i + 1) as usize);
        }

        // Verify data integrity
        assert_eq!(history.best_fitness_history[25], 25.0);
        assert_eq!(history.average_fitness_history[25], 12.5);
        assert_eq!(history.diversity_history[25], 0.75);
        assert_eq!(history.generation_times[25], Duration::from_millis(125));
    }
}
