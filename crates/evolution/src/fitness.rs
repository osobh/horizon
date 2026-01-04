//! Fitness evaluation for evolution

use serde::{Deserialize, Serialize};
use stratoswarm_agent_core::agent::{Agent, EvolutionMetrics};

/// Fitness evaluation function
pub trait FitnessFunction: Send + Sync {
    fn evaluate(&self, individual: &[u8]) -> f64;
}

/// XP-based fitness function for agent evolution
pub trait XPFitnessFunction: Send + Sync {
    /// Evaluate fitness based on agent XP and performance metrics
    fn evaluate_agent_fitness(
        &self,
        agent: &Agent,
    ) -> impl std::future::Future<Output = f64> + Send;

    /// Calculate performance-based XP rewards
    fn calculate_xp_reward(
        &self,
        fitness_improvement: f64,
        evolution_metrics: &EvolutionMetrics,
    ) -> u64;

    /// Determine if agent should evolve based on XP thresholds
    fn should_evolve(&self, agent: &Agent) -> impl std::future::Future<Output = bool> + Send;
}

/// Fitness score for an individual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessScore {
    pub value: f64,
    pub individual_id: uuid::Uuid,
}

/// Agent-based fitness score with XP integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFitnessScore {
    pub fitness: f64,
    pub agent_id: stratoswarm_agent_core::agent::AgentId,
    pub xp_contribution: u64,
    pub level: u32,
    pub performance_metrics: EvolutionMetrics,
}

/// Agent performance-based fitness function
pub struct AgentPerformanceFitnessFunction {
    /// Weight for XP in fitness calculation (0.0 to 1.0)
    pub xp_weight: f64,
    /// Weight for success rate in fitness calculation (0.0 to 1.0)
    pub success_rate_weight: f64,
    /// Weight for processing speed in fitness calculation (0.0 to 1.0)
    pub processing_speed_weight: f64,
    /// Weight for memory efficiency in fitness calculation (0.0 to 1.0)
    pub memory_efficiency_weight: f64,
    /// Minimum XP required for positive fitness
    pub min_xp_threshold: u64,
}

impl Default for AgentPerformanceFitnessFunction {
    fn default() -> Self {
        Self {
            xp_weight: 0.4,
            success_rate_weight: 0.3,
            processing_speed_weight: 0.2,
            memory_efficiency_weight: 0.1,
            min_xp_threshold: 50,
        }
    }
}

impl AgentPerformanceFitnessFunction {
    pub fn new(
        xp_weight: f64,
        success_rate_weight: f64,
        processing_speed_weight: f64,
        memory_efficiency_weight: f64,
        min_xp_threshold: u64,
    ) -> Self {
        let total_weight =
            xp_weight + success_rate_weight + processing_speed_weight + memory_efficiency_weight;
        assert!(
            (total_weight - 1.0).abs() < 0.001,
            "Weights must sum to 1.0"
        );

        Self {
            xp_weight,
            success_rate_weight,
            processing_speed_weight,
            memory_efficiency_weight,
            min_xp_threshold,
        }
    }
}

impl XPFitnessFunction for AgentPerformanceFitnessFunction {
    async fn evaluate_agent_fitness(&self, agent: &Agent) -> f64 {
        let stats = agent.stats().await;

        // Return 0 if below minimum XP threshold
        if stats.current_xp < self.min_xp_threshold {
            return 0.0;
        }

        // Normalize XP (assuming max reasonable XP is 25000 for level 15)
        let normalized_xp = (stats.current_xp as f64 / 25000.0).min(1.0);

        // Calculate success rate
        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.0
        };

        // Calculate processing speed (inverse of average completion time)
        let processing_speed = if stats.goals_processed > 0 {
            let avg_time = stats.total_execution_time.as_secs_f64() / stats.goals_processed as f64;
            // Normalize to 0-1 range (assuming 10 seconds is excellent, 120 seconds is poor)
            ((120.0 - avg_time) / 110.0).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Calculate memory efficiency (assume current implementation has basic efficiency)
        let memory_efficiency = if stats.memory_usage > 0 {
            // Simple efficiency metric: available memory / max memory
            let max_memory = agent.config.max_memory as f64;
            (1.0 - (stats.memory_usage as f64 / max_memory)).max(0.0)
        } else {
            1.0 // No memory usage is perfect efficiency
        };

        // Weighted fitness calculation

        (normalized_xp * self.xp_weight)
            + (success_rate * self.success_rate_weight)
            + (processing_speed * self.processing_speed_weight)
            + (memory_efficiency * self.memory_efficiency_weight)
    }

    fn calculate_xp_reward(
        &self,
        fitness_improvement: f64,
        _evolution_metrics: &EvolutionMetrics,
    ) -> u64 {
        // Base evolution XP reward
        let base_reward = 100u64;

        // Bonus for significant fitness improvement
        let improvement_bonus = if fitness_improvement > 0.1 {
            (fitness_improvement * 200.0) as u64
        } else {
            0
        };

        base_reward + improvement_bonus
    }

    async fn should_evolve(&self, agent: &Agent) -> bool {
        agent.check_evolution_readiness().await
    }
}

/// Level-based fitness function that primarily uses agent level for fitness
pub struct LevelBasedFitnessFunction {
    pub level_multiplier: f64,
    pub xp_bonus_factor: f64,
}

impl Default for LevelBasedFitnessFunction {
    fn default() -> Self {
        Self {
            level_multiplier: 0.1,
            xp_bonus_factor: 0.0001,
        }
    }
}

impl XPFitnessFunction for LevelBasedFitnessFunction {
    async fn evaluate_agent_fitness(&self, agent: &Agent) -> f64 {
        let stats = agent.stats().await;

        // Primary fitness based on level
        let level_fitness = stats.level as f64 * self.level_multiplier;

        // Bonus for XP within current level
        let xp_bonus = stats.current_xp as f64 * self.xp_bonus_factor;

        level_fitness + xp_bonus
    }

    fn calculate_xp_reward(
        &self,
        fitness_improvement: f64,
        _evolution_metrics: &EvolutionMetrics,
    ) -> u64 {
        // Simple reward based on fitness improvement
        (fitness_improvement * 500.0).max(50.0) as u64
    }

    async fn should_evolve(&self, agent: &Agent) -> bool {
        agent.check_evolution_readiness().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use stratoswarm_agent_core::agent::{Agent, AgentConfig};
    use tokio;

    async fn create_test_agent_with_stats(
        xp: u64,
        goals_completed: u64,
        goals_failed: u64,
        memory_usage: usize,
    ) -> Agent {
        let config = AgentConfig {
            name: "test_agent".to_string(),
            max_memory: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        };

        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();

        // Award XP
        if xp > 0 {
            agent
                .award_xp(xp, "Test XP".to_string(), "test".to_string())
                .await
                .unwrap();
        }

        // Simulate goal completions
        for _ in 0..goals_completed {
            agent.update_goal_stats(true, Duration::from_secs(30)).await;
        }

        for _ in 0..goals_failed {
            agent
                .update_goal_stats(false, Duration::from_secs(60))
                .await;
        }

        // Note: In real use, memory usage would be tracked by the agent automatically
        // For testing purposes, we'll simulate it differently
        if memory_usage > 0 {
            // We can't directly set memory usage, so we'll note it for reference
            // In a real scenario, memory usage is tracked by the agent's operations
        }

        agent
    }

    #[tokio::test]
    async fn test_agent_performance_fitness_function_creation() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        assert_eq!(fitness_fn.xp_weight, 0.4);
        assert_eq!(fitness_fn.success_rate_weight, 0.3);
        assert_eq!(fitness_fn.processing_speed_weight, 0.2);
        assert_eq!(fitness_fn.memory_efficiency_weight, 0.1);
        assert_eq!(fitness_fn.min_xp_threshold, 50);
    }

    #[tokio::test]
    async fn test_agent_performance_fitness_function_custom_weights() {
        let fitness_fn = AgentPerformanceFitnessFunction::new(0.5, 0.2, 0.2, 0.1, 100);
        assert_eq!(fitness_fn.xp_weight, 0.5);
        assert_eq!(fitness_fn.min_xp_threshold, 100);
    }

    #[tokio::test]
    #[should_panic(expected = "Weights must sum to 1.0")]
    fn test_agent_performance_fitness_function_invalid_weights() {
        AgentPerformanceFitnessFunction::new(0.5, 0.5, 0.5, 0.5, 100);
    }

    #[tokio::test]
    async fn test_evaluate_agent_fitness_below_threshold() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(25, 0, 0, 0).await; // Below 50 XP threshold

        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;
        assert_eq!(fitness, 0.0);
    }

    #[tokio::test]
    async fn test_evaluate_agent_fitness_basic() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(100, 5, 1, 0).await; // 100 XP, 5 success, 1 failure

        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Should be positive and reasonable
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);

        // XP component: 100/25000 = 0.004 * 0.4 = 0.0016
        // Success rate: 5/6 = 0.833 * 0.3 = 0.25
        // Memory efficiency: 1.0 (no memory usage) * 0.1 = 0.1
        // Processing speed depends on execution time but should be positive
        let expected_min_fitness = 0.0016 + 0.25 + 0.1; // ~0.3516 + processing speed
        assert!(fitness >= expected_min_fitness); // Should be at least this much
    }

    #[tokio::test]
    async fn test_evaluate_agent_fitness_high_xp() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(25000, 10, 0, 0).await; // Max XP, perfect success rate, no memory usage

        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Should be high fitness (close to 1.0)
        // XP: 25000/25000 = 1.0 * 0.4 = 0.4
        // Success rate: 10/10 = 1.0 * 0.3 = 0.3
        // Memory efficiency: 1.0 * 0.1 = 0.1
        // Processing speed varies but should be positive
        assert!(fitness >= 0.8); // Should be quite high
        assert!(fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_calculate_xp_reward() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let metrics = EvolutionMetrics {
            avg_completion_time: Duration::from_secs(30),
            success_rate: 0.8,
            memory_efficiency: 0.7,
            processing_speed: 1.2,
        };

        // Small improvement
        let reward1 = fitness_fn.calculate_xp_reward(0.05, &metrics);
        assert_eq!(reward1, 100); // Base reward only

        // Significant improvement
        let reward2 = fitness_fn.calculate_xp_reward(0.2, &metrics);
        assert_eq!(reward2, 100 + 40); // Base + bonus (0.2 * 200)
    }

    #[tokio::test]
    async fn test_should_evolve() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(50, 0, 0, 0).await; // Below evolution threshold

        let should_evolve = fitness_fn.should_evolve(&agent).await;
        assert!(!should_evolve);

        // Award enough XP to reach evolution threshold
        agent
            .award_xp(50, "More XP".to_string(), "test".to_string())
            .await
            .unwrap(); // Total 100 XP
        let should_evolve = fitness_fn.should_evolve(&agent).await;
        assert!(should_evolve);
    }

    #[tokio::test]
    async fn test_level_based_fitness_function() {
        let fitness_fn = LevelBasedFitnessFunction::default();

        // Test with level 1 agent
        let agent = create_test_agent_with_stats(50, 0, 0, 0).await;
        let fitness1 = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Level 1: 1 * 0.1 = 0.1, XP bonus: 50 * 0.0001 = 0.005, total = 0.105
        assert!((fitness1 - 0.105).abs() < 0.001);

        // Test with higher level agent
        let agent2 = create_test_agent_with_stats(1000, 0, 0, 0).await; // Should be level 5
        let fitness2 = fitness_fn.evaluate_agent_fitness(&agent2).await;
        assert!(fitness2 > fitness1);
    }

    #[tokio::test]
    async fn test_level_based_calculate_xp_reward() {
        let fitness_fn = LevelBasedFitnessFunction::default();
        let metrics = EvolutionMetrics {
            avg_completion_time: Duration::from_secs(30),
            success_rate: 0.8,
            memory_efficiency: 0.7,
            processing_speed: 1.2,
        };

        let reward = fitness_fn.calculate_xp_reward(0.2, &metrics);
        assert_eq!(reward, 100); // 0.2 * 500 = 100

        let small_reward = fitness_fn.calculate_xp_reward(0.05, &metrics);
        assert_eq!(small_reward, 50); // min of 50
    }

    #[tokio::test]
    async fn test_agent_fitness_score_serialization() {
        let metrics = EvolutionMetrics {
            avg_completion_time: Duration::from_secs(60),
            success_rate: 0.8,
            memory_efficiency: 0.7,
            processing_speed: 1.0,
        };

        let score = AgentFitnessScore {
            fitness: 0.85,
            agent_id: stratoswarm_agent_core::agent::AgentId::new(),
            xp_contribution: 150,
            level: 3,
            performance_metrics: metrics,
        };

        // Test serialization
        let json = serde_json::to_string(&score).unwrap();
        let deserialized: AgentFitnessScore = serde_json::from_str(&json).unwrap();

        assert_eq!(score.fitness, deserialized.fitness);
        assert_eq!(score.xp_contribution, deserialized.xp_contribution);
        assert_eq!(score.level, deserialized.level);
    }

    #[tokio::test]
    async fn test_fitness_with_no_goals() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(200, 0, 0, 0).await; // No goals processed

        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Should still get XP and memory efficiency components
        // XP: 200/25000 = 0.008 * 0.4 = 0.0032
        // Success rate: 0 (no goals) * 0.3 = 0
        // Processing speed: 0 (no goals) * 0.2 = 0
        // Memory efficiency: 1.0 * 0.1 = 0.1
        let expected = 0.0032 + 0.0 + 0.0 + 0.1;
        assert!((fitness - expected).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_fitness_with_perfect_memory_efficiency() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();
        let agent = create_test_agent_with_stats(200, 1, 0, 0).await; // No memory usage

        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Should have perfect memory efficiency (1.0) contributing to fitness
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);

        // XP: 200/25000 * 0.4 = 0.0032, Success: 1.0 * 0.3 = 0.3, Memory: 1.0 * 0.1 = 0.1, Processing: >0
        let expected_min = 0.0032 + 0.3 + 0.1; // Should be at least 0.4032
        assert!(fitness >= expected_min);
    }

    #[tokio::test]
    async fn test_processing_speed_calculation() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();

        // Create agent with fast processing (30 seconds per goal)
        let agent = create_test_agent_with_stats(200, 2, 0, 0).await;
        let fitness1 = fitness_fn.evaluate_agent_fitness(&agent).await;

        // Create agent with slower processing by simulating longer goal execution
        let agent2 = create_test_agent_with_stats(200, 0, 0, 0).await;
        // Simulate slower processing by using longer execution times
        agent2
            .update_goal_stats(true, Duration::from_secs(120))
            .await; // 120 seconds per goal
        agent2
            .update_goal_stats(true, Duration::from_secs(120))
            .await; // 120 seconds per goal
        let fitness2 = fitness_fn.evaluate_agent_fitness(&agent2).await;

        // Agent with faster processing should have higher fitness
        assert!(fitness1 > fitness2);
    }

    #[tokio::test]
    async fn test_fitness_boundary_conditions() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();

        // Test with agent exactly at XP threshold
        let agent = create_test_agent_with_stats(50, 0, 0, 0).await; // Exactly at threshold
        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;
        assert!(fitness > 0.0);

        // Test with agent just below threshold
        let agent2 = create_test_agent_with_stats(49, 10, 0, 0).await; // Below threshold despite good performance
        let fitness2 = fitness_fn.evaluate_agent_fitness(&agent2).await;
        assert_eq!(fitness2, 0.0);
    }

    #[tokio::test]
    async fn test_comprehensive_fitness_evaluation() {
        let fitness_fn = AgentPerformanceFitnessFunction::default();

        // Create high-performing agent (high XP, high success rate)
        let high_perf_agent = create_test_agent_with_stats(5000, 20, 1, 0).await;
        let high_fitness = fitness_fn.evaluate_agent_fitness(&high_perf_agent).await;

        // Create low-performing agent (low XP, poor success rate)
        let low_perf_agent = create_test_agent_with_stats(100, 1, 10, 0).await;
        let low_fitness = fitness_fn.evaluate_agent_fitness(&low_perf_agent).await;

        // High-performing agent should have significantly higher fitness
        assert!(high_fitness > low_fitness);
        assert!(high_fitness > 0.4); // Should be quite high due to good XP and success rate
        assert!(low_fitness < 0.2); // Should be low due to poor success rate
    }
}
