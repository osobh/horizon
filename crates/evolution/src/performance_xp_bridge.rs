//! Performance metrics to XP conversion system
//! 
//! This module provides sophisticated conversion from various performance metrics
//! to XP rewards, enabling fine-grained tracking and rewarding of agent improvements.

use crate::EvolutionError;
use exorust_agent_core::agent::{Agent, AgentStats, EvolutionMetrics, EvolutionResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Performance metric types that can be tracked and converted to XP
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetricType {
    /// Goal completion success rate
    SuccessRate,
    /// Average task completion time
    CompletionSpeed,
    /// Memory usage efficiency
    MemoryEfficiency,
    /// GPU utilization optimization
    GpuEfficiency,
    /// Error recovery rate
    ErrorRecovery,
    /// Learning curve steepness
    LearningRate,
    /// Adaptation speed to new tasks
    AdaptationSpeed,
    /// Consistency across different contexts
    Consistency,
    /// Collaboration effectiveness with other agents
    CollaborationScore,
    /// Innovation and creative problem solving
    Innovation,
    /// Resource usage optimization
    ResourceOptimization,
    /// Fault tolerance and robustness
    FaultTolerance,
    /// Custom metric with string identifier
    Custom(String),
}

/// Performance measurement with trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub metric_type: PerformanceMetricType,
    pub current_value: f64,
    pub baseline_value: f64,
    pub historical_values: Vec<(chrono::DateTime<chrono::Utc>, f64)>,
    pub trend_direction: TrendDirection,
    pub improvement_rate: f64,
    pub statistical_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// XP conversion configuration for different metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricXPConfig {
    /// Base XP reward for any improvement
    pub base_reward: u64,
    /// Multiplier for the improvement magnitude
    pub improvement_multiplier: f64,
    /// Bonus for sustained improvement trends
    pub trend_bonus_multiplier: f64,
    /// Bonus for high confidence improvements
    pub confidence_bonus: u64,
    /// Minimum improvement required for any XP
    pub improvement_threshold: f64,
    /// Maximum XP that can be awarded for this metric per evaluation
    pub max_xp_per_evaluation: u64,
}

impl Default for MetricXPConfig {
    fn default() -> Self {
        Self {
            base_reward: 25,
            improvement_multiplier: 100.0,
            trend_bonus_multiplier: 1.5,
            confidence_bonus: 10,
            improvement_threshold: 0.01,
            max_xp_per_evaluation: 200,
        }
    }
}

/// Comprehensive performance to XP conversion engine
#[derive(Debug, Clone)]
pub struct PerformanceXPConverter {
    /// Configuration for different metric types
    metric_configs: HashMap<PerformanceMetricType, MetricXPConfig>,
    /// Historical performance tracking
    performance_history: HashMap<exorust_agent_core::agent::AgentId, Vec<PerformanceMeasurement>>,
    /// Global performance multipliers
    global_multiplier: f64,
    /// Evaluation window for trend analysis
    evaluation_window_size: usize,
}

impl Default for PerformanceXPConverter {
    fn default() -> Self {
        let mut metric_configs = HashMap::new();
        
        // Configure different metric types with appropriate rewards
        metric_configs.insert(PerformanceMetricType::SuccessRate, MetricXPConfig {
            base_reward: 40,
            improvement_multiplier: 150.0,
            trend_bonus_multiplier: 1.8,
            confidence_bonus: 15,
            improvement_threshold: 0.02,
            max_xp_per_evaluation: 250,
        });
        
        metric_configs.insert(PerformanceMetricType::CompletionSpeed, MetricXPConfig {
            base_reward: 30,
            improvement_multiplier: 120.0,
            trend_bonus_multiplier: 1.6,
            confidence_bonus: 12,
            improvement_threshold: 0.05,
            max_xp_per_evaluation: 200,
        });
        
        metric_configs.insert(PerformanceMetricType::MemoryEfficiency, MetricXPConfig {
            base_reward: 35,
            improvement_multiplier: 140.0,
            trend_bonus_multiplier: 1.7,
            confidence_bonus: 18,
            improvement_threshold: 0.03,
            max_xp_per_evaluation: 220,
        });
        
        metric_configs.insert(PerformanceMetricType::GpuEfficiency, MetricXPConfig {
            base_reward: 50,
            improvement_multiplier: 200.0,
            trend_bonus_multiplier: 2.0,
            confidence_bonus: 25,
            improvement_threshold: 0.02,
            max_xp_per_evaluation: 300,
        });
        
        metric_configs.insert(PerformanceMetricType::ErrorRecovery, MetricXPConfig {
            base_reward: 45,
            improvement_multiplier: 180.0,
            trend_bonus_multiplier: 1.9,
            confidence_bonus: 20,
            improvement_threshold: 0.01,
            max_xp_per_evaluation: 280,
        });
        
        metric_configs.insert(PerformanceMetricType::CollaborationScore, MetricXPConfig {
            base_reward: 55,
            improvement_multiplier: 220.0,
            trend_bonus_multiplier: 2.2,
            confidence_bonus: 30,
            improvement_threshold: 0.04,
            max_xp_per_evaluation: 350,
        });

        Self {
            metric_configs,
            performance_history: HashMap::new(),
            global_multiplier: 1.0,
            evaluation_window_size: 10,
        }
    }
}

impl PerformanceXPConverter {
    /// Create a new converter with custom configuration
    pub fn new(
        metric_configs: HashMap<PerformanceMetricType, MetricXPConfig>,
        global_multiplier: f64,
        evaluation_window_size: usize,
    ) -> Self {
        Self {
            metric_configs,
            performance_history: HashMap::new(),
            global_multiplier,
            evaluation_window_size,
        }
    }

    /// Record a performance measurement for an agent
    pub fn record_performance(
        &mut self,
        agent_id: exorust_agent_core::agent::AgentId,
        measurement: PerformanceMeasurement,
    ) {
        self.performance_history
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(measurement);
        
        // Keep history within reasonable bounds
        let history = self.performance_history.get_mut(&agent_id).unwrap();
        if history.len() > self.evaluation_window_size * 3 {
            history.drain(0..history.len() - self.evaluation_window_size * 2);
        }
    }

    /// Convert agent performance metrics to XP rewards
    pub async fn convert_agent_performance_to_xp(
        &mut self,
        agent: &Agent,
    ) -> Result<PerformanceXPBreakdown, EvolutionError> {
        let agent_id = agent.id();
        let stats = agent.stats().await;
        
        // Generate performance measurements from agent stats
        let measurements = self.generate_measurements_from_stats(&stats);
        
        // Record new measurements
        for measurement in measurements {
            self.record_performance(agent_id, measurement);
        }
        
        // Calculate XP rewards for each metric
        let mut breakdown = PerformanceXPBreakdown::default();
        
        if let Some(history) = self.performance_history.get(&agent_id) {
            for measurement in history.iter().rev().take(self.evaluation_window_size) {
                let xp_reward = self.calculate_metric_xp_reward(measurement);
                
                breakdown.metric_rewards.insert(
                    measurement.metric_type.clone(),
                    xp_reward,
                );
                breakdown.total_xp += xp_reward.total_xp;
            }
        }
        
        // Apply global multiplier
        breakdown.total_xp = (breakdown.total_xp as f64 * self.global_multiplier) as u64;
        
        // Apply agent-level bonuses
        let level_bonus = self.calculate_level_bonus(&stats);
        breakdown.level_bonus = level_bonus;
        breakdown.total_xp += level_bonus;
        
        // Apply consistency bonuses
        let consistency_bonus = self.calculate_consistency_bonus(agent_id);
        breakdown.consistency_bonus = consistency_bonus;
        breakdown.total_xp += consistency_bonus;

        Ok(breakdown)
    }

    /// Generate performance measurements from agent statistics
    fn generate_measurements_from_stats(&self, stats: &AgentStats) -> Vec<PerformanceMeasurement> {
        let mut measurements = Vec::new();
        let now = chrono::Utc::now();
        
        // Success rate measurement
        if stats.goals_processed > 0 {
            let success_rate = stats.goals_succeeded as f64 / stats.goals_processed as f64;
            measurements.push(PerformanceMeasurement {
                metric_type: PerformanceMetricType::SuccessRate,
                current_value: success_rate,
                baseline_value: 0.5, // Assume 50% baseline
                historical_values: vec![(now, success_rate)],
                trend_direction: TrendDirection::Stable,
                improvement_rate: 0.0,
                statistical_confidence: if stats.goals_processed >= 10 { 0.8 } else { 0.5 },
            });
        }
        
        // Completion speed measurement
        if stats.goals_processed > 0 {
            let avg_time = stats.total_execution_time.as_secs_f64() / stats.goals_processed as f64;
            let speed_score = (120.0 - avg_time) / 110.0; // Normalize to 0-1 scale
            measurements.push(PerformanceMeasurement {
                metric_type: PerformanceMetricType::CompletionSpeed,
                current_value: speed_score.max(0.0),
                baseline_value: 0.5,
                historical_values: vec![(now, speed_score.max(0.0))],
                trend_direction: TrendDirection::Stable,
                improvement_rate: 0.0,
                statistical_confidence: if stats.goals_processed >= 5 { 0.7 } else { 0.4 },
            });
        }
        
        // Memory efficiency (simplified)
        let memory_efficiency = if stats.memory_usage > 0 {
            // Assuming we can calculate efficiency from usage patterns
            1.0 - (stats.memory_usage as f64 / (1024.0 * 1024.0 * 1024.0)) // Normalize to 1GB
        } else {
            1.0
        };
        
        measurements.push(PerformanceMeasurement {
            metric_type: PerformanceMetricType::MemoryEfficiency,
            current_value: memory_efficiency.max(0.0).min(1.0),
            baseline_value: 0.7,
            historical_values: vec![(now, memory_efficiency.max(0.0).min(1.0))],
            trend_direction: TrendDirection::Stable,
            improvement_rate: 0.0,
            statistical_confidence: 0.6,
        });
        
        // Learning rate based on XP gain patterns
        if stats.xp_history.len() >= 3 {
            let recent_xp_gains: Vec<u64> = stats.xp_history.iter()
                .take(5)
                .map(|entry| entry.amount)
                .collect();
            
            let learning_score = if recent_xp_gains.len() >= 2 {
                // Check for increasing gains (learning improvement)
                let avg_gain = recent_xp_gains.iter().sum::<u64>() as f64 / recent_xp_gains.len() as f64;
                (avg_gain / 100.0).min(1.0) // Normalize
            } else {
                0.5
            };
            
            measurements.push(PerformanceMeasurement {
                metric_type: PerformanceMetricType::LearningRate,
                current_value: learning_score,
                baseline_value: 0.3,
                historical_values: vec![(now, learning_score)],
                trend_direction: TrendDirection::Stable,
                improvement_rate: 0.0,
                statistical_confidence: if stats.xp_history.len() >= 10 { 0.8 } else { 0.5 },
            });
        }

        measurements
    }

    /// Calculate XP reward for a specific performance measurement
    fn calculate_metric_xp_reward(&self, measurement: &PerformanceMeasurement) -> MetricXPReward {
        let config = self.metric_configs.get(&measurement.metric_type)
            .unwrap_or(&MetricXPConfig::default());
        
        let improvement = measurement.current_value - measurement.baseline_value;
        
        if improvement < config.improvement_threshold {
            return MetricXPReward {
                metric_type: measurement.metric_type.clone(),
                base_xp: 0,
                improvement_xp: 0,
                trend_bonus: 0,
                confidence_bonus: 0,
                total_xp: 0,
            };
        }
        
        let base_xp = config.base_reward;
        let improvement_xp = (improvement * config.improvement_multiplier) as u64;
        
        let trend_bonus = match measurement.trend_direction {
            TrendDirection::Improving => (base_xp as f64 * (config.trend_bonus_multiplier - 1.0)) as u64,
            TrendDirection::Stable => base_xp / 4,
            _ => 0,
        };
        
        let confidence_bonus = if measurement.statistical_confidence > 0.7 {
            config.confidence_bonus
        } else {
            0
        };
        
        let total_xp = (base_xp + improvement_xp + trend_bonus + confidence_bonus)
            .min(config.max_xp_per_evaluation);
        
        MetricXPReward {
            metric_type: measurement.metric_type.clone(),
            base_xp,
            improvement_xp,
            trend_bonus,
            confidence_bonus,
            total_xp,
        }
    }

    /// Calculate level-based bonus
    fn calculate_level_bonus(&self, stats: &AgentStats) -> u64 {
        // Higher level agents get bonus for maintaining performance
        if stats.level >= 5 {
            stats.level as u64 * 5
        } else {
            0
        }
    }

    /// Calculate consistency bonus for maintaining good performance
    fn calculate_consistency_bonus(&self, agent_id: exorust_agent_core::agent::AgentId) -> u64 {
        if let Some(history) = self.performance_history.get(&agent_id) {
            if history.len() >= 5 {
                // Check for consistent performance across different metrics
                let consistent_metrics = history.iter()
                    .filter(|m| m.statistical_confidence > 0.6)
                    .count();
                
                if consistent_metrics >= 3 {
                    50 // Consistency bonus
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Set configuration for a specific metric type
    pub fn set_metric_config(&mut self, metric_type: PerformanceMetricType, config: MetricXPConfig) {
        self.metric_configs.insert(metric_type, config);
    }

    /// Get current performance trends for an agent
    pub fn get_performance_trends(
        &self,
        agent_id: &exorust_agent_core::agent::AgentId,
    ) -> Option<&Vec<PerformanceMeasurement>> {
        self.performance_history.get(agent_id)
    }

    /// Calculate evolution-specific XP rewards
    pub fn calculate_evolution_performance_xp(
        &self,
        evolution_result: &EvolutionResult,
    ) -> u64 {
        let previous_metrics = &evolution_result.previous_metrics;
        let new_metrics = &evolution_result.new_metrics;
        
        let mut total_xp = 0u64;
        
        // Success rate improvement
        let success_improvement = new_metrics.success_rate - previous_metrics.success_rate;
        if success_improvement > 0.0 {
            total_xp += (success_improvement * 200.0) as u64;
        }
        
        // Processing speed improvement
        let speed_improvement = new_metrics.processing_speed - previous_metrics.processing_speed;
        if speed_improvement > 0.0 {
            total_xp += (speed_improvement * 150.0) as u64;
        }
        
        // Memory efficiency improvement
        let memory_improvement = new_metrics.memory_efficiency - previous_metrics.memory_efficiency;
        if memory_improvement > 0.0 {
            total_xp += (memory_improvement * 180.0) as u64;
        }
        
        // Completion time improvement (inverse relationship)
        let time_improvement = previous_metrics.avg_completion_time.as_secs_f64() - 
                              new_metrics.avg_completion_time.as_secs_f64();
        if time_improvement > 0.0 {
            total_xp += (time_improvement * 2.0) as u64; // 2 XP per second improved
        }
        
        // Level jump bonus
        let level_jump = evolution_result.new_level - evolution_result.previous_level;
        if level_jump > 1 {
            total_xp += level_jump as u64 * 100; // 100 XP per level jumped
        }
        
        // Cap the reward
        total_xp.min(500)
    }
}

/// XP reward breakdown for a specific performance metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricXPReward {
    pub metric_type: PerformanceMetricType,
    pub base_xp: u64,
    pub improvement_xp: u64,
    pub trend_bonus: u64,
    pub confidence_bonus: u64,
    pub total_xp: u64,
}

/// Complete performance-to-XP conversion breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceXPBreakdown {
    pub metric_rewards: HashMap<PerformanceMetricType, MetricXPReward>,
    pub level_bonus: u64,
    pub consistency_bonus: u64,
    pub total_xp: u64,
}

impl PerformanceXPBreakdown {
    /// Get a detailed summary of the XP breakdown
    pub fn detailed_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Performance XP Breakdown:\n");
        
        for (metric_type, reward) in &self.metric_rewards {
            if reward.total_xp > 0 {
                summary.push_str(&format!(
                    "  {:?}: {} XP (base: {}, improvement: {}, bonus: {})\n",
                    metric_type,
                    reward.total_xp,
                    reward.base_xp,
                    reward.improvement_xp,
                    reward.trend_bonus + reward.confidence_bonus
                ));
            }
        }
        
        if self.level_bonus > 0 {
            summary.push_str(&format!("  Level Bonus: {} XP\n", self.level_bonus));
        }
        
        if self.consistency_bonus > 0 {
            summary.push_str(&format!("  Consistency Bonus: {} XP\n", self.consistency_bonus));
        }
        
        summary.push_str(&format!("  Total: {} XP", self.total_xp));
        summary
    }
    
    /// Get a brief summary of the XP gained
    pub fn brief_summary(&self) -> String {
        let metric_count = self.metric_rewards.values().filter(|r| r.total_xp > 0).count();
        format!(
            "{} XP from {} performance improvements{}{}",
            self.total_xp,
            metric_count,
            if self.level_bonus > 0 { format!(" (+{} level bonus)", self.level_bonus) } else { String::new() },
            if self.consistency_bonus > 0 { format!(" (+{} consistency)", self.consistency_bonus) } else { String::new() }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use exorust_agent_core::agent::{Agent, AgentConfig};

    async fn create_test_agent_with_performance(
        goals_completed: u64,
        goals_failed: u64,
        execution_time_secs: u64,
    ) -> Agent {
        let config = AgentConfig {
            name: "performance_test_agent".to_string(),
            ..Default::default()
        };
        
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Simulate goal completions
        for _ in 0..goals_completed {
            agent.update_goal_stats(true, Duration::from_secs(execution_time_secs)).await;
        }
        
        for _ in 0..goals_failed {
            agent.update_goal_stats(false, Duration::from_secs(execution_time_secs + 20)).await;
        }
        
        agent
    }

    #[tokio::test]
    async fn test_performance_xp_converter_creation() {
        let converter = PerformanceXPConverter::default();
        
        assert!(converter.metric_configs.contains_key(&PerformanceMetricType::SuccessRate));
        assert!(converter.metric_configs.contains_key(&PerformanceMetricType::CompletionSpeed));
        assert!(converter.metric_configs.contains_key(&PerformanceMetricType::MemoryEfficiency));
        assert_eq!(converter.global_multiplier, 1.0);
    }

    #[tokio::test]
    async fn test_performance_measurement_generation() {
        let converter = PerformanceXPConverter::default();
        let agent = create_test_agent_with_performance(8, 2, 30).await;
        let stats = agent.stats().await;
        
        let measurements = converter.generate_measurements_from_stats(&stats);
        
        assert!(!measurements.is_empty());
        assert!(measurements.iter().any(|m| m.metric_type == PerformanceMetricType::SuccessRate));
        assert!(measurements.iter().any(|m| m.metric_type == PerformanceMetricType::CompletionSpeed));
    }

    #[tokio::test]
    async fn test_xp_conversion() {
        let mut converter = PerformanceXPConverter::default();
        let agent = create_test_agent_with_performance(10, 0, 25).await; // Perfect performance, fast execution
        
        let breakdown = converter.convert_agent_performance_to_xp(&agent).await.unwrap();
        
        assert!(breakdown.total_xp > 0);
        assert!(!breakdown.metric_rewards.is_empty());
        
        // Should have rewards for success rate and speed
        assert!(breakdown.metric_rewards.contains_key(&PerformanceMetricType::SuccessRate));
        assert!(breakdown.metric_rewards.contains_key(&PerformanceMetricType::CompletionSpeed));
    }

    #[tokio::test]
    async fn test_poor_performance_low_xp() {
        let mut converter = PerformanceXPConverter::default();
        let agent = create_test_agent_with_performance(1, 9, 180).await; // Poor performance, slow execution
        
        let breakdown = converter.convert_agent_performance_to_xp(&agent).await.unwrap();
        
        // Should get minimal XP for poor performance
        assert!(breakdown.total_xp < 100);
    }

    #[tokio::test]
    async fn test_evolution_performance_xp() {
        let converter = PerformanceXPConverter::default();
        
        let evolution_result = EvolutionResult {
            previous_level: 2,
            new_level: 3,
            xp_at_evolution: 250,
            evolution_timestamp: chrono::Utc::now(),
            capabilities_gained: vec!["optimization".to_string()],
            previous_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(60),
                success_rate: 0.6,
                memory_efficiency: 0.6,
                processing_speed: 1.0,
            },
            new_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(45),
                success_rate: 0.8,
                memory_efficiency: 0.8,
                processing_speed: 1.3,
            },
        };
        
        let xp = converter.calculate_evolution_performance_xp(&evolution_result);
        assert!(xp > 0);
        // Should get XP for all improvements: success rate, speed, memory, time
        assert!(xp > 100); // Significant improvements should yield substantial XP
    }

    #[tokio::test]
    async fn test_consistency_bonus() {
        let mut converter = PerformanceXPConverter::default();
        let agent = create_test_agent_with_performance(5, 1, 40).await;
        
        // Record multiple consistent measurements
        let agent_id = agent.id();
        for i in 0..6 {
            let measurement = PerformanceMeasurement {
                metric_type: PerformanceMetricType::SuccessRate,
                current_value: 0.8 + i as f64 * 0.01, // Slightly improving
                baseline_value: 0.5,
                historical_values: vec![(chrono::Utc::now(), 0.8)],
                trend_direction: TrendDirection::Improving,
                improvement_rate: 0.05,
                statistical_confidence: 0.8,
            };
            converter.record_performance(agent_id, measurement);
        }
        
        let breakdown = converter.convert_agent_performance_to_xp(&agent).await.unwrap();
        assert!(breakdown.consistency_bonus > 0);
    }

    #[tokio::test]
    async fn test_performance_breakdown_summary() {
        let mut breakdown = PerformanceXPBreakdown::default();
        breakdown.total_xp = 250;
        breakdown.level_bonus = 25;
        breakdown.consistency_bonus = 50;
        
        breakdown.metric_rewards.insert(
            PerformanceMetricType::SuccessRate,
            MetricXPReward {
                metric_type: PerformanceMetricType::SuccessRate,
                base_xp: 40,
                improvement_xp: 100,
                trend_bonus: 20,
                confidence_bonus: 15,
                total_xp: 175,
            }
        );
        
        let summary = breakdown.detailed_summary();
        assert!(summary.contains("SuccessRate"));
        assert!(summary.contains("250 XP"));
        assert!(summary.contains("Level Bonus"));
        assert!(summary.contains("Consistency Bonus"));
        
        let brief = breakdown.brief_summary();
        assert!(brief.contains("250 XP"));
        assert!(brief.contains("1 performance"));
    }

    #[tokio::test]
    async fn test_custom_metric_configuration() {
        let mut converter = PerformanceXPConverter::default();
        
        let custom_config = MetricXPConfig {
            base_reward: 100,
            improvement_multiplier: 300.0,
            trend_bonus_multiplier: 2.5,
            confidence_bonus: 50,
            improvement_threshold: 0.1,
            max_xp_per_evaluation: 500,
        };
        
        converter.set_metric_config(
            PerformanceMetricType::Custom("innovation_score".to_string()),
            custom_config.clone(),
        );
        
        assert_eq!(
            converter.metric_configs
                .get(&PerformanceMetricType::Custom("innovation_score".to_string()))
                .unwrap()
                .base_reward,
            100
        );
    }
}