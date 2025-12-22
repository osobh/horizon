//! XP reward calculation system for evolution outcomes

use crate::EvolutionError;
use exorust_agent_core::agent::{Agent, EvolutionResult, EvolutionMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// XP reward category with base amounts and multipliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XPRewardCategory {
    pub base_reward: u64,
    pub level_multiplier: f64,
    pub performance_multiplier: f64,
    pub improvement_threshold: f64,
}

impl Default for XPRewardCategory {
    fn default() -> Self {
        Self {
            base_reward: 100,
            level_multiplier: 1.1,
            performance_multiplier: 1.0,
            improvement_threshold: 0.05,
        }
    }
}

/// Comprehensive XP reward calculator for evolution outcomes
#[derive(Debug, Clone)]
pub struct EvolutionXPRewardCalculator {
    /// Reward categories mapped by type
    pub reward_categories: HashMap<String, XPRewardCategory>,
    /// Global multiplier for all rewards
    pub global_multiplier: f64,
    /// Maximum reward cap per evolution
    pub max_reward_per_evolution: u64,
    /// Bonus for consecutive successful evolutions
    pub streak_bonus_multiplier: f64,
}

impl Default for EvolutionXPRewardCalculator {
    fn default() -> Self {
        let mut reward_categories = HashMap::new();
        
        // Evolution outcome rewards
        reward_categories.insert("level_up".to_string(), XPRewardCategory {
            base_reward: 200,
            level_multiplier: 1.2,
            performance_multiplier: 1.5,
            improvement_threshold: 0.0,
        });
        
        reward_categories.insert("capability_gained".to_string(), XPRewardCategory {
            base_reward: 150,
            level_multiplier: 1.1,
            performance_multiplier: 1.3,
            improvement_threshold: 0.0,
        });
        
        reward_categories.insert("performance_improvement".to_string(), XPRewardCategory {
            base_reward: 100,
            level_multiplier: 1.05,
            performance_multiplier: 2.0,
            improvement_threshold: 0.1,
        });
        
        reward_categories.insert("speed_improvement".to_string(), XPRewardCategory {
            base_reward: 80,
            level_multiplier: 1.1,
            performance_multiplier: 1.8,
            improvement_threshold: 0.05,
        });
        
        reward_categories.insert("memory_efficiency_gain".to_string(), XPRewardCategory {
            base_reward: 75,
            level_multiplier: 1.0,
            performance_multiplier: 1.5,
            improvement_threshold: 0.08,
        });
        
        reward_categories.insert("success_rate_improvement".to_string(), XPRewardCategory {
            base_reward: 120,
            level_multiplier: 1.15,
            performance_multiplier: 1.7,
            improvement_threshold: 0.1,
        });

        Self {
            reward_categories,
            global_multiplier: 1.0,
            max_reward_per_evolution: 1000,
            streak_bonus_multiplier: 1.1,
        }
    }
}

impl EvolutionXPRewardCalculator {
    /// Create a new calculator with custom settings
    pub fn new(
        global_multiplier: f64,
        max_reward_per_evolution: u64,
        streak_bonus_multiplier: f64,
    ) -> Self {
        Self {
            global_multiplier,
            max_reward_per_evolution,
            streak_bonus_multiplier,
            ..Default::default()
        }
    }

    /// Calculate total XP reward for an evolution result
    pub async fn calculate_evolution_reward(
        &self,
        agent: &Agent,
        evolution_result: &EvolutionResult,
    ) -> Result<XPRewardBreakdown, EvolutionError> {
        let mut breakdown = XPRewardBreakdown::default();
        let agent_stats = agent.stats().await;

        // Level up reward
        if evolution_result.new_level > evolution_result.previous_level {
            let level_diff = evolution_result.new_level - evolution_result.previous_level;
            let reward = self.calculate_category_reward(
                "level_up",
                level_diff as f64,
                evolution_result.new_level,
                &evolution_result.new_metrics,
                &evolution_result.previous_metrics,
            );
            breakdown.level_up_reward = reward;
        }

        // Capability rewards
        let capability_reward = evolution_result.capabilities_gained.len() as f64 * 
            self.calculate_category_reward(
                "capability_gained",
                1.0,
                evolution_result.new_level,
                &evolution_result.new_metrics,
                &evolution_result.previous_metrics,
            );
        breakdown.capability_reward = capability_reward as u64;

        // Performance improvement rewards
        breakdown.performance_rewards = self.calculate_performance_rewards(
            evolution_result.new_level,
            &evolution_result.new_metrics,
            &evolution_result.previous_metrics,
        );

        // Calculate base total
        let base_total = breakdown.level_up_reward + 
            breakdown.capability_reward + 
            breakdown.performance_rewards.iter().sum::<u64>();

        // Apply global multiplier
        let multiplied_total = (base_total as f64 * self.global_multiplier) as u64;

        // Apply streak bonus if applicable
        let streak_bonus = self.calculate_streak_bonus(&agent_stats);
        let total_with_streak = (multiplied_total as f64 * streak_bonus) as u64;

        // Apply cap
        breakdown.total_reward = total_with_streak.min(self.max_reward_per_evolution);
        breakdown.capped = total_with_streak > self.max_reward_per_evolution;
        breakdown.streak_multiplier = streak_bonus;

        Ok(breakdown)
    }

    /// Calculate reward for a specific category
    fn calculate_category_reward(
        &self,
        category: &str,
        improvement_factor: f64,
        agent_level: u32,
        new_metrics: &EvolutionMetrics,
        _previous_metrics: &EvolutionMetrics,
    ) -> u64 {
        if let Some(category_config) = self.reward_categories.get(category) {
            let base = category_config.base_reward as f64;
            let level_bonus = base * (category_config.level_multiplier.powf(agent_level as f64 - 1.0) - 1.0);
            let performance_bonus = base * (new_metrics.success_rate * category_config.performance_multiplier);
            let improvement_bonus = base * (improvement_factor * 0.5);

            (base + level_bonus + performance_bonus + improvement_bonus) as u64
        } else {
            0
        }
    }

    /// Calculate performance-specific rewards
    fn calculate_performance_rewards(
        &self,
        agent_level: u32,
        new_metrics: &EvolutionMetrics,
        previous_metrics: &EvolutionMetrics,
    ) -> Vec<u64> {
        let mut rewards = Vec::new();

        // Speed improvement
        let speed_improvement = new_metrics.processing_speed - previous_metrics.processing_speed;
        if speed_improvement > 0.05 {
            let reward = self.calculate_category_reward(
                "speed_improvement",
                speed_improvement,
                agent_level,
                new_metrics,
                previous_metrics,
            );
            rewards.push(reward);
        }

        // Memory efficiency improvement
        let memory_improvement = new_metrics.memory_efficiency - previous_metrics.memory_efficiency;
        if memory_improvement > 0.08 {
            let reward = self.calculate_category_reward(
                "memory_efficiency_gain",
                memory_improvement,
                agent_level,
                new_metrics,
                previous_metrics,
            );
            rewards.push(reward);
        }

        // Success rate improvement
        let success_improvement = new_metrics.success_rate - previous_metrics.success_rate;
        if success_improvement > 0.1 {
            let reward = self.calculate_category_reward(
                "success_rate_improvement",
                success_improvement,
                agent_level,
                new_metrics,
                previous_metrics,
            );
            rewards.push(reward);
        }

        rewards
    }

    /// Calculate streak bonus based on recent successful evolutions
    fn calculate_streak_bonus(&self, _agent_stats: &exorust_agent_core::agent::AgentStats) -> f64 {
        // For now, return base multiplier
        // In a real implementation, we'd track evolution success streaks
        self.streak_bonus_multiplier
    }

    /// Add or update a reward category
    pub fn set_reward_category(&mut self, category: String, config: XPRewardCategory) {
        self.reward_categories.insert(category, config);
    }

    /// Get current reward category
    pub fn get_reward_category(&self, category: &str) -> Option<&XPRewardCategory> {
        self.reward_categories.get(category)
    }

    /// Calculate bonus XP for cross-agent collaboration
    pub fn calculate_collaboration_bonus(
        &self,
        collaborating_agents: &[&Agent],
        shared_improvement: f64,
    ) -> u64 {
        let base_collaboration_reward = 50;
        let agent_count_multiplier = (collaborating_agents.len() as f64).sqrt();
        let improvement_multiplier = shared_improvement + 1.0;

        (base_collaboration_reward as f64 * agent_count_multiplier * improvement_multiplier * self.global_multiplier) as u64
    }

    /// Calculate XP penalty for failed evolution attempts
    pub fn calculate_evolution_penalty(
        &self,
        agent_level: u32,
        failure_reason: &str,
    ) -> u64 {
        let base_penalty = match failure_reason {
            "insufficient_xp" => 0, // No penalty for not being ready
            "evolution_failed" => 20,
            "resource_exhaustion" => 30,
            "constraint_violation" => 40,
            _ => 10,
        };

        // Scale penalty with level (higher level agents lose more for failures)
        (base_penalty as f64 * (1.0 + agent_level as f64 * 0.1)) as u64
    }
}

/// Detailed breakdown of XP rewards for an evolution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XPRewardBreakdown {
    pub level_up_reward: u64,
    pub capability_reward: u64,
    pub performance_rewards: Vec<u64>,
    pub collaboration_bonus: u64,
    pub total_reward: u64,
    pub streak_multiplier: f64,
    pub capped: bool,
}

impl XPRewardBreakdown {
    /// Get a summary description of the rewards
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if self.level_up_reward > 0 {
            parts.push(format!("Level up: {}", self.level_up_reward));
        }

        if self.capability_reward > 0 {
            parts.push(format!("Capabilities: {}", self.capability_reward));
        }

        if !self.performance_rewards.is_empty() {
            let total_perf: u64 = self.performance_rewards.iter().sum();
            parts.push(format!("Performance: {}", total_perf));
        }

        if self.collaboration_bonus > 0 {
            parts.push(format!("Collaboration: {}", self.collaboration_bonus));
        }

        let summary = if parts.is_empty() {
            "No rewards".to_string()
        } else {
            parts.join(", ")
        };

        if self.capped {
            format!("{} (capped at {})", summary, self.total_reward)
        } else {
            format!("{} (total: {})", summary, self.total_reward)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use exorust_agent_core::agent::{Agent, AgentConfig};
    use std::time::Duration;

    async fn create_test_agent_with_level(level: u32) -> Agent {
        let config = AgentConfig {
            name: format!("test_agent_level_{}", level),
            ..Default::default()
        };
        
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Award enough XP to reach the specified level
        if level > 1 {
            let xp_needed = match level {
                2 => 100,
                3 => 250,
                4 => 500,
                5 => 1000,
                _ => 1000 + (level - 5) as u64 * 500,
            };
            agent.award_xp(xp_needed, "Test level setup".to_string(), "test".to_string()).await.unwrap();
        }
        
        agent
    }

    #[tokio::test]
    async fn test_reward_calculator_creation() {
        let calculator = EvolutionXPRewardCalculator::default();
        
        assert!(calculator.reward_categories.contains_key("level_up"));
        assert!(calculator.reward_categories.contains_key("capability_gained"));
        assert_eq!(calculator.global_multiplier, 1.0);
        assert_eq!(calculator.max_reward_per_evolution, 1000);
    }

    #[tokio::test]
    async fn test_level_up_reward_calculation() {
        let calculator = EvolutionXPRewardCalculator::default();
        let agent = create_test_agent_with_level(2).await;
        
        let evolution_result = EvolutionResult {
            previous_level: 2,
            new_level: 3,
            xp_at_evolution: 250,
            evolution_timestamp: chrono::Utc::now(),
            capabilities_gained: vec!["memory_management".to_string()],
            previous_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(60),
                success_rate: 0.7,
                memory_efficiency: 0.7,
                processing_speed: 1.0,
            },
            new_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(50),
                success_rate: 0.8,
                memory_efficiency: 0.8,
                processing_speed: 1.2,
            },
        };

        let breakdown = calculator.calculate_evolution_reward(&agent, &evolution_result).await.unwrap();
        
        assert!(breakdown.level_up_reward > 0);
        assert!(breakdown.capability_reward > 0);
        assert!(breakdown.total_reward > 0);
        assert!(!breakdown.capped); // Should not be capped for reasonable rewards
    }

    #[tokio::test]
    async fn test_performance_improvement_rewards() {
        let calculator = EvolutionXPRewardCalculator::default();
        let agent = create_test_agent_with_level(3).await;
        
        let evolution_result = EvolutionResult {
            previous_level: 3,
            new_level: 3, // No level change
            xp_at_evolution: 300,
            evolution_timestamp: chrono::Utc::now(),
            capabilities_gained: vec![],
            previous_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(80),
                success_rate: 0.6,
                memory_efficiency: 0.6,
                processing_speed: 1.0,
            },
            new_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(40),
                success_rate: 0.8,
                memory_efficiency: 0.8,
                processing_speed: 1.5, // Significant improvement
            },
        };

        let breakdown = calculator.calculate_evolution_reward(&agent, &evolution_result).await.unwrap();
        
        assert_eq!(breakdown.level_up_reward, 0); // No level change
        assert_eq!(breakdown.capability_reward, 0); // No new capabilities
        assert!(!breakdown.performance_rewards.is_empty()); // Should have performance rewards
        assert!(breakdown.total_reward > 0);
    }

    #[tokio::test]
    async fn test_reward_capping() {
        let mut calculator = EvolutionXPRewardCalculator::default();
        calculator.max_reward_per_evolution = 100; // Very low cap for testing
        
        let agent = create_test_agent_with_level(5).await;
        
        let evolution_result = EvolutionResult {
            previous_level: 5,
            new_level: 6,
            xp_at_evolution: 1000,
            evolution_timestamp: chrono::Utc::now(),
            capabilities_gained: vec!["advanced_analytics".to_string(), "gpu_acceleration".to_string()],
            previous_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(120),
                success_rate: 0.5,
                memory_efficiency: 0.5,
                processing_speed: 1.0,
            },
            new_metrics: EvolutionMetrics {
                avg_completion_time: Duration::from_secs(30),
                success_rate: 0.95,
                memory_efficiency: 0.9,
                processing_speed: 2.0,
            },
        };

        let breakdown = calculator.calculate_evolution_reward(&agent, &evolution_result).await.unwrap();
        
        assert!(breakdown.capped);
        assert_eq!(breakdown.total_reward, 100); // Should be capped
    }

    #[tokio::test]
    async fn test_collaboration_bonus() {
        let calculator = EvolutionXPRewardCalculator::default();
        
        let agent1 = create_test_agent_with_level(3).await;
        let agent2 = create_test_agent_with_level(4).await;
        let agent3 = create_test_agent_with_level(2).await;
        
        let collaborating_agents = vec![&agent1, &agent2, &agent3];
        let shared_improvement = 0.3; // 30% shared improvement
        
        let bonus = calculator.calculate_collaboration_bonus(&collaborating_agents, shared_improvement);
        
        assert!(bonus > 0);
        // Should scale with number of agents and improvement
    }

    #[tokio::test]
    async fn test_evolution_penalty() {
        let calculator = EvolutionXPRewardCalculator::default();
        
        let penalty1 = calculator.calculate_evolution_penalty(1, "evolution_failed");
        let penalty5 = calculator.calculate_evolution_penalty(5, "evolution_failed");
        let penalty_insufficient = calculator.calculate_evolution_penalty(3, "insufficient_xp");
        
        assert!(penalty5 > penalty1); // Higher level should have higher penalty
        assert_eq!(penalty_insufficient, 0); // No penalty for insufficient XP
        assert!(penalty1 > 0);
    }

    #[tokio::test]
    async fn test_custom_reward_categories() {
        let mut calculator = EvolutionXPRewardCalculator::default();
        
        let custom_category = XPRewardCategory {
            base_reward: 500,
            level_multiplier: 2.0,
            performance_multiplier: 3.0,
            improvement_threshold: 0.2,
        };
        
        calculator.set_reward_category("custom_achievement".to_string(), custom_category.clone());
        
        let retrieved = calculator.get_reward_category("custom_achievement").unwrap();
        assert_eq!(retrieved.base_reward, 500);
        assert_eq!(retrieved.level_multiplier, 2.0);
    }

    #[tokio::test]
    async fn test_reward_breakdown_summary() {
        let breakdown = XPRewardBreakdown {
            level_up_reward: 200,
            capability_reward: 150,
            performance_rewards: vec![80, 75, 120],
            collaboration_bonus: 50,
            total_reward: 675,
            streak_multiplier: 1.1,
            capped: false,
        };
        
        let summary = breakdown.summary();
        assert!(summary.contains("Level up: 200"));
        assert!(summary.contains("Capabilities: 150"));
        assert!(summary.contains("Performance: 275")); // Sum of performance rewards
        assert!(summary.contains("total: 675"));
        assert!(!summary.contains("capped"));
        
        let capped_breakdown = XPRewardBreakdown {
            total_reward: 1000,
            capped: true,
            ..breakdown
        };
        
        let capped_summary = capped_breakdown.summary();
        assert!(capped_summary.contains("capped at 1000"));
    }
}