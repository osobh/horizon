//! XP system integration bridge for DGM (Discovered Agent Growth Mode) engine

use super::engine::DgmEngine;
use super::improvement::{GrowthPattern, GrowthHistory};
use crate::traits::EvolvableAgent;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use stratoswarm_evolution::{XPFitnessFunction, AgentEvolutionEngine, XPEvolutionEngine, XPEvolutionStats, AgentFitnessScore, EvolutionXPRewardCalculator, XPRewardBreakdown};
use stratoswarm_agent_core::agent::{Agent, AgentId, EvolutionResult, EvolutionMetrics};
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;

/// DGM-specific XP fitness function that rewards self-improvement and pattern discovery
pub struct DgmXPFitnessFunction {
    /// Weight for self-improvement patterns
    pub self_improvement_weight: f64,
    /// Weight for pattern discovery success
    pub pattern_discovery_weight: f64,
    /// Weight for growth momentum
    pub growth_momentum_weight: f64,
    /// Weight for agent performance metrics
    pub performance_weight: f64,
    /// Minimum self-modifications required for evolution
    pub min_modifications_threshold: u32,
}

impl Default for DgmXPFitnessFunction {
    fn default() -> Self {
        Self {
            self_improvement_weight: 0.35,
            pattern_discovery_weight: 0.25,
            growth_momentum_weight: 0.2,
            performance_weight: 0.2,
            min_modifications_threshold: 3,
        }
    }
}

impl DgmXPFitnessFunction {
    pub fn new(
        self_improvement_weight: f64,
        pattern_discovery_weight: f64,
        growth_momentum_weight: f64,
        performance_weight: f64,
        min_modifications_threshold: u32,
    ) -> Self {
        Self {
            self_improvement_weight,
            pattern_discovery_weight,
            growth_momentum_weight,
            performance_weight,
            min_modifications_threshold,
        }
    }

    /// Evaluate self-improvement fitness based on agent's capability to modify itself
    fn evaluate_self_improvement_fitness(&self, agent: &Agent) -> f64 {
        // Check agent stats for signs of self-improvement
        let stats = futures::executor::block_on(agent.stats());
        
        // Factor in XP growth rate as a sign of self-improvement
        let xp_growth_rate = if stats.xp_history.len() >= 2 {
            let recent_gains: u64 = stats.xp_history.iter().take(10).map(|entry| entry.amount).sum();
            (recent_gains as f64 / 10.0) / 100.0 // Normalize to 0-1 scale
        } else {
            0.0
        };

        // Factor in goal success improvements over time
        let improvement_trend = if stats.goals_processed >= 5 {
            let recent_success_rate = if stats.goals_processed >= 10 {
                // Compare recent vs earlier performance
                let recent_successes = stats.goals_succeeded.saturating_sub(stats.goals_processed / 2);
                let recent_attempts = stats.goals_processed / 2;
                if recent_attempts > 0 {
                    recent_successes as f64 / recent_attempts as f64
                } else {
                    0.0
                }
            } else {
                stats.goals_succeeded as f64 / stats.goals_processed as f64
            };
            recent_success_rate
        } else {
            0.5 // Neutral score for new agents
        };

        // Capability progression score
        let capability_score = (stats.level as f64 - 1.0) * 0.1; // Each level adds 0.1 to fitness

        (xp_growth_rate + improvement_trend + capability_score).min(1.0)
    }

    /// Evaluate pattern discovery contribution to fitness
    fn evaluate_pattern_discovery_fitness(&self, _agent: &Agent, growth_history: Option<&GrowthHistory>) -> f64 {
        if let Some(history) = growth_history {
            // Score based on pattern discovery rate and success
            let discovery_rate = history.get_discovery_rate();
            let pattern_success_rate = history.get_pattern_success_rate();
            
            (discovery_rate * 0.6 + pattern_success_rate * 0.4).min(1.0)
        } else {
            0.0
        }
    }

    /// Evaluate growth momentum fitness
    fn evaluate_growth_momentum_fitness(&self, agent: &Agent) -> f64 {
        let stats = futures::executor::block_on(agent.stats());
        
        // Check for consistent XP gains as momentum indicator
        if stats.xp_history.len() >= 5 {
            let recent_gains: Vec<u64> = stats.xp_history.iter()
                .take(5)
                .map(|entry| entry.amount)
                .collect();
            
            // Check for increasing or consistent gains
            let is_growing = recent_gains.windows(2)
                .all(|window| window[1] >= window[0] || (window[0] - window[1]) <= 5);
            
            if is_growing {
                0.8 // High momentum score
            } else {
                0.3 // Lower but not zero momentum
            }
        } else {
            0.5 // Neutral score for new agents
        }
    }
}

#[async_trait]
impl XPFitnessFunction for DgmXPFitnessFunction {
    async fn evaluate_agent_fitness(&self, agent: &Agent) -> f64 {
        let stats = agent.stats().await;
        
        // Core DGM fitness components
        let self_improvement_score = self.evaluate_self_improvement_fitness(agent);
        let pattern_discovery_score = self.evaluate_pattern_discovery_fitness(agent, None); // Would need access to growth history
        let growth_momentum_score = self.evaluate_growth_momentum_fitness(agent);
        
        // Performance component
        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.0
        };
        
        // Processing efficiency component
        let processing_efficiency = if stats.goals_processed > 0 {
            let avg_time = stats.total_execution_time.as_secs_f64() / stats.goals_processed as f64;
            ((90.0 - avg_time) / 80.0).max(0.0).min(1.0) // Normalize for DGM (longer analysis times expected)
        } else {
            0.5
        };
        
        let performance_score = (success_rate + processing_efficiency) / 2.0;
        
        // Weighted combination
        (self_improvement_score * self.self_improvement_weight) +
        (pattern_discovery_score * self.pattern_discovery_weight) +
        (growth_momentum_score * self.growth_momentum_weight) +
        (performance_score * self.performance_weight)
    }

    fn calculate_xp_reward(&self, fitness_improvement: f64, evolution_metrics: &EvolutionMetrics) -> u64 {
        let base_reward = 175u64; // DGM base reward
        
        // Self-improvement bonus (DGM specialty)
        let self_improvement_bonus = (fitness_improvement * 400.0) as u64;
        
        // Pattern discovery bonus
        let pattern_bonus = if fitness_improvement > 0.15 {
            100u64 // Significant pattern discovery
        } else if fitness_improvement > 0.08 {
            50u64  // Moderate pattern discovery
        } else {
            0u64
        };
        
        // Growth momentum bonus
        let momentum_bonus = ((evolution_metrics.processing_speed - 1.0) * 75.0).max(0.0) as u64;
        
        // Memory efficiency bonus (important for self-modification)
        let memory_bonus = (evolution_metrics.memory_efficiency * 60.0) as u64;
        
        base_reward + self_improvement_bonus + pattern_bonus + momentum_bonus + memory_bonus
    }

    async fn should_evolve(&self, agent: &Agent) -> bool {
        let stats = agent.stats().await;
        
        // DGM agents should evolve when they've demonstrated self-improvement capability
        if stats.current_xp >= 125 { // Slightly higher threshold for DGM
            return true;
        }
        
        // Also check if agent has made enough self-modifications
        let modification_count = stats.xp_history.iter()
            .filter(|entry| entry.category.contains("modification") || entry.category.contains("pattern"))
            .count() as u32;
            
        if modification_count >= self.min_modifications_threshold {
            return true;
        }
        
        // Standard evolution readiness check
        agent.check_evolution_readiness().await
    }
}

/// DGM XP engine combining growth pattern discovery with agent XP progression
pub struct DgmXPEngine {
    /// Core DGM engine
    pub dgm_engine: DgmEngine,
    /// XP-aware agent evolution engine
    pub xp_engine: AgentEvolutionEngine<DgmXPFitnessFunction>,
    /// XP reward calculator for DGM-specific rewards
    pub reward_calculator: Arc<RwLock<EvolutionXPRewardCalculator>>,
    /// Current agent population with XP tracking
    pub agent_population: Arc<RwLock<Vec<Agent>>>,
    /// Pattern discovery XP tracker
    pub pattern_xp_tracker: Arc<RwLock<std::collections::HashMap<String, u64>>>,
}

impl DgmXPEngine {
    pub fn new(dgm_engine: DgmEngine, xp_fitness_function: DgmXPFitnessFunction) -> Self {
        let xp_engine = AgentEvolutionEngine::with_defaults(xp_fitness_function);
        let mut reward_calculator = EvolutionXPRewardCalculator::default();
        
        // Add DGM-specific reward categories
        reward_calculator.set_reward_category(
            "self_modification".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 150,
                level_multiplier: 1.4,
                performance_multiplier: 1.7,
                improvement_threshold: 0.05,
            }
        );
        
        reward_calculator.set_reward_category(
            "pattern_discovery".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 200,
                level_multiplier: 1.5,
                performance_multiplier: 2.2,
                improvement_threshold: 0.1,
            }
        );
        
        reward_calculator.set_reward_category(
            "growth_momentum".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 125,
                level_multiplier: 1.3,
                performance_multiplier: 1.8,
                improvement_threshold: 0.08,
            }
        );
        
        reward_calculator.set_reward_category(
            "meta_improvement".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 300,
                level_multiplier: 1.6,
                performance_multiplier: 2.5,
                improvement_threshold: 0.15,
            }
        );

        Self {
            dgm_engine,
            xp_engine,
            reward_calculator: Arc::new(RwLock::new(reward_calculator)),
            agent_population: Arc::new(RwLock::new(Vec::new())),
            pattern_xp_tracker: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Initialize DGM population with XP tracking and self-improvement capability
    pub async fn initialize_dgm_xp_population(&self, size: usize) -> EvolutionEngineResult<Vec<Agent>> {
        let population = self.dgm_engine.generate_initial_population(size).await?;
        let mut agents = Vec::new();

        for individual in &population.individuals {
            let agent = individual.entity.agent.clone();
            agent.initialize().await.map_err(|e| EvolutionEngineError::InitializationError {
                message: format!("Failed to initialize DGM agent: {}", e),
            })?;
            
            // Award initial XP for self-modification potential
            let initial_xp = self.calculate_self_modification_potential_xp(&individual.entity);
            agent.award_xp(initial_xp, "Initial self-modification potential".to_string(), "dgm_initialization".to_string())
                .await.map_err(|e| EvolutionEngineError::InitializationError {
                    message: format!("Failed to award initial DGM XP: {}", e),
                })?;
            
            agents.push(agent);
        }

        *self.agent_population.write().await = agents.clone();
        Ok(agents)
    }

    /// Calculate XP based on self-modification potential
    fn calculate_self_modification_potential_xp(&self, evolvable_agent: &EvolvableAgent) -> u64 {
        let genome = &evolvable_agent.genome;
        
        // Reward complex architectures that can self-modify
        let architecture_complexity = genome.architecture.network_topology.len() as u64 * 8;
        let memory_for_modification = (genome.architecture.memory_capacity / 200_000) as u64; // More memory = more self-modification space
        let processing_power = genome.architecture.processing_units as u64 * 15;
        
        // Reward exploration-oriented behavior (needed for self-improvement)
        let exploration_bonus = (genome.behavior.exploration_rate * 50.0) as u64;
        let learning_bonus = (genome.behavior.learning_rate * 1000.0) as u64; // Scale up learning rate
        
        (architecture_complexity + memory_for_modification + processing_power + exploration_bonus + learning_bonus).min(300)
    }

    /// Evolve with DGM self-improvement and XP progression
    pub async fn evolve_with_dgm_xp(&mut self) -> EvolutionEngineResult<(Vec<EvolutionResult>, XPEvolutionStats, DgmXPStats)> {
        let mut agent_population = self.agent_population.write().await;
        
        // Award XP for discovered patterns before evolution
        self.award_pattern_discovery_xp(&agent_population).await?;
        
        // Perform XP-based agent evolution
        let evolution_results = self.xp_engine.evolve_agent_population(&mut agent_population).await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("DGM XP evolution failed: {:?}", e),
            })?;

        // Award DGM-specific XP bonuses
        for (agent, evolution_result) in agent_population.iter().zip(&evolution_results) {
            let reward_breakdown = self.reward_calculator.read().await
                .calculate_evolution_reward(agent, evolution_result).await
                .map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to calculate DGM XP reward: {:?}", e),
                })?;

            // Award the calculated reward with DGM-specific context
            if reward_breakdown.total_reward > 0 {
                agent.award_xp(
                    reward_breakdown.total_reward,
                    format!("DGM self-improvement reward: {}", reward_breakdown.summary()),
                    "dgm_evolution".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award DGM XP: {}", e),
                })?;
            }

            // Additional bonus for meta-level improvements
            if evolution_result.new_level > evolution_result.previous_level + 1 {
                agent.award_xp(
                    150,
                    "Multi-level breakthrough in self-improvement".to_string(),
                    "dgm_breakthrough".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award DGM breakthrough XP: {}", e),
                })?;
            }
        }

        // Get statistics
        let xp_stats = self.xp_engine.get_xp_evolution_stats().await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to get XP stats: {:?}", e),
            })?;

        let dgm_stats = self.get_dgm_xp_stats().await?;

        Ok((evolution_results, xp_stats, dgm_stats))
    }

    /// Award XP for pattern discovery achievements
    async fn award_pattern_discovery_xp(&self, agents: &[Agent]) -> EvolutionEngineResult<()> {
        let growth_patterns = self.dgm_engine.growth_patterns.read();
        
        if !growth_patterns.is_empty() {
            let pattern_count = growth_patterns.len();
            let discovery_bonus = (pattern_count as u64 * 25).min(200); // Up to 200 XP for many patterns
            
            // Award pattern discovery XP to all agents (collective intelligence)
            for agent in agents {
                agent.award_xp(
                    discovery_bonus,
                    format!("Collective pattern discovery: {} patterns", pattern_count),
                    "pattern_discovery".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award pattern discovery XP: {}", e),
                })?;
            }
            
            // Track pattern XP
            let mut tracker = self.pattern_xp_tracker.write().await;
            for pattern in growth_patterns.values() {
                tracker.insert(pattern.id.clone(), discovery_bonus);
            }
        }

        Ok(())
    }

    /// Get DGM-specific XP statistics
    async fn get_dgm_xp_stats(&self) -> EvolutionEngineResult<DgmXPStats> {
        let agent_population = self.agent_population.read().await;
        let growth_patterns = self.dgm_engine.growth_patterns.read();
        let pattern_xp_tracker = self.pattern_xp_tracker.read().await;
        
        // Calculate self-modification metrics
        let mut total_self_modifications = 0u64;
        let mut agents_with_modifications = 0u64;
        let mut pattern_usage_count = 0u64;
        
        for agent in agent_population.iter() {
            let stats = agent.stats().await;
            let modifications = stats.xp_history.iter()
                .filter(|entry| entry.category.contains("modification") || entry.category.contains("dgm"))
                .count() as u64;
            
            if modifications > 0 {
                total_self_modifications += modifications;
                agents_with_modifications += 1;
            }
            
            pattern_usage_count += stats.xp_history.iter()
                .filter(|entry| entry.category.contains("pattern"))
                .count() as u64;
        }
        
        let avg_modifications_per_agent = if !agent_population.is_empty() {
            total_self_modifications as f64 / agent_population.len() as f64
        } else {
            0.0
        };
        
        let pattern_discovery_xp: u64 = pattern_xp_tracker.values().sum();
        let velocity = *self.dgm_engine.improvement_velocity.read();
        
        Ok(DgmXPStats {
            patterns_discovered: growth_patterns.len() as u64,
            pattern_discovery_xp_awarded: pattern_discovery_xp,
            total_self_modifications,
            agents_with_modifications,
            avg_modifications_per_agent,
            pattern_usage_count,
            improvement_velocity: velocity,
            self_assessment_available: self.dgm_engine.get_self_assessment_report().is_some(),
        })
    }

    /// Award XP for successful self-modifications
    pub async fn award_self_modification_xp(
        &self,
        agent_id: &AgentId,
        modification_type: &str,
        improvement_score: f64,
    ) -> EvolutionEngineResult<()> {
        let agent_population = self.agent_population.read().await;
        
        if let Some(agent) = agent_population.iter().find(|a| a.id() == *agent_id) {
            let base_xp = match modification_type {
                "architecture_change" => 80,
                "behavior_optimization" => 60,
                "pattern_application" => 100,
                "novel_discovery" => 150,
                _ => 40,
            };
            
            let improvement_bonus = (improvement_score * 100.0) as u64;
            let total_xp = base_xp + improvement_bonus;
            
            agent.award_xp(
                total_xp,
                format!("Self-modification: {} (improvement: {:.2})", modification_type, improvement_score),
                "self_modification".to_string(),
            ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to award self-modification XP: {}", e),
            })?;
        }

        Ok(())
    }
}

/// DGM-specific XP statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DgmXPStats {
    pub patterns_discovered: u64,
    pub pattern_discovery_xp_awarded: u64,
    pub total_self_modifications: u64,
    pub agents_with_modifications: u64,
    pub avg_modifications_per_agent: f64,
    pub pattern_usage_count: u64,
    pub improvement_velocity: f64,
    pub self_assessment_available: bool,
}

impl DgmXPStats {
    pub fn summary(&self) -> String {
        format!(
            "DGM-XP Statistics:\n\
            Patterns Discovered: {} (XP Awarded: {})\n\
            Self-Modifications: {} total, {} agents active (avg: {:.1})\n\
            Pattern Usage: {} applications\n\
            Improvement Velocity: {:.3}\n\
            Self-Assessment: {}",
            self.patterns_discovered,
            self.pattern_discovery_xp_awarded,
            self.total_self_modifications,
            self.agents_with_modifications,
            self.avg_modifications_per_agent,
            self.pattern_usage_count,
            self.improvement_velocity,
            if self.self_assessment_available { "Active" } else { "Inactive" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgm::config::DgmConfig;
    use stratoswarm_agent_core::agent::AgentConfig;

    #[tokio::test]
    async fn test_dgm_xp_fitness_function() {
        let fitness_fn = DgmXPFitnessFunction::default();
        
        let config = AgentConfig {
            name: "test_dgm_agent".to_string(),
            ..Default::default()
        };
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Award some XP with self-modification pattern
        agent.award_xp(100, "Self-modification test".to_string(), "self_modification".to_string()).await.unwrap();
        
        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_dgm_should_evolve_conditions() {
        let fitness_fn = DgmXPFitnessFunction::default();
        
        let config = AgentConfig::default();
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Should not evolve initially
        assert!(!fitness_fn.should_evolve(&agent).await);
        
        // Award enough modifications to trigger evolution
        for i in 0..4 {
            agent.award_xp(20, format!("Modification {}", i), "modification".to_string()).await.unwrap();
        }
        
        assert!(fitness_fn.should_evolve(&agent).await);
    }

    #[tokio::test]
    async fn test_dgm_xp_reward_calculation() {
        let fitness_fn = DgmXPFitnessFunction::default();
        
        let evolution_metrics = EvolutionMetrics {
            avg_completion_time: std::time::Duration::from_secs(45),
            success_rate: 0.85,
            memory_efficiency: 0.9,
            processing_speed: 1.3,
        };
        
        let reward = fitness_fn.calculate_xp_reward(0.25, &evolution_metrics);
        assert!(reward >= 175); // Should be at least base reward
        assert!(reward > 300); // Should include significant bonuses for DGM
    }
}