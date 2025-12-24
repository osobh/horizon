//! XP system integration bridge for ADAS engine

use super::engine::AdasEngine;
use crate::traits::EvolvableAgent;
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use stratoswarm_evolution::{XPFitnessFunction, AgentEvolutionEngine, XPEvolutionEngine, XPEvolutionStats, AgentFitnessScore, EvolutionXPRewardCalculator, XPRewardBreakdown};
use stratoswarm_agent_core::agent::{Agent, AgentId, EvolutionResult, EvolutionMetrics};
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;

/// XP-aware ADAS fitness function that evaluates agents based on their architecture and behavior evolution
pub struct AdasXPFitnessFunction {
    /// Base architecture fitness weight
    pub architecture_weight: f64,
    /// Base behavior fitness weight  
    pub behavior_weight: f64,
    /// XP multiplier for higher level agents
    pub xp_level_multiplier: f64,
    /// Performance tracking weight
    pub performance_weight: f64,
}

impl Default for AdasXPFitnessFunction {
    fn default() -> Self {
        Self {
            architecture_weight: 0.4,
            behavior_weight: 0.3,
            xp_level_multiplier: 0.1,
            performance_weight: 0.2,
        }
    }
}

impl AdasXPFitnessFunction {
    pub fn new(
        architecture_weight: f64,
        behavior_weight: f64,
        xp_level_multiplier: f64,
        performance_weight: f64,
    ) -> Self {
        Self {
            architecture_weight,
            behavior_weight,
            xp_level_multiplier,
            performance_weight,
        }
    }

    /// Evaluate architecture genes fitness
    fn evaluate_architecture_fitness(&self, evolvable_agent: &EvolvableAgent) -> f64 {
        let genome = &evolvable_agent.genome;
        
        // Architecture complexity score
        let complexity_score = (genome.architecture.network_topology.len() as f64 / 10.0).min(1.0);
        let memory_score = (genome.architecture.memory_capacity as f64 / 8_000_000.0).min(1.0); // Normalize to ~8MB
        let processing_score = (genome.architecture.processing_units as f64 / 16.0).min(1.0); // Normalize to 16 units
        
        // Balanced architecture score
        (complexity_score + memory_score + processing_score) / 3.0
    }

    /// Evaluate behavior genes fitness
    fn evaluate_behavior_fitness(&self, evolvable_agent: &EvolvableAgent) -> f64 {
        let behavior = &evolvable_agent.genome.behavior;
        
        // Balanced behavior parameters (not too extreme)
        let exploration_score = 1.0 - (behavior.exploration_rate - 0.5).abs() * 2.0;
        let learning_score = behavior.learning_rate.min(1.0); // Higher learning is generally better
        let risk_score = 1.0 - (behavior.risk_tolerance - 0.3).abs() / 0.7; // Moderate risk tolerance is optimal
        
        (exploration_score.max(0.0) + learning_score + risk_score.max(0.0)) / 3.0
    }
}

#[async_trait]
impl XPFitnessFunction for AdasXPFitnessFunction {
    async fn evaluate_agent_fitness(&self, agent: &Agent) -> f64 {
        let stats = agent.stats().await;
        
        // Base fitness components
        let xp_component = (stats.current_xp as f64 / 25000.0).min(1.0) * self.xp_level_multiplier;
        let level_component = (stats.level as f64 * 0.05).min(0.5);
        
        // Performance component
        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.0
        };
        let performance_component = success_rate * self.performance_weight;
        
        // Calculate processing speed score
        let processing_speed_score = if stats.goals_processed > 0 {
            let avg_time = stats.total_execution_time.as_secs_f64() / stats.goals_processed as f64;
            ((60.0 - avg_time) / 50.0).max(0.0).min(1.0) // Normalize to 0-1, 10s is excellent
        } else {
            0.5
        };
        
        // Combined fitness
        xp_component + level_component + performance_component + (processing_speed_score * 0.2)
    }

    fn calculate_xp_reward(&self, fitness_improvement: f64, evolution_metrics: &EvolutionMetrics) -> u64 {
        let base_reward = 150u64; // Higher base for ADAS
        
        // Architecture evolution bonus
        let architecture_bonus = (fitness_improvement * 300.0) as u64;
        
        // Performance bonus
        let performance_bonus = (evolution_metrics.success_rate * 100.0) as u64;
        let speed_bonus = ((evolution_metrics.processing_speed - 1.0) * 50.0).max(0.0) as u64;
        
        base_reward + architecture_bonus + performance_bonus + speed_bonus
    }

    async fn should_evolve(&self, agent: &Agent) -> bool {
        let stats = agent.stats().await;
        
        // ADAS agents should evolve more frequently due to architectural changes
        if stats.current_xp >= 75 { // Lower threshold than standard
            return true;
        }
        
        // Also check readiness through standard XP system
        agent.check_evolution_readiness().await
    }
}

/// XP-integrated ADAS engine that combines architectural evolution with agent XP progression
pub struct AdasXPEngine {
    /// Core ADAS engine
    pub adas_engine: AdasEngine,
    /// XP-aware agent evolution engine
    pub xp_engine: AgentEvolutionEngine<AdasXPFitnessFunction>,
    /// XP reward calculator for ADAS-specific rewards
    pub reward_calculator: Arc<RwLock<EvolutionXPRewardCalculator>>,
    /// Current agent population with XP tracking
    pub agent_population: Arc<RwLock<Vec<Agent>>>,
}

impl AdasXPEngine {
    pub fn new(adas_engine: AdasEngine, xp_fitness_function: AdasXPFitnessFunction) -> Self {
        let xp_engine = AgentEvolutionEngine::with_defaults(xp_fitness_function);
        let mut reward_calculator = EvolutionXPRewardCalculator::default();
        
        // Add ADAS-specific reward categories
        reward_calculator.set_reward_category(
            "architecture_evolution".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 200,
                level_multiplier: 1.3,
                performance_multiplier: 1.8,
                improvement_threshold: 0.05,
            }
        );
        
        reward_calculator.set_reward_category(
            "behavior_optimization".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 175,
                level_multiplier: 1.2,
                performance_multiplier: 1.6,
                improvement_threshold: 0.08,
            }
        );
        
        reward_calculator.set_reward_category(
            "meta_learning_improvement".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 250,
                level_multiplier: 1.4,
                performance_multiplier: 2.0,
                improvement_threshold: 0.1,
            }
        );

        Self {
            adas_engine,
            xp_engine,
            reward_calculator: Arc::new(RwLock::new(reward_calculator)),
            agent_population: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize agent population with XP tracking
    pub async fn initialize_xp_population(&self, size: usize) -> EvolutionEngineResult<Vec<Agent>> {
        let population = self.adas_engine.generate_initial_population(size).await?;
        let mut agents = Vec::new();

        for individual in &population.individuals {
            let agent = individual.entity.agent.clone();
            agent.initialize().await.map_err(|e| EvolutionEngineError::InitializationError {
                message: format!("Failed to initialize agent: {}", e),
            })?;
            
            // Award initial XP based on genome complexity
            let initial_xp = self.calculate_genome_complexity_xp(&individual.entity);
            agent.award_xp(initial_xp, "Initial genome complexity".to_string(), "initialization".to_string())
                .await.map_err(|e| EvolutionEngineError::InitializationError {
                    message: format!("Failed to award initial XP: {}", e),
                })?;
            
            agents.push(agent);
        }

        *self.agent_population.write().await = agents.clone();
        Ok(agents)
    }

    /// Calculate XP reward based on genome complexity
    fn calculate_genome_complexity_xp(&self, evolvable_agent: &EvolvableAgent) -> u64 {
        let genome = &evolvable_agent.genome;
        
        // Architecture complexity
        let arch_complexity = genome.architecture.network_topology.len() as u64 * 5 +
                             (genome.architecture.memory_capacity / 100_000) as u64 +
                             genome.architecture.processing_units as u64 * 2;
        
        // Behavior complexity (balanced parameters get more XP)
        let behavior_balance_score = {
            let exploration_balance = (1.0 - (genome.behavior.exploration_rate - 0.5).abs() * 2.0).max(0.0);
            let risk_balance = (1.0 - (genome.behavior.risk_tolerance - 0.3).abs() / 0.7).max(0.0);
            ((exploration_balance + risk_balance) * 25.0) as u64
        };
        
        (arch_complexity + behavior_balance_score).min(200) // Cap at 200 XP
    }

    /// Evolve both architecture and XP-based agent progression
    pub async fn evolve_with_xp(&mut self) -> EvolutionEngineResult<(Vec<EvolutionResult>, XPEvolutionStats)> {
        let mut agent_population = self.agent_population.write().await;
        
        // Perform XP-based agent evolution
        let evolution_results = self.xp_engine.evolve_agent_population(&mut agent_population).await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("XP evolution failed: {:?}", e),
            })?;

        // Award ADAS-specific XP bonuses for architectural improvements
        for (agent, evolution_result) in agent_population.iter().zip(&evolution_results) {
            let reward_breakdown = self.reward_calculator.read().await
                .calculate_evolution_reward(agent, evolution_result).await
                .map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to calculate ADAS XP reward: {:?}", e),
                })?;

            // Award the calculated reward
            if reward_breakdown.total_reward > 0 {
                agent.award_xp(
                    reward_breakdown.total_reward,
                    format!("ADAS evolution reward: {}", reward_breakdown.summary()),
                    "adas_evolution".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award ADAS XP: {}", e),
                })?;
            }
        }

        // Get XP evolution statistics
        let xp_stats = self.xp_engine.get_xp_evolution_stats().await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to get XP stats: {:?}", e),
            })?;

        Ok((evolution_results, xp_stats))
    }

    /// Get comprehensive ADAS + XP statistics
    pub async fn get_comprehensive_stats(&self) -> EvolutionEngineResult<AdasXPStats> {
        let adas_metrics = self.adas_engine.metrics();
        let (total_workflows, current_iteration, best_performance) = self.adas_engine.get_meta_agent_stats();
        let xp_stats = self.xp_engine.get_xp_evolution_stats().await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to get XP stats: {:?}", e),
            })?;

        // Calculate agent level distribution
        let agent_population = self.agent_population.read().await;
        let mut level_distribution = std::collections::HashMap::new();
        let mut total_xp = 0u64;
        
        for agent in agent_population.iter() {
            let stats = agent.stats().await;
            *level_distribution.entry(stats.level).or_insert(0u32) += 1;
            total_xp += stats.current_xp;
        }

        Ok(AdasXPStats {
            // ADAS stats
            adas_generation: adas_metrics.generation,
            adas_best_fitness: adas_metrics.best_fitness,
            adas_diversity: adas_metrics.diversity,
            meta_workflows_discovered: total_workflows as u64,
            meta_search_iteration: current_iteration as u64,
            best_meta_performance: best_performance,
            
            // XP stats
            xp_evolution_stats: xp_stats,
            agent_level_distribution: level_distribution,
            total_agent_xp: total_xp,
            average_agent_level: if !agent_population.is_empty() {
                agent_population.iter().map(|a| a.calculate_level()).sum::<f64>() / agent_population.len() as f64
            } else {
                1.0
            },
        })
    }

    /// Award collaborative XP when multiple agents work together on architectural improvements
    pub async fn award_collaboration_xp(&self, participating_agents: &[AgentId], improvement_score: f64) -> EvolutionEngineResult<()> {
        let agent_population = self.agent_population.read().await;
        let collaborating_agents: Vec<&Agent> = agent_population.iter()
            .filter(|a| participating_agents.contains(&a.id()))
            .collect();

        if !collaborating_agents.is_empty() {
            let collaboration_bonus = self.reward_calculator.read().await
                .calculate_collaboration_bonus(&collaborating_agents, improvement_score);

            for agent in &collaborating_agents {
                agent.award_xp(
                    collaboration_bonus,
                    format!("ADAS collaboration bonus: {:.2} improvement", improvement_score),
                    "adas_collaboration".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award collaboration XP: {}", e),
                })?;
            }
        }

        Ok(())
    }
}

/// Comprehensive statistics combining ADAS and XP system metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdasXPStats {
    // ADAS-specific metrics
    pub adas_generation: u64,
    pub adas_best_fitness: f64,
    pub adas_diversity: f64,
    pub meta_workflows_discovered: u64,
    pub meta_search_iteration: u64,
    pub best_meta_performance: f64,
    
    // XP system metrics
    pub xp_evolution_stats: XPEvolutionStats,
    pub agent_level_distribution: std::collections::HashMap<u32, u32>,
    pub total_agent_xp: u64,
    pub average_agent_level: f64,
}

impl AdasXPStats {
    /// Get a summary of the combined evolution progress
    pub fn summary(&self) -> String {
        format!(
            "ADAS-XP Evolution Summary:\n\
            ADAS Generation: {} (Fitness: {:.3}, Diversity: {:.3})\n\
            Meta Workflows: {} (Iteration: {}, Best: {:.3})\n\
            XP Agents Evolved: {} (Success Rate: {:.2}%)\n\
            Agent Levels: Avg {:.1}, Total XP: {}",
            self.adas_generation,
            self.adas_best_fitness,
            self.adas_diversity,
            self.meta_workflows_discovered,
            self.meta_search_iteration,
            self.best_meta_performance,
            self.xp_evolution_stats.total_agents_evolved,
            self.xp_evolution_stats.evolution_success_rate * 100.0,
            self.average_agent_level,
            self.total_agent_xp
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adas::config::AdasConfig;
    use stratoswarm_agent_core::agent::AgentConfig;

    #[tokio::test]
    async fn test_adas_xp_fitness_function() {
        let fitness_fn = AdasXPFitnessFunction::default();
        
        // Create a test agent with some XP
        let config = AgentConfig {
            name: "test_adas_agent".to_string(),
            ..Default::default()
        };
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        agent.award_xp(200, "Test XP".to_string(), "test".to_string()).await.unwrap();
        
        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_adas_xp_should_evolve() {
        let fitness_fn = AdasXPFitnessFunction::default();
        
        let config = AgentConfig {
            name: "test_agent".to_string(),
            ..Default::default()
        };
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Should not evolve initially
        assert!(!fitness_fn.should_evolve(&agent).await);
        
        // Award enough XP to trigger evolution
        agent.award_xp(100, "Test XP".to_string(), "test".to_string()).await.unwrap();
        assert!(fitness_fn.should_evolve(&agent).await);
    }

    #[tokio::test]
    async fn test_adas_xp_reward_calculation() {
        let fitness_fn = AdasXPFitnessFunction::default();
        
        let evolution_metrics = EvolutionMetrics {
            avg_completion_time: std::time::Duration::from_secs(30),
            success_rate: 0.8,
            memory_efficiency: 0.7,
            processing_speed: 1.5,
        };
        
        let reward = fitness_fn.calculate_xp_reward(0.2, &evolution_metrics);
        assert!(reward >= 150); // Should be at least base reward
        assert!(reward > 200); // Should include bonuses
    }
}