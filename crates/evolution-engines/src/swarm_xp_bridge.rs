//! XP system integration bridge for SwarmAgentic engine

use crate::swarm::{SwarmEngine, SwarmConfig};
use crate::swarm_particle::Particle;
use crate::traits::{EvolvableAgent, EvolutionEngine};
use crate::error::{EvolutionEngineError, EvolutionEngineResult};
use stratoswarm_evolution::{XPFitnessFunction, AgentEvolutionEngine, XPEvolutionEngine, XPEvolutionStats, AgentFitnessScore, EvolutionXPRewardCalculator, XPRewardBreakdown};
use stratoswarm_agent_core::agent::{Agent, AgentId, EvolutionResult, EvolutionMetrics};
use std::sync::Arc;
use tokio::sync::RwLock;
/// Swarm-specific XP fitness function that rewards collective optimization and social learning
pub struct SwarmXPFitnessFunction {
    /// Weight for individual particle performance
    pub individual_performance_weight: f64,
    /// Weight for social learning contribution
    pub social_learning_weight: f64,
    /// Weight for swarm convergence participation
    pub convergence_participation_weight: f64,
    /// Weight for exploration vs exploitation balance
    pub exploration_balance_weight: f64,
    /// Minimum swarm interactions required for evolution
    pub min_interactions_threshold: u32,
}

impl Default for SwarmXPFitnessFunction {
    fn default() -> Self {
        Self {
            individual_performance_weight: 0.3,
            social_learning_weight: 0.35,
            convergence_participation_weight: 0.2,
            exploration_balance_weight: 0.15,
            min_interactions_threshold: 5,
        }
    }
}

impl SwarmXPFitnessFunction {
    pub fn new(
        individual_performance_weight: f64,
        social_learning_weight: f64,
        convergence_participation_weight: f64,
        exploration_balance_weight: f64,
        min_interactions_threshold: u32,
    ) -> Self {
        Self {
            individual_performance_weight,
            social_learning_weight,
            convergence_participation_weight,
            exploration_balance_weight,
            min_interactions_threshold,
        }
    }

    /// Evaluate social learning contribution based on agent's interaction patterns
    fn evaluate_social_learning_fitness(&self, agent: &Agent) -> f64 {
        let stats = futures::executor::block_on(agent.stats());
        
        // Look for social learning indicators in XP history
        let social_xp_count = stats.xp_history.iter()
            .filter(|entry| entry.category.contains("social") || 
                           entry.category.contains("swarm") || 
                           entry.category.contains("collaboration"))
            .count();
        
        // Social learning score based on interactions
        let social_score = (social_xp_count as f64 / 20.0).min(1.0); // Normalize to max 20 interactions
        
        // Factor in collaborative success patterns
        let collaboration_success = if stats.goals_processed >= 5 {
            // Agents that consistently collaborate should have higher success rates
            let base_success_rate = stats.goals_succeeded as f64 / stats.goals_processed as f64;
            if social_xp_count > 0 {
                base_success_rate * 1.2 // Boost for collaborative agents
            } else {
                base_success_rate
            }
        } else {
            0.5
        };
        
        (social_score * 0.6 + collaboration_success.min(1.0) * 0.4)
    }

    /// Evaluate convergence participation (how well agent follows swarm dynamics)
    fn evaluate_convergence_participation_fitness(&self, agent: &Agent) -> f64 {
        let stats = futures::executor::block_on(agent.stats());
        
        // Check for consistent performance improvement (sign of following swarm trends)
        if stats.xp_history.len() >= 5 {
            let recent_xp: Vec<u64> = stats.xp_history.iter()
                .take(5)
                .map(|entry| entry.amount)
                .collect();
            
            // Check for trend following (consistent gains indicate good swarm participation)
            let has_consistency = recent_xp.windows(2)
                .filter(|window| (window[1] as i64 - window[0] as i64).abs() <= 20)
                .count() >= 3;
            
            if has_consistency {
                0.8 // High convergence participation
            } else {
                0.4 // Lower participation
            }
        } else {
            0.5 // Neutral for new agents
        }
    }

    /// Evaluate exploration vs exploitation balance
    fn evaluate_exploration_balance_fitness(&self, agent: &Agent, evolvable_agent: Option<&EvolvableAgent>) -> f64 {
        let stats = futures::executor::block_on(agent.stats());
        
        // Balance component from behavior genes
        let behavior_balance = if let Some(evolvable) = evolvable_agent {
            let behavior = &evolvable.genome.behavior;
            
            // Optimal exploration rate for swarms is around 0.3-0.7
            let exploration_optimality = 1.0 - (behavior.exploration_rate - 0.5).abs() * 2.0;
            
            // Risk tolerance should be moderate for good swarm behavior
            let risk_optimality = 1.0 - (behavior.risk_tolerance - 0.4).abs() / 0.6;
            
            (exploration_optimality.max(0.0) + risk_optimality.max(0.0)) / 2.0
        } else {
            0.5
        };
        
        // Performance stability indicates good balance
        let performance_stability = if stats.goals_processed >= 8 {
            let recent_success_rate = if stats.goals_processed >= 16 {
                let recent_successes = stats.goals_succeeded.saturating_sub(stats.goals_processed / 2);
                let recent_attempts = stats.goals_processed / 2;
                recent_successes as f64 / recent_attempts as f64
            } else {
                stats.goals_succeeded as f64 / stats.goals_processed as f64
            };
            
            // Stable performance indicates good exploration/exploitation balance
            if recent_success_rate >= 0.6 {
                0.9
            } else if recent_success_rate >= 0.4 {
                0.6
            } else {
                0.3
            }
        } else {
            0.5
        };
        
        (behavior_balance * 0.4 + performance_stability * 0.6)
    }
}

impl XPFitnessFunction for SwarmXPFitnessFunction {
    async fn evaluate_agent_fitness(&self, agent: &Agent) -> f64 {
        let stats = agent.stats().await;
        
        // Individual performance component
        let success_rate = if stats.goals_processed > 0 {
            stats.goals_succeeded as f64 / stats.goals_processed as f64
        } else {
            0.0
        };
        
        let processing_efficiency = if stats.goals_processed > 0 {
            let avg_time = stats.total_execution_time.as_secs_f64() / stats.goals_processed as f64;
            ((75.0 - avg_time) / 65.0).max(0.0).min(1.0) // Swarms may be slower due to coordination
        } else {
            0.5
        };
        
        let individual_performance = (success_rate + processing_efficiency) / 2.0;
        
        // Social learning component
        let social_learning_score = self.evaluate_social_learning_fitness(agent);
        
        // Convergence participation component
        let convergence_score = self.evaluate_convergence_participation_fitness(agent);
        
        // Exploration balance component (requires evolvable agent access, simplified for now)
        let exploration_balance_score = self.evaluate_exploration_balance_fitness(agent, None);
        
        // Weighted combination
        (individual_performance * self.individual_performance_weight) +
        (social_learning_score * self.social_learning_weight) +
        (convergence_score * self.convergence_participation_weight) +
        (exploration_balance_score * self.exploration_balance_weight)
    }

    fn calculate_xp_reward(&self, fitness_improvement: f64, evolution_metrics: &EvolutionMetrics) -> u64 {
        let base_reward = 160u64; // Swarm base reward
        
        // Social learning bonus (key swarm strength)
        let social_bonus = (fitness_improvement * 350.0) as u64;
        
        // Convergence participation bonus
        let convergence_bonus = if fitness_improvement > 0.12 {
            120u64 // Strong convergence participation
        } else if fitness_improvement > 0.06 {
            60u64  // Moderate participation
        } else {
            0u64
        };
        
        // Collective intelligence bonus
        let collective_bonus = (evolution_metrics.success_rate * 80.0) as u64;
        
        // Exploration balance bonus
        let balance_bonus = if evolution_metrics.processing_speed > 1.0 && evolution_metrics.processing_speed < 2.0 {
            40u64 // Good balance
        } else {
            0u64
        };
        
        base_reward + social_bonus + convergence_bonus + collective_bonus + balance_bonus
    }

    async fn should_evolve(&self, agent: &Agent) -> bool {
        let stats = agent.stats().await;
        
        // Swarm agents should evolve when they've demonstrated social learning
        if stats.current_xp >= 110 { // Moderate threshold for swarm agents
            return true;
        }
        
        // Check for sufficient swarm interactions
        let interaction_count = stats.xp_history.iter()
            .filter(|entry| entry.category.contains("social") || 
                           entry.category.contains("swarm") ||
                           entry.category.contains("collaboration"))
            .count() as u32;
            
        if interaction_count >= self.min_interactions_threshold {
            return true;
        }
        
        // Standard evolution readiness check
        agent.check_evolution_readiness().await
    }
}

/// Swarm XP engine combining particle swarm optimization with agent XP progression
pub struct SwarmXPEngine {
    /// Core swarm engine
    pub swarm_engine: SwarmEngine,
    /// XP-aware agent evolution engine
    pub xp_engine: AgentEvolutionEngine<SwarmXPFitnessFunction>,
    /// XP reward calculator for swarm-specific rewards
    pub reward_calculator: Arc<RwLock<EvolutionXPRewardCalculator>>,
    /// Current agent population with XP tracking
    pub agent_population: Arc<RwLock<Vec<Agent>>>,
    /// Social interaction XP tracker
    pub social_xp_tracker: Arc<RwLock<std::collections::HashMap<AgentId, u64>>>,
    /// Swarm convergence history
    pub convergence_history: Arc<RwLock<Vec<(u64, f64, f64)>>>, // (generation, best_fitness, convergence_rate)
}

impl SwarmXPEngine {
    pub fn new(swarm_engine: SwarmEngine, xp_fitness_function: SwarmXPFitnessFunction) -> Self {
        let xp_engine = AgentEvolutionEngine::with_defaults(xp_fitness_function);
        let mut reward_calculator = EvolutionXPRewardCalculator::default();
        
        // Add Swarm-specific reward categories
        reward_calculator.set_reward_category(
            "social_learning".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 130,
                level_multiplier: 1.3,
                performance_multiplier: 1.9,
                improvement_threshold: 0.05,
            }
        );
        
        reward_calculator.set_reward_category(
            "swarm_convergence".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 110,
                level_multiplier: 1.2,
                performance_multiplier: 1.6,
                improvement_threshold: 0.08,
            }
        );
        
        reward_calculator.set_reward_category(
            "collective_intelligence".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 180,
                level_multiplier: 1.4,
                performance_multiplier: 2.1,
                improvement_threshold: 0.1,
            }
        );
        
        reward_calculator.set_reward_category(
            "particle_optimization".to_string(),
            stratoswarm_evolution::XPRewardCategory {
                base_reward: 140,
                level_multiplier: 1.25,
                performance_multiplier: 1.8,
                improvement_threshold: 0.07,
            }
        );

        Self {
            swarm_engine,
            xp_engine,
            reward_calculator: Arc::new(RwLock::new(reward_calculator)),
            agent_population: Arc::new(RwLock::new(Vec::new())),
            social_xp_tracker: Arc::new(RwLock::new(std::collections::HashMap::new())),
            convergence_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize swarm population with XP tracking and social capabilities
    pub async fn initialize_swarm_xp_population(&self, size: usize) -> EvolutionEngineResult<Vec<Agent>> {
        let population = self.swarm_engine.generate_initial_population(size).await?;
        let mut agents = Vec::new();

        for individual in &population.individuals {
            let agent = individual.entity.agent.clone();
            agent.initialize().await.map_err(|e| EvolutionEngineError::InitializationError {
                message: format!("Failed to initialize swarm agent: {}", e),
            })?;
            
            // Award initial XP for swarm participation potential
            let initial_xp = self.calculate_swarm_potential_xp(&individual.entity);
            agent.award_xp(initial_xp, "Initial swarm participation potential".to_string(), "swarm_initialization".to_string())
                .await.map_err(|e| EvolutionEngineError::InitializationError {
                    message: format!("Failed to award initial swarm XP: {}", e),
                })?;
            
            agents.push(agent);
        }

        *self.agent_population.write().await = agents.clone();
        Ok(agents)
    }

    /// Calculate XP based on swarm participation potential
    fn calculate_swarm_potential_xp(&self, evolvable_agent: &EvolvableAgent) -> u64 {
        let genome = &evolvable_agent.genome;
        
        // Reward social behavior genes
        let exploration_bonus = (genome.behavior.exploration_rate * 80.0) as u64;
        let learning_bonus = (genome.behavior.learning_rate * 800.0) as u64; // High learning rate is valuable for swarms
        let risk_balance_bonus = ((1.0 - (genome.behavior.risk_tolerance - 0.4).abs() / 0.6) * 50.0) as u64;
        
        // Architecture suitable for swarm communication
        let communication_potential = genome.architecture.network_topology.len() as u64 * 6; // More layers = better communication
        let processing_coordination = genome.architecture.processing_units as u64 * 10;
        
        (exploration_bonus + learning_bonus + risk_balance_bonus + communication_potential + processing_coordination).min(250)
    }

    /// Evolve with swarm intelligence and XP progression
    pub async fn evolve_with_swarm_xp(&mut self) -> EvolutionEngineResult<(Vec<EvolutionResult>, XPEvolutionStats, SwarmXPStats)> {
        let mut agent_population = self.agent_population.write().await;
        
        // Award XP for social interactions before evolution
        self.award_social_interaction_xp(&agent_population).await?;
        
        // Perform XP-based agent evolution
        let evolution_results = self.xp_engine.evolve_agent_population(&mut agent_population).await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Swarm XP evolution failed: {:?}", e),
            })?;

        // Award swarm-specific XP bonuses
        for (agent, evolution_result) in agent_population.iter().zip(&evolution_results) {
            let reward_breakdown = self.reward_calculator.read().await
                .calculate_evolution_reward(agent, evolution_result).await
                .map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to calculate swarm XP reward: {:?}", e),
                })?;

            // Award the calculated reward with swarm-specific context
            if reward_breakdown.total_reward > 0 {
                agent.award_xp(
                    reward_breakdown.total_reward,
                    format!("Swarm intelligence reward: {}", reward_breakdown.summary()),
                    "swarm_evolution".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award swarm XP: {}", e),
                })?;
            }

            // Bonus for exceptional social learning breakthroughs
            if evolution_result.new_level > evolution_result.previous_level && 
               evolution_result.new_metrics.success_rate > 0.8 {
                agent.award_xp(
                    100,
                    "Exceptional swarm intelligence breakthrough".to_string(),
                    "swarm_breakthrough".to_string(),
                ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                    message: format!("Failed to award swarm breakthrough XP: {}", e),
                })?;
            }
        }

        // Record convergence history
        let swarm_metrics = self.swarm_engine.metrics();
        self.convergence_history.write().await.push((
            swarm_metrics.generation as u64,
            swarm_metrics.best_fitness,
            swarm_metrics.convergence_rate,
        ));

        // Get statistics
        let xp_stats = self.xp_engine.get_xp_evolution_stats().await
            .map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to get XP stats: {:?}", e),
            })?;

        let swarm_stats = self.get_swarm_xp_stats().await?;

        Ok((evolution_results, xp_stats, swarm_stats))
    }

    /// Award XP for social interactions and collective behaviors
    async fn award_social_interaction_xp(&self, agents: &[Agent]) -> EvolutionEngineResult<()> {
        let interaction_bonus = 30u64; // Base social interaction bonus
        
        // Award social XP to all agents (representing swarm interactions)
        for agent in agents {
            agent.award_xp(
                interaction_bonus,
                "Swarm social interaction".to_string(),
                "social_learning".to_string(),
            ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to award social XP: {}", e),
            })?;
            
            // Track social XP
            let mut tracker = self.social_xp_tracker.write().await;
            *tracker.entry(agent.id()).or_insert(0) += interaction_bonus;
        }

        Ok(())
    }

    /// Get swarm-specific XP statistics
    async fn get_swarm_xp_stats(&self) -> EvolutionEngineResult<SwarmXPStats> {
        let agent_population = self.agent_population.read().await;
        let social_xp_tracker = self.social_xp_tracker.read().await;
        let convergence_history = self.convergence_history.read().await;
        let swarm_metrics = self.swarm_engine.metrics();
        
        // Calculate social interaction metrics
        let total_social_xp: u64 = social_xp_tracker.values().sum();
        let agents_with_social_interactions = social_xp_tracker.len() as u64;
        
        let mut collective_performance_improvements = 0u64;
        let mut social_learning_events = 0u64;
        
        for agent in agent_population.iter() {
            let stats = agent.stats().await;
            
            // Count social learning events
            social_learning_events += stats.xp_history.iter()
                .filter(|entry| entry.category.contains("social") || entry.category.contains("swarm"))
                .count() as u64;
            
            // Count performance improvements
            if stats.goals_processed >= 5 {
                let success_rate = stats.goals_succeeded as f64 / stats.goals_processed as f64;
                if success_rate > 0.7 {
                    collective_performance_improvements += 1;
                }
            }
        }
        
        // Calculate convergence metrics
        let convergence_rate = swarm_metrics.convergence_rate;
        let swarm_diversity = swarm_metrics.diversity_score;
        
        // Best global fitness achieved
        let best_global_fitness = convergence_history.iter()
            .map(|(_, fitness, _)| *fitness)
            .fold(0.0f64, f64::max);
        
        Ok(SwarmXPStats {
            total_social_xp_awarded: total_social_xp,
            agents_with_social_interactions,
            social_learning_events,
            collective_performance_improvements,
            swarm_convergence_rate: convergence_rate,
            swarm_diversity,
            best_global_fitness,
            convergence_generations: convergence_history.len() as u64,
        })
    }

    /// Award XP for achieving swarm convergence milestones
    pub async fn award_convergence_milestone_xp(
        &self,
        milestone_type: &str,
        convergence_quality: f64,
    ) -> EvolutionEngineResult<()> {
        let agent_population = self.agent_population.read().await;
        
        let milestone_xp = match milestone_type {
            "global_optimum_found" => 200,
            "swarm_convergence" => 150,
            "collective_breakthrough" => 180,
            "diversity_maintained" => 100,
            _ => 50,
        };
        
        let quality_bonus = (convergence_quality * 100.0) as u64;
        let total_xp = milestone_xp + quality_bonus;
        
        // Award to all agents in the swarm
        for agent in agent_population.iter() {
            agent.award_xp(
                total_xp,
                format!("Swarm milestone: {} (quality: {:.2})", milestone_type, convergence_quality),
                "swarm_convergence".to_string(),
            ).await.map_err(|e| EvolutionEngineError::EvolutionError {
                message: format!("Failed to award convergence milestone XP: {}", e),
            })?;
        }

        Ok(())
    }
}

/// Swarm-specific XP statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmXPStats {
    pub total_social_xp_awarded: u64,
    pub agents_with_social_interactions: u64,
    pub social_learning_events: u64,
    pub collective_performance_improvements: u64,
    pub swarm_convergence_rate: f64,
    pub swarm_diversity: f64,
    pub best_global_fitness: f64,
    pub convergence_generations: u64,
}

impl SwarmXPStats {
    pub fn summary(&self) -> String {
        format!(
            "Swarm-XP Statistics:\n\
            Social XP Awarded: {} (across {} agents)\n\
            Social Learning Events: {}\n\
            Collective Improvements: {} agents\n\
            Swarm Convergence: {:.2}% (Diversity: {:.2})\n\
            Best Global Fitness: {:.3} (over {} generations)",
            self.total_social_xp_awarded,
            self.agents_with_social_interactions,
            self.social_learning_events,
            self.collective_performance_improvements,
            self.swarm_convergence_rate * 100.0,
            self.swarm_diversity,
            self.best_global_fitness,
            self.convergence_generations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stratoswarm_agent_core::agent::AgentConfig;

    #[tokio::test]
    async fn test_swarm_xp_fitness_function() {
        let fitness_fn = SwarmXPFitnessFunction::default();
        
        let config = AgentConfig {
            name: "test_swarm_agent".to_string(),
            ..Default::default()
        };
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Award some social learning XP
        agent.award_xp(80, "Social learning test".to_string(), "social_learning".to_string()).await.unwrap();
        
        let fitness = fitness_fn.evaluate_agent_fitness(&agent).await;
        assert!(fitness > 0.0);
        assert!(fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_swarm_should_evolve_conditions() {
        let fitness_fn = SwarmXPFitnessFunction::default();
        
        let config = AgentConfig::default();
        let agent = Agent::new(config).unwrap();
        agent.initialize().await.unwrap();
        
        // Should not evolve initially
        assert!(!fitness_fn.should_evolve(&agent).await);
        
        // Award enough social interactions to trigger evolution
        for i in 0..6 {
            agent.award_xp(15, format!("Social interaction {}", i), "social".to_string()).await.unwrap();
        }
        
        assert!(fitness_fn.should_evolve(&agent).await);
    }

    #[tokio::test]
    async fn test_swarm_xp_reward_calculation() {
        let fitness_fn = SwarmXPFitnessFunction::default();
        
        let evolution_metrics = EvolutionMetrics {
            avg_completion_time: std::time::Duration::from_secs(40),
            success_rate: 0.9,
            memory_efficiency: 0.8,
            processing_speed: 1.4,
        };
        
        let reward = fitness_fn.calculate_xp_reward(0.18, &evolution_metrics);
        assert!(reward >= 160); // Should be at least base reward
        assert!(reward > 300); // Should include significant bonuses for swarm intelligence
    }
}