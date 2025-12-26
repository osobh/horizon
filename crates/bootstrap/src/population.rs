//! Population Controller - Manages agent population dynamics

use crate::{
    config::{BootstrapConfig, BootstrapPhase},
    dna::{AgentDNA, PerformanceMetrics},
};
use anyhow::Result;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Population controller manages the agent ecosystem
#[derive(Debug)]
pub struct PopulationController {
    config: BootstrapConfig,
    agents: HashMap<Uuid, AgentRecord>,
    current_mutation_rate: f32,
    generation_counter: u32,
    phase: BootstrapPhase,
    autonomous_mode: bool,
    population_stats: PopulationStats,
}

/// Individual agent record in population
#[derive(Debug, Clone)]
pub struct AgentRecord {
    pub dna: AgentDNA,
    pub status: AgentStatus,
    pub last_metrics: Option<PerformanceMetrics>,
    pub registration_time: u64,
    pub last_activity_time: u64,
}

/// Agent status in population
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Active,
    Inactive,
    Failing,
    Terminated,
}

/// Population statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PopulationStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub average_fitness: f32,
    pub diversity_score: f32,
    pub generation_stats: HashMap<u32, usize>,
    pub type_distribution: HashMap<String, usize>,
}

impl PopulationController {
    /// Create a new population controller
    pub fn new(config: BootstrapConfig) -> Result<Self> {
        Ok(Self {
            config,
            agents: HashMap::new(),
            current_mutation_rate: 0.25, // Start with high mutation rate
            generation_counter: 0,
            phase: BootstrapPhase::Genesis,
            autonomous_mode: false,
            population_stats: PopulationStats::default(),
        })
    }

    /// Register a new agent in the population
    pub async fn register_agent(&mut self, dna: AgentDNA) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("System time error: {}", e))?
            .as_secs();

        let record = AgentRecord {
            dna: dna.clone(),
            status: AgentStatus::Active,
            last_metrics: None,
            registration_time: current_time,
            last_activity_time: current_time,
        };

        self.agents.insert(dna.id, record);
        self.update_population_stats().await;

        tracing::debug!(
            "Registered agent {} (type: {:?}, generation: {})",
            dna.id,
            dna.core_traits.agent_type,
            dna.generation
        );

        Ok(())
    }

    /// Update agent metrics and status
    pub async fn update_agent_metrics(
        &mut self,
        agent_id: Uuid,
        metrics: PerformanceMetrics,
    ) -> Result<()> {
        if let Some(record) = self.agents.get_mut(&agent_id) {
            record.last_metrics = Some(metrics.clone());
            record.last_activity_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| anyhow::anyhow!("System time error: {}", e))?
                .as_secs();

            // Update status based on metrics
            record.status = if metrics.goals_failed > metrics.goals_completed * 3 {
                AgentStatus::Failing
            } else {
                AgentStatus::Active
            };

            self.update_population_stats().await;
        }

        Ok(())
    }

    /// Remove terminated agents from population
    pub async fn remove_agent(&mut self, agent_id: Uuid) -> Result<()> {
        if let Some(mut record) = self.agents.remove(&agent_id) {
            record.status = AgentStatus::Terminated;
            tracing::debug!("Removed agent {agent_id} from population");
            self.update_population_stats().await;
        }
        Ok(())
    }

    /// Calculate current population health score
    pub async fn health_score(&self) -> Result<f32> {
        if self.agents.is_empty() {
            return Ok(0.0);
        }

        let active_count = self
            .agents
            .values()
            .filter(|r| r.status == AgentStatus::Active)
            .count() as f32;

        let failing_count = self
            .agents
            .values()
            .filter(|r| r.status == AgentStatus::Failing)
            .count() as f32;

        let total_count = self.agents.len() as f32;

        // Health score based on active vs failing ratio
        let health = (active_count - failing_count * 0.5) / total_count;

        Ok(health.max(0.0).min(1.0))
    }

    /// Calculate population diversity score
    pub async fn diversity_score(&self) -> Result<f32> {
        if self.agents.is_empty() {
            return Ok(0.0);
        }

        // Calculate trait diversity
        let mut trait_variances: Vec<f32> = Vec::new();

        // Collect all variable traits
        let traits: Vec<_> = self
            .agents
            .values()
            .map(|r| &r.dna.variable_traits)
            .collect();

        if traits.is_empty() {
            return Ok(0.0);
        }

        // Calculate variance for each trait
        let exploration_rates: Vec<f32> = traits.iter().map(|t| t.exploration_rate).collect();
        trait_variances.push(Self::calculate_variance(&exploration_rates));

        let learning_rates: Vec<f32> = traits.iter().map(|t| t.learning_rate).collect();
        trait_variances.push(Self::calculate_variance(&learning_rates));

        let risk_tolerances: Vec<f32> = traits.iter().map(|t| t.risk_tolerance).collect();
        trait_variances.push(Self::calculate_variance(&risk_tolerances));

        let cooperation_rates: Vec<f32> = traits.iter().map(|t| t.cooperation_rate).collect();
        trait_variances.push(Self::calculate_variance(&cooperation_rates));

        // Average variance as diversity score
        let average_variance = trait_variances.iter().sum::<f32>() / trait_variances.len() as f32;

        // Normalize to 0-1 range (variance of 0.25 = max diversity)
        Ok((average_variance / 0.25).min(1.0))
    }

    /// Run a single evolution cycle
    pub async fn run_evolution_cycle(&self) -> Result<()> {
        tracing::info!(
            "Running evolution cycle with mutation rate: {:.3}",
            self.current_mutation_rate
        );

        // Selection: identify high and low fitness agents
        let (survivors, candidates_for_termination) = self.select_agents().await?;

        // Reproduction: create offspring from high-fitness agents
        let offspring = self.create_offspring(&survivors).await?;

        tracing::info!(
            "Evolution cycle: {} survivors, {} offspring, {} to terminate",
            survivors.len(),
            offspring.len(),
            candidates_for_termination.len()
        );

        // Note: In a complete implementation, we would:
        // 1. Actually terminate low-fitness agents
        // 2. Instantiate new offspring agents
        // 3. Update the population records

        Ok(())
    }

    /// Set current mutation rate
    pub async fn set_mutation_rate(&mut self, rate: f32) -> Result<()> {
        self.current_mutation_rate = rate.clamp(0.0, 1.0);
        tracing::debug!(
            "Updated mutation rate to: {:.3}",
            self.current_mutation_rate
        );
        Ok(())
    }

    /// Enable autonomous operation mode
    pub async fn enable_autonomous_mode(&mut self) -> Result<()> {
        self.autonomous_mode = true;
        tracing::info!("Autonomous mode enabled - population will self-manage");
        Ok(())
    }

    /// Get population statistics
    pub fn get_stats(&self) -> &PopulationStats {
        &self.population_stats
    }

    /// Update population statistics
    async fn update_population_stats(&mut self) {
        let total_agents = self.agents.len();
        let active_agents = self
            .agents
            .values()
            .filter(|r| r.status == AgentStatus::Active)
            .count();

        // Calculate average fitness
        let fitness_scores: Vec<f32> = self
            .agents
            .values()
            .filter_map(|r| r.dna.fitness_history.last().copied())
            .collect();

        let average_fitness = if fitness_scores.is_empty() {
            0.0
        } else {
            fitness_scores.iter().sum::<f32>() / fitness_scores.len() as f32
        };

        // Calculate generation distribution
        let mut generation_stats = HashMap::new();
        for record in self.agents.values() {
            *generation_stats.entry(record.dna.generation).or_insert(0) += 1;
        }

        // Calculate type distribution
        let mut type_distribution = HashMap::new();
        for record in self.agents.values() {
            let type_name = format!("{:?}", record.dna.core_traits.agent_type);
            *type_distribution.entry(type_name).or_insert(0) += 1;
        }

        // Calculate diversity score
        let diversity_score = self.diversity_score().await.unwrap_or(0.0);

        self.population_stats = PopulationStats {
            total_agents,
            active_agents,
            average_fitness,
            diversity_score,
            generation_stats,
            type_distribution,
        };
    }

    /// Select agents for survival and reproduction
    async fn select_agents(&self) -> Result<(Vec<AgentDNA>, Vec<Uuid>)> {
        let mut survivors = Vec::new();
        let mut candidates_for_termination = Vec::new();

        for (id, record) in &self.agents {
            if record.status == AgentStatus::Active {
                if let Some(fitness) = record.dna.fitness_history.last() {
                    if *fitness > 20.0 {
                        // Fitness threshold for survival
                        survivors.push(record.dna.clone());
                    } else if *fitness < 5.0 {
                        candidates_for_termination.push(*id);
                    }
                }
            } else if record.status == AgentStatus::Failing {
                candidates_for_termination.push(*id);
            }
        }

        Ok((survivors, candidates_for_termination))
    }

    /// Create offspring from surviving agents
    async fn create_offspring(&self, survivors: &[AgentDNA]) -> Result<Vec<AgentDNA>> {
        let mut offspring = Vec::new();

        for i in 0..survivors.len() {
            if offspring.len() >= self.config.population.max_size {
                break;
            }

            let parent = &survivors[i];

            // Select partner for crossover (if available)
            let partner = if i + 1 < survivors.len() {
                Some(&survivors[i + 1])
            } else if !survivors.is_empty() {
                Some(&survivors[0])
            } else {
                None
            };

            // Create offspring through reproduction
            if let Ok(child_dna) = parent.reproduce(partner, self.current_mutation_rate) {
                offspring.push(child_dna);
            }
        }

        Ok(offspring)
    }

    /// Calculate variance of a trait across population
    fn calculate_variance(values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

impl Default for PopulationStats {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_agents: 0,
            average_fitness: 0.0,
            diversity_score: 0.0,
            generation_stats: HashMap::new(),
            type_distribution: HashMap::new(),
        }
    }
}
