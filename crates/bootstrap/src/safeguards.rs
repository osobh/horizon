//! Bootstrap Safeguards - Safety mechanisms for the bootstrap process

use crate::{
    config::{BootstrapConfig, SafeguardConfig},
    population::PopulationController,
};
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bootstrap safeguards monitor and intervene when necessary
#[derive(Debug)]
pub struct BootstrapSafeguards {
    config: SafeguardConfig,
    population_history: VecDeque<PopulationSnapshot>,
    failure_count: usize,
    last_intervention: Option<std::time::Instant>,
    emergency_mode: bool,
}

/// Snapshot of population state for trend analysis
#[derive(Debug, Clone)]
struct PopulationSnapshot {
    timestamp: u64,
    total_agents: usize,
    active_agents: usize,
    average_fitness: f32,
    diversity_score: f32,
    resource_usage: f32,
}

/// Types of safeguard interventions
#[derive(Debug, Clone)]
pub enum SafeguardIntervention {
    /// Reduce mutation rate to stabilize population
    ReduceMutationRate { new_rate: f32 },
    /// Increase mutation rate to improve diversity
    IncreaseMutationRate { new_rate: f32 },
    /// Inject new template agents
    InjectTemplateAgents { count: usize },
    /// Remove failing agents
    CullFailingAgents { threshold: f32 },
    /// Emergency population reset
    EmergencyReset,
    /// Resource limitation enforcement
    EnforceResourceLimits,
    /// None - no intervention needed
    None,
}

impl BootstrapSafeguards {
    /// Create new bootstrap safeguards
    pub fn new(config: BootstrapConfig) -> Result<Self> {
        Ok(Self {
            config: config.safeguards,
            population_history: VecDeque::with_capacity(100), // Keep last 100 snapshots
            failure_count: 0,
            last_intervention: None,
            emergency_mode: false,
        })
    }

    /// Check safeguards and intervene if necessary
    pub async fn check_and_intervene(&mut self) -> Result<SafeguardIntervention> {
        // Update population snapshot
        self.update_population_snapshot().await?;

        // Check various safeguard conditions
        let interventions = vec![
            self.check_population_explosion().await?,
            self.check_diversity_crisis().await?,
            self.check_resource_exhaustion().await?,
            self.check_population_collapse().await?,
            self.check_fitness_stagnation().await?,
        ];

        // Select the most critical intervention
        for intervention in interventions {
            if !matches!(intervention, SafeguardIntervention::None) {
                self.record_intervention(&intervention).await?;
                return Ok(intervention);
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Check for population explosion
    async fn check_population_explosion(&self) -> Result<SafeguardIntervention> {
        if !self.config.prevent_explosion {
            return Ok(SafeguardIntervention::None);
        }

        if let (Some(current), Some(previous)) = (
            self.population_history.back(),
            self.population_history
                .get(self.population_history.len().saturating_sub(10)),
        ) {
            let growth_rate = (current.total_agents as f32 / previous.total_agents as f32) - 1.0;

            if growth_rate > 0.5 {
                // 50% growth is concerning
                tracing::warn!(
                    "Population explosion detected: growth rate {:.2}%",
                    growth_rate * 100.0
                );

                // Reduce mutation rate to slow reproduction
                return Ok(SafeguardIntervention::ReduceMutationRate { new_rate: 0.05 });
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Check for diversity crisis
    async fn check_diversity_crisis(&self) -> Result<SafeguardIntervention> {
        if !self.config.diversity_protection {
            return Ok(SafeguardIntervention::None);
        }

        if let Some(current) = self.population_history.back() {
            if current.diversity_score < 0.2 {
                // Critical diversity threshold
                tracing::warn!(
                    "Diversity crisis detected: score {:.3}",
                    current.diversity_score
                );

                // Increase mutation rate and inject new templates
                if current.diversity_score < 0.1 {
                    return Ok(SafeguardIntervention::InjectTemplateAgents { count: 5 });
                } else {
                    return Ok(SafeguardIntervention::IncreaseMutationRate { new_rate: 0.4 });
                }
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Check for resource exhaustion
    async fn check_resource_exhaustion(&self) -> Result<SafeguardIntervention> {
        if !self.config.resource_protection {
            return Ok(SafeguardIntervention::None);
        }

        if let Some(current) = self.population_history.back() {
            if current.resource_usage > 0.9 {
                // 90% resource usage threshold
                tracing::warn!(
                    "Resource exhaustion detected: usage {:.1}%",
                    current.resource_usage * 100.0
                );

                return Ok(SafeguardIntervention::EnforceResourceLimits);
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Check for population collapse
    async fn check_population_collapse(&self) -> Result<SafeguardIntervention> {
        if let Some(current) = self.population_history.back() {
            let active_ratio = current.active_agents as f32 / current.total_agents.max(1) as f32;

            if active_ratio < 0.3 || current.total_agents < 3 {
                tracing::error!(
                    "Population collapse detected: {} total, {} active",
                    current.total_agents,
                    current.active_agents
                );

                if self.config.emergency_reset && current.total_agents < 2 {
                    return Ok(SafeguardIntervention::EmergencyReset);
                } else {
                    return Ok(SafeguardIntervention::InjectTemplateAgents { count: 10 });
                }
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Check for fitness stagnation
    async fn check_fitness_stagnation(&self) -> Result<SafeguardIntervention> {
        if self.population_history.len() < 20 {
            return Ok(SafeguardIntervention::None);
        }

        // Check if fitness hasn't improved in the last 20 snapshots
        let recent_fitness: Vec<f32> = self
            .population_history
            .iter()
            .rev()
            .take(20)
            .map(|s| s.average_fitness)
            .collect();

        if let (Some(&first), Some(&last)) = (recent_fitness.last(), recent_fitness.first()) {
            let improvement = (last - first) / first.max(1.0);

            if improvement < 0.05 {
                // Less than 5% improvement
                tracing::warn!(
                    "Fitness stagnation detected: {:.1}% improvement over 20 cycles",
                    improvement * 100.0
                );

                // Increase mutation rate to encourage exploration
                return Ok(SafeguardIntervention::IncreaseMutationRate { new_rate: 0.3 });
            }
        }

        Ok(SafeguardIntervention::None)
    }

    /// Update population snapshot for monitoring
    async fn update_population_snapshot(&mut self) -> Result<()> {
        // In a complete implementation, this would query the actual population controller
        // For now, we'll create a mock snapshot
        let snapshot = PopulationSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                ?
                .as_secs(),
            total_agents: 15, // Mock data
            active_agents: 12,
            average_fitness: 25.0,
            diversity_score: 0.4,
            resource_usage: 0.6,
        };

        self.population_history.push_back(snapshot);

        // Keep only last 100 snapshots
        if self.population_history.len() > 100 {
            self.population_history.pop_front();
        }

        Ok(())
    }

    /// Record intervention and update failure count
    async fn record_intervention(&mut self, intervention: &SafeguardIntervention) -> Result<()> {
        self.last_intervention = Some(std::time::Instant::now());

        match intervention {
            SafeguardIntervention::EmergencyReset => {
                self.emergency_mode = true;
                self.failure_count += 1;
                tracing::error!(
                    "EMERGENCY RESET triggered - failure count: {}",
                    self.failure_count
                );
            }
            SafeguardIntervention::InjectTemplateAgents { count } => {
                tracing::warn!(
                    "Injecting {} template agents for population recovery",
                    count
                );
            }
            SafeguardIntervention::CullFailingAgents { threshold } => {
                tracing::warn!("Culling agents with fitness below {:.2}", threshold);
            }
            SafeguardIntervention::ReduceMutationRate { new_rate } => {
                tracing::info!(
                    "Reducing mutation rate to {:.3} for stabilization",
                    new_rate
                );
            }
            SafeguardIntervention::IncreaseMutationRate { new_rate } => {
                tracing::info!("Increasing mutation rate to {:.3} for diversity", new_rate);
            }
            SafeguardIntervention::EnforceResourceLimits => {
                tracing::warn!("Enforcing resource limits due to exhaustion");
            }
            SafeguardIntervention::None => {}
        }

        // Check if we've exceeded max failures
        if self.failure_count >= self.config.max_failures {
            tracing::error!("Maximum failure count reached: {}", self.failure_count);
            self.emergency_mode = true;
        }

        Ok(())
    }

    /// Execute a safeguard intervention
    pub async fn execute_intervention(
        &mut self,
        intervention: SafeguardIntervention,
        population: Arc<RwLock<PopulationController>>,
    ) -> Result<()> {
        match intervention {
            SafeguardIntervention::ReduceMutationRate { new_rate } => {
                let mut pop = population.write().await;
                pop.set_mutation_rate(new_rate).await?;
            }
            SafeguardIntervention::IncreaseMutationRate { new_rate } => {
                let mut pop = population.write().await;
                pop.set_mutation_rate(new_rate).await?;
            }
            SafeguardIntervention::InjectTemplateAgents { count } => {
                // In complete implementation, would create and register new template agents
                tracing::info!("Would inject {} template agents", count);
            }
            SafeguardIntervention::CullFailingAgents {
                threshold: _threshold,
            } => {
                // In complete implementation, would remove low-fitness agents
                tracing::info!("Would cull failing agents");
            }
            SafeguardIntervention::EmergencyReset => {
                // In complete implementation, would reset entire population
                tracing::error!("Would perform emergency population reset");
                self.emergency_mode = true;
            }
            SafeguardIntervention::EnforceResourceLimits => {
                // In complete implementation, would apply strict resource limits
                tracing::info!("Would enforce strict resource limits");
            }
            SafeguardIntervention::None => {}
        }

        Ok(())
    }

    /// Check if system is in emergency mode
    pub fn is_emergency_mode(&self) -> bool {
        self.emergency_mode
    }

    /// Get time since last intervention
    pub fn time_since_last_intervention(&self) -> Option<std::time::Duration> {
        self.last_intervention.map(|time| time.elapsed())
    }

    /// Get current failure count
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }

    /// Reset safeguards to normal operation
    pub fn reset(&mut self) {
        self.emergency_mode = false;
        self.failure_count = 0;
        self.last_intervention = None;
        self.population_history.clear();
        tracing::info!("Bootstrap safeguards reset to normal operation");
    }
}
