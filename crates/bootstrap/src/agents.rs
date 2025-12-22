//! Template agents for bootstrap initialization

use crate::dna::{AgentDNA, AgentType, PerformanceMetrics};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Base trait for template agents
#[async_trait::async_trait]
pub trait TemplateAgent: Send + Sync {
    /// Get the agent's DNA
    fn dna(&self) -> &AgentDNA;

    /// Get mutable DNA reference
    fn dna_mut(&mut self) -> &mut AgentDNA;

    /// Execute agent's primary function
    async fn execute_primary_function(&mut self) -> Result<PerformanceMetrics>;

    /// Create offspring agent
    async fn replicate(
        &self,
        partner: Option<&dyn TemplateAgent>,
        mutation_rate: f32,
    ) -> Result<Box<dyn TemplateAgent>>;

    /// Update agent based on experience
    async fn learn_from_experience(&mut self, metrics: &PerformanceMetrics) -> Result<()>;

    /// Check if agent should terminate
    fn should_terminate(&self) -> bool;
}

/// PrimeAgent - Most basic self-replicating agent
pub struct PrimeAgent {
    dna: AgentDNA,
    performance_history: Vec<PerformanceMetrics>,
    last_execution: Option<std::time::Instant>,
}

impl PrimeAgent {
    /// Create a new PrimeAgent
    pub async fn new() -> Result<Self> {
        let dna = AgentDNA::create_template(AgentType::Prime);

        Ok(Self {
            dna,
            performance_history: Vec::new(),
            last_execution: None,
        })
    }

    /// Basic goal execution capability
    async fn execute_basic_goal(&mut self) -> Result<PerformanceMetrics> {
        let start_time = std::time::Instant::now();
        let mut metrics = PerformanceMetrics {
            survival_time: 0,
            goals_completed: 0,
            goals_failed: 0,
            offspring_created: 0,
            resources_used: 0,
            resources_allocated: self.dna.core_traits.base_memory as u64 * 1024 * 1024,
            kernels_synthesized: 0,
            successful_interactions: 0,
        };

        // Simulate basic computation work
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Mock successful execution
        if rand::random::<f32>() > 0.2 {
            // 80% success rate
            metrics.goals_completed += 1;
            metrics.kernels_synthesized += 1;
            self.dna.update_experience("basic_computation", true, 0.5);
        } else {
            metrics.goals_failed += 1;
            self.dna.update_experience("basic_computation", false, 0.8);
        }

        metrics.survival_time = start_time.elapsed().as_millis().max(1) as u64; // Ensure at least 1ms
        metrics.resources_used = (self.dna.variable_traits.efficiency_modifier
            * metrics.resources_allocated as f32) as u64;

        self.performance_history.push(metrics.clone());
        Ok(metrics)
    }

    /// Mock goal execution pipeline
    async fn execute_goal_pipeline(&mut self) -> Result<()> {
        // Mock execution delay
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        Ok(())
    }
}

#[async_trait::async_trait]
impl TemplateAgent for PrimeAgent {
    fn dna(&self) -> &AgentDNA {
        &self.dna
    }

    fn dna_mut(&mut self) -> &mut AgentDNA {
        &mut self.dna
    }

    async fn execute_primary_function(&mut self) -> Result<PerformanceMetrics> {
        self.last_execution = Some(std::time::Instant::now());
        self.execute_basic_goal().await
    }

    async fn replicate(
        &self,
        partner: Option<&dyn TemplateAgent>,
        mutation_rate: f32,
    ) -> Result<Box<dyn TemplateAgent>> {
        let partner_dna = partner.map(|p| p.dna());
        let offspring_dna = self.dna.reproduce(partner_dna, mutation_rate)?;

        let mut offspring = Self::new().await?;
        offspring.dna = offspring_dna;

        Ok(Box::new(offspring))
    }

    async fn learn_from_experience(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        let fitness = self.dna.calculate_fitness(metrics);
        self.dna.fitness_history.push(fitness);

        // Adapt variable traits based on performance
        if fitness > 50.0 {
            // Good performance - slightly reduce exploration, increase exploitation
            self.dna.variable_traits.exploration_rate *= 0.95;
            self.dna.variable_traits.efficiency_modifier *= 1.02;
        } else {
            // Poor performance - increase exploration
            self.dna.variable_traits.exploration_rate *= 1.05;
            self.dna.variable_traits.exploration_rate =
                self.dna.variable_traits.exploration_rate.min(1.0);
        }

        Ok(())
    }

    fn should_terminate(&self) -> bool {
        // Terminate if fitness is consistently low
        if self.dna.fitness_history.len() >= 5 {
            let recent_avg = self.dna.fitness_history.iter().rev().take(5).sum::<f32>() / 5.0;
            recent_avg < 10.0
        } else {
            false
        }
    }
}

/// ReplicatorAgent - Specialized in creating new agents
pub struct ReplicatorAgent {
    dna: AgentDNA,
    replication_count: u32,
    agent_templates: Vec<AgentType>,
}

impl ReplicatorAgent {
    pub async fn new() -> Result<Self> {
        let dna = AgentDNA::create_template(AgentType::Replicator);

        Ok(Self {
            dna,
            replication_count: 0,
            agent_templates: vec![AgentType::Prime, AgentType::Specialized],
        })
    }

    async fn create_new_agent(&mut self) -> Result<PerformanceMetrics> {
        let start_time = std::time::Instant::now();
        let mut metrics = PerformanceMetrics {
            survival_time: 0,
            goals_completed: 0,
            goals_failed: 0,
            offspring_created: 0,
            resources_used: 0,
            resources_allocated: self.dna.core_traits.base_memory as u64 * 1024 * 1024,
            kernels_synthesized: 0,
            successful_interactions: 1,
        };

        // Select template type based on current needs
        let template_type = self.select_template_type();

        // Create new agent (mock implementation)
        match self.instantiate_template(template_type).await {
            Ok(_) => {
                self.replication_count += 1;
                metrics.offspring_created += 1;
                metrics.goals_completed += 1;

                // Update DNA with successful replication
                self.dna.lineage.children.push(Uuid::new_v4());
                self.dna.update_experience("agent_creation", true, 0.6);
            }
            Err(_) => {
                metrics.goals_failed += 1;
                self.dna.update_experience("agent_creation", false, 0.9);
            }
        }

        metrics.survival_time = start_time.elapsed().as_millis().max(1) as u64; // Ensure at least 1ms
        metrics.resources_used = (0.7 * metrics.resources_allocated as f32) as u64; // Replicator is efficient

        Ok(metrics)
    }

    fn select_template_type(&self) -> AgentType {
        // Simple selection logic - could be more sophisticated
        use rand::seq::SliceRandom;
        *self
            .agent_templates
            .choose(&mut rand::thread_rng())
            .unwrap_or(&AgentType::Prime)
    }

    async fn instantiate_template(&self, agent_type: AgentType) -> Result<()> {
        // Mock implementation - in real system would create actual agent instance
        tracing::info!("Creating new agent of type: {:?}", agent_type);

        // Simulate creation time
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        Ok(())
    }
}

#[async_trait::async_trait]
impl TemplateAgent for ReplicatorAgent {
    fn dna(&self) -> &AgentDNA {
        &self.dna
    }

    fn dna_mut(&mut self) -> &mut AgentDNA {
        &mut self.dna
    }

    async fn execute_primary_function(&mut self) -> Result<PerformanceMetrics> {
        self.create_new_agent().await
    }

    async fn replicate(
        &self,
        partner: Option<&dyn TemplateAgent>,
        mutation_rate: f32,
    ) -> Result<Box<dyn TemplateAgent>> {
        let partner_dna = partner.map(|p| p.dna());
        let offspring_dna = self.dna.reproduce(partner_dna, mutation_rate)?;

        let mut offspring = Self::new().await?;
        offspring.dna = offspring_dna;

        Ok(Box::new(offspring))
    }

    async fn learn_from_experience(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        let fitness = self.dna.calculate_fitness(metrics);
        self.dna.fitness_history.push(fitness);

        // Replicator adapts based on creation success
        if metrics.offspring_created > 0 {
            self.dna.variable_traits.efficiency_modifier *= 1.01;
            self.dna.variable_traits.cooperation_rate *= 1.02;
        } else {
            self.dna.variable_traits.learning_rate *= 1.05;
        }

        Ok(())
    }

    fn should_terminate(&self) -> bool {
        // Replicators are valuable - only terminate if completely failing
        if self.dna.fitness_history.len() >= 10 {
            let recent_avg = self.dna.fitness_history.iter().rev().take(10).sum::<f32>() / 10.0;
            recent_avg < 5.0
        } else {
            false
        }
    }
}

/// EvolutionAgent - Drives mutation and fitness evaluation
pub struct EvolutionAgent {
    dna: AgentDNA,
    evaluations_performed: u32,
}

impl EvolutionAgent {
    pub async fn new() -> Result<Self> {
        let dna = AgentDNA::create_template(AgentType::Evolution);

        Ok(Self {
            dna,
            evaluations_performed: 0,
        })
    }

    async fn evaluate_population(&mut self) -> Result<PerformanceMetrics> {
        let start_time = std::time::Instant::now();
        let mut metrics = PerformanceMetrics {
            survival_time: 0,
            goals_completed: 0,
            goals_failed: 0,
            offspring_created: 0,
            resources_used: 0,
            resources_allocated: self.dna.core_traits.base_memory as u64 * 1024 * 1024,
            kernels_synthesized: 0,
            successful_interactions: 0,
        };

        // Mock fitness evaluation
        match self.perform_fitness_evaluation().await {
            Ok(evaluations) => {
                self.evaluations_performed += evaluations as u32;
                metrics.goals_completed += 1;
                metrics.successful_interactions += evaluations as u32;

                self.dna.update_experience("fitness_evaluation", true, 0.4);
            }
            Err(_) => {
                metrics.goals_failed += 1;
                self.dna.update_experience("fitness_evaluation", false, 0.7);
            }
        }

        metrics.survival_time = start_time.elapsed().as_millis().max(1) as u64; // Ensure at least 1ms
        metrics.resources_used = (1.2 * metrics.resources_allocated as f32) as u64; // Evolution is resource intensive

        Ok(metrics)
    }

    async fn perform_fitness_evaluation(&mut self) -> Result<u64> {
        // Mock implementation - would evaluate actual agent population
        tracing::info!("Performing fitness evaluation on population");

        // Simulate evaluation work
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Return number of agents evaluated
        Ok(rand::random::<u8>() as u64 % 20 + 5) // 5-24 agents evaluated
    }
}

#[async_trait::async_trait]
impl TemplateAgent for EvolutionAgent {
    fn dna(&self) -> &AgentDNA {
        &self.dna
    }

    fn dna_mut(&mut self) -> &mut AgentDNA {
        &mut self.dna
    }

    async fn execute_primary_function(&mut self) -> Result<PerformanceMetrics> {
        self.evaluate_population().await
    }

    async fn replicate(
        &self,
        partner: Option<&dyn TemplateAgent>,
        mutation_rate: f32,
    ) -> Result<Box<dyn TemplateAgent>> {
        let partner_dna = partner.map(|p| p.dna());
        let offspring_dna = self.dna.reproduce(partner_dna, mutation_rate)?;

        let mut offspring = Self::new().await?;
        offspring.dna = offspring_dna;

        Ok(Box::new(offspring))
    }

    async fn learn_from_experience(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        let fitness = self.dna.calculate_fitness(metrics);
        self.dna.fitness_history.push(fitness);

        // Evolution agents adapt their evaluation strategy
        if fitness > 30.0 {
            self.dna.variable_traits.specialization *= 1.01;
            self.dna.variable_traits.exploration_rate *= 0.98; // Become more focused
        } else {
            self.dna.variable_traits.exploration_rate *= 1.03; // Explore more
        }

        Ok(())
    }

    fn should_terminate(&self) -> bool {
        // Evolution agents are critical - very conservative termination
        if self.dna.fitness_history.len() >= 15 {
            let recent_avg = self.dna.fitness_history.iter().rev().take(15).sum::<f32>() / 15.0;
            recent_avg < 1.0
        } else {
            false
        }
    }
}
