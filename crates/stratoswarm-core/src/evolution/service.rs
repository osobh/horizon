//! Evolution service for processing evolution commands through channels.

use crate::channels::{EvolutionMessage, SystemEvent};
use tokio::sync::{broadcast, mpsc};

/// Configuration for the evolution service.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Number of genes per individual
    pub genome_size: usize,
    /// Mutation rate (0.0 to 1.0)
    pub mutation_rate: f64,
    /// Crossover rate (0.0 to 1.0)
    pub crossover_rate: f64,
    /// Number of elite individuals to preserve
    pub elite_count: usize,
    /// Maximum generations to run
    pub max_generations: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            genome_size: 64,
            mutation_rate: 0.01,
            crossover_rate: 0.8,
            elite_count: 5,
            max_generations: 1000,
        }
    }
}

/// Metrics tracked during evolution.
#[derive(Debug, Clone, Default)]
pub struct EvolutionMetrics {
    /// Current generation number
    pub generation: u64,
    /// Best fitness seen
    pub best_fitness: f64,
    /// Average fitness of current population
    pub average_fitness: f64,
    /// Fitness standard deviation
    pub fitness_std_dev: f64,
    /// Time spent on fitness evaluation (ms)
    pub evaluation_time_ms: u64,
    /// Time spent on selection (ms)
    pub selection_time_ms: u64,
    /// Time spent on mutation/crossover (ms)
    pub reproduction_time_ms: u64,
}

/// Evolution service that processes commands from a channel.
#[derive(Debug)]
pub struct EvolutionService {
    config: EvolutionConfig,
    metrics: EvolutionMetrics,
}

impl EvolutionService {
    /// Create a new evolution service with the given configuration.
    pub fn new(config: EvolutionConfig) -> Self {
        Self {
            config,
            metrics: EvolutionMetrics::default(),
        }
    }

    /// Run the evolution service, processing commands from the receiver.
    pub async fn run(
        &mut self,
        mut evolution_rx: mpsc::Receiver<EvolutionMessage>,
        events_tx: broadcast::Sender<SystemEvent>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        while let Some(message) = evolution_rx.recv().await {
            match message {
                EvolutionMessage::Step { generation } => {
                    self.metrics.generation = generation;
                    // Process evolution step
                }
                EvolutionMessage::EvaluateFitness { individual_ids } => {
                    // Fitness evaluation - broadcast completion events
                    for id in individual_ids {
                        let _ = events_tx.send(SystemEvent::FitnessImproved {
                            individual_id: id,
                            old_fitness: 0.0,
                            new_fitness: self.metrics.best_fitness,
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                        });
                    }
                }
                EvolutionMessage::Selection { strategy: _, count: _ } => {
                    // Selection handled in GPU pipeline
                }
                EvolutionMessage::Mutation { individual_ids: _, rate: _ } => {
                    // Mutation handled in GPU pipeline
                }
                EvolutionMessage::GetBest { count: _ } => {
                    // Return best individuals
                }
            }
        }
        Ok(())
    }

    /// Get current metrics.
    pub fn metrics(&self) -> &EvolutionMetrics {
        &self.metrics
    }

    /// Get configuration.
    pub fn config(&self) -> &EvolutionConfig {
        &self.config
    }
}
