//! Evolution streaming pipeline core implementation

use crate::{
    archive::AgentArchive, evaluation::GpuBatchEvaluator, mutation::MutationProcessor, AgentGenome,
    EvolutionStreamingError, SelectionStrategy,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::timeout;

use super::{
    builder::EvolutionStreamingPipelineBuilder, result::EvolutionCycleResult, stats::PipelineStats,
};

/// Evolution streaming pipeline for coordinated agent processing
#[derive(Clone)]
pub struct EvolutionStreamingPipeline {
    pub(super) mutation_processor: Arc<RwLock<MutationProcessor>>,
    pub(super) evaluation_processor: Arc<RwLock<GpuBatchEvaluator>>,
    pub(super) archive: Arc<AgentArchive>,
    pub(super) selection_strategy: SelectionStrategy,
    pub(super) batch_size: usize,
    pub(super) pipeline_timeout: Duration,
    pub(super) stats: Arc<RwLock<PipelineStats>>,
}

impl EvolutionStreamingPipeline {
    /// Create a new pipeline builder
    pub fn builder() -> EvolutionStreamingPipelineBuilder {
        EvolutionStreamingPipelineBuilder::new()
    }

    /// Run a single evolution cycle
    pub async fn run_cycle(&self) -> Result<EvolutionCycleResult, EvolutionStreamingError> {
        let cycle_start = Instant::now();
        let mut cycle_result = EvolutionCycleResult::default();

        // Step 1: Selection
        let selection_start = Instant::now();
        let selected_agents = self
            .archive
            .select_agents(self.batch_size, self.selection_strategy.clone())
            .await?;
        cycle_result.selection_time = selection_start.elapsed();
        cycle_result.selected_count = selected_agents.len();

        if selected_agents.is_empty() {
            return Err(EvolutionStreamingError::PipelineError(
                "No agents available for selection".to_string(),
            ));
        }

        // Step 2: Mutation
        let mutation_start = Instant::now();
        let mutated_agents = {
            let mutation_processor = self.mutation_processor.write().await;
            mutation_processor
                .process_agent_batch(selected_agents)
                .await?
        };
        cycle_result.mutation_time = mutation_start.elapsed();
        cycle_result.mutated_count = mutated_agents.len();

        // Step 3: Evaluation
        let evaluation_start = Instant::now();
        let agent_genomes: Vec<AgentGenome> =
            mutated_agents.into_iter().map(|ma| ma.mutated).collect();

        let evaluation_results = {
            let evaluation_processor = self.evaluation_processor.read().await;
            evaluation_processor.evaluate_batch(agent_genomes).await?
        };
        cycle_result.evaluation_time = evaluation_start.elapsed();
        cycle_result.evaluated_count = evaluation_results.len();

        // Step 4: Archive updates
        let archive_start = Instant::now();
        let mut novel_agents = 0;
        for result in evaluation_results {
            if let Some((genome, _)) = self.find_agent_by_id(result.agent_id).await {
                if self.archive.add_if_novel(genome, result.fitness).await? {
                    novel_agents += 1;
                }
            }
        }
        cycle_result.archive_time = archive_start.elapsed();
        cycle_result.novel_agents = novel_agents;

        cycle_result.total_time = cycle_start.elapsed();

        // Update pipeline statistics
        {
            let mut stats = self.stats.write().await;
            stats.cycles_completed += 1;
            stats.agents_processed += cycle_result.selected_count as u64;
            stats.mutations_generated += cycle_result.mutated_count as u64;
            stats.evaluations_completed += cycle_result.evaluated_count as u64;
            stats.archive_updates += cycle_result.novel_agents as u64;
            stats.total_processing_time += cycle_result.total_time;
            stats.average_cycle_time =
                stats.total_processing_time / stats.cycles_completed.max(1) as u32;

            if cycle_result.total_time.as_secs_f64() > 0.0 {
                stats.throughput_agents_per_sec =
                    cycle_result.selected_count as f64 / cycle_result.total_time.as_secs_f64();
            }
        }

        Ok(cycle_result)
    }

    /// Run multiple evolution cycles
    pub async fn run_cycles(
        &self,
        cycle_count: usize,
    ) -> Result<Vec<EvolutionCycleResult>, EvolutionStreamingError> {
        let mut results = Vec::with_capacity(cycle_count);

        for cycle_idx in 0..cycle_count {
            match timeout(self.pipeline_timeout, self.run_cycle()).await {
                Ok(Ok(result)) => {
                    results.push(result);
                }
                Ok(Err(e)) => {
                    eprintln!("Cycle {} failed: {}", cycle_idx, e);
                    return Err(e);
                }
                Err(_) => {
                    return Err(EvolutionStreamingError::PipelineError(format!(
                        "Cycle {} timed out after {:?}",
                        cycle_idx, self.pipeline_timeout
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Run evolution pipeline continuously
    pub async fn run_continuous(
        &self,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> Result<(), EvolutionStreamingError> {
        let mut cycle_count = 0u64;

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    println!("Evolution pipeline shutting down after {cycle_count} cycles");
                    break;
                }

                cycle_result = self.run_cycle() => {
                    match cycle_result {
                        Ok(result) => {
                            cycle_count += 1;
                            if cycle_count % 10 == 0 {
                                println!("Completed {} evolution cycles, latest: {:?}", cycle_count, result);
                            }
                        }
                        Err(e) => {
                            eprintln!("Evolution cycle failed: {}", e);
                            // Continue running unless it's a critical error
                            if matches!(e, EvolutionStreamingError::ResourceExhausted { .. }) {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add initial agents to bootstrap the evolution process
    pub async fn bootstrap_with_agents(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<usize, EvolutionStreamingError> {
        let mut added_count = 0;

        for agent in agents {
            // Give initial agents a baseline fitness
            let initial_fitness = 0.5;
            if self.archive.add_if_novel(agent, initial_fitness).await? {
                added_count += 1;
            }
        }

        Ok(added_count)
    }

    /// Get current pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    /// Get archive statistics
    pub async fn get_archive_stats(&self) -> crate::archive::ArchiveStatistics {
        self.archive.get_stats()
    }

    /// Get current archive size
    pub fn archive_size(&self) -> usize {
        self.archive.size()
    }

    /// Check if archive is empty
    pub fn is_archive_empty(&self) -> bool {
        self.archive.is_empty()
    }

    /// Get best fitness from archive
    pub async fn best_fitness(&self) -> Option<f64> {
        self.archive.best_fitness().await
    }

    // Helper methods

    /// Find agent by ID in recent processing
    async fn find_agent_by_id(&self, _agent_id: crate::AgentId) -> Option<(AgentGenome, f64)> {
        // This is a simplified implementation
        // In a real system, we'd track recent agents more efficiently
        None // Placeholder - would need to track recent mutations
    }
}
