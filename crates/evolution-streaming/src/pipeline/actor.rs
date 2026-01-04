//! Actor-based evolution pipeline for cancel-safe async operations
//!
//! This module provides an actor-based implementation of the evolution pipeline
//! that eliminates shared mutable state and provides cancel-safe operations.
//!
//! # Cancel Safety
//!
//! The actor model ensures cancel safety by:
//! 1. Actor owns all mutable state exclusively - no Arc<RwLock<...>>
//! 2. Callers communicate via message passing, not shared memory
//! 3. If a caller's future is cancelled, the actor continues processing
//! 4. Graceful shutdown via explicit Shutdown message

use crate::{
    archive::AgentArchive, evaluation::GpuBatchEvaluator, mutation::MutationProcessor, AgentGenome,
    EvolutionStreamingError, SelectionStrategy,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

use super::{result::EvolutionCycleResult, stats::PipelineStats};

/// Requests that can be sent to the evolution pipeline actor
#[derive(Debug)]
pub enum PipelineRequest {
    /// Run a single evolution cycle
    RunCycle {
        reply: oneshot::Sender<Result<EvolutionCycleResult, EvolutionStreamingError>>,
    },
    /// Run multiple evolution cycles
    RunCycles {
        count: usize,
        reply: oneshot::Sender<Result<Vec<EvolutionCycleResult>, EvolutionStreamingError>>,
    },
    /// Bootstrap with initial agents
    Bootstrap {
        agents: Vec<AgentGenome>,
        reply: oneshot::Sender<Result<usize, EvolutionStreamingError>>,
    },
    /// Get current statistics
    GetStats {
        reply: oneshot::Sender<PipelineStats>,
    },
    /// Get archive statistics
    GetArchiveStats {
        reply: oneshot::Sender<crate::archive::ArchiveStatistics>,
    },
    /// Get best fitness
    GetBestFitness { reply: oneshot::Sender<Option<f64>> },
    /// Graceful shutdown
    Shutdown,
}

/// Evolution pipeline actor that owns all mutable state
///
/// This actor processes requests sequentially, eliminating the need for
/// Arc<RwLock<...>> and providing inherent cancel safety.
pub struct EvolutionPipelineActor {
    /// Mutation processor - owned, not shared
    mutation_processor: MutationProcessor,
    /// Evaluation processor - owned, not shared
    evaluation_processor: GpuBatchEvaluator,
    /// Agent archive - shared for efficiency (read-mostly)
    archive: Arc<AgentArchive>,
    /// Selection strategy
    selection_strategy: SelectionStrategy,
    /// Batch size for processing
    batch_size: usize,
    /// Pipeline timeout
    pipeline_timeout: Duration,
    /// Pipeline statistics - owned, not shared
    stats: PipelineStats,
    /// Request receiver
    inbox: mpsc::Receiver<PipelineRequest>,
}

impl EvolutionPipelineActor {
    /// Create a new actor with its components
    pub fn new(
        mutation_processor: MutationProcessor,
        evaluation_processor: GpuBatchEvaluator,
        archive: Arc<AgentArchive>,
        selection_strategy: SelectionStrategy,
        batch_size: usize,
        pipeline_timeout: Duration,
        inbox: mpsc::Receiver<PipelineRequest>,
    ) -> Self {
        Self {
            mutation_processor,
            evaluation_processor,
            archive,
            selection_strategy,
            batch_size,
            pipeline_timeout,
            stats: PipelineStats::default(),
            inbox,
        }
    }

    /// Run the actor's message processing loop
    ///
    /// This method runs until a Shutdown message is received or the
    /// channel is closed. Each request is processed to completion
    /// before the next is handled.
    pub async fn run(mut self) {
        while let Some(request) = self.inbox.recv().await {
            match request {
                PipelineRequest::RunCycle { reply } => {
                    let result = self.run_cycle().await;
                    // Ignore send errors - caller may have been cancelled
                    let _ = reply.send(result);
                }
                PipelineRequest::RunCycles { count, reply } => {
                    let result = self.run_cycles(count).await;
                    let _ = reply.send(result);
                }
                PipelineRequest::Bootstrap { agents, reply } => {
                    let result = self.bootstrap_with_agents(agents).await;
                    let _ = reply.send(result);
                }
                PipelineRequest::GetStats { reply } => {
                    let _ = reply.send(self.stats.clone());
                }
                PipelineRequest::GetArchiveStats { reply } => {
                    let _ = reply.send(self.archive.get_stats());
                }
                PipelineRequest::GetBestFitness { reply } => {
                    let result = self.archive.best_fitness().await;
                    let _ = reply.send(result);
                }
                PipelineRequest::Shutdown => {
                    // Graceful shutdown
                    break;
                }
            }
        }
        // Actor cleanup happens here via Drop
    }

    /// Run a single evolution cycle (internal implementation)
    async fn run_cycle(&mut self) -> Result<EvolutionCycleResult, EvolutionStreamingError> {
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

        // Step 2: Mutation - no lock needed, we own the processor
        let mutation_start = Instant::now();
        let mutated_agents = self
            .mutation_processor
            .process_agent_batch(selected_agents)
            .await?;
        cycle_result.mutation_time = mutation_start.elapsed();
        cycle_result.mutated_count = mutated_agents.len();

        // Step 3: Evaluation - no lock needed, we own the evaluator
        let evaluation_start = Instant::now();
        let agent_genomes: Vec<AgentGenome> =
            mutated_agents.into_iter().map(|ma| ma.mutated).collect();

        let evaluation_results = self
            .evaluation_processor
            .evaluate_batch(agent_genomes)
            .await?;
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

        // Update statistics - no lock needed, we own stats
        self.stats.cycles_completed += 1;
        self.stats.agents_processed += cycle_result.selected_count as u64;
        self.stats.mutations_generated += cycle_result.mutated_count as u64;
        self.stats.evaluations_completed += cycle_result.evaluated_count as u64;
        self.stats.archive_updates += cycle_result.novel_agents as u64;
        self.stats.total_processing_time += cycle_result.total_time;
        self.stats.average_cycle_time =
            self.stats.total_processing_time / self.stats.cycles_completed.max(1) as u32;

        if cycle_result.total_time.as_secs_f64() > 0.0 {
            self.stats.throughput_agents_per_sec =
                cycle_result.selected_count as f64 / cycle_result.total_time.as_secs_f64();
        }

        Ok(cycle_result)
    }

    /// Run multiple evolution cycles
    async fn run_cycles(
        &mut self,
        cycle_count: usize,
    ) -> Result<Vec<EvolutionCycleResult>, EvolutionStreamingError> {
        let mut results = Vec::with_capacity(cycle_count);

        for cycle_idx in 0..cycle_count {
            match tokio::time::timeout(self.pipeline_timeout, self.run_cycle()).await {
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

    /// Add initial agents to bootstrap the evolution process
    async fn bootstrap_with_agents(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<usize, EvolutionStreamingError> {
        let mut added_count = 0;

        for agent in agents {
            let initial_fitness = 0.5;
            if self.archive.add_if_novel(agent, initial_fitness).await? {
                added_count += 1;
            }
        }

        Ok(added_count)
    }

    /// Find agent by ID (placeholder - would need tracking implementation)
    async fn find_agent_by_id(&self, _agent_id: crate::AgentId) -> Option<(AgentGenome, f64)> {
        None
    }
}

/// Handle for interacting with the evolution pipeline actor
///
/// This handle is cheap to clone and can be used from multiple tasks.
/// All operations are cancel-safe - if the caller's future is dropped,
/// the actor continues processing.
#[derive(Clone)]
pub struct EvolutionPipelineHandle {
    sender: mpsc::Sender<PipelineRequest>,
}

impl EvolutionPipelineHandle {
    /// Create a new handle from a sender
    pub fn new(sender: mpsc::Sender<PipelineRequest>) -> Self {
        Self { sender }
    }

    /// Run a single evolution cycle
    ///
    /// This operation is cancel-safe. If the caller's future is dropped,
    /// the cycle continues to completion in the actor.
    pub async fn run_cycle(&self) -> Result<EvolutionCycleResult, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::RunCycle { reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))?
    }

    /// Run multiple evolution cycles
    pub async fn run_cycles(
        &self,
        count: usize,
    ) -> Result<Vec<EvolutionCycleResult>, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::RunCycles { count, reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))?
    }

    /// Bootstrap with initial agents
    pub async fn bootstrap_with_agents(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<usize, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::Bootstrap { agents, reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))?
    }

    /// Get current pipeline statistics
    pub async fn get_stats(&self) -> Result<PipelineStats, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::GetStats { reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))
    }

    /// Get archive statistics
    pub async fn get_archive_stats(
        &self,
    ) -> Result<crate::archive::ArchiveStatistics, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::GetArchiveStats { reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))
    }

    /// Get best fitness
    pub async fn get_best_fitness(&self) -> Result<Option<f64>, EvolutionStreamingError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(PipelineRequest::GetBestFitness { reply: tx })
            .await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor stopped".to_string()))?;

        rx.await
            .map_err(|_| EvolutionStreamingError::PipelineError("Actor dropped".to_string()))
    }

    /// Request graceful shutdown
    pub async fn shutdown(&self) -> Result<(), EvolutionStreamingError> {
        self.sender
            .send(PipelineRequest::Shutdown)
            .await
            .map_err(|_| {
                EvolutionStreamingError::PipelineError("Actor already stopped".to_string())
            })
    }

    /// Run continuous evolution until shutdown
    ///
    /// This spawns the actor task and runs cycles until `shutdown()` is called
    /// on a clone of this handle.
    pub async fn run_continuous(&self) -> Result<u64, EvolutionStreamingError> {
        let mut cycle_count = 0u64;

        loop {
            match self.run_cycle().await {
                Ok(result) => {
                    cycle_count += 1;
                    if cycle_count % 10 == 0 {
                        println!(
                            "Completed {} evolution cycles, latest: {:?}",
                            cycle_count, result
                        );
                    }
                }
                Err(EvolutionStreamingError::PipelineError(msg)) if msg.contains("stopped") => {
                    // Actor shutdown - this is expected
                    break;
                }
                Err(e) => {
                    eprintln!("Evolution cycle failed: {}", e);
                    if matches!(e, EvolutionStreamingError::ResourceExhausted { .. }) {
                        return Err(e);
                    }
                }
            }
        }

        Ok(cycle_count)
    }
}

/// Create an actor and its handle
///
/// # Returns
/// A tuple of (actor, handle). The actor should be spawned as a task,
/// and the handle used to communicate with it.
///
/// # Example
/// ```ignore
/// let (actor, handle) = create_pipeline_actor(
///     mutation_processor,
///     evaluation_processor,
///     archive,
///     selection_strategy,
///     batch_size,
///     timeout,
/// );
///
/// // Spawn the actor
/// tokio::spawn(actor.run());
///
/// // Use the handle
/// let result = handle.run_cycle().await?;
///
/// // Shutdown
/// handle.shutdown().await?;
/// ```
pub fn create_pipeline_actor(
    mutation_processor: MutationProcessor,
    evaluation_processor: GpuBatchEvaluator,
    archive: Arc<AgentArchive>,
    selection_strategy: SelectionStrategy,
    batch_size: usize,
    pipeline_timeout: Duration,
) -> (EvolutionPipelineActor, EvolutionPipelineHandle) {
    let (tx, rx) = mpsc::channel(32);

    let actor = EvolutionPipelineActor::new(
        mutation_processor,
        evaluation_processor,
        archive,
        selection_strategy,
        batch_size,
        pipeline_timeout,
        rx,
    );

    let handle = EvolutionPipelineHandle::new(tx);

    (actor, handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would go here - testing the actor model
    // Key tests:
    // 1. Cancel safety - verify actor continues if caller drops
    // 2. Shutdown - verify graceful shutdown works
    // 3. Request ordering - verify FIFO processing
}
