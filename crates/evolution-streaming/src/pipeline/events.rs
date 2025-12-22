//! Event-driven evolution processor

use crate::{AgentGenome, EvolutionEvent, EvolutionStreamingError};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::core::EvolutionStreamingPipeline;

/// Event-driven evolution processor
pub struct EvolutionEventProcessor {
    pipeline: Arc<EvolutionStreamingPipeline>,
    event_rx: mpsc::Receiver<EvolutionEvent>,
    event_tx: mpsc::Sender<EvolutionEvent>,
}

impl EvolutionEventProcessor {
    /// Create a new event processor
    pub fn new(pipeline: Arc<EvolutionStreamingPipeline>) -> (Self, mpsc::Sender<EvolutionEvent>) {
        let (event_tx, event_rx) = mpsc::channel(1000);

        (
            Self {
                pipeline,
                event_rx,
                event_tx: event_tx.clone(),
            },
            event_tx,
        )
    }

    /// Process evolution events
    pub async fn run(&mut self) -> Result<(), EvolutionStreamingError> {
        while let Some(event) = self.event_rx.recv().await {
            match event {
                EvolutionEvent::MutationRequest {
                    agent,
                    mutation_count,
                } => {
                    self.handle_mutation_request(agent, mutation_count).await?;
                }
                EvolutionEvent::EvaluationRequest { agent } => {
                    self.handle_evaluation_request(agent).await?;
                }
                EvolutionEvent::ArchiveUpdate { agent, fitness } => {
                    self.handle_archive_update(agent, fitness).await?;
                }
                EvolutionEvent::PipelineStats {
                    throughput,
                    latency,
                    queue_depth,
                } => {
                    self.handle_pipeline_stats(throughput, latency, queue_depth)
                        .await?;
                }
                _ => {
                    // Handle other event types
                }
            }
        }

        Ok(())
    }

    async fn handle_mutation_request(
        &self,
        agent: AgentGenome,
        _mutation_count: u32,
    ) -> Result<(), EvolutionStreamingError> {
        let mutated_agents = {
            let mutation_processor = self.pipeline.mutation_processor.write().await;
            mutation_processor.process_agent_batch(vec![agent]).await?
        };

        // Send mutation complete events
        for mutated_agent in mutated_agents {
            let _ = self
                .event_tx
                .send(EvolutionEvent::MutationComplete { mutated_agent })
                .await;
        }

        Ok(())
    }

    async fn handle_evaluation_request(
        &self,
        agent: AgentGenome,
    ) -> Result<(), EvolutionStreamingError> {
        let results = {
            let evaluation_processor = self.pipeline.evaluation_processor.read().await;
            evaluation_processor.evaluate_batch(vec![agent]).await?
        };

        // Send evaluation complete events
        for result in results {
            let _ = self
                .event_tx
                .send(EvolutionEvent::EvaluationComplete { result })
                .await;
        }

        Ok(())
    }

    async fn handle_archive_update(
        &self,
        agent: AgentGenome,
        fitness: f64,
    ) -> Result<(), EvolutionStreamingError> {
        self.pipeline.archive.add_if_novel(agent, fitness).await?;
        Ok(())
    }

    async fn handle_pipeline_stats(
        &self,
        _throughput: f64,
        _latency: u64,
        _queue_depth: usize,
    ) -> Result<(), EvolutionStreamingError> {
        // Update internal metrics
        Ok(())
    }
}
