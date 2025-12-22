//! Evolution streaming pipeline builder

use crate::{
    archive::AgentArchive,
    evaluation::{BenchmarkSuite, GpuBatchEvaluator},
    mutation::MutationProcessor,
    EvolutionStreamingError, SelectionStrategy,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use super::{stats::PipelineStats, EvolutionStreamingPipeline};

/// Evolution streaming pipeline builder
pub struct EvolutionStreamingPipelineBuilder {
    mutation_processor: Option<MutationProcessor>,
    evaluation_processor: Option<GpuBatchEvaluator>,
    archive: Option<AgentArchive>,
    selection_strategy: SelectionStrategy,
    batch_size: usize,
    pipeline_timeout: Duration,
}

impl EvolutionStreamingPipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            mutation_processor: None,
            evaluation_processor: None,
            archive: None,
            selection_strategy: SelectionStrategy::Elite { count: 10 },
            batch_size: 32,
            pipeline_timeout: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Configure mutation processor
    pub fn with_mutation_processor(mut self, processor: MutationProcessor) -> Self {
        self.mutation_processor = Some(processor);
        self
    }

    /// Configure evaluation processor
    pub fn with_evaluation_processor(mut self, processor: GpuBatchEvaluator) -> Self {
        self.evaluation_processor = Some(processor);
        self
    }

    /// Configure agent archive
    pub fn with_archive(mut self, archive: AgentArchive) -> Self {
        self.archive = Some(archive);
        self
    }

    /// Configure selection strategy
    pub fn with_selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Configure batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Configure pipeline timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.pipeline_timeout = timeout;
        self
    }

    /// Build the evolution streaming pipeline
    pub async fn build(self) -> Result<EvolutionStreamingPipeline, EvolutionStreamingError> {
        let mutation_processor = self.mutation_processor.unwrap_or_else(|| {
            MutationProcessor::new("default-mutation".to_string())
                .with_mutations_per_agent(4)
                .with_parallelism(8)
        });

        let evaluation_processor = self.evaluation_processor.unwrap_or_else(|| {
            GpuBatchEvaluator::new("default-evaluation".to_string())
                .with_batch_size(self.batch_size)
                .with_benchmark_suite(BenchmarkSuite::default())
        });

        let archive = self.archive.unwrap_or_else(|| {
            AgentArchive::new(1000) // Default archive size
        });

        Ok(EvolutionStreamingPipeline {
            mutation_processor: Arc::new(RwLock::new(mutation_processor)),
            evaluation_processor: Arc::new(RwLock::new(evaluation_processor)),
            archive: Arc::new(archive),
            selection_strategy: self.selection_strategy,
            batch_size: self.batch_size,
            pipeline_timeout: self.pipeline_timeout,
            stats: Arc::new(RwLock::new(PipelineStats::default())),
        })
    }
}

impl Default for EvolutionStreamingPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
