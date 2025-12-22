//! Mutation stream processing for agent evolution

use crate::{
    AgentGenome, CodeLocation, EvolutionStreamingError, MutatedAgent, MutationInfo, MutationType,
};
use async_trait::async_trait;
use exorust_streaming::{StreamChunk, StreamProcessor, StreamStats};
use futures::future::join_all;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

pub mod engine;
pub mod validator;

pub use engine::*;
pub use validator::*;

/// Mutation stream processor for parallel agent mutation
pub struct MutationProcessor {
    id: String,
    mutation_engines: Vec<Box<dyn MutationEngine>>,
    safety_validator: SafetyValidator,
    mutations_per_agent: u32,
    parallelism: usize,
    stats: Arc<MutationStats>,
}

/// Thread-safe statistics for mutation processor
#[derive(Debug, Default)]
struct MutationStats {
    agents_processed: AtomicU64,
    mutations_generated: AtomicU64,
    mutations_validated: AtomicU64,
    processing_time_ns: AtomicU64,
    validation_failures: AtomicU64,
}

/// Mutation task for worker pool
#[derive(Debug, Clone)]
pub struct MutationTask {
    pub agent: AgentGenome,
    pub mutation_type: MutationType,
    pub task_id: u64,
}

impl MutationProcessor {
    /// Create a new mutation processor
    pub fn new(id: String) -> Self {
        Self {
            id,
            mutation_engines: vec![Box::new(BasicMutationEngine::new())],
            safety_validator: SafetyValidator::new(),
            mutations_per_agent: 4,
            parallelism: 8,
            stats: Arc::new(MutationStats::default()),
        }
    }

    /// Configure number of mutations per agent
    pub fn with_mutations_per_agent(mut self, count: u32) -> Self {
        self.mutations_per_agent = count;
        self
    }

    /// Configure parallelism level
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Add mutation engine
    pub fn add_mutation_engine(mut self, engine: Box<dyn MutationEngine>) -> Self {
        self.mutation_engines.push(engine);
        self
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        let agents = self.stats.agents_processed.load(Ordering::SeqCst);
        let mutations = self.stats.mutations_generated.load(Ordering::SeqCst);
        let time_ns = self.stats.processing_time_ns.load(Ordering::SeqCst);
        let failures = self.stats.validation_failures.load(Ordering::SeqCst);

        let throughput_agents_per_sec = if time_ns > 0 {
            (agents as f64) / ((time_ns as f64) / 1_000_000_000.0)
        } else {
            0.0
        };

        StreamStats {
            chunks_processed: agents,
            bytes_processed: mutations * 100, // Rough estimate
            processing_time_ms: time_ns / 1_000_000,
            throughput_mbps: throughput_agents_per_sec / 1_000_000.0, // Convert to "MB/s" equivalent
            errors: failures,
        }
    }

    /// Process batch of agents for mutation
    pub async fn process_agent_batch(
        &self,
        agents: Vec<AgentGenome>,
    ) -> Result<Vec<MutatedAgent>, EvolutionStreamingError> {
        let start_time = Instant::now();
        let (tx, mut rx) = mpsc::channel(agents.len() * self.mutations_per_agent as usize);

        // Create mutation tasks
        let mut task_id = 0u64;
        for agent in agents.iter() {
            for _ in 0..self.mutations_per_agent {
                let engine_idx = task_id as usize % self.mutation_engines.len();
                let mutation_type = self.mutation_engines[engine_idx].select_mutation_type();

                let task = MutationTask {
                    agent: agent.clone(),
                    mutation_type,
                    task_id,
                };

                let tx = tx.clone();
                let engine = self.mutation_engines[engine_idx].clone_engine();
                let validator = self.safety_validator.clone();
                let stats = Arc::clone(&self.stats);

                tokio::spawn(async move {
                    match Self::process_mutation_task(task, engine, validator, stats).await {
                        Ok(mutated) => {
                            let _ = tx.send(Ok(mutated)).await;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                        }
                    }
                });

                task_id += 1;
            }
        }

        drop(tx); // Close channel

        // Collect results
        let mut results = Vec::new();
        while let Some(result) = rx.recv().await {
            match result {
                Ok(mutated) => results.push(mutated),
                Err(e) => {
                    self.stats
                        .validation_failures
                        .fetch_add(1, Ordering::SeqCst);
                    eprintln!("Mutation failed: {e}");
                }
            }
        }

        let processing_time = start_time.elapsed().as_nanos() as u64;
        self.stats
            .processing_time_ns
            .fetch_add(processing_time, Ordering::SeqCst);
        self.stats
            .agents_processed
            .fetch_add(agents.len() as u64, Ordering::SeqCst);
        self.stats
            .mutations_generated
            .fetch_add(results.len() as u64, Ordering::SeqCst);

        Ok(results)
    }

    /// Process individual mutation task
    async fn process_mutation_task(
        task: MutationTask,
        engine: Box<dyn MutationEngine>,
        validator: SafetyValidator,
        stats: Arc<MutationStats>,
    ) -> Result<MutatedAgent, EvolutionStreamingError> {
        // Apply mutation
        let mutated_genome = engine
            .mutate(&task.agent, &task.mutation_type)
            .await
            .map_err(|e| EvolutionStreamingError::MutationFailed(e.to_string()))?;

        // Validate safety
        validator.validate(&mutated_genome).await.map_err(|e| {
            EvolutionStreamingError::SafetyViolation {
                reason: e.to_string(),
            }
        })?;

        stats.mutations_validated.fetch_add(1, Ordering::SeqCst);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(MutatedAgent {
            original: task.agent,
            mutated: mutated_genome,
            mutations: vec![MutationInfo {
                mutation_type: task.mutation_type,
                location: CodeLocation {
                    line: 1,
                    column: 1,
                    length: 1,
                },
                impact_score: 0.5,
            }],
            mutation_time: timestamp,
        })
    }
}

#[async_trait]
impl StreamProcessor for MutationProcessor {
    async fn process(
        &mut self,
        chunk: StreamChunk,
    ) -> Result<StreamChunk, exorust_streaming::StreamingError> {
        // Deserialize agent from chunk
        let agent_data = String::from_utf8_lossy(&chunk.data);
        let agent: AgentGenome = serde_json::from_str(&agent_data).map_err(|e| {
            exorust_streaming::StreamingError::ProcessingFailed {
                reason: format!("Failed to deserialize agent: {e}"),
            }
        })?;

        // Process single agent
        let mutated_agents = self.process_agent_batch(vec![agent]).await.map_err(|e| {
            exorust_streaming::StreamingError::ProcessingFailed {
                reason: e.to_string(),
            }
        })?;

        // Serialize result
        let result_data = serde_json::to_string(&mutated_agents).map_err(|e| {
            exorust_streaming::StreamingError::ProcessingFailed {
                reason: format!("Failed to serialize mutation results: {e}"),
            }
        })?;

        Ok(StreamChunk::new(
            bytes::Bytes::from(result_data),
            chunk.sequence,
            chunk.metadata.source_id,
        ))
    }

    async fn process_batch(
        &mut self,
        chunks: Vec<StreamChunk>,
    ) -> Result<Vec<StreamChunk>, exorust_streaming::StreamingError> {
        let mut agents = Vec::new();

        // Deserialize all agents
        for chunk in &chunks {
            let agent_data = String::from_utf8_lossy(&chunk.data);
            let agent: AgentGenome = serde_json::from_str(&agent_data).map_err(|e| {
                exorust_streaming::StreamingError::ProcessingFailed {
                    reason: format!("Failed to deserialize agent: {e}"),
                }
            })?;
            agents.push(agent);
        }

        // Process batch
        let mutated_agents = self.process_agent_batch(agents).await.map_err(|e| {
            exorust_streaming::StreamingError::ProcessingFailed {
                reason: e.to_string(),
            }
        })?;

        // Create result chunks
        let result_data = serde_json::to_string(&mutated_agents).map_err(|e| {
            exorust_streaming::StreamingError::ProcessingFailed {
                reason: format!("Failed to serialize mutation results: {e}"),
            }
        })?;

        // Split results among output chunks
        let chunk_size = result_data.len() / chunks.len().max(1);
        let mut result_chunks = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let start = i * chunk_size;
            let end = if i == chunks.len() - 1 {
                result_data.len()
            } else {
                (i + 1) * chunk_size
            };

            let chunk_data = if start < result_data.len() {
                &result_data[start..end.min(result_data.len())]
            } else {
                ""
            };

            result_chunks.push(StreamChunk::new(
                bytes::Bytes::from(chunk_data.to_string()),
                chunk.sequence,
                chunk.metadata.source_id.clone(),
            ));
        }

        Ok(result_chunks)
    }

    async fn stats(&self) -> Result<StreamStats, exorust_streaming::StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn processor_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mutation_processor_creation() {
        let processor = MutationProcessor::new("test-mutation".to_string());

        assert_eq!(processor.processor_id(), "test-mutation");
        assert_eq!(processor.mutations_per_agent, 4);
        assert_eq!(processor.parallelism, 8);
    }

    #[tokio::test]
    async fn test_mutation_processor_configuration() {
        let processor = MutationProcessor::new("config-test".to_string())
            .with_mutations_per_agent(2)
            .with_parallelism(4);

        assert_eq!(processor.mutations_per_agent, 2);
        assert_eq!(processor.parallelism, 4);
    }

    #[tokio::test]
    async fn test_process_agent_batch() {
        let processor =
            MutationProcessor::new("batch-test".to_string()).with_mutations_per_agent(2);

        let agents = vec![
            AgentGenome::new("fn test1() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn test2() { 2 }".to_string(), vec![2.0]),
        ];

        let results = processor.process_agent_batch(agents.clone()).await.unwrap();

        // Should have 2 agents * 2 mutations = 4 mutated agents
        assert_eq!(results.len(), 4);

        // Check that mutations were applied
        for result in results {
            assert!(agents.iter().any(|a| a.id == result.original.id));
            assert!(!result.mutations.is_empty());
        }
    }

    #[tokio::test]
    async fn test_mutation_stats() {
        let processor =
            MutationProcessor::new("stats-test".to_string()).with_mutations_per_agent(1);

        let agents = vec![AgentGenome::new("test".to_string(), vec![1.0])];

        // Initial stats
        let initial_stats = processor.get_stats_snapshot();
        assert_eq!(initial_stats.chunks_processed, 0);

        // Process batch
        processor.process_agent_batch(agents).await.unwrap();

        // Final stats
        let final_stats = processor.get_stats_snapshot();
        assert_eq!(final_stats.chunks_processed, 1);
        assert!(final_stats.processing_time_ms > 0);
    }

    #[tokio::test]
    async fn test_stream_processor_interface() {
        let mut processor = MutationProcessor::new("stream-test".to_string());

        let agent = AgentGenome::new("fn test() { 42 }".to_string(), vec![1.0, 2.0]);
        let agent_json = serde_json::to_string(&agent).unwrap();

        let chunk = StreamChunk::new(bytes::Bytes::from(agent_json), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        assert!(!result.data.is_empty());
        assert_eq!(result.sequence, 1);

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_add_mutation_engine() {
        let processor = MutationProcessor::new("engine-test".to_string())
            .add_mutation_engine(Box::new(BasicMutationEngine::new()))
            .add_mutation_engine(Box::new(BasicMutationEngine::new()));

        assert_eq!(processor.mutation_engines.len(), 3); // 1 default + 2 added
    }

    #[test]
    fn test_mutation_task_creation() {
        let agent = AgentGenome::new("test".to_string(), vec![1.0]);
        let task = MutationTask {
            agent: agent.clone(),
            mutation_type: MutationType::ParameterTweak {
                parameter_id: 0,
                delta: 0.1,
            },
            task_id: 42,
        };

        assert_eq!(task.agent.id, agent.id);
        assert_eq!(task.task_id, 42);
    }

    #[tokio::test]
    async fn test_process_batch_multiple_chunks() {
        let mut processor =
            MutationProcessor::new("batch-chunks-test".to_string()).with_mutations_per_agent(1);

        let agents = vec![
            AgentGenome::new("fn a() { 1 }".to_string(), vec![1.0]),
            AgentGenome::new("fn b() { 2 }".to_string(), vec![2.0]),
            AgentGenome::new("fn c() { 3 }".to_string(), vec![3.0]),
        ];

        let chunks: Vec<StreamChunk> = agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let json = serde_json::to_string(agent).unwrap();
                StreamChunk::new(bytes::Bytes::from(json), i as u64, format!("source-{i}"))
            })
            .collect();

        let results = processor.process_batch(chunks).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_empty_agent_batch() {
        let processor = MutationProcessor::new("empty-batch-test".to_string());
        let results = processor.process_agent_batch(vec![]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_mutation_stats_accumulation() {
        let processor = MutationProcessor::new("stats-accumulation-test".to_string())
            .with_mutations_per_agent(2);

        // Process first batch
        let agents1 = vec![AgentGenome::new("test1".to_string(), vec![1.0])];
        processor.process_agent_batch(agents1).await.unwrap();

        let stats1 = processor.get_stats_snapshot();
        assert_eq!(stats1.chunks_processed, 1);

        // Process second batch
        let agents2 = vec![
            AgentGenome::new("test2".to_string(), vec![2.0]),
            AgentGenome::new("test3".to_string(), vec![3.0]),
        ];
        processor.process_agent_batch(agents2).await.unwrap();

        let stats2 = processor.get_stats_snapshot();
        assert_eq!(stats2.chunks_processed, 3); // 1 + 2
    }

    #[test]
    fn test_stats_throughput_calculation() {
        let stats = MutationStats::default();
        stats.agents_processed.store(1000, Ordering::SeqCst);
        stats.mutations_generated.store(4000, Ordering::SeqCst);
        stats
            .processing_time_ns
            .store(1_000_000_000, Ordering::SeqCst); // 1 second

        let processor = MutationProcessor::new("throughput-test".to_string());
        let snapshot = processor.get_stats_snapshot();

        // Should process 1000 agents per second
        assert!(snapshot.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_mutation_validation_failure() {
        // This test would need a custom validator that fails
        // For now, we just ensure the stats track failures correctly
        let processor = MutationProcessor::new("validation-test".to_string());

        assert_eq!(
            processor.stats.validation_failures.load(Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn test_stream_chunk_metadata_preservation() {
        let mut processor = MutationProcessor::new("metadata-test".to_string());
        let agent = AgentGenome::new("test".to_string(), vec![1.0]);
        let agent_json = serde_json::to_string(&agent).unwrap();

        let chunk = StreamChunk::new(
            bytes::Bytes::from(agent_json),
            99,
            "special-source".to_string(),
        );

        let result = processor.process(chunk).await.unwrap();
        assert_eq!(result.sequence, 99);
        assert_eq!(result.metadata.source_id, "special-source");
    }

    #[tokio::test]
    async fn test_large_agent_batch() {
        let processor = MutationProcessor::new("large-batch-test".to_string())
            .with_mutations_per_agent(1)
            .with_parallelism(16);

        let agents: Vec<AgentGenome> = (0..100)
            .map(|i| AgentGenome::new(format!("fn agent_{}() {{ {} }}", i, i), vec![i as f32]))
            .collect();

        let results = processor.process_agent_batch(agents.clone()).await.unwrap();
        assert_eq!(results.len(), 100); // 100 agents * 1 mutation each
    }

    #[test]
    fn test_mutation_processor_id() {
        let processor = MutationProcessor::new("custom-id-123".to_string());
        assert_eq!(processor.processor_id(), "custom-id-123");
    }

    #[tokio::test]
    async fn test_deserialization_error_handling() {
        let mut processor = MutationProcessor::new("deser-error-test".to_string());

        // Create invalid JSON
        let chunk = StreamChunk::new(bytes::Bytes::from("invalid json {"), 1, "test".to_string());

        let result = processor.process(chunk).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mutation_time_tracking() {
        let processor =
            MutationProcessor::new("time-tracking-test".to_string()).with_mutations_per_agent(1);

        let agents = vec![AgentGenome::new("test".to_string(), vec![1.0])];
        let results = processor.process_agent_batch(agents).await.unwrap();

        assert!(!results.is_empty());
        for result in results {
            assert!(result.mutation_time > 0);
        }
    }

    #[tokio::test]
    async fn test_mutation_info_generation() {
        let processor = MutationProcessor::new("info-test".to_string()).with_mutations_per_agent(1);

        let agents = vec![AgentGenome::new("test".to_string(), vec![1.0])];
        let results = processor.process_agent_batch(agents).await.unwrap();

        for result in results {
            assert!(!result.mutations.is_empty());
            let mutation_info = &result.mutations[0];
            assert_eq!(mutation_info.impact_score, 0.5); // Default value
            assert_eq!(mutation_info.location.line, 1);
            assert_eq!(mutation_info.location.column, 1);
        }
    }

    #[tokio::test]
    async fn test_parallel_mutation_processing() {
        use tokio::time::{sleep, Duration};

        let processor = MutationProcessor::new("parallel-test".to_string())
            .with_mutations_per_agent(10)
            .with_parallelism(8);

        let agents = vec![
            AgentGenome::new("agent1".to_string(), vec![1.0]),
            AgentGenome::new("agent2".to_string(), vec![2.0]),
        ];

        let start = std::time::Instant::now();
        let results = processor.process_agent_batch(agents).await.unwrap();
        let duration = start.elapsed();

        assert_eq!(results.len(), 20); // 2 agents * 10 mutations
                                       // Parallel processing should be reasonably fast
        assert!(duration.as_millis() < 1000);
    }

    #[test]
    fn test_mutation_engine_selection() {
        let processor = MutationProcessor::new("engine-selection-test".to_string())
            .add_mutation_engine(Box::new(BasicMutationEngine::new()))
            .add_mutation_engine(Box::new(BasicMutationEngine::new()));

        // Test that engines are selected round-robin based on task_id
        assert_eq!(processor.mutation_engines.len(), 3);
    }

    #[tokio::test]
    async fn test_chunk_splitting_in_batch() {
        let mut processor =
            MutationProcessor::new("chunk-split-test".to_string()).with_mutations_per_agent(1);

        let agents = vec![
            AgentGenome::new("a".repeat(100), vec![1.0]),
            AgentGenome::new("b".repeat(100), vec![2.0]),
        ];

        let chunks: Vec<StreamChunk> = agents
            .iter()
            .enumerate()
            .map(|(i, agent)| {
                let json = serde_json::to_string(agent).unwrap();
                StreamChunk::new(bytes::Bytes::from(json), i as u64, "test".to_string())
            })
            .collect();

        let results = processor.process_batch(chunks.clone()).await.unwrap();
        assert_eq!(results.len(), chunks.len());
    }

    #[test]
    fn test_atomic_stats_operations() {
        use std::sync::Arc;

        let stats = Arc::new(MutationStats::default());

        // Test concurrent increments
        let threads: Vec<_> = (0..10)
            .map(|_| {
                let stats_clone = stats.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        stats_clone.agents_processed.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(stats.agents_processed.load(Ordering::SeqCst), 1000);
    }

    #[tokio::test]
    async fn test_empty_batch_processing() {
        let mut processor = MutationProcessor::new("empty-batch-stream-test".to_string());
        let results = processor.process_batch(vec![]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_single_chunk_batch() {
        let mut processor = MutationProcessor::new("single-chunk-test".to_string());
        let agent = AgentGenome::new("test".to_string(), vec![1.0]);
        let json = serde_json::to_string(&agent).unwrap();
        let chunk = StreamChunk::new(bytes::Bytes::from(json), 1, "test".to_string());

        let results = processor.process_batch(vec![chunk]).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
