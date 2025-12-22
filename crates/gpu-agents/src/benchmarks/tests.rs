//! Tests for benchmark implementations
//! Following TDD workflow: RED -> GREEN -> REFACTOR

use crate::benchmarks::progress_writer::ProgressWriter;

#[cfg(test)]
mod scalability_tests {

    use crate::{GpuSwarm, GpuSwarmConfig};

    #[test]
    fn test_scalability_benchmark_should_run_real_gpu_operations() {
        // RED PHASE: This test should fail until we implement real GPU operations
        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 1000,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_collective_knowledge: false,
            enable_knowledge_graph: false,
        };

        let mut swarm = GpuSwarm::new(config).expect("Failed to create swarm");
        assert!(
            swarm.initialize(1000).is_ok(),
            "Swarm initialization should succeed"
        );

        // Should actually run GPU kernels
        let result = swarm.step();
        assert!(result.is_ok(), "Step should execute GPU operations");

        // Should have measurable GPU memory usage and kernel time
        let metrics = swarm.metrics();
        assert!(
            metrics.gpu_memory_used > 100_000,
            "Should use significant GPU memory (>100KB)"
        );
        assert!(
            metrics.agent_count == 1000,
            "Should have exactly 1000 agents"
        );
        assert!(
            metrics.kernel_time_ms > 0.0,
            "Kernel execution time should be measurable"
        );

        // Verify GPU kernels actually executed by checking metrics change
        swarm.step().expect("Second step should succeed");
        let new_metrics = swarm.metrics();
        assert!(
            new_metrics.kernel_time_ms > 0.0,
            "Second kernel should have execution time"
        );
    }

    #[test]
    fn test_scalability_benchmark_should_measure_throughput() {
        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 10_000,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_collective_knowledge: false,
            enable_knowledge_graph: false,
        };
        let mut swarm = GpuSwarm::new(config).expect("Failed to create swarm");
        swarm.initialize(10_000).expect("Failed to initialize");

        let start = std::time::Instant::now();
        let steps = 10;
        for _ in 0..steps {
            swarm.step().expect("Step failed");
        }
        let elapsed = start.elapsed();

        // Get metrics to verify real GPU execution
        let metrics = swarm.metrics();
        assert!(
            metrics.kernel_time_ms > 0.0,
            "Should have kernel execution time"
        );

        // Should process at least 10,000 agents per second on GPU
        let agents_per_second = (10_000 * steps) as f64 / elapsed.as_secs_f64();
        assert!(
            agents_per_second > 10_000.0,
            "GPU throughput too low: {} agents/sec (expected >10,000)",
            agents_per_second
        );

        // Kernel time should be reasonable (not just placeholder)
        let avg_kernel_time = metrics.kernel_time_ms / steps as f32;
        assert!(
            avg_kernel_time < 100.0,
            "Kernel time too high: {}ms (expected <100ms per step)",
            avg_kernel_time
        );
    }
}

#[cfg(test)]
mod llm_integration_tests {

    use crate::llm::LlmConfig;
    use crate::{GpuSwarm, GpuSwarmConfig};

    #[test]
    fn test_llm_benchmark_should_run_real_llm_operations() {
        // RED PHASE: This test should fail until we implement real LLM integration
        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 100,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: true,
            enable_collective_intelligence: true,
            enable_collective_knowledge: false,
            enable_knowledge_graph: false,
        };

        let mut swarm = GpuSwarm::new(config).expect("Failed to create swarm");
        swarm.initialize(100).expect("Failed to initialize");

        // Enable LLM reasoning
        let llm_config = LlmConfig::default();
        swarm
            .enable_llm_reasoning(llm_config)
            .expect("Failed to enable LLM");

        // Run with LLM
        let result = swarm.step_with_llm();
        assert!(result.is_ok(), "LLM step should succeed");

        // Query collective intelligence
        let query_result = swarm
            .query_collective_intelligence("Test query")
            .expect("Query should succeed");

        // Should have real LLM inference, not placeholder
        let metrics = swarm.metrics();
        assert!(
            metrics.llm_inference_time_ms > 50.0,
            "LLM inference time too low: {}ms (expected >50ms for real inference)",
            metrics.llm_inference_time_ms
        );
        assert!(
            metrics.llm_buffer_memory > 1_000_000,
            "LLM buffer memory too small: {} bytes (expected >1MB for real model)",
            metrics.llm_buffer_memory
        );

        // Query result should not be a placeholder response
        assert!(
            !query_result.contains("northwest"),
            "Query result appears to be placeholder: {}",
            query_result
        );
        assert!(
            query_result.len() > 50,
            "Query result too short for real LLM response: {} chars",
            query_result.len()
        );

        // Test batch inference performance
        let start = std::time::Instant::now();
        for _ in 0..5 {
            swarm.step_with_llm().expect("LLM step failed");
        }
        let elapsed = start.elapsed();
        let avg_inference_time = elapsed.as_millis() as f32 / 5.0;
        assert!(
            avg_inference_time > 100.0,
            "LLM inference too fast for real model: {}ms (expected >100ms)",
            avg_inference_time
        );
    }
}

#[cfg(test)]
mod knowledge_graph_tests {

    use crate::knowledge::{GraphQuery, KnowledgeEdge, KnowledgeGraph, KnowledgeNode};
    use crate::{GpuSwarm, GpuSwarmConfig};

    #[test]
    fn test_knowledge_graph_benchmark_should_build_real_graph() {
        // RED PHASE: This test should fail until we implement real knowledge graph
        let config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 1000,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_collective_knowledge: true,
            enable_knowledge_graph: true,
        };

        let mut swarm = GpuSwarm::new(config).expect("Failed to create swarm");
        swarm.initialize(1000).expect("Failed to initialize");

        // Build knowledge graph
        let mut graph = KnowledgeGraph::new();

        // Add nodes
        for i in 0..1000 {
            graph.add_node(KnowledgeNode {
                id: i as u32,
                content: format!("Node {}", i),
                node_type: "concept".to_string(),
                embedding: vec![0.1; 768], // Default embedding
            });
        }

        // Add edges
        for i in 0..999 {
            graph.add_edge(KnowledgeEdge {
                source_id: i as u32,
                target_id: (i + 1) as u32,
                relationship: "next".to_string(),
                weight: 1.0,
            });
        }

        // Attach to swarm
        swarm
            .attach_knowledge_graph(graph)
            .expect("Failed to attach graph");

        // Query the graph
        let query = GraphQuery {
            query_type: "similarity".to_string(),
            embedding: vec![0.1; 768],
            max_results: 10,
        };

        let results = swarm.query_knowledge_graph(&query).expect("Query failed");
        assert!(!results.is_empty(), "Should return results");

        // Verify real GPU-based knowledge graph operations
        let kg_metrics = swarm.knowledge_graph_metrics();
        assert!(
            kg_metrics.node_count == 1000,
            "Should have exactly 1000 nodes on GPU"
        );
        assert!(
            kg_metrics.gpu_memory_used > 1_000_000,
            "GPU memory usage too low: {} bytes (expected >1MB for 1000 nodes)",
            kg_metrics.gpu_memory_used
        );

        // Measure GPU query throughput
        let start = std::time::Instant::now();
        for _ in 0..100 {
            swarm.query_knowledge_graph(&query).expect("Query failed");
        }
        let elapsed = start.elapsed();
        let queries_per_second = 100.0 / elapsed.as_secs_f64();
        assert!(
            queries_per_second > 1000.0,
            "GPU query throughput too low: {:.1} queries/sec (expected >1000 for GPU acceleration)",
            queries_per_second
        );

        // Verify results are not placeholder responses
        assert!(
            results.len() >= 5,
            "Should return multiple results for similarity search"
        );
        assert!(
            results[0].score > 0.5,
            "Top result should have high similarity score: {}",
            results[0].score
        );
        assert!(
            !results[0].content.starts_with("Result"),
            "Result appears to be placeholder: '{}'",
            results[0].content
        );

        // Test GPU-based vector similarity (should not be simple linear pattern)
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let is_linear_pattern = scores.windows(2).all(|w| (w[0] - w[1] - 0.1).abs() < 0.01);
        assert!(
            !is_linear_pattern,
            "Similarity scores appear to be placeholder linear pattern: {:?}",
            scores
        );
    }
}

#[cfg(test)]
mod evolution_tests {

    use crate::evolution::{
        EvolutionConfig, EvolutionManager, FitnessObjective, MutationStrategy, SelectionStrategy,
    };
    use crate::{GpuSwarm, GpuSwarmConfig};

    #[test]
    fn test_evolution_benchmark_should_run_real_evolution() {
        // RED PHASE: This test should fail until we implement real evolution
        let swarm_config = GpuSwarmConfig {
            device_id: 0,
            max_agents: 1000,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 10,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_collective_knowledge: false,
            enable_knowledge_graph: false,
        };

        let mut swarm = GpuSwarm::new(swarm_config).expect("Failed to create swarm");
        swarm.initialize(1000).expect("Failed to initialize");

        let evolution_config = EvolutionConfig {
            population_size: 1000,
            selection_strategy: SelectionStrategy::Tournament,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism_percentage: 0.01, // 10 agents out of 1000
            tournament_size: 3,
            mutation_strategy: MutationStrategy::Adaptive,
            max_generations: 100,
            convergence_threshold: 0.001,
            enable_archive: true,
            archive_size_limit: 1000,
            objectives: vec![FitnessObjective::Performance, FitnessObjective::Novelty],
            novelty_k_nearest: 15,
            behavioral_descriptor_size: 10,
        };

        let mut evolution =
            EvolutionManager::new(evolution_config).expect("Failed to create evolution manager");

        // Run evolution steps
        let generations = 10;
        let start = std::time::Instant::now();

        for _ in 0..generations {
            // Swarm step
            swarm.step().expect("Step failed");

            // Run real GPU evolution
            evolution
                .evolve_generation(&mut swarm)
                .expect("Evolution failed");

            // Check for convergence
            if evolution.has_converged() {
                break;
            }
        }

        let elapsed = start.elapsed();
        let generations_per_second = generations as f64 / elapsed.as_secs_f64();

        // Should have reasonable performance
        assert!(generations_per_second > 1.0, "Evolution too slow");

        // Check evolution metrics
        let current_gen = evolution.current_generation();
        assert!(
            current_gen > 0,
            "Should have evolved at least one generation"
        );
        assert!(
            current_gen <= generations,
            "Should not exceed max generations"
        );
    }
}

#[cfg(test)]
mod progress_writer_tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_progress_writer_writes_to_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let log_path = temp_file.path().to_str().unwrap();

        let writer = ProgressWriter::new(log_path).unwrap();
        writer.log("Test message")?;
        writer.log_progress(0.5)?;
        writer.log_gpu_usage(75.0)?;

        let contents = fs::read_to_string(log_path)?;
        assert!(contents.contains("Test message"));
        assert!(contents.contains("Progress: 50%"));
        assert!(contents.contains("GPU Usage: 75%"));
    }
}
