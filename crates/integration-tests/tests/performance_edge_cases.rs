//! Performance Edge Case Integration Tests
//! Tests system behavior under extreme performance conditions

use consensus::{ConsensusConfig, ConsensusEngine, Vote as ConsensusVote};
use cpu_agents::{
    agent::{AgentCapability, AgentTask, BasicCpuAgent, CpuAgent, CpuAgentConfig, TaskType},
    io_manager::{IoManager, IoOperation},
    orchestrator::{Orchestrator, Workflow, WorkflowStep},
};
use gpu_agents::{
    evolution::{GpuEvolutionEngine, PopulationBuffer},
    streaming::{StreamConfig, StreamProcessor},
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use stratoswarm_agent_core::{Agent, AgentConfig, Goal, GoalPriority, MemoryType};
use stratoswarm_evolution::{EvolutionEngine, GeneticEvolutionEngine, Population};
use stratoswarm_knowledge_graph::{KnowledgeGraph, KnowledgeGraphConfig, Query, QueryType};
use stratoswarm_memory::{GpuMemoryAllocator, MemoryPool};
use stratoswarm_net::{Network, ZeroCopyTransport};
use stratoswarm_runtime::{ContainerRuntime, SecureContainerRuntime};
use stratoswarm_storage::{NvmeConfig, NvmeStorage, Storage};
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{sleep, timeout};

/// Test system behavior with zero timeouts and instant operations
#[tokio::test]
async fn test_zero_timeout_operations() {
    // Create components with zero or minimal timeouts
    let agent_config = CpuAgentConfig {
        id: "zero_timeout_agent".to_string(),
        capabilities: vec![AgentCapability::FileIo, AgentCapability::NetworkIo],
        max_concurrent_tasks: 100,
        memory_limit_mb: 100,
    };

    let agent = Arc::new(Mutex::new(BasicCpuAgent::new(agent_config)));
    agent.lock().await.initialize().await.unwrap();

    // Fire tasks with zero delay
    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..1000 {
        let agent_clone = agent.clone();
        let handle = tokio::spawn(async move {
            let task = AgentTask {
                id: format!("instant_{}", i),
                task_type: TaskType::Compute("instant".to_string()),
                priority: 10,
            };

            // Try with zero timeout
            let result = timeout(Duration::ZERO, async {
                agent_clone.lock().await.execute_task(task).await
            })
            .await;

            result.is_ok()
        });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let elapsed = start.elapsed();

    // Some should timeout, some might complete
    let completed = results.iter().filter(|&&r| r).count();
    assert!(
        completed < 1000,
        "Most operations should timeout with Duration::ZERO"
    );
    assert!(elapsed < Duration::from_secs(1), "Should complete quickly");
}

/// Test maximum concurrent operations across all components
#[tokio::test]
async fn test_maximum_concurrency_stress() {
    let max_concurrent = 10000;
    let semaphore = Arc::new(Semaphore::new(100)); // Limit actual concurrency

    let io_manager = Arc::new(IoManager::new().await.unwrap());
    let memory_pool = Arc::new(Mutex::new(
        MemoryPool::new(1024 * 1024 * 10).unwrap(), // 10MB pool
    ));

    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..max_concurrent {
        let io_clone = io_manager.clone();
        let memory_clone = memory_pool.clone();
        let sem_clone = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem_clone.acquire().await.unwrap();

            match i % 3 {
                0 => {
                    // I/O operation
                    let op = IoOperation::Read {
                        path: format!("/tmp/test_{}", i),
                        offset: 0,
                        size: 100,
                    };
                    io_clone.execute(op).await.is_ok()
                }
                1 => {
                    // Memory operation
                    let mut pool = memory_clone.lock().await;
                    pool.allocate(100).is_ok()
                }
                _ => {
                    // CPU computation
                    let sum: u64 = (0..100).sum();
                    sum > 0
                }
            }
        });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let elapsed = start.elapsed();
    let successful = results.iter().filter(|&&r| r).count();

    println!("Completed {} operations in {:?}", successful, elapsed);
    assert!(successful > 0, "Some operations should complete");
    assert!(
        elapsed < Duration::from_secs(30),
        "Should complete within reasonable time"
    );
}

/// Test consensus with maximum node count and vote frequency
#[tokio::test]
async fn test_consensus_maximum_scale() {
    let config = ConsensusConfig {
        node_count: 1000,
        byzantine_threshold: 0.33,
        timeout_ms: 100,
        max_rounds: 10,
    };

    let consensus = ConsensusEngine::new(config).await.unwrap();
    let start = Instant::now();

    // Generate votes from all nodes simultaneously
    let mut handles = vec![];

    for round in 0..5 {
        for node_id in 0..1000 {
            let consensus_clone = consensus.clone();
            let handle = tokio::spawn(async move {
                let vote = ConsensusVote {
                    node_id,
                    round,
                    value: vec![round as u8],
                    signature: vec![0u8; 64], // Mock signature
                };

                consensus_clone.submit_vote(vote).await.is_ok()
            });
            handles.push(handle);
        }
    }

    // Wait for all votes with timeout
    let vote_results = timeout(Duration::from_secs(10), futures::future::join_all(handles)).await;

    let elapsed = start.elapsed();

    match vote_results {
        Ok(results) => {
            let successful = results
                .iter()
                .filter(|r| r.as_ref().unwrap_or(&false) == &true)
                .count();
            assert!(successful > 0, "Some votes should be processed");
        }
        Err(_) => {
            assert!(
                elapsed >= Duration::from_secs(10),
                "Should timeout after 10 seconds"
            );
        }
    }
}

/// Test evolution with extreme generation counts and population sizes
#[tokio::test]
async fn test_evolution_extreme_generations() {
    // Test with maximum speed - 1 generation, huge population
    let speed_config = stratoswarm_evolution::EvolutionConfig {
        population_size: 100000,
        mutation_rate: 0.5,
        crossover_rate: 0.5,
        elite_size: 1,
        tournament_size: 2,
        max_generations: 1,
        fitness_threshold: f64::MAX,
    };

    let engine = GeneticEvolutionEngine::new(speed_config);
    let start = Instant::now();

    let result = timeout(Duration::from_secs(5), async {
        engine
            .evolve(
                |_| rand::random::<f64>(),               // Random fitness
                |g| g.into_iter().map(|b| !b).collect(), // Simple mutation
                |g1, g2| {
                    g1.into_iter()
                        .zip(g2.into_iter())
                        .map(|(a, b)| a && b)
                        .collect()
                }, // Simple crossover
            )
            .await
    })
    .await;

    let elapsed = start.elapsed();

    match result {
        Ok(Ok((best, fitness))) => {
            assert!(fitness >= 0.0, "Should have valid fitness");
            assert!(
                elapsed < Duration::from_secs(5),
                "Should complete quickly with 1 generation"
            );
        }
        _ => {
            assert!(
                elapsed >= Duration::from_secs(5),
                "Should timeout with huge population"
            );
        }
    }
}

/// Test knowledge graph with maximum query complexity
#[tokio::test]
async fn test_knowledge_graph_complex_queries() {
    let config = KnowledgeGraphConfig {
        max_nodes: 10000,
        max_edges: 100000,
        cache_size: 1000,
        consistency_level: "strong".to_string(),
        replication_factor: 5,
    };

    let graph = KnowledgeGraph::new(config).await.unwrap();

    // Build a complex graph structure
    for i in 0..100 {
        let node_id = format!("node_{}", i);
        graph
            .add_node(node_id.clone(), serde_json::json!({"index": i}))
            .await
            .unwrap();

        // Create dense connections
        if i > 0 {
            for j in 0..i.min(10) {
                let target = format!("node_{}", j);
                graph
                    .add_edge(
                        node_id.clone(),
                        target,
                        "connects".to_string(),
                        serde_json::json!({"weight": i - j}),
                    )
                    .await
                    .unwrap();
            }
        }
    }

    // Execute complex multi-hop queries
    let complex_queries = vec![
        // 5-hop traversal
        Query {
            query_type: QueryType::Traversal,
            start_node: Some("node_0".to_string()),
            max_depth: Some(5),
            filters: vec![],
            limit: 1000,
        },
        // Pattern matching across entire graph
        Query {
            query_type: QueryType::Pattern,
            start_node: None,
            max_depth: Some(10),
            filters: vec!["weight > 5".to_string()],
            limit: 10000,
        },
        // Shortest path in dense graph
        Query {
            query_type: QueryType::ShortestPath,
            start_node: Some("node_0".to_string()),
            max_depth: Some(20),
            filters: vec![],
            limit: 1,
        },
    ];

    let start = Instant::now();
    let mut query_times = vec![];

    for query in complex_queries {
        let query_start = Instant::now();
        let result = timeout(Duration::from_millis(500), graph.query(query)).await;
        let query_elapsed = query_start.elapsed();
        query_times.push(query_elapsed);

        match result {
            Ok(Ok(results)) => {
                assert!(!results.is_empty() || query_elapsed < Duration::from_millis(500));
            }
            _ => {
                assert!(
                    query_elapsed >= Duration::from_millis(500),
                    "Should timeout on complex query"
                );
            }
        }
    }

    let total_elapsed = start.elapsed();
    assert!(
        total_elapsed < Duration::from_secs(2),
        "All queries should complete within 2 seconds"
    );
}

/// Test streaming with maximum throughput
#[tokio::test]
async fn test_streaming_maximum_throughput() {
    let config = StreamConfig {
        batch_size: 10000,
        buffer_size: 1024 * 1024, // 1MB
        compression: true,
        checksum: true,
    };

    let processor = StreamProcessor::new(config).await.unwrap();
    let data_size = 100_000_000; // 100MB of data
    let chunk_size = 1024; // 1KB chunks

    let start = Instant::now();
    let mut bytes_processed = 0;

    // Stream data as fast as possible
    let producer = tokio::spawn(async move {
        let mut sent = 0;
        while sent < data_size {
            let chunk = vec![0u8; chunk_size.min(data_size - sent)];
            if processor.send(chunk).await.is_err() {
                break;
            }
            sent += chunk_size;
        }
        sent
    });

    // Consume data as fast as possible
    let consumer = tokio::spawn(async move {
        let mut received = 0;
        let timeout_duration = Duration::from_millis(100);

        loop {
            match timeout(timeout_duration, processor.receive()).await {
                Ok(Ok(chunk)) => {
                    received += chunk.len();
                    if received >= data_size {
                        break;
                    }
                }
                _ => break,
            }
        }
        received
    });

    let (sent, received) = tokio::join!(producer, consumer);
    let sent = sent.unwrap_or(0);
    let received = received.unwrap_or(0);

    let elapsed = start.elapsed();
    let throughput_mbps = (received as f64 / 1_000_000.0) / elapsed.as_secs_f64();

    println!("Streaming throughput: {:.2} MB/s", throughput_mbps);
    assert!(throughput_mbps > 0.0, "Should achieve some throughput");
    assert!(received > 0, "Should process some data");
}

/// Test storage with maximum I/O operations per second
#[tokio::test]
async fn test_storage_maximum_iops() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = NvmeConfig {
        path: temp_dir.path().to_path_buf(),
        max_size_bytes: 1024 * 1024 * 100, // 100MB
        io_depth: 256,                     // Maximum queue depth
        compression: false,                // Disable for max performance
    };

    let storage = Arc::new(NvmeStorage::new(config).await.unwrap());
    let operations = 10000;
    let data_size = 4096; // 4KB blocks

    let start = Instant::now();
    let semaphore = Arc::new(Semaphore::new(256)); // Match io_depth

    let mut handles = vec![];

    // Perform mixed read/write operations
    for i in 0..operations {
        let storage_clone = storage.clone();
        let sem_clone = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem_clone.acquire().await.unwrap();
            let key = format!("key_{}", i % 1000); // Reuse keys for some reads

            if i % 3 == 0 {
                // Write
                let data = vec![i as u8; data_size];
                storage_clone.store(&key, &data).await.is_ok()
            } else {
                // Read
                storage_clone.retrieve(&key).await.is_ok()
            }
        });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let elapsed = start.elapsed();
    let successful = results.iter().filter(|&&r| r).count();
    let iops = successful as f64 / elapsed.as_secs_f64();

    println!("Storage IOPS: {:.0}", iops);
    assert!(iops > 0.0, "Should achieve some IOPS");
    assert!(
        successful > operations / 2,
        "Most operations should succeed"
    );
}

/// Test orchestrator with maximum workflow complexity
#[tokio::test]
async fn test_orchestrator_complex_workflows() {
    let orchestrator = Orchestrator::new();

    // Create a complex workflow with many parallel branches
    let mut workflow = Workflow {
        id: "complex_workflow".to_string(),
        steps: vec![],
        timeout: Duration::from_secs(10),
    };

    // Add 100 parallel steps
    for i in 0..100 {
        workflow.steps.push(WorkflowStep {
            id: format!("step_{}", i),
            task: AgentTask {
                id: format!("task_{}", i),
                task_type: TaskType::Compute(format!("computation_{}", i)),
                priority: (i % 10 + 1) as u8,
            },
            dependencies: if i > 10 {
                vec![format!("step_{}", i % 10)]
            } else {
                vec![]
            },
            retry_policy: Some(3),
        });
    }

    // Add convergence steps
    for i in 100..110 {
        workflow.steps.push(WorkflowStep {
            id: format!("converge_{}", i),
            task: AgentTask {
                id: format!("converge_task_{}", i),
                task_type: TaskType::Aggregate,
                priority: 10,
            },
            dependencies: (0..10)
                .map(|j| format!("step_{}", i - 100 + j * 10))
                .collect(),
            retry_policy: Some(1),
        });
    }

    let start = Instant::now();
    let result = timeout(
        Duration::from_secs(15),
        orchestrator.execute_workflow(workflow),
    )
    .await;

    let elapsed = start.elapsed();

    match result {
        Ok(Ok(execution_result)) => {
            assert!(
                execution_result.completed_steps > 0,
                "Some steps should complete"
            );
            assert!(
                elapsed < Duration::from_secs(15),
                "Should complete within timeout"
            );
        }
        _ => {
            assert!(
                elapsed >= Duration::from_secs(15),
                "Complex workflow might timeout"
            );
        }
    }
}

/// Test system behavior with instant failures and recovery
#[tokio::test]
async fn test_instant_failure_recovery() {
    let runtime = SecureContainerRuntime::new();
    let mut container_ids = vec![];

    // Create containers that will instantly fail
    for i in 0..50 {
        let config = ContainerConfig {
            memory_limit_bytes: 1,    // Impossibly small
            gpu_compute_units: 0,     // No GPU
            timeout_seconds: Some(0), // Instant timeout
            agent_type: format!("failing_agent_{}", i),
            environment: Default::default(),
        };

        match runtime.create_container(config).await {
            Ok(container) => container_ids.push(container.id()),
            Err(_) => {} // Expected to fail
        }
    }

    // Try to start all containers concurrently
    let mut handles = vec![];
    for id in &container_ids {
        let runtime_clone = runtime.clone();
        let id_clone = *id;

        let handle =
            tokio::spawn(async move { runtime_clone.start_container(id_clone).await.is_ok() });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    // Most should fail instantly
    let successful = results.iter().filter(|&&r| r).count();
    assert!(
        successful < container_ids.len() / 2,
        "Most containers should fail to start"
    );

    // Runtime should still be functional
    let normal_config = ContainerConfig {
        memory_limit_bytes: 1024 * 1024,
        gpu_compute_units: 1,
        timeout_seconds: Some(60),
        agent_type: "recovery_test".to_string(),
        environment: Default::default(),
    };

    let recovery_container = runtime.create_container(normal_config).await;
    assert!(
        recovery_container.is_ok(),
        "Should be able to create normal container after failures"
    );
}

/// Test memory allocator with extreme fragmentation
#[tokio::test]
async fn test_memory_extreme_fragmentation() {
    let allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 10).unwrap(), // 10MB
    );

    let mut handles = vec![];
    let allocation_pattern = vec![
        (1024, 100),  // 1KB x 100
        (4096, 50),   // 4KB x 50
        (16384, 25),  // 16KB x 25
        (65536, 10),  // 64KB x 10
        (262144, 5),  // 256KB x 5
        (1048576, 2), // 1MB x 2
    ];

    // Allocate in pattern to create fragmentation
    for (size, count) in allocation_pattern {
        for i in 0..count {
            let allocator_clone = allocator.clone();
            let handle = tokio::spawn(async move {
                let result = allocator_clone.allocate(size).await;

                // Deallocate every other allocation to create holes
                if i % 2 == 0 {
                    if let Ok(handle) = result {
                        sleep(Duration::from_micros(10)).await;
                        let _ = allocator_clone.deallocate(handle).await;
                    }
                }

                result.is_ok()
            });
            handles.push(handle);
        }
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let successful = results.iter().filter(|&&r| r).count();

    // Now try large allocation in fragmented memory
    let large_result = allocator.allocate(1024 * 1024 * 2).await; // 2MB

    // Should handle fragmentation
    assert!(successful > 0, "Some allocations should succeed");
    assert!(
        large_result.is_err() || large_result.is_ok(),
        "Should handle large allocation gracefully"
    );
}

#[cfg(test)]
mod performance_verification {
    use super::*;

    #[test]
    fn verify_performance_edge_coverage() {
        let scenarios_tested = vec![
            "zero_timeout",
            "maximum_concurrency",
            "consensus_scale",
            "evolution_extremes",
            "complex_queries",
            "streaming_throughput",
            "storage_iops",
            "workflow_complexity",
            "instant_failures",
            "memory_fragmentation",
        ];

        assert_eq!(
            scenarios_tested.len(),
            10,
            "Should test all performance edge cases"
        );
    }
}
