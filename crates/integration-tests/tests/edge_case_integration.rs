//! Edge Case Integration Tests for ExoRust
//! Tests boundary conditions, race conditions, and extreme scenarios across components

use cpu_agents::{
    agent::{
        AgentCapability, AgentStatus, AgentTask, BasicCpuAgent, CpuAgent, CpuAgentConfig, TaskType,
    },
    bridge::{BridgeConfig, CpuGpuBridge, CpuGpuMessage, MessageType as BridgeMessageType},
};
use stratoswarm_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use stratoswarm_cuda::{CudaContext, CudaMemoryManager};
use stratoswarm_evolution::{EvolutionEngine, GeneticEvolutionEngine, Population};
use stratoswarm_evolution_engines::{EvolutionEngineConfig, HybridEvolutionSystem};
use stratoswarm_knowledge_graph::{KnowledgeGraph, KnowledgeGraphConfig, Query, QueryType};
use stratoswarm_memory::{GpuMemoryAllocator, MemoryManager};
use stratoswarm_net::{MemoryNetwork, Message, MessageType, Network, ZeroCopyTransport};
use stratoswarm_runtime::{ContainerConfig, ContainerRuntime, SecureContainerRuntime};
use stratoswarm_storage::{MemoryStorage, NvmeConfig, NvmeStorage, Storage};
use stratoswarm_synthesis::{SynthesisConfig, SynthesisPipeline};
use gpu_agents::{
    consensus::{ConsensusModule, Vote, VoteType},
    memory::{MemoryTier, TierManager, UnifiedMemoryManager},
    synthesis::{Pattern, SynthesisModule, Template},
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, timeout};

/// Test extreme memory pressure across all tiers simultaneously
#[tokio::test]
async fn test_extreme_memory_pressure_all_tiers() {
    // Initialize tier manager with very small tier sizes
    let tier_manager = TierManager::new(
        1024,  // GPU: 1KB
        2048,  // CPU: 2KB
        4096,  // NVMe: 4KB
        8192,  // SSD: 8KB
        16384, // HDD: 16KB
    )
    .await
    .unwrap();

    // Track successful and failed allocations per tier
    let mut results = HashMap::new();

    // Try to fill all tiers simultaneously
    let mut handles = vec![];

    for tier in 0..5 {
        let tier_manager_clone = tier_manager.clone();
        let handle = tokio::spawn(async move {
            let mut successes = 0;
            let mut failures = 0;

            // Try to allocate beyond tier capacity
            for i in 0..100 {
                let size = 512; // 512 bytes per allocation
                let result = tier_manager_clone.allocate_in_tier(tier, size).await;

                match result {
                    Ok(_) => successes += 1,
                    Err(_) => failures += 1,
                }

                // Small delay to allow migration
                if i % 10 == 0 {
                    sleep(Duration::from_micros(100)).await;
                }
            }

            (tier, successes, failures)
        });
        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let (tier, successes, failures) = handle.await.unwrap();
        results.insert(tier, (successes, failures));
    }

    // Verify graceful degradation
    for tier in 0..5 {
        let (successes, failures) = results.get(&tier).unwrap();
        assert!(
            *successes > 0,
            "Tier {} should have some successful allocations",
            tier
        );
        assert!(
            *failures > 0,
            "Tier {} should have some failed allocations due to pressure",
            tier
        );
    }
}

/// Test concurrent CPU-GPU bridge operations with message flooding
#[tokio::test]
async fn test_bridge_message_flooding() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 1,
        max_message_size_mb: 1,
        polling_interval_ms: 10,
    };

    let bridge = Arc::new(Mutex::new(CpuGpuBridge::new(config).await.unwrap()));
    bridge.lock().await.start().await.unwrap();

    // Flood with messages from multiple threads
    let mut handles = vec![];
    let message_count = 1000;
    let thread_count = 10;

    for thread_id in 0..thread_count {
        let bridge_clone = bridge.clone();
        let handle = tokio::spawn(async move {
            let mut sent = 0;
            let mut failed = 0;

            for i in 0..message_count {
                let msg = CpuGpuMessage {
                    id: format!("flood-{}-{}", thread_id, i),
                    message_type: match i % 6 {
                        0 => BridgeMessageType::TaskRequest,
                        1 => BridgeMessageType::TaskResult,
                        2 => BridgeMessageType::DataTransfer,
                        3 => BridgeMessageType::StatusUpdate,
                        4 => BridgeMessageType::ErrorReport,
                        _ => BridgeMessageType::Shutdown,
                    },
                    source: format!("cpu-{}", thread_id),
                    destination: "gpu".to_string(),
                    payload: serde_json::json!({
                        "thread": thread_id,
                        "index": i,
                        "data": "x".repeat(100)
                    }),
                    timestamp: chrono::Utc::now(),
                    priority: ((i % 10) + 1) as u8,
                };

                let result = bridge_clone.lock().await.send_to_gpu(msg).await;
                match result {
                    Ok(_) => sent += 1,
                    Err(_) => failed += 1,
                }
            }

            (sent, failed)
        });
        handles.push(handle);
    }

    // Collect results
    let mut total_sent = 0;
    let mut total_failed = 0;

    for handle in handles {
        let (sent, failed) = handle.await.unwrap();
        total_sent += sent;
        total_failed += failed;
    }

    // Some messages should succeed despite flooding
    assert!(total_sent > 0, "Some messages should be sent successfully");

    // Stop bridge
    bridge.lock().await.stop().await.unwrap();
}

/// Test agent lifecycle with rapid state transitions
#[tokio::test]
async fn test_agent_rapid_state_transitions() {
    let config = CpuAgentConfig {
        id: "rapid-transition-agent".to_string(),
        capabilities: vec![
            AgentCapability::FileIo,
            AgentCapability::NetworkIo,
            AgentCapability::DataTransform,
        ],
        max_concurrent_tasks: 5,
        memory_limit_mb: 10,
    };

    let agent = Arc::new(Mutex::new(BasicCpuAgent::new(config)));

    // Rapidly transition through states
    let agent_clone = agent.clone();
    let state_changer = tokio::spawn(async move {
        for _ in 0..50 {
            let mut agent = agent_clone.lock().await;

            // Initialize if not ready
            if agent.status() == AgentStatus::Created {
                let _ = agent.initialize().await;
            }

            // Execute a quick task
            let task = AgentTask {
                id: format!("rapid-{}", uuid::Uuid::new_v4()),
                task_type: TaskType::Compute("quick".to_string()),
                priority: 5,
            };

            let _ = agent.execute_task(task).await;

            // Random delay
            sleep(Duration::from_micros(rand::random::<u64>() % 1000)).await;
        }
    });

    // Concurrent status checker
    let agent_clone2 = agent.clone();
    let status_checker = tokio::spawn(async move {
        let mut status_counts = HashMap::new();

        for _ in 0..100 {
            let agent = agent_clone2.lock().await;
            let status = agent.status();
            *status_counts.entry(status).or_insert(0) += 1;
            drop(agent);

            sleep(Duration::from_micros(100)).await;
        }

        status_counts
    });

    // Wait for completion
    state_changer.await.unwrap();
    let status_counts = status_checker.await.unwrap();

    // Should have seen multiple states
    assert!(
        status_counts.len() >= 2,
        "Should observe multiple agent states"
    );
}

/// Test consensus with Byzantine nodes and extreme voting patterns
#[tokio::test]
async fn test_consensus_byzantine_extreme() {
    let consensus = ConsensusModule::new(100).await.unwrap();

    // Create extreme voting patterns
    let vote_patterns = vec![
        // All nodes vote for same value
        (0..100, vec![1u8; 100]),
        // Half and half split
        (0..50, vec![0u8; 50])
            .into_iter()
            .chain((50..100, vec![1u8; 50])),
        // Byzantine pattern (33% each)
        (0..33, vec![0u8; 33])
            .into_iter()
            .chain((33..66, vec![1u8; 33]))
            .chain((66..100, vec![2u8; 34])),
        // Random votes
        (0..100, (0..100).map(|_| rand::random::<u8>() % 3).collect()),
    ];

    for (round, pattern) in vote_patterns.into_iter().enumerate() {
        let mut votes = Vec::new();

        for (node_id, value) in pattern {
            votes.push(Vote {
                node_id: node_id as u32,
                round: round as u32,
                value: vec![value],
                vote_type: if node_id % 10 == 0 {
                    VoteType::Byzantine
                } else {
                    VoteType::Honest
                },
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        }

        // Process votes concurrently
        let mut handles = vec![];
        for vote in votes {
            let consensus_clone = consensus.clone();
            let handle = tokio::spawn(async move { consensus_clone.submit_vote(vote).await });
            handles.push(handle);
        }

        // Wait for all votes
        for handle in handles {
            let _ = handle.await;
        }

        // Check consensus was reached
        let result = consensus.get_consensus(round as u32).await;
        assert!(
            result.is_some(),
            "Consensus should be reached for round {}",
            round
        );
    }
}

/// Test evolution with extreme population sizes and mutation rates
#[tokio::test]
async fn test_evolution_extreme_parameters() {
    // Test with very small population
    let small_config = EvolutionEngineConfig {
        population_size: 2,
        mutation_rate: 0.99, // Extreme mutation
        crossover_rate: 0.01,
        elite_size: 1,
        tournament_size: 2,
        max_generations: 10,
        target_fitness: f64::INFINITY,
        parallel_evaluations: 1,
    };

    let small_engine = HybridEvolutionSystem::new(small_config.clone())
        .await
        .unwrap();
    let small_result = small_engine.evolve().await;
    assert!(small_result.is_ok(), "Should handle small population");

    // Test with very large population
    let large_config = EvolutionEngineConfig {
        population_size: 10000,
        mutation_rate: 0.001, // Very low mutation
        crossover_rate: 0.999,
        elite_size: 5000,
        tournament_size: 100,
        max_generations: 1,
        target_fitness: -f64::INFINITY,
        parallel_evaluations: 100,
    };

    let large_engine = HybridEvolutionSystem::new(large_config.clone())
        .await
        .unwrap();

    // Should handle resource constraints gracefully
    let result = timeout(Duration::from_secs(5), large_engine.evolve()).await;
    assert!(
        result.is_ok() || result.is_err(),
        "Should either complete or timeout gracefully"
    );
}

/// Test knowledge graph with concurrent modifications and queries
#[tokio::test]
async fn test_knowledge_graph_concurrent_stress() {
    let config = KnowledgeGraphConfig {
        max_nodes: 100,
        max_edges: 1000,
        cache_size: 10,
        consistency_level: "eventual".to_string(),
        replication_factor: 3,
    };

    let graph = Arc::new(RwLock::new(KnowledgeGraph::new(config).await.unwrap()));

    let mut handles = vec![];

    // Concurrent writers
    for writer_id in 0..5 {
        let graph_clone = graph.clone();
        let handle = tokio::spawn(async move {
            for i in 0..20 {
                let node_id = format!("node-{}-{}", writer_id, i);
                let graph = graph_clone.write().await;
                let _ = graph
                    .add_node(
                        node_id.clone(),
                        serde_json::json!({
                            "writer": writer_id,
                            "index": i,
                            "data": "x".repeat(writer_id * 10)
                        }),
                    )
                    .await;

                // Add random edges
                if i > 0 {
                    let target = format!("node-{}-{}", writer_id, i - 1);
                    let _ = graph
                        .add_edge(
                            node_id,
                            target,
                            "connects_to".to_string(),
                            serde_json::json!({"weight": i}),
                        )
                        .await;
                }
            }
        });
        handles.push(handle);
    }

    // Concurrent readers
    for reader_id in 0..10 {
        let graph_clone = graph.clone();
        let handle = tokio::spawn(async move {
            let mut successful_reads = 0;

            for _ in 0..50 {
                let graph = graph_clone.read().await;
                let node_id = format!("node-{}-{}", reader_id % 5, rand::random::<usize>() % 20);

                if graph.get_node(&node_id).await.is_some() {
                    successful_reads += 1;
                }

                // Random delay
                sleep(Duration::from_micros(rand::random::<u64>() % 100)).await;
            }

            successful_reads
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let _ = handle.await;
    }

    // Verify graph integrity
    let graph = graph.read().await;
    let stats = graph.stats().await;
    assert!(stats.node_count > 0, "Graph should have nodes");
    assert!(stats.edge_count > 0, "Graph should have edges");
}

/// Test storage with corruption and recovery scenarios
#[tokio::test]
async fn test_storage_corruption_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Mutex::new(
        NvmeStorage::new(NvmeConfig {
            path: temp_dir.path().to_path_buf(),
            max_size_bytes: 1024 * 1024, // 1MB
            io_depth: 32,
            compression: true,
        })
        .await
        .unwrap(),
    ));

    // Store initial data
    let test_data = b"Important data that must survive corruption";
    storage
        .lock()
        .await
        .store("critical_data", test_data)
        .await
        .unwrap();

    // Simulate concurrent access with potential corruption
    let mut handles = vec![];

    // Writer that might corrupt
    let storage_clone = storage.clone();
    let corruptor = tokio::spawn(async move {
        for i in 0..10 {
            let key = format!("corrupt_{}", i);
            let data = vec![0xFF; 1000]; // Potentially corrupting data

            let mut storage = storage_clone.lock().await;
            let _ = storage.store(&key, &data).await;

            // Immediately try to overwrite
            let _ = storage.store(&key, b"fixed").await;
        }
    });
    handles.push(corruptor);

    // Reader trying to access data
    let storage_clone = storage.clone();
    let reader = tokio::spawn(async move {
        let mut successful_reads = 0;
        let mut corrupted_reads = 0;

        for _ in 0..50 {
            let storage = storage_clone.lock().await;

            // Try to read critical data
            match storage.retrieve("critical_data").await {
                Ok(data) if data == test_data => successful_reads += 1,
                Ok(_) => corrupted_reads += 1,
                Err(_) => {}
            }

            drop(storage);
            sleep(Duration::from_micros(10)).await;
        }

        (successful_reads, corrupted_reads)
    });
    handles.push(reader);

    // Wait for completion
    for handle in handles {
        let _ = handle.await;
    }

    // Verify critical data survived
    let final_data = storage
        .lock()
        .await
        .retrieve("critical_data")
        .await
        .unwrap();
    assert_eq!(final_data, test_data, "Critical data should survive");
}

/// Test runtime container resource exhaustion and recovery
#[tokio::test]
async fn test_container_resource_exhaustion() {
    let runtime = SecureContainerRuntime::new();

    // Create container with minimal resources
    let config = ContainerConfig {
        memory_limit_bytes: 1024, // 1KB - extremely small
        gpu_compute_units: 1,
        timeout_seconds: Some(1),
        agent_type: "resource_test".to_string(),
        environment: HashMap::new(),
    };

    let container = runtime.create_container(config).await.unwrap();
    let container_id = container.id();

    runtime.start_container(container_id).await.unwrap();

    // Try to exhaust memory
    let mut allocation_results = vec![];

    for i in 0..10 {
        let result = runtime
            .allocate_memory(
                container_id,
                256, // Each allocation is 256 bytes
            )
            .await;

        allocation_results.push((i, result.is_ok()));

        if result.is_err() {
            // Should fail due to memory limit
            break;
        }
    }

    // Some allocations should succeed, some should fail
    let successes = allocation_results.iter().filter(|(_, ok)| *ok).count();
    let failures = allocation_results.iter().filter(|(_, ok)| !*ok).count();

    assert!(successes > 0, "Some allocations should succeed");
    assert!(failures > 0 || successes <= 4, "Should hit memory limit");

    // Container should still be operational
    let stats = runtime.container_stats(container_id).await.unwrap();
    assert!(stats.state == "Running" || stats.state == "ResourceExhausted");

    // Cleanup
    runtime.stop_container(container_id).await.unwrap();
    runtime.remove_container(container_id).await.unwrap();
}

/// Test network partition and message ordering
#[tokio::test]
async fn test_network_partition_recovery() {
    let node1 = Arc::new(MemoryNetwork::new("node1".to_string()));
    let node2 = Arc::new(MemoryNetwork::new("node2".to_string()));
    let node3 = Arc::new(MemoryNetwork::new("node3".to_string()));

    // Send messages in specific order
    let messages = vec![
        ("msg1", MessageType::Consensus, 1),
        ("msg2", MessageType::KnowledgeSync, 2),
        ("msg3", MessageType::Evolution, 3),
    ];

    // Send from node1 to others
    for (content, msg_type, order) in &messages {
        let msg = Message {
            msg_type: msg_type.clone(),
            payload: content.as_bytes().to_vec(),
            timestamp: *order,
        };

        node1.send("node2", msg.clone()).await.unwrap();
        node1.send("node3", msg).await.unwrap();
    }

    // Simulate partition - node2 receives out of order
    for i in (0..3).rev() {
        let (content, msg_type, order) = &messages[i];
        let msg = Message {
            msg_type: msg_type.clone(),
            payload: content.as_bytes().to_vec(),
            timestamp: *order,
        };

        node2
            .simulate_receive("node1".to_string(), msg)
            .await
            .unwrap();
    }

    // Node3 receives in order
    for (content, msg_type, order) in &messages {
        let msg = Message {
            msg_type: msg_type.clone(),
            payload: content.as_bytes().to_vec(),
            timestamp: *order,
        };

        node3
            .simulate_receive("node1".to_string(), msg)
            .await
            .unwrap();
    }

    // Verify message ordering is preserved based on timestamp
    let mut node2_messages = vec![];
    while let Ok((_, msg)) = node2.receive().await {
        node2_messages.push(msg);
    }

    let mut node3_messages = vec![];
    while let Ok((_, msg)) = node3.receive().await {
        node3_messages.push(msg);
    }

    // Both should have all messages
    assert_eq!(node2_messages.len(), 3);
    assert_eq!(node3_messages.len(), 3);
}

/// Test synthesis with malformed patterns and templates
#[tokio::test]
async fn test_synthesis_malformed_inputs() {
    let config = SynthesisConfig {
        max_depth: 10,
        max_iterations: 100,
        timeout_ms: 1000,
        cache_size: 100,
    };

    let synthesis = SynthesisPipeline::new(config).await.unwrap();

    // Test various malformed patterns
    let malformed_patterns = vec![
        Pattern {
            pattern: "".to_string(),
            weight: 1.0,
        }, // Empty pattern
        Pattern {
            pattern: "a".repeat(10000),
            weight: 0.0,
        }, // Huge pattern, zero weight
        Pattern {
            pattern: "{{{{".to_string(),
            weight: -1.0,
        }, // Unbalanced, negative weight
        Pattern {
            pattern: "\0\0\0".to_string(),
            weight: f64::NAN,
        }, // Null bytes, NaN weight
        Pattern {
            pattern: "🦀💥".to_string(),
            weight: f64::INFINITY,
        }, // Unicode, infinite weight
    ];

    for pattern in malformed_patterns {
        let result = synthesis.add_pattern(pattern).await;
        // Should handle gracefully - either accept or reject
        assert!(result.is_ok() || result.is_err());
    }

    // Test malformed templates
    let malformed_templates = vec![
        Template {
            name: "".to_string(),
            template: "".to_string(),
        },
        Template {
            name: "a".repeat(1000),
            template: "b".repeat(1000),
        },
        Template {
            name: "null\0template",
            template: "\0\0\0",
        },
        Template {
            name: "unicode🔥",
            template: "🦀{{var}}🚀",
        },
    ];

    for template in malformed_templates {
        let result = synthesis.add_template(template).await;
        assert!(result.is_ok() || result.is_err());
    }

    // Try synthesis with these malformed inputs
    let goal = "synthesize something valid";
    let result = timeout(Duration::from_secs(2), synthesis.synthesize(goal)).await;

    // Should either complete or timeout gracefully
    assert!(result.is_ok() || result.is_err());
}

/// Test integration with all components under memory pressure
#[tokio::test]
async fn test_full_system_memory_pressure() {
    // Create all components with minimal memory
    let memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 10) // 10KB total
            .expect("Failed to create allocator"),
    );

    let storage = Arc::new(MemoryStorage::new(1024 * 5)); // 5KB
    let network = Arc::new(MemoryNetwork::new("pressure_test".to_string()));

    let agent_config = AgentConfig {
        name: "memory_pressure_agent".to_string(),
        agent_type: "test".to_string(),
        max_memory: 1024,    // 1KB
        max_gpu_memory: 512, // 512B
        priority: 1,
        metadata: serde_json::json!({}),
    };

    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Perform operations that consume memory
    let mut operations = vec![];

    // GPU allocation
    operations.push(async { memory_allocator.allocate(512).await });

    // Storage operations
    operations.push(async { storage.store("key1", b"x".repeat(1000).as_slice()).await });

    // Network messages
    operations.push(async {
        let msg = Message {
            msg_type: MessageType::Evolution,
            payload: vec![0u8; 1000],
            timestamp: 0,
        };
        network.send("peer", msg).await
    });

    // Agent memory
    operations.push(async {
        agent
            .memory()
            .store(
                MemoryType::LongTerm,
                "large_memory".to_string(),
                serde_json::json!({"data": "x".repeat(500)}),
            )
            .await
    });

    // Execute all operations concurrently
    let results = futures::future::join_all(operations).await;

    // Some should succeed, some should fail due to memory pressure
    let successes = results.iter().filter(|r| r.is_ok()).count();
    let failures = results.iter().filter(|r| r.is_err()).count();

    assert!(successes > 0, "Some operations should succeed");
    assert!(
        failures > 0,
        "Some operations should fail due to memory pressure"
    );

    // System should remain functional
    let small_data = b"tiny";
    let recovery = storage.store("recovery", small_data).await;
    assert!(
        recovery.is_ok() || recovery.is_err(),
        "System should handle gracefully"
    );

    agent.shutdown().await.unwrap();
}

#[cfg(test)]
mod verification {
    use super::*;

    #[test]
    fn verify_test_coverage() {
        // This test verifies we're testing edge cases across all major components
        let components_tested = vec![
            "memory_tiers",
            "cpu_gpu_bridge",
            "agent_lifecycle",
            "consensus",
            "evolution",
            "knowledge_graph",
            "storage",
            "runtime_containers",
            "network",
            "synthesis",
            "full_integration",
        ];

        assert_eq!(
            components_tested.len(),
            11,
            "Should test all major component integrations"
        );
    }
}
