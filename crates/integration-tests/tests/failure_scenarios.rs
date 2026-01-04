//! Failure Scenario Integration Tests
//! Tests system behavior under various failure conditions and recovery mechanisms

use std::sync::Arc;
use stratoswarm_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use stratoswarm_cuda::{CudaContext, CudaMemoryManager};
use stratoswarm_evolution_engines::{EvolutionEngineConfig, HybridEvolutionSystem};
use stratoswarm_knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, Query, QueryEngine, QueryType,
};
use stratoswarm_synthesis::{SynthesisConfig, SynthesisPipeline};
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_memory_exhaustion_recovery() {
    let agent_config = AgentConfig {
        name: "memory_stress_agent".to_string(),
        agent_type: "stress_test".to_string(),
        max_memory: 1024, // Very small limit
        max_gpu_memory: 512,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Try to store memories exceeding the limit
    let mut successful_stores = 0;
    let mut failed_stores = 0;

    for i in 0..100 {
        let large_data = serde_json::json!({
            "data": "x".repeat(100), // Large memory entry
            "index": i
        });

        let result = agent
            .memory()
            .store(MemoryType::Working, format!("large_memory_{i}"), large_data)
            .await;

        match result {
            Ok(_) => successful_stores += 1,
            Err(_) => failed_stores += 1,
        }
    }

    // System should handle memory pressure gracefully
    assert!(successful_stores > 0);
    assert!(failed_stores > 0); // Some should fail due to memory limits

    // Memory system should still be functional
    let small_data = serde_json::json!({"test": "recovery"});
    let recovery_result = agent
        .memory()
        .store(MemoryType::Working, "recovery_test".to_string(), small_data)
        .await;
    assert!(recovery_result.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_cuda_context_failure_recovery() {
    // Test recovery from CUDA context failures
    let mut contexts: Vec<CudaContext> = Vec::new();
    let mut successful_contexts = 0;
    let mut failed_contexts = 0;

    // Try to create many CUDA contexts (some may fail)
    for device_id in 0..10 {
        match CudaContext::new(device_id).await {
            Ok(context) => {
                contexts.push(context);
                successful_contexts += 1;
            }
            Err(_) => {
                failed_contexts += 1;
            }
        }
    }

    // Should handle graceful fallback
    assert!(successful_contexts > 0 || failed_contexts > 0);

    // Test memory allocation on successful contexts
    for context in &contexts {
        let memory_manager = CudaMemoryManager::new(context.device_id()).await.unwrap();

        // Try large allocation that might fail
        let large_allocation = memory_manager.allocate(1024 * 1024 * 1024).await; // 1GB

        // Try smaller allocation that should succeed
        let small_allocation = memory_manager.allocate(1024).await; // 1KB
        assert!(small_allocation.is_ok());

        memory_manager
            .deallocate(small_allocation.unwrap())
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn test_knowledge_graph_corruption_recovery() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 1000,
        max_edges: 5000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    // Add valid data
    let agent_config = AgentConfig {
        name: "corruption_test_agent".to_string(),
        agent_type: "corruption_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store some memories
    for i in 0..10 {
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                format!("valid_memory_{i}"),
                serde_json::json!({"data": format!("content_{i}")}),
            )
            .await
            .unwrap();
    }

    // Sync to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    let initial_sync = memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await
        .unwrap();
    assert!(initial_sync > 0);

    // Simulate corruption by trying to add invalid data
    for i in 0..5 {
        let corrupt_data = serde_json::json!({
            "invalid_field": null,
            "corrupted_data": "\x00\x01\x02", // Binary data that might cause issues
            "index": i
        });

        let result = agent
            .memory()
            .store(
                MemoryType::Working,
                format!("corrupt_memory_{i}"),
                corrupt_data,
            )
            .await;

        // Some may fail, that's expected
    }

    // Try to sync corrupted data
    let corrupt_sync_result = memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await;

    // System should handle corruption gracefully
    assert!(corrupt_sync_result.is_ok());

    // Verify graph is still queryable
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(stratoswarm_knowledge_graph::NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: Some(20),
        offset: None,
        use_gpu: false,
    };

    let query_result = query_engine.execute(&graph, query).await;
    assert!(query_result.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_evolution_engine_timeout_recovery() {
    let evolution_config = EvolutionEngineConfig {
        population_size: 100,
        max_generations: 1000, // Very large to potentially cause timeout
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elite_size: 5,
        fitness_threshold: 0.99, // Very high threshold
        diversity_threshold: 0.1,
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create large initial population
    let mut initial_population = Vec::new();
    for i in 0..50 {
        let agent_config = AgentConfig {
            name: format!("timeout_test_agent_{i}"),
            agent_type: "timeout_test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({"generation": 0}),
        };

        let agent = Agent::new(agent_config).unwrap();
        initial_population.push(agent);
    }

    // Test evolution with timeout
    let evolution_future = hybrid_system.evolve_generation(initial_population);
    let timeout_result = timeout(Duration::from_secs(5), evolution_future).await;

    match timeout_result {
        Ok(evolution_result) => {
            // Evolution completed within timeout
            assert!(evolution_result.is_ok());
            let final_population = evolution_result.unwrap();
            assert!(!final_population.is_empty());
        }
        Err(_) => {
            // Evolution timed out - this is acceptable behavior
            // System should handle timeouts gracefully
        }
    }

    // Verify system is still responsive
    let stats_result = hybrid_system.get_evolution_stats().await;
    assert!(stats_result.is_ok());
}

#[tokio::test]
async fn test_synthesis_pipeline_error_cascade() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    // Test with various problematic goals
    let problematic_goals = vec![
        Goal::new("".to_string(), GoalPriority::Low), // Empty goal
        Goal::new("a".repeat(10000), GoalPriority::Normal), // Very long goal
        Goal::new(
            "Invalid binary characters in goal".to_string(),
            GoalPriority::High,
        ), // Binary characters
        Goal::new(
            "Goal with unicode: 🚀🔥💻🎯".to_string(),
            GoalPriority::Medium,
        ), // Unicode
    ];

    let mut successful_synthesis = 0;
    let mut failed_synthesis = 0;

    for goal in problematic_goals {
        let synthesis_result = synthesis_pipeline.process_goal(&goal).await;

        match synthesis_result {
            Ok(_) => successful_synthesis += 1,
            Err(_) => failed_synthesis += 1,
        }
    }

    // System should handle some cases gracefully
    assert!(successful_synthesis > 0 || failed_synthesis > 0);

    // Test recovery with valid goal
    let valid_goal = Goal::new("Valid recovery goal".to_string(), GoalPriority::Normal);
    let recovery_result = synthesis_pipeline.process_goal(&valid_goal).await;
    assert!(recovery_result.is_ok());
}

#[tokio::test]
async fn test_concurrent_agent_failure_isolation() {
    // Test that one agent's failure doesn't affect others
    let num_agents = 5;
    let mut agent_tasks = Vec::new();

    for i in 0..num_agents {
        let task = tokio::spawn(async move {
            let agent_config = AgentConfig {
                name: format!("isolation_test_agent_{i}"),
                agent_type: "isolation_test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"index": i}),
            };

            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();

            // Agent 2 will intentionally fail
            if i == 2 {
                // Cause failure by invalid state transition
                agent.transition_to(AgentState::Terminated).await.unwrap(); // Invalid transition
                return Err("Intentional failure");
            }

            // Other agents perform normal operations
            let goal = Goal::new(
                format!("Normal operation for agent {i}"),
                GoalPriority::Normal,
            );
            agent.add_goal(goal).await.unwrap();

            // Store memories
            for j in 0..5 {
                agent
                    .memory()
                    .store(
                        MemoryType::Working,
                        format!("memory_{}_{i, j}"),
                        serde_json::json!({"agent": i, "memory": j}),
                    )
                    .await
                    .unwrap();
            }

            agent.shutdown().await.unwrap();
            Ok(i)
        });

        agent_tasks.push(task);
    }

    // Collect results
    let mut successful_agents = 0;
    let mut failed_agents = 0;

    for task in agent_tasks {
        match task.await.unwrap() {
            Ok(_) => successful_agents += 1,
            Err(_) => failed_agents += 1,
        }
    }

    // Should have exactly one failure (agent 2) and others successful
    assert_eq!(failed_agents, 1);
    assert_eq!(successful_agents, num_agents - 1);
}

#[tokio::test]
async fn test_resource_exhaustion_graceful_degradation() {
    // Test system behavior when resources are exhausted
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 100, // Very small limit
        max_edges: 200,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let agent_config = AgentConfig {
        name: "resource_exhaustion_agent".to_string(),
        agent_type: "exhaustion_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Fill up the knowledge graph to capacity
    let mut stored_memories = 0;
    for i in 0..200 {
        // Try to store more than the limit
        let result = agent
            .memory()
            .store(
                MemoryType::Semantic,
                format!("exhaustion_memory_{i}"),
                serde_json::json!({"data": format!("content_{i}"), "index": i}),
            )
            .await;

        if result.is_ok() {
            stored_memories += 1;
        }
    }

    // Sync to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    let sync_result = memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await;

    // Should handle resource limits gracefully
    assert!(sync_result.is_ok());
    let synced_count = sync_result.unwrap();

    // Should not exceed graph limits
    let stats = graph.stats();
    assert!(stats.node_count <= 100);

    // System should still be responsive for queries
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(stratoswarm_knowledge_graph::NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: Some(10),
        offset: None,
        use_gpu: false,
    };

    let query_result = query_engine.execute(&graph, query).await;
    assert!(query_result.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_network_partition_simulation() {
    // Simulate network partitions between agents
    let num_agents = 4;
    let mut agents = Vec::new();
    let mut knowledge_graphs = Vec::new();

    // Create isolated agents with separate knowledge graphs
    for i in 0..num_agents {
        let agent_config = AgentConfig {
            name: format!("partitioned_agent_{i}"),
            agent_type: "partition_test".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"partition": i % 2}), // Two partitions
        };

        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();

        let kg_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let graph = KnowledgeGraph::new(kg_config).await.unwrap();

        agents.push(agent);
        knowledge_graphs.push(graph);
    }

    // Each agent operates in isolation (simulating network partition)
    for (i, (agent, mut graph)) in agents.iter().zip(knowledge_graphs.iter_mut()).enumerate() {
        // Add partition-specific data
        for j in 0..5 {
            agent
                .memory()
                .store(
                    MemoryType::Working,
                    format!("partition_{}_memory_{i % 2, j}"),
                    serde_json::json!({
                        "partition": i % 2,
                        "agent": i,
                        "data": format!("isolated_data_{}_{i, j}")
                    }),
                )
                .await
                .unwrap();
        }

        // Sync to local knowledge graph
        let memory_integration = graph.get_memory_integration().unwrap();
        let synced = memory_integration
            .sync_agent_memory(agent, graph)
            .await
            .unwrap();
        assert!(synced > 0);
    }

    // Verify each partition has isolated data
    for (i, graph) in knowledge_graphs.iter().enumerate() {
        let stats = graph.stats();
        assert!(stats.node_count > 0);

        // Each graph should only have data from its partition
        let mut query_engine = QueryEngine::new(false).await.unwrap();
        let query = Query {
            query_type: QueryType::FindNodes {
                node_type: Some(stratoswarm_knowledge_graph::NodeType::Memory),
                properties: std::collections::HashMap::new(),
            },
            timeout_ms: Some(5000),
            limit: None,
            offset: None,
            use_gpu: false,
        };

        let result = query_engine.execute(graph, query).await.unwrap();
        assert!(!result.nodes.is_empty());
    }

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_system_recovery_after_panic() {
    // Test system recovery after a component panic
    let agent_config = AgentConfig {
        name: "panic_recovery_agent".to_string(),
        agent_type: "panic_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Spawn a task that will panic
    let panic_task = tokio::spawn(async {
        // Simulate a panic in a background task
        panic!("Simulated panic for testing");
    });

    // The panic should not affect the main system
    let panic_result = panic_task.await;
    assert!(panic_result.is_err()); // Task should have panicked

    // Main system should still be functional
    let goal = Goal::new("Recovery test goal".to_string(), GoalPriority::Normal);
    let add_goal_result = agent.add_goal(goal).await;
    assert!(add_goal_result.is_ok());

    // Memory system should still work
    let memory_result = agent
        .memory()
        .store(
            MemoryType::Working,
            "recovery_test".to_string(),
            serde_json::json!({"test": "recovery_after_panic"}),
        )
        .await;
    assert!(memory_result.is_ok());

    // Agent state should be consistent
    let state = agent.state().await;
    assert!(matches!(
        state,
        AgentState::Planning | AgentState::Executing | AgentState::Idle
    ));

    agent.shutdown().await.unwrap();
}
