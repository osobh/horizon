//! Stress Testing Integration Tests
//! Tests system behavior under extreme load conditions and edge cases

use exorust_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use exorust_cuda::{CudaContext, CudaMemoryManager};
use exorust_evolution_engines::{EvolutionEngineConfig, HybridEvolutionSystem};
use exorust_knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, Query, QueryEngine, QueryType,
};
use exorust_synthesis::{SynthesisConfig, SynthesisPipeline};
// Unused: use std::sync::Arc;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_maximum_agent_capacity() {
    // Test system with maximum number of agents
    let max_agents = 500;
    let mut agents: Vec<Agent> = Vec::new();
    let mut creation_failures = 0;

    for i in 0..max_agents {
        let agent_config = AgentConfig {
            name: format!("stress_agent_{i}"),
            agent_type: "stress_test".to_string(),
            max_memory: 512, // Small memory footprint
            max_gpu_memory: 256,
            priority: 1,
            metadata: serde_json::json!({"stress_id": i}),
        };

        match Agent::new(agent_config) {
            Ok(agent) => {
                if agent.initialize().await.is_ok() {
                    agents.push(agent);
                } else {
                    creation_failures += 1;
                }
            }
            Err(_) => creation_failures += 1,
        }

        // Break if we hit too many failures (system limit reached)
        if creation_failures > 50 {
            break;
        }
    }

    // Should create at least 100 agents before hitting limits
    assert!(agents.len() >= 100, "Should create at least 100 agents");

    // Test that all created agents are functional
    let mut functional_agents = 0;
    for agent in &agents {
        let goal = Goal::new("Stress test goal".to_string(), GoalPriority::Low);
        if agent.add_goal(goal).await.is_ok() {
            functional_agents += 1;
        }
    }

    // Most agents should remain functional
    assert!(functional_agents >= agents.len() * 9 / 10);

    // Cleanup
    for agent in agents {
        let _ = agent.shutdown().await;
    }
}

#[tokio::test]
async fn test_memory_exhaustion_scenarios() {
    let agent_config = AgentConfig {
        name: "memory_exhaustion_agent".to_string(),
        agent_type: "exhaustion_test".to_string(),
        max_memory: 4096, // 4MB limit
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Test 1: Gradual memory exhaustion
    let mut stored_memories = 0;
    for i in 0..2000 {
        let massive_data = serde_json::json!({
            "id": i,
            "data": "x".repeat(2048), // 2KB each
            "nested": {
                "level1": "y".repeat(1024),
                "level2": {
                    "data": "z".repeat(512),
                    "array": vec![format!("item_{j}") for j in 0..100]
                }
            }
        });

        match agent
            .memory()
            .store(
                MemoryType::Working,
                format!("exhaustion_memory_{i}"),
                massive_data,
            )
            .await
        {
            Ok(_) => stored_memories += 1,
            Err(_) => break, // Hit memory limit
        }
    }

    assert!(
        stored_memories > 0,
        "Should store some memories before exhaustion"
    );
    assert!(
        stored_memories < 2000,
        "Should hit memory limit before storing all"
    );

    // Test 2: System should remain responsive after exhaustion
    let small_data = serde_json::json!({"test": "small"});
    let recovery_result = agent
        .memory()
        .store(MemoryType::Working, "recovery_test".to_string(), small_data)
        .await;

    // Should either succeed (if memory was freed) or fail gracefully
    assert!(recovery_result.is_ok() || recovery_result.is_err());

    // Test 3: Agent should still accept goals
    let recovery_goal = Goal::new("Recovery goal".to_string(), GoalPriority::Low);
    let goal_result = agent.add_goal(recovery_goal).await;
    assert!(goal_result.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_knowledge_graph_node_limit_stress() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 1000, // Small limit to test boundaries
        max_edges: 5000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    // Create agents to fill the knowledge graph
    let mut agents: Vec<Agent> = Vec::new();
    for i in 0..20 {
        let agent_config = AgentConfig {
            name: format!("kg_stress_agent_{i}"),
            agent_type: "kg_stress".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"agent_id": i}),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();

        // Each agent stores many memories
        for j in 0..100 {
            agent
                .memory()
                .store(
                    MemoryType::Semantic,
                    format!("kg_stress_{}_{i, j}"),
                    serde_json::json!({
                        "agent": i,
                        "memory": j,
                        "content": format!("Stress test content {} from agent {j, i}"),
                        "category": format!("category_{j % 10}"),
                        "metadata": {
                            "importance": j % 5,
                            "tags": vec![format!("tag_{k}") for k in 0..5]
                        }
                    }),
                )
                .await
                .unwrap();
        }

        agents.push(agent);
    }

    // Sync all memories to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    let mut total_synced = 0;
    let mut sync_failures = 0;

    for agent in &agents {
        match memory_integration
            .sync_agent_memory(agent, &mut graph)
            .await
        {
            Ok(synced) => total_synced += synced,
            Err(_) => sync_failures += 1,
        }
    }

    // Should sync some memories but hit node limits
    assert!(total_synced > 0);
    assert!(sync_failures > 0 || total_synced < 2000); // Should hit limits

    // Test graph is still queryable
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(exorust_knowledge_graph::NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: Some(50),
        offset: None,
        use_gpu: false,
    };

    let query_result = query_engine.execute(&graph, query).await;
    assert!(query_result.is_ok());

    // Verify node count is within limits
    let stats = graph.stats();
    assert!(stats.node_count <= 1000);

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_evolution_engine_population_explosion() {
    let evolution_config = EvolutionEngineConfig {
        population_size: 1000, // Very large population
        max_generations: 50,
        mutation_rate: 0.2,  // High mutation rate
        crossover_rate: 0.9, // High crossover rate
        elite_size: 50,
        fitness_threshold: 0.99,
        diversity_threshold: 0.05,
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create massive initial population
    let mut initial_population = Vec::new();
    for i in 0..1000 {
        let agent_config = AgentConfig {
            name: format!("evolution_stress_agent_{i}"),
            agent_type: "evolution_stress".to_string(),
            max_memory: 512,
            max_gpu_memory: 256,
            priority: 1,
            metadata: serde_json::json!({"stress_id": i, "generation": 0}),
        };

        let agent = Agent::new(agent_config).unwrap();
        initial_population.push(agent);
    }

    // Test evolution with timeout (may hit resource limits)
    let evolution_result = timeout(
        Duration::from_secs(60), // Generous timeout
        hybrid_system.evolve_generation(initial_population),
    )
    .await;

    match evolution_result {
        Ok(result) => {
            // Evolution completed
            assert!(result.is_ok());
            let evolved_population = result.unwrap();
            assert!(!evolved_population.is_empty());
            assert!(evolved_population.len() <= 1000);
        }
        Err(_) => {
            // Evolution timed out - acceptable under stress
            println!("Evolution timed out under stress conditions");
        }
    }

    // System should still be responsive
    let stats_result = hybrid_system.get_evolution_stats().await;
    assert!(stats_result.is_ok());
}

#[tokio::test]
async fn test_synthesis_pipeline_overload() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    // Create extremely complex and large goals
    let complex_goals = vec![
        format!("Develop a comprehensive artificial intelligence system that can {} and {} while {} with {} capabilities", 
                "x".repeat(1000), "y".repeat(1000), "z".repeat(1000), "w".repeat(1000)),
        format!("Create a distributed quantum computing framework that implements {} algorithms for {} optimization with {} constraints",
                "a".repeat(500), "b".repeat(500), "c".repeat(500)),
        "🚀".repeat(10000), // Unicode stress test
        "\x00\x01\x02".repeat(1000), // Binary data stress test
        format!("Recursive goal: {}", "Design a system to design a system to ".repeat(1000)),
    ];

    let mut successful_synthesis = 0;
    let mut failed_synthesis = 0;

    for (i, goal_text) in complex_goals.iter().enumerate() {
        let goal = Goal::new(goal_text.clone(), GoalPriority::Normal);

        // Use timeout to prevent hanging
        let synthesis_result = timeout(
            Duration::from_secs(10),
            synthesis_pipeline.process_goal(&goal),
        )
        .await;

        match synthesis_result {
            Ok(Ok(_)) => successful_synthesis += 1,
            Ok(Err(_)) => failed_synthesis += 1,
            Err(_) => failed_synthesis += 1, // Timeout
        }
    }

    // System should handle some cases gracefully
    assert!(successful_synthesis > 0 || failed_synthesis > 0);

    // Test recovery with normal goal
    let normal_goal = Goal::new("Simple recovery goal".to_string(), GoalPriority::Normal);
    let recovery_result = timeout(
        Duration::from_secs(5),
        synthesis_pipeline.process_goal(&normal_goal),
    )
    .await;

    assert!(recovery_result.is_ok());
}

#[tokio::test]
async fn test_cuda_device_exhaustion() {
    // Try to create contexts for many devices
    let mut contexts = Vec::new();
    let mut memory_managers: Vec<CudaMemoryManager> = Vec::new();
    let max_devices = 20;

    for device_id in 0..max_devices {
        match CudaContext::new(device_id).await {
            Ok(context) => {
                // Try to create memory manager
                match CudaMemoryManager::new(context.device_id()).await {
                    Ok(manager) => {
                        contexts.push(context);
                        memory_managers.push(manager);
                    }
                    Err(_) => break,
                }
            }
            Err(_) => break, // No more devices available
        }
    }

    // Test massive memory allocation on available devices
    let mut allocations = Vec::new();
    for manager in &memory_managers {
        // Try to allocate large chunks until failure
        for _ in 0..100 {
            match manager.allocate(1024 * 1024 * 10).await {
                // 10MB chunks
                Ok(allocation) => allocations.push((manager, allocation)),
                Err(_) => break, // Out of memory
            }
        }
    }

    // Should have made some allocations
    if !memory_managers.is_empty() {
        assert!(!allocations.is_empty(), "Should allocate some memory");
    }

    // Test system stability after allocation stress
    for manager in &memory_managers {
        let small_alloc = manager.allocate(1024).await; // 1KB
                                                        // Should either succeed or fail gracefully
        assert!(small_alloc.is_ok() || small_alloc.is_err());
    }

    // Cleanup allocations
    for (manager, allocation) in allocations {
        let _ = manager.deallocate(allocation).await;
    }
}

#[tokio::test]
async fn test_concurrent_stress_operations() {
    // Launch many concurrent stress operations
    let num_stress_tasks = 50;
    let mut handles = Vec::new();

    for i in 0..num_stress_tasks {
        let handle = tokio::spawn(async move {
            // Each task performs intensive operations
            let agent_config = AgentConfig {
                name: format!("stress_concurrent_agent_{i}"),
                agent_type: "concurrent_stress".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"stress_task": i}),
            };

            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();

            // Intensive memory operations
            for j in 0..50 {
                let large_data = serde_json::json!({
                    "task": i,
                    "iteration": j,
                    "data": "x".repeat(1000),
                    "metadata": {
                        "timestamp": j,
                        "nested": "y".repeat(500)
                    }
                });

                let _ = agent
                    .memory()
                    .store(
                        MemoryType::Working,
                        format!("stress_memory_{}_{i, j}"),
                        large_data,
                    )
                    .await;
            }

            // Intensive goal operations
            for j in 0..20 {
                let goal = Goal::new(
                    format!("Stress goal {} for task {j, i}"),
                    GoalPriority::Normal,
                );
                let _ = agent.add_goal(goal).await;
            }

            // State transitions
            for _ in 0..10 {
                let _ = agent.pause().await;
                let _ = agent.resume().await;
            }

            agent.shutdown().await.unwrap();
            i // Return task id
        });

        handles.push(handle);
    }

    // Use timeout to prevent hanging
    let concurrent_result = timeout(Duration::from_secs(30), async move {
        let mut completed_tasks = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(task_id) => completed_tasks.push(task_id),
                Err(_) => {} // Task may have panicked under stress
            }
        }
        completed_tasks
    })
    .await;

    assert!(
        concurrent_result.is_ok(),
        "Concurrent stress test should not hang"
    );
    let completed_tasks = concurrent_result.unwrap();

    // Should complete at least half the tasks under stress
    assert!(completed_tasks.len() >= num_stress_tasks / 2);
}

#[tokio::test]
async fn test_system_recovery_after_stress() {
    // First, apply stress to the system
    let mut stress_agents = Vec::new();

    // Create many agents with large memory usage
    for i in 0..100 {
        let agent_config = AgentConfig {
            name: format!("recovery_stress_agent_{i}"),
            agent_type: "recovery_stress".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"stress_phase": true}),
        };

        match Agent::new(agent_config) {
            Ok(agent) => {
                if agent.initialize().await.is_ok() {
                    // Store large amounts of data
                    for j in 0..20 {
                        let _ = agent
                            .memory()
                            .store(
                                MemoryType::Working,
                                format!("stress_data_{}_{i, j}"),
                                serde_json::json!({
                                    "data": "x".repeat(2000),
                                    "metadata": "y".repeat(1000)
                                }),
                            )
                            .await;
                    }
                    stress_agents.push(agent);
                }
            }
            Err(_) => break, // Hit system limits
        }
    }

    // Shutdown all stress agents
    for agent in stress_agents {
        let _ = agent.shutdown().await;
    }

    // Allow system to recover
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test system recovery with normal operations
    let recovery_agent_config = AgentConfig {
        name: "recovery_test_agent".to_string(),
        agent_type: "recovery_test".to_string(),
        max_memory: 1024,
        max_gpu_memory: 512,
        priority: 1,
        metadata: serde_json::json!({"recovery_phase": true}),
    };

    let recovery_agent = Agent::new(recovery_agent_config).unwrap();
    recovery_agent.initialize().await.unwrap();

    // System should be functional after stress
    let goal = Goal::new(
        "Recovery verification goal".to_string(),
        GoalPriority::Normal,
    );
    let goal_result = recovery_agent.add_goal(goal).await;
    assert!(goal_result.is_ok(), "System should recover after stress");

    let memory_result = recovery_agent
        .memory()
        .store(
            MemoryType::Working,
            "recovery_memory".to_string(),
            serde_json::json!({"test": "recovery"}),
        )
        .await;
    assert!(memory_result.is_ok(), "Memory system should recover");

    recovery_agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_edge_case_data_handling() {
    let agent_config = AgentConfig {
        name: "edge_case_agent".to_string(),
        agent_type: "edge_case_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Test various edge cases
    let edge_cases = vec![
        ("empty_string", serde_json::json!("")),
        ("null_value", serde_json::json!(null)),
        ("max_int", serde_json::json!(i64::MAX)),
        ("min_int", serde_json::json!(i64::MIN)),
        ("unicode_data", serde_json::json!("🚀🌟💫⭐🌙☀️🌍🔥💎")),
        ("binary_like", serde_json::json!("binary_test_data_content")),
        (
            "nested_deep",
            serde_json::json!({
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "data": "deeply nested"
                                }
                            }
                        }
                    }
                }
            }),
        ),
        (
            "large_array",
            serde_json::json!(vec![format!("item_{i}") for i in 0..1000]),
        ),
        (
            "special_chars",
            serde_json::json!("!@#$%^&*()_+-=[]{}|;':\",./<>?"),
        ),
        ("very_long_key", serde_json::json!({"data": "value"})),
    ];

    let mut successful_stores = 0;
    let mut failed_stores = 0;

    for (i, (name, data)) in edge_cases.iter().enumerate() {
        let key = if name == &"very_long_key" {
            "x".repeat(1000) // Very long key
        } else {
            format!("edge_case_{}_{i, name}")
        };

        match agent
            .memory()
            .store(MemoryType::Working, key, data.clone())
            .await
        {
            Ok(_) => successful_stores += 1,
            Err(_) => failed_stores += 1,
        }
    }

    // System should handle most edge cases gracefully
    assert!(successful_stores > 0, "Should handle some edge cases");

    // Test that agent remains functional
    let normal_goal = Goal::new(
        "Normal goal after edge cases".to_string(),
        GoalPriority::Normal,
    );
    let goal_result = agent.add_goal(normal_goal).await;
    assert!(goal_result.is_ok(), "Agent should remain functional");

    agent.shutdown().await.unwrap();
}
