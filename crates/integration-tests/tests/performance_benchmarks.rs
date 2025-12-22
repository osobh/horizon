//! Performance Benchmark Integration Tests
//! Tests system performance under various load conditions and scenarios

use exorust_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use exorust_cuda::{CudaContext, CudaMemoryManager};
use exorust_evolution_engines::{EvolutionEngineConfig, HybridEvolutionSystem};
use exorust_knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, Query, QueryEngine, QueryType,
};
use exorust_synthesis::{SynthesisConfig, SynthesisPipeline};
// Unused: use std::sync::Arc;
use tokio::time::{timeout, Duration, Instant};

#[tokio::test]
async fn test_high_throughput_agent_creation() {
    let start_time = Instant::now();
    let num_agents = 100;
    let mut agents: Vec<Agent> = Vec::new();

    // Create agents in batches for performance
    for batch in 0..10 {
        let mut batch_agents = Vec::new();
        for i in 0..10 {
            let agent_id = batch * 10 + i;
            let agent_config = AgentConfig {
                name: format!("perf_agent_{agent_id}"),
                agent_type: "performance_test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"batch": batch, "id": agent_id}),
            };

            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();
            batch_agents.push(agent);
        }
        agents.extend(batch_agents);
    }

    let creation_time = start_time.elapsed();

    // Benchmark: Should create 100 agents in under 5 seconds
    assert!(creation_time < Duration::from_secs(5));
    assert_eq!(agents.len(), num_agents);

    // Test concurrent goal assignment
    let goal_start = Instant::now();
    let mut goal_tasks = Vec::new();

    for (i, agent) in agents.iter().enumerate() {
        let goal = Goal::new(format!("Performance test goal {i}"), GoalPriority::Normal);
        goal_tasks.push(agent.add_goal(goal));
    }

    // Wait for all goals to be assigned
    for task in goal_tasks {
        task.await.unwrap();
    }

    let goal_assignment_time = goal_start.elapsed();

    // Benchmark: Should assign 100 goals in under 2 seconds
    assert!(goal_assignment_time < Duration::from_secs(2));

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_memory_intensive_operations() {
    let agent_config = AgentConfig {
        name: "memory_intensive_agent".to_string(),
        agent_type: "memory_test".to_string(),
        max_memory: 10240, // 10MB
        max_gpu_memory: 5120,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    let start_time = Instant::now();
    let num_memories = 1000;

    // Store large number of memories
    for i in 0..num_memories {
        let large_data = serde_json::json!({
            "id": i,
            "data": "x".repeat(1000), // 1KB each
            "metadata": {
                "timestamp": i,
                "category": format!("category_{i % 10}")
            }
        });

        agent
            .memory()
            .store(MemoryType::Working, format!("memory_{i}"), large_data)
            .await
            .unwrap();
    }

    let storage_time = start_time.elapsed();

    // Benchmark: Should store 1000 memories in under 3 seconds
    assert!(storage_time < Duration::from_secs(3));

    // Test memory retrieval performance
    let retrieval_start = Instant::now();
    let mut retrieved_count = 0;

    for i in (0..num_memories).step_by(10) {
        let result = agent
            .memory()
            .retrieve(MemoryType::Working, &format!("memory_{i}"))
            .await;
        if result.is_ok() {
            retrieved_count += 1;
        }
    }

    let retrieval_time = retrieval_start.elapsed();

    // Benchmark: Should retrieve 100 memories in under 1 second
    assert!(retrieval_time < Duration::from_secs(1));
    assert!(retrieved_count > 90); // Most should succeed

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_knowledge_graph_query_performance() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 10000,
        max_edges: 50000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    // Create multiple agents with different memory patterns
    let mut agents: Vec<Agent> = Vec::new();
    for i in 0..20 {
        let agent_config = AgentConfig {
            name: format!("kg_perf_agent_{i}"),
            agent_type: "kg_performance".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"agent_id": i}),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();

        // Store varied memories
        for j in 0..50 {
            agent
                .memory()
                .store(
                    MemoryType::Semantic,
                    format!("semantic_{}_{i, j}"),
                    serde_json::json!({
                        "content": format!("Knowledge item {} from agent {j, i}"),
                        "category": format!("category_{j % 5}"),
                        "importance": j % 10
                    }),
                )
                .await
                .unwrap();
        }

        agents.push(agent);
    }

    // Sync all agent memories to knowledge graph
    let sync_start = Instant::now();
    let memory_integration = graph.get_memory_integration().unwrap();

    for agent in &agents {
        memory_integration
            .sync_agent_memory(agent, &mut graph)
            .await
            .unwrap();
    }

    let sync_time = sync_start.elapsed();

    // Benchmark: Should sync 1000 memories in under 5 seconds
    assert!(sync_time < Duration::from_secs(5));

    // Test query performance
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query_start = Instant::now();

    for i in 0..50 {
        let query = Query {
            query_type: QueryType::FindNodes {
                node_type: Some(exorust_knowledge_graph::NodeType::Memory),
                properties: std::collections::HashMap::new(),
            },
            timeout_ms: Some(1000),
            limit: Some(10),
            offset: Some(i * 10),
            use_gpu: false,
        };

        let result = query_engine.execute(&graph, query).await;
        assert!(result.is_ok());
    }

    let query_time = query_start.elapsed();

    // Benchmark: Should execute 50 queries in under 2 seconds
    assert!(query_time < Duration::from_secs(2));

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_evolution_engine_scalability() {
    let evolution_config = EvolutionEngineConfig {
        population_size: 200, // Large population
        max_generations: 20,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elite_size: 10,
        fitness_threshold: 0.95,
        diversity_threshold: 0.1,
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create large initial population
    let population_start = Instant::now();
    let mut initial_population = Vec::new();

    for i in 0..200 {
        let agent_config = AgentConfig {
            name: format!("evolution_agent_{i}"),
            agent_type: "evolution_scalability".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({"index": i, "generation": 0}),
        };

        let agent = Agent::new(agent_config).unwrap();
        initial_population.push(agent);
    }

    let population_creation_time = population_start.elapsed();

    // Benchmark: Should create 200 agents in under 3 seconds
    assert!(population_creation_time < Duration::from_secs(3));

    // Test evolution performance
    let evolution_start = Instant::now();
    let evolution_result = timeout(
        Duration::from_secs(30),
        hybrid_system.evolve_generation(initial_population),
    )
    .await;

    assert!(evolution_result.is_ok(), "Evolution should not timeout");
    let evolved_population = evolution_result.unwrap().unwrap();

    let evolution_time = evolution_start.elapsed();

    // Benchmark: Should evolve 200 agents in under 30 seconds
    assert!(evolution_time < Duration::from_secs(30));
    assert!(!evolved_population.is_empty());
    assert!(evolved_population.len() <= 200);
}

#[tokio::test]
async fn test_synthesis_pipeline_throughput() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    let goals = vec![
        "Optimize neural network architecture for image recognition",
        "Develop efficient sorting algorithm for large datasets",
        "Create distributed consensus protocol for blockchain",
        "Design machine learning model for natural language processing",
        "Implement real-time data processing pipeline",
        "Build fault-tolerant distributed storage system",
        "Optimize database query performance for analytics",
        "Develop computer vision algorithm for object detection",
        "Create recommendation system for e-commerce platform",
        "Design reinforcement learning agent for game playing",
    ];

    let synthesis_start = Instant::now();
    let mut successful_synthesis = 0;

    // Process multiple goals concurrently
    let mut synthesis_tasks = Vec::new();

    for (i, goal_text) in goals.iter().enumerate() {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::Normal);
        let task = synthesis_pipeline.process_goal(&goal);
        synthesis_tasks.push(task);
    }

    // Wait for all synthesis tasks
    for task in synthesis_tasks {
        if task.await.is_ok() {
            successful_synthesis += 1;
        }
    }

    let synthesis_time = synthesis_start.elapsed();

    // Benchmark: Should process 10 goals in under 5 seconds
    assert!(synthesis_time < Duration::from_secs(5));
    assert!(successful_synthesis >= 8); // Most should succeed
}

#[tokio::test]
async fn test_cuda_memory_allocation_performance() {
    // Test CUDA context creation performance
    let context_start = Instant::now();
    let mut contexts: Vec<CudaContext> = Vec::new();

    for device_id in 0..5 {
        match CudaContext::new(device_id).await {
            Ok(context) => contexts.push(context),
            Err(_) => {} // Some may fail, that's expected
        }
    }

    let context_creation_time = context_start.elapsed();

    // Benchmark: Should create contexts in under 2 seconds
    assert!(context_creation_time < Duration::from_secs(2));

    if !contexts.is_empty() {
        // Test memory allocation performance
        let alloc_start = Instant::now();
        let mut allocations = Vec::new();

        for context in &contexts {
            let memory_manager = CudaMemoryManager::new(context.device_id()).await.unwrap();

            // Allocate multiple small buffers
            for _ in 0..10 {
                if let Ok(allocation) = memory_manager.allocate(1024 * 1024).await {
                    allocations.push((memory_manager, allocation));
                }
            }
        }

        let allocation_time = alloc_start.elapsed();

        // Benchmark: Should allocate memory buffers quickly
        assert!(allocation_time < Duration::from_secs(3));

        // Test deallocation performance
        let dealloc_start = Instant::now();

        for (memory_manager, allocation) in allocations {
            memory_manager.deallocate(allocation).await.unwrap();
        }

        let deallocation_time = dealloc_start.elapsed();

        // Benchmark: Should deallocate quickly
        assert!(deallocation_time < Duration::from_secs(1));
    }
}

#[tokio::test]
async fn test_concurrent_system_operations() {
    // Test system performance under concurrent load
    let num_concurrent_operations = 20;
    let mut handles = Vec::new();

    let start_time = Instant::now();

    // Launch concurrent operations
    for i in 0..num_concurrent_operations {
        let handle = tokio::spawn(async move {
            // Each operation does a mini-workflow
            let agent_config = AgentConfig {
                name: format!("concurrent_agent_{i}"),
                agent_type: "concurrent_performance".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"operation_id": i}),
            };

            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();

            // Add goals
            for j in 0..5 {
                let goal = Goal::new(
                    format!("Concurrent goal {} for operation {j, i}"),
                    GoalPriority::Normal,
                );
                agent.add_goal(goal).await.unwrap();
            }

            // Store memories
            for j in 0..10 {
                agent
                    .memory()
                    .store(
                        MemoryType::Working,
                        format!("concurrent_memory_{}_{i, j}"),
                        serde_json::json!({"operation": i, "memory": j}),
                    )
                    .await
                    .unwrap();
            }

            agent.shutdown().await.unwrap();
            i // Return operation id
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut completed_operations = Vec::new();
    for handle in handles {
        let operation_id = handle.await.unwrap();
        completed_operations.push(operation_id);
    }

    let total_time = start_time.elapsed();

    // Benchmark: Should complete 20 concurrent operations in under 10 seconds
    assert!(total_time < Duration::from_secs(10));
    assert_eq!(completed_operations.len(), num_concurrent_operations);

    // Verify all operations completed
    completed_operations.sort();
    for i in 0..num_concurrent_operations {
        assert!(completed_operations.contains(&i));
    }
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    // Test system behavior under memory pressure
    let agent_config = AgentConfig {
        name: "memory_pressure_agent".to_string(),
        agent_type: "pressure_test".to_string(),
        max_memory: 2048, // Limited memory
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    let pressure_start = Instant::now();
    let mut successful_stores = 0;
    let mut failed_stores = 0;

    // Try to store more data than the limit allows
    for i in 0..500 {
        let large_data = serde_json::json!({
            "id": i,
            "data": "x".repeat(5000), // 5KB each
            "additional_info": {
                "timestamp": i,
                "category": format!("category_{i % 20}"),
                "metadata": "y".repeat(1000)
            }
        });

        let result = agent
            .memory()
            .store(
                MemoryType::Working,
                format!("pressure_memory_{i}"),
                large_data,
            )
            .await;

        match result {
            Ok(_) => successful_stores += 1,
            Err(_) => failed_stores += 1,
        }
    }

    let pressure_time = pressure_start.elapsed();

    // Benchmark: Should handle memory pressure in reasonable time
    assert!(pressure_time < Duration::from_secs(5));

    // Should have some successes and some failures due to memory limits
    assert!(successful_stores > 0);
    assert!(failed_stores > 0);

    // System should remain responsive
    let test_goal = Goal::new("Test goal under pressure".to_string(), GoalPriority::Low);
    let goal_result = agent.add_goal(test_goal).await;
    assert!(goal_result.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_large_scale_integration_workflow() {
    // Test complete workflow with many components
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 5000,
        max_edges: 25000,
        ..Default::default()
    };
    let mut knowledge_graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    let workflow_start = Instant::now();

    // Stage 1: Create multiple agents
    let mut agents: Vec<Agent> = Vec::new();
    for i in 0..25 {
        let agent_config = AgentConfig {
            name: format!("workflow_agent_{i}"),
            agent_type: "large_scale_workflow".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"workflow_id": i}),
        };

        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();
        agents.push(agent);
    }

    // Stage 2: Process goals through synthesis
    let goals = [
        "Develop machine learning pipeline",
        "Optimize data processing algorithms",
        "Create distributed computing framework",
        "Build real-time analytics system",
        "Design scalable microservices architecture",
    ];

    let mut synthesis_results = Vec::new();
    for (i, goal_text) in goals.iter().enumerate() {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::Normal);
        let agent = &agents[i % agents.len()];
        agent.add_goal(goal.clone()).await.unwrap();

        let synthesis_result = synthesis_pipeline.process_goal(&goal).await.unwrap();
        synthesis_results.push(synthesis_result);
    }

    // Stage 3: Store memories and sync to knowledge graph
    for (i, agent) in agents.iter().enumerate() {
        for j in 0..20 {
            agent
                .memory()
                .store(
                    MemoryType::Semantic,
                    format!("workflow_memory_{}_{i, j}"),
                    serde_json::json!({
                        "agent_id": i,
                        "memory_id": j,
                        "content": format!("Workflow knowledge {} from agent {j, i}"),
                        "category": format!("category_{j % 5}")
                    }),
                )
                .await
                .unwrap();
        }
    }

    let memory_integration = knowledge_graph.get_memory_integration().unwrap();
    for agent in &agents {
        memory_integration
            .sync_agent_memory(agent, &mut knowledge_graph)
            .await
            .unwrap();
    }

    // Stage 4: Query and validate
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(exorust_knowledge_graph::NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: Some(100),
        offset: None,
        use_gpu: false,
    };

    let query_result = query_engine.execute(&knowledge_graph, query).await.unwrap();
    assert!(!query_result.nodes.is_empty());

    let workflow_time = workflow_start.elapsed();

    // Benchmark: Complete large-scale workflow should finish in reasonable time
    assert!(workflow_time < Duration::from_secs(15));

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}
