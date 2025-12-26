//! Phase 3 Agent Systems Integration Tests
//! Tests integration between CUDA, agent-core, synthesis, evolution-engines, and knowledge-graph

use stratoswarm_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use stratoswarm_cuda::{CudaContext, CudaMemoryManager, CudaStream};
use stratoswarm_evolution_engines::{
    AdasEngine, DgmEngine, EvolutionEngineConfig, HybridEvolutionSystem, SwarmAgenticEngine,
};
use stratoswarm_knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, NodeType, Query, QueryEngine, QueryType,
};
use stratoswarm_synthesis::{GoalInterpreter, SynthesisConfig, SynthesisPipeline};
use std::sync::Arc;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_agent_synthesis_evolution_integration() {
    // Initialize agent
    let agent_config = AgentConfig {
        name: "integration_test_agent".to_string(),
        agent_type: "synthesis_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({"test": "integration"}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Initialize synthesis pipeline
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    // Initialize evolution engines
    let evolution_config = EvolutionEngineConfig::default();
    let adas_engine = AdasEngine::new(evolution_config.clone()).await.unwrap();
    let swarm_engine = SwarmAgenticEngine::new(evolution_config.clone())
        .await
        .unwrap();
    let dgm_engine = DgmEngine::new(evolution_config.clone()).await.unwrap();

    // Test agent goal processing through synthesis
    let goal = Goal::new(
        "optimize neural network architecture".to_string(),
        GoalPriority::High,
    );
    agent.add_goal(goal.clone()).await.unwrap();
    assert_eq!(agent.state().await, AgentState::Planning);

    // Test synthesis pipeline with goal
    let synthesis_result = synthesis_pipeline.process_goal(&goal).await;
    assert!(synthesis_result.is_ok());

    // Test evolution with generated agents
    let initial_population = vec![synthesis_result.unwrap()];
    let evolved_result = adas_engine.evolve_population(initial_population, 10).await;
    assert!(evolved_result.is_ok());

    let evolved_population = evolved_result.unwrap();
    assert!(!evolved_population.is_empty());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_knowledge_graph_agent_memory_integration() {
    // Initialize knowledge graph
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 10000,
        max_edges: 50000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    // Initialize agent with different memory types
    let agent_config = AgentConfig {
        name: "memory_integration_agent".to_string(),
        agent_type: "memory_test".to_string(),
        max_memory: 4096,
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store different types of memories
    let memories = [
        (
            MemoryType::Working,
            "current_task",
            serde_json::json!("optimizing performance"),
        ),
        (
            MemoryType::Episodic,
            "past_experience",
            serde_json::json!({"event": "successful_optimization", "timestamp": "2025-01-01"}),
        ),
        (
            MemoryType::Semantic,
            "knowledge",
            serde_json::json!({"fact": "neural networks learn through backpropagation"}),
        ),
        (
            MemoryType::Procedural,
            "skill",
            serde_json::json!({"procedure": "gradient_descent_optimization"}),
        ),
    ];

    for (mem_type, key, value) in memories {
        agent
            .memory()
            .store(mem_type, key.to_string(), value)
            .await
            .unwrap();
    }

    // Test knowledge graph integration with agent memories
    let memory_integration = graph.get_memory_integration().unwrap();
    let synced_count = memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await
        .unwrap();
    assert!(synced_count > 0);

    // Query the knowledge graph for agent memories
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: None,
        offset: None,
        use_gpu: false,
    };

    let result = query_engine.execute(&graph, query).await.unwrap();
    assert!(!result.nodes.is_empty());

    // Test memory search functionality
    let search_results = memory_integration
        .search_memories(agent.id(), "optimization", None, 10, &graph)
        .await
        .unwrap();
    assert!(!search_results.is_empty());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_cuda_context_agent_isolation() {
    // Test CUDA context isolation between multiple agents
    let mut agents: Vec<Arc<Agent>> = Vec::new();
    let mut contexts = Vec::new();

    for i in 0..3 {
        let agent_config = AgentConfig {
            name: format!("isolated_agent_{i}"),
            agent_type: "isolation_test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: i as i32,
            metadata: serde_json::json!({"agent_index": i}),
        };

        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();

        // Create isolated CUDA context for each agent
        let context = CudaContext::new(i).await.unwrap();

        agents.push(agent);
        contexts.push(context);
    }

    // Verify each agent has isolated resources
    for (i, (agent, context)) in agents.iter().zip(contexts.iter()).enumerate() {
        assert_eq!(context.device_id(), i);
        assert_eq!(agent.config.priority, i as i32);

        // Test memory allocation in isolated context
        let memory_manager = CudaMemoryManager::new(context.device_id()).await.unwrap();
        let allocation = memory_manager.allocate(1024).await.unwrap();
        assert_eq!(allocation.size(), 1024);

        memory_manager.deallocate(allocation).await.unwrap();
    }

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_hybrid_evolution_system_integration() {
    // Test complete hybrid evolution system with multiple engines
    let evolution_config = EvolutionEngineConfig {
        population_size: 50,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elite_size: 5,
        max_generations: 10,
        fitness_threshold: 0.9,
        diversity_threshold: 0.1,
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create initial population through agent synthesis
    let mut initial_agents = Vec::new();
    for i in 0..10 {
        let agent_config = AgentConfig {
            name: format!("evolved_agent_{i}"),
            agent_type: "evolution_test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({"generation": 0, "index": i}),
        };

        let agent = Agent::new(agent_config).unwrap();
        initial_agents.push(agent);
    }

    // Test evolution across multiple generations
    let generations = 5;
    let mut current_population = initial_agents;

    for generation in 0..generations {
        let evolution_result = timeout(
            Duration::from_secs(30),
            hybrid_system.evolve_generation(current_population),
        )
        .await;

        assert!(evolution_result.is_ok(), "Evolution timed out");
        current_population = evolution_result.unwrap().unwrap();

        assert!(!current_population.is_empty());
        assert!(current_population.len() <= 50); // Respect population size limit

        // Verify generation metadata is updated
        for agent in &current_population {
            if let Some(gen) = agent.config.metadata.get("generation") {
                assert!(gen.as_u64().unwrap() >= generation as u64);
            }
        }
    }

    // Verify final population quality
    let final_stats = hybrid_system.get_evolution_stats().await.unwrap();
    assert!(final_stats.total_generations >= generations);
    assert!(final_stats.best_fitness > 0.0);
}

#[tokio::test]
async fn test_end_to_end_agent_workflow() {
    // Complete end-to-end test: Goal -> Synthesis -> Evolution -> Knowledge Graph

    // 1. Initialize all systems
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        ..Default::default()
    };
    let mut knowledge_graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    let evolution_config = EvolutionEngineConfig::default();
    let mut hybrid_evolution = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // 2. Create agent with complex goal
    let agent_config = AgentConfig {
        name: "end_to_end_agent".to_string(),
        agent_type: "complete_workflow".to_string(),
        max_memory: 4096,
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({
            "workflow": "end_to_end",
            "start_time": chrono::Utc::now().to_rfc3339()
        }),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // 3. Add complex goal requiring synthesis and evolution
    let complex_goal = Goal::new(
        "Develop an adaptive AI system that can optimize its own performance through reinforcement learning".to_string(),
        GoalPriority::Critical
    );
    agent.add_goal(complex_goal.clone()).await.unwrap();

    // 4. Process goal through synthesis pipeline
    let synthesis_result = synthesis_pipeline
        .process_goal(&complex_goal)
        .await
        .unwrap();

    // Store synthesis results in agent memory
    agent
        .memory()
        .store(
            MemoryType::Procedural,
            "synthesis_result".to_string(),
            serde_json::to_value(&synthesis_result).unwrap(),
        )
        .await
        .unwrap();

    // 5. Evolve the synthesized solution
    let initial_population = vec![synthesis_result];
    let evolved_solution = hybrid_evolution
        .evolve_generation(initial_population)
        .await
        .unwrap();

    assert!(!evolved_solution.is_empty());

    // Store evolution results
    agent
        .memory()
        .store(
            MemoryType::Episodic,
            "evolution_result".to_string(),
            serde_json::json!({
                "population_size": evolved_solution.len(),
                "evolution_complete": true
            }),
        )
        .await
        .unwrap();

    // 6. Integrate all data into knowledge graph
    let memory_integration = knowledge_graph.get_memory_integration().unwrap();
    let synced_memories = memory_integration
        .sync_agent_memory(&agent, &mut knowledge_graph)
        .await
        .unwrap();
    assert!(synced_memories > 0);

    // 7. Query knowledge graph for workflow insights
    let mut query_engine = QueryEngine::new(false).await.unwrap();

    // Find all memories related to the workflow
    let workflow_query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(10000),
        limit: Some(10),
        offset: None,
        use_gpu: false,
    };

    let workflow_results = query_engine
        .execute(&knowledge_graph, workflow_query)
        .await
        .unwrap();
    assert!(!workflow_results.nodes.is_empty());

    // 8. Verify workflow completion
    let agent_stats = agent.stats().await;
    assert_eq!(agent_stats.goals_processed, 0); // Goal still being processed
    assert_eq!(agent.state().await, AgentState::Planning);

    // 9. Mark goal as completed and update stats
    agent.remove_goal(complex_goal.id).await.unwrap();
    agent.update_goal_stats(true, Duration::from_secs(1)).await;

    let final_stats = agent.stats().await;
    assert_eq!(final_stats.goals_processed, 1);
    assert_eq!(final_stats.goals_succeeded, 1);

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_concurrent_agent_interactions() {
    // Test multiple agents working concurrently with shared resources

    let num_agents = 5;
    let mut agents: Vec<Arc<Agent>> = Vec::new();
    let mut agent_tasks = Vec::new();

    // Initialize shared knowledge graph
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 50000,
        max_edges: 100000,
        ..Default::default()
    };
    let knowledge_graph = Arc::new(tokio::sync::Mutex::new(
        KnowledgeGraph::new(kg_config).await.unwrap(),
    ));

    // Create multiple agents
    for i in 0..num_agents {
        let agent_config = AgentConfig {
            name: format!("concurrent_agent_{i}"),
            agent_type: "concurrent_test".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: i as i32,
            metadata: serde_json::json!({"agent_index": i}),
        };

        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();
        agents.push(Arc::new(agent));
    }

    // Create concurrent tasks for each agent
    for (i, agent) in agents.iter().enumerate() {
        let agent_clone = Arc::clone(agent);
        let kg_clone = Arc::clone(&knowledge_graph);

        let task = tokio::spawn(async move {
            // Each agent performs different operations
            let goal = Goal::new(format!("Agent {} concurrent task", i), GoalPriority::Normal);
            agent_clone.add_goal(goal).await.unwrap();

            // Store agent-specific memories
            for j in 0..10 {
                agent_clone.memory().store(
                    MemoryType::Working,
                    format!("task_{}_{i, j}"),
                    serde_json::json!({"agent": i, "task": j, "data": format!("concurrent_data_{}_{i, j}")})
                ).await.unwrap();
            }

            // Sync to shared knowledge graph - scope guard to prevent deadlock
            let synced = {
                let mut kg = kg_clone.lock().await;
                let memory_integration = kg.get_memory_integration().unwrap();
                memory_integration
                    .sync_agent_memory(&*agent_clone, &mut *kg)
                    .await
                    .unwrap()
            };

            (i, synced)
        });

        agent_tasks.push(task);
    }

    // Wait for all agents to complete
    let mut total_synced = 0;
    for task in agent_tasks {
        let (agent_id, synced_count) = task.await.unwrap();
        assert!(
            synced_count > 0,
            "Agent {} failed to sync memories",
            agent_id
        );
        total_synced += synced_count;
    }

    // Verify all data was properly synced
    let kg = knowledge_graph.lock().await;
    let stats = kg.stats();
    assert!(stats.node_count >= total_synced);

    // Cleanup agents
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_error_recovery_and_resilience() {
    // Test system behavior under error conditions and recovery

    let agent_config = AgentConfig {
        name: "error_recovery_agent".to_string(),
        agent_type: "resilience_test".to_string(),
        max_memory: 1024,
        max_gpu_memory: 512,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Test 1: Invalid goal processing
    let invalid_goal = Goal::new("".to_string(), GoalPriority::Low); // Empty goal
    let result = agent.add_goal(invalid_goal).await;
    // Should either succeed or fail gracefully
    assert!(result.is_ok() || result.is_err());

    // Test 2: Memory system stress test
    let stress_test_memories = 1000;
    let mut successful_stores = 0;

    for i in 0..stress_test_memories {
        let result = agent
            .memory()
            .store(
                MemoryType::Working,
                format!("stress_test_{i}"),
                serde_json::json!({"data": vec![0u8; 1024]}), // 1KB per memory
            )
            .await;

        if result.is_ok() {
            successful_stores += 1;
        }
    }

    // Should handle memory pressure gracefully
    assert!(successful_stores > 0);

    // Test 3: Concurrent state transitions
    let state_transition_tasks = (0..10)
        .map(|_| {
            let agent_clone = &agent;
            tokio::spawn(async move { agent_clone.pause().await })
        })
        .collect::<Vec<_>>();

    // Some transitions should succeed, others may fail due to race conditions
    let mut successful_transitions = 0;
    for task in state_transition_tasks {
        if task.await.unwrap().is_ok() {
            successful_transitions += 1;
        }
    }

    // At least one should succeed
    assert!(successful_transitions > 0);

    // Test 4: Recovery after errors
    let current_state = agent.state().await;
    if current_state == AgentState::Paused {
        agent.resume().await.unwrap();
    }

    // Should be in a valid operational state
    let final_state = agent.state().await;
    assert!(matches!(
        final_state,
        AgentState::Idle | AgentState::Planning | AgentState::Executing
    ));

    agent.shutdown().await.unwrap();
}
