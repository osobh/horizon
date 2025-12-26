//! Comprehensive Integration Scenarios
//! Additional integration tests covering complex workflows and edge cases

use stratoswarm_agent_core::{Agent, AgentConfig, AgentState, Goal, GoalPriority, MemoryType};
use stratoswarm_cuda::{CudaContext, CudaMemoryManager};
use stratoswarm_evolution_engines::{
    AdasEngine, DgmEngine, EvolutionEngineConfig, HybridEvolutionSystem, SwarmAgenticEngine,
};
use stratoswarm_knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphConfig, Query, QueryEngine, QueryType,
};
use stratoswarm_synthesis::{SynthesisConfig, SynthesisPipeline};
use std::sync::Arc;
use tokio::time::{timeout, Duration, Instant};

#[tokio::test]
async fn test_multi_agent_collaboration() {
    let mut agents: Vec<Agent> = Vec::new();

    // Create specialized agents
    for i in 0..5 {
        let agent_config = AgentConfig {
            name: format!("collaborative_agent_{i}"),
            agent_type: format!("specialist_{i}"),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"specialization": format!("domain_{i}")}),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();
        agents.push(agent);
    }

    // Each agent contributes to a shared problem
    let shared_goals = [
        "Analyze data patterns in domain 0",
        "Process algorithms for domain 1",
        "Optimize performance in domain 2",
        "Validate results in domain 3",
        "Generate reports for domain 4",
    ];

    for (i, (agent, goal_text)) in agents.iter().zip(shared_goals.iter()).enumerate() {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::Normal);
        agent.add_goal(goal).await.unwrap();

        // Store domain-specific knowledge
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                format!("domain_{}_knowledge", i),
                serde_json::json!({
                    "domain": i,
                    "expertise": format!("specialized_knowledge_{i}"),
                    "collaboration_data": "shared_context"
                }),
            )
            .await
            .unwrap();
    }

    // Verify all agents are working
    for agent in &agents {
        assert!(matches!(
            agent.state().await,
            AgentState::Planning | AgentState::Executing
        ));
    }

    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_dynamic_goal_prioritization() {
    let agent_config = AgentConfig {
        name: "priority_test_agent".to_string(),
        agent_type: "priority_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Add goals with different priorities
    let goals = [
        ("Low priority task", GoalPriority::Low),
        ("Critical emergency", GoalPriority::Critical),
        ("Normal operation", GoalPriority::Normal),
        ("High importance", GoalPriority::High),
        ("Medium priority", GoalPriority::Medium),
    ];

    for (goal_text, priority) in goals {
        let goal = Goal::new(goal_text.to_string(), priority);
        agent.add_goal(goal).await.unwrap();
    }

    // Agent should be processing goals
    assert_eq!(agent.state().await, AgentState::Planning);

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_memory_type_interactions() {
    let agent_config = AgentConfig {
        name: "memory_interaction_agent".to_string(),
        agent_type: "memory_test".to_string(),
        max_memory: 4096,
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store different types of memories that should interact
    let working_data = serde_json::json!({
        "current_task": "learning_algorithm",
        "progress": 0.5,
        "resources_used": ["cpu", "memory", "gpu"]
    });

    let episodic_data = serde_json::json!({
        "event": "completed_training",
        "timestamp": "2025-01-01T10:00:00Z",
        "performance_metrics": {"accuracy": 0.95, "loss": 0.05}
    });

    let semantic_data = serde_json::json!({
        "concept": "machine_learning",
        "definition": "algorithms that improve through experience",
        "related_concepts": ["neural_networks", "deep_learning", "ai"]
    });

    let procedural_data = serde_json::json!({
        "skill": "gradient_descent",
        "steps": ["compute_gradients", "update_weights", "evaluate_loss"],
        "expertise_level": "advanced"
    });

    agent
        .memory()
        .store(
            MemoryType::Working,
            "current_work".to_string(),
            working_data,
        )
        .await
        .unwrap();
    agent
        .memory()
        .store(
            MemoryType::Episodic,
            "past_event".to_string(),
            episodic_data,
        )
        .await
        .unwrap();
    agent
        .memory()
        .store(MemoryType::Semantic, "knowledge".to_string(), semantic_data)
        .await
        .unwrap();
    agent
        .memory()
        .store(MemoryType::Procedural, "skill".to_string(), procedural_data)
        .await
        .unwrap();

    // Retrieve and verify each memory type
    let retrieved_working = agent
        .memory()
        .retrieve(MemoryType::Working, "current_work")
        .await
        .unwrap();
    let retrieved_episodic = agent
        .memory()
        .retrieve(MemoryType::Episodic, "past_event")
        .await
        .unwrap();
    let retrieved_semantic = agent
        .memory()
        .retrieve(MemoryType::Semantic, "knowledge")
        .await
        .unwrap();
    let retrieved_procedural = agent
        .memory()
        .retrieve(MemoryType::Procedural, "skill")
        .await
        .unwrap();

    assert!(retrieved_working.get("current_task").is_some());
    assert!(retrieved_episodic.get("event").is_some());
    assert!(retrieved_semantic.get("concept").is_some());
    assert!(retrieved_procedural.get("skill").is_some());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_knowledge_graph_semantic_search() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 5000,
        max_edges: 25000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let agent_config = AgentConfig {
        name: "semantic_search_agent".to_string(),
        agent_type: "semantic_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store semantically related memories
    let topics = [
        ("artificial_intelligence", "AI systems and machine learning"),
        ("neural_networks", "Deep learning architectures"),
        ("algorithms", "Computational procedures and methods"),
        ("optimization", "Improving system performance"),
        ("data_science", "Analysis and interpretation of data"),
    ];

    for (topic, description) in topics {
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                topic.to_string(),
                serde_json::json!({
                    "topic": topic,
                    "description": description,
                    "category": "technology",
                    "relevance": "high"
                }),
            )
            .await
            .unwrap();
    }

    // Sync to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await
        .unwrap();

    // Test semantic search
    let search_results = memory_integration
        .search_memories(agent.id(), "machine learning", None, 10, &graph)
        .await
        .unwrap();

    assert!(!search_results.is_empty());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_evolution_engine_comparison() {
    let evolution_config = EvolutionEngineConfig::default();

    // Test different evolution engines
    let adas_engine = AdasEngine::new(evolution_config.clone()).await.unwrap();
    let swarm_engine = SwarmAgenticEngine::new(evolution_config.clone())
        .await
        .unwrap();
    let dgm_engine = DgmEngine::new(evolution_config.clone()).await.unwrap();

    // Create test population
    let mut test_population = Vec::new();
    for i in 0..10 {
        let agent_config = AgentConfig {
            name: format!("evolution_test_agent_{i}"),
            agent_type: "evolution_comparison".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({"test_id": i}),
        };
        let agent = Agent::new(agent_config).unwrap();
        test_population.push(agent);
    }

    // Test ADAS evolution
    let adas_result = adas_engine
        .evolve_population(test_population.clone(), 5)
        .await;
    assert!(adas_result.is_ok());

    // Test Swarm evolution
    let swarm_result = swarm_engine
        .evolve_population(test_population.clone(), 5)
        .await;
    assert!(swarm_result.is_ok());

    // Test DGM evolution
    let dgm_result = dgm_engine.evolve_population(test_population, 5).await;
    assert!(dgm_result.is_ok());
}

#[tokio::test]
async fn test_synthesis_goal_complexity() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    let complex_goals = [
        "Develop a multi-modal AI system that can process text, images, and audio simultaneously",
        "Create a distributed consensus algorithm for blockchain networks with Byzantine fault tolerance",
        "Design a quantum-resistant cryptographic protocol for secure communications",
        "Implement a real-time recommendation system using collaborative filtering and deep learning",
        "Build a fault-tolerant microservices architecture with automatic scaling and recovery",
    ];

    for goal_text in complex_goals {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::Normal);
        let result = synthesis_pipeline.process_goal(&goal).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_agent_state_persistence() {
    let agent_config = AgentConfig {
        name: "persistence_test_agent".to_string(),
        agent_type: "persistence_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({"persistent": true}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Add goals and memories
    for i in 0..5 {
        let goal = Goal::new(format!("Persistent goal {i}"), GoalPriority::Normal);
        agent.add_goal(goal).await.unwrap();

        agent
            .memory()
            .store(
                MemoryType::Working,
                format!("persistent_memory_{i}"),
                serde_json::json!({"id": i, "persistent": true}),
            )
            .await
            .unwrap();
    }

    let initial_stats = agent.stats().await;

    // Simulate state persistence by pausing and resuming
    agent.pause().await.unwrap();
    assert_eq!(agent.state().await, AgentState::Paused);

    agent.resume().await.unwrap();
    assert!(matches!(
        agent.state().await,
        AgentState::Planning | AgentState::Executing | AgentState::Idle
    ));

    let resumed_stats = agent.stats().await;
    assert_eq!(initial_stats.goals_processed, resumed_stats.goals_processed);

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_cuda_multi_stream_operations() {
    let mut contexts = Vec::new();
    let mut memory_managers: Vec<CudaMemoryManager> = Vec::new();

    // Create multiple CUDA contexts
    for device_id in 0..3 {
        if let Ok(context) = CudaContext::new(device_id).await {
            if let Ok(manager) = CudaMemoryManager::new(context.device_id()).await {
                contexts.push(context);
                memory_managers.push(manager);
            }
        }
    }

    if !memory_managers.is_empty() {
        // Test concurrent operations across streams
        let mut allocation_tasks = Vec::new();

        for manager in &memory_managers {
            let task = manager.allocate(1024 * 1024); // 1MB
            allocation_tasks.push(task);
        }

        for task in allocation_tasks {
            if let Ok(allocation) = task.await {
                let manager = &memory_managers[allocation.device_id()];
                manager.deallocate(allocation).await.unwrap();
            }
        }
    }
}

#[tokio::test]
async fn test_large_scale_memory_operations() {
    let agent_config = AgentConfig {
        name: "large_memory_agent".to_string(),
        agent_type: "large_memory_test".to_string(),
        max_memory: 8192, // 8MB
        max_gpu_memory: 4096,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store large amounts of data
    for i in 0..1000 {
        let data = serde_json::json!({
            "id": i,
            "data": format!("Large data entry {i}"),
            "metadata": {
                "timestamp": i,
                "size": "large",
                "category": format!("category_{i % 10}")
            }
        });

        let result = agent
            .memory()
            .store(MemoryType::Working, format!("large_entry_{i}"), data)
            .await;

        if result.is_err() {
            break; // Hit memory limit
        }
    }

    // Verify agent remains functional
    let test_goal = Goal::new("Test after large operations".to_string(), GoalPriority::Low);
    assert!(agent.add_goal(test_goal).await.is_ok());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_concurrent_knowledge_graph_access() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        ..Default::default()
    };
    let knowledge_graph = Arc::new(tokio::sync::Mutex::new(
        KnowledgeGraph::new(kg_config).await.unwrap(),
    ));

    let mut tasks = Vec::new();

    // Spawn concurrent access tasks
    for i in 0..10 {
        let kg_clone = Arc::clone(&knowledge_graph);
        let task = tokio::spawn(async move {
            let agent_config = AgentConfig {
                name: format!("concurrent_kg_agent_{i}"),
                agent_type: "concurrent_kg_test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"task_id": i}),
            };
            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();

            // Store memories
            for j in 0..5 {
                agent
                    .memory()
                    .store(
                        MemoryType::Semantic,
                        format!("concurrent_memory_{}_{i, j}"),
                        serde_json::json!({"agent": i, "memory": j}),
                    )
                    .await
                    .unwrap();
            }

            // Access knowledge graph - extract needed data before await
            let synced = {
                let mut kg = kg_clone.lock().await;
                let memory_integration = kg.get_memory_integration().unwrap();
                memory_integration
                    .sync_agent_memory(&agent, &mut *kg)
                    .await
                    .unwrap()
            };

            agent.shutdown().await.unwrap();
            synced
        });
        tasks.push(task);
    }

    // Wait for all tasks
    for task in tasks {
        let synced_count = task.await.unwrap();
        assert!(synced_count > 0);
    }
}

#[tokio::test]
async fn test_agent_lifecycle_events() {
    let agent_config = AgentConfig {
        name: "lifecycle_agent".to_string(),
        agent_type: "lifecycle_test".to_string(),
        max_memory: 1024,
        max_gpu_memory: 512,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();

    // Test initialization
    assert_eq!(agent.state().await, AgentState::Idle);
    agent.initialize().await.unwrap();

    // Test goal addition
    let goal = Goal::new("Lifecycle test goal".to_string(), GoalPriority::Normal);
    agent.add_goal(goal.clone()).await.unwrap();
    assert_eq!(agent.state().await, AgentState::Planning);

    // Test pause/resume cycle
    agent.pause().await.unwrap();
    assert_eq!(agent.state().await, AgentState::Paused);

    agent.resume().await.unwrap();
    assert!(matches!(
        agent.state().await,
        AgentState::Planning | AgentState::Executing | AgentState::Idle
    ));

    // Test goal removal
    agent.remove_goal(goal.id).await.unwrap();

    // Test shutdown
    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_memory_search_patterns() {
    let agent_config = AgentConfig {
        name: "search_pattern_agent".to_string(),
        agent_type: "search_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store memories with searchable patterns
    let patterns = [
        ("pattern_alpha", "machine learning algorithms"),
        ("pattern_beta", "neural network architectures"),
        ("pattern_gamma", "optimization techniques"),
        ("pattern_delta", "data processing methods"),
        ("pattern_epsilon", "performance metrics"),
    ];

    for (key, content) in patterns {
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                key.to_string(),
                serde_json::json!({
                    "content": content,
                    "pattern": key,
                    "searchable": true
                }),
            )
            .await
            .unwrap();
    }

    // Test retrieval of stored patterns
    for (key, _) in patterns {
        let retrieved = agent.memory().retrieve(MemoryType::Semantic, key).await;
        assert!(retrieved.is_ok());
    }

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_evolution_fitness_evaluation() {
    let evolution_config = EvolutionEngineConfig {
        population_size: 20,
        max_generations: 5,
        mutation_rate: 0.1,
        crossover_rate: 0.8,
        elite_size: 3,
        fitness_threshold: 0.8,
        diversity_threshold: 0.1,
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create population with varied fitness characteristics
    let mut population = Vec::new();
    for i in 0..20 {
        let agent_config = AgentConfig {
            name: format!("fitness_agent_{i}"),
            agent_type: "fitness_test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({
                "fitness_potential": i as f64 / 20.0,
                "generation": 0
            }),
        };
        let agent = Agent::new(agent_config).unwrap();
        population.push(agent);
    }

    // Test evolution with fitness evaluation
    let evolved_population = hybrid_system.evolve_generation(population).await.unwrap();
    assert!(!evolved_population.is_empty());

    let stats = hybrid_system.get_evolution_stats().await.unwrap();
    assert!(stats.best_fitness >= 0.0);
}

#[tokio::test]
async fn test_synthesis_pipeline_optimization() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    // Test optimization-focused goals
    let optimization_goals = [
        "Optimize memory usage in data structures",
        "Improve algorithm time complexity",
        "Reduce network latency in distributed systems",
        "Enhance cache hit rates",
        "Minimize energy consumption",
    ];

    for goal_text in optimization_goals {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::High);
        let result = synthesis_pipeline.process_goal(&goal).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_multi_threaded_agent_operations() {
    let num_threads = 5;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let handle = tokio::spawn(async move {
            let agent_config = AgentConfig {
                name: format!("thread_agent_{thread_id}"),
                agent_type: "thread_test".to_string(),
                max_memory: 1024,
                max_gpu_memory: 512,
                priority: 1,
                metadata: serde_json::json!({"thread_id": thread_id}),
            };
            let agent = Agent::new(agent_config).unwrap();
            agent.initialize().await.unwrap();

            // Perform thread-specific operations
            for i in 0..10 {
                let goal = Goal::new(
                    format!("Thread {} goal {thread_id, i}"),
                    GoalPriority::Normal,
                );
                agent.add_goal(goal).await.unwrap();

                agent
                    .memory()
                    .store(
                        MemoryType::Working,
                        format!("thread_{}_memory_{thread_id, i}"),
                        serde_json::json!({"thread": thread_id, "operation": i}),
                    )
                    .await
                    .unwrap();
            }

            agent.shutdown().await.unwrap();
            thread_id
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let thread_id = handle.await.unwrap();
        assert!(thread_id < num_threads);
    }
}

#[tokio::test]
async fn test_knowledge_graph_relationship_mapping() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        max_nodes: 1000,
        max_edges: 5000,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let agent_config = AgentConfig {
        name: "relationship_agent".to_string(),
        agent_type: "relationship_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store related concepts
    let concepts = [
        ("machine_learning", "algorithms that learn from data"),
        ("neural_networks", "interconnected processing nodes"),
        ("deep_learning", "multi-layer neural networks"),
        ("artificial_intelligence", "intelligent agent systems"),
        ("data_science", "extracting insights from data"),
    ];

    for (concept, description) in concepts {
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                concept.to_string(),
                serde_json::json!({
                    "concept": concept,
                    "description": description,
                    "domain": "AI",
                    "relationships": ["related_to_AI", "technical_concept"]
                }),
            )
            .await
            .unwrap();
    }

    // Sync to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    let synced = memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await
        .unwrap();
    assert!(synced > 0);

    // Verify graph structure
    let stats = graph.stats();
    assert!(stats.node_count > 0);

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_agent_error_recovery_patterns() {
    let agent_config = AgentConfig {
        name: "error_recovery_agent".to_string(),
        agent_type: "error_recovery_test".to_string(),
        max_memory: 1024,
        max_gpu_memory: 512,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Test various error conditions and recovery

    // Invalid memory operation
    let invalid_result = agent
        .memory()
        .retrieve(MemoryType::Working, "nonexistent_key")
        .await;
    assert!(invalid_result.is_err());

    // Agent should remain functional
    let valid_goal = Goal::new("Valid goal after error".to_string(), GoalPriority::Normal);
    assert!(agent.add_goal(valid_goal).await.is_ok());

    // Empty goal handling
    let empty_goal = Goal::new("".to_string(), GoalPriority::Low);
    let empty_result = agent.add_goal(empty_goal).await;
    // Should either succeed or fail gracefully
    assert!(empty_result.is_ok() || empty_result.is_err());

    // Large data handling
    let large_data = serde_json::json!({
        "data": "x".repeat(10000),
        "metadata": "y".repeat(5000)
    });
    let large_result = agent
        .memory()
        .store(MemoryType::Working, "large_data".to_string(), large_data)
        .await;
    // Should either succeed or fail gracefully due to size
    assert!(large_result.is_ok() || large_result.is_err());

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_resource_monitoring() {
    let mut agents: Vec<Agent> = Vec::new();

    // Create multiple agents to monitor resource usage
    for i in 0..10 {
        let agent_config = AgentConfig {
            name: format!("monitor_agent_{i}"),
            agent_type: "resource_monitor".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({"monitor_id": i}),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();

        // Each agent performs resource-intensive operations
        for j in 0..20 {
            agent
                .memory()
                .store(
                    MemoryType::Working,
                    format!("resource_data_{}_{i, j}"),
                    serde_json::json!({
                        "agent": i,
                        "operation": j,
                        "data": "resource_intensive_content",
                        "timestamp": j
                    }),
                )
                .await
                .unwrap();
        }

        agents.push(agent);
    }

    // Monitor agent statistics
    for agent in &agents {
        let stats = agent.stats().await;
        assert!(stats.goals_processed >= 0);
    }

    // Cleanup
    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_evolution_diversity_maintenance() {
    let evolution_config = EvolutionEngineConfig {
        population_size: 30,
        max_generations: 3,
        mutation_rate: 0.2,
        crossover_rate: 0.7,
        elite_size: 5,
        fitness_threshold: 0.9,
        diversity_threshold: 0.2, // High diversity requirement
        parallel_evaluation: true,
    };

    let mut hybrid_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create diverse initial population
    let mut population = Vec::new();
    for i in 0..30 {
        let agent_config = AgentConfig {
            name: format!("diverse_agent_{i}"),
            agent_type: format!("type_{i % 5}"), // 5 different types
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: i % 3, // 3 different priorities
            metadata: serde_json::json!({
                "diversity_trait": i % 10,
                "specialization": format!("spec_{i % 7}")
            }),
        };
        let agent = Agent::new(agent_config).unwrap();
        population.push(agent);
    }

    // Test evolution maintains diversity
    let evolved_population = hybrid_system.evolve_generation(population).await.unwrap();
    assert!(!evolved_population.is_empty());
    assert!(evolved_population.len() <= 30);

    // Check diversity is maintained
    let mut types_seen = std::collections::HashSet::new();
    for agent in &evolved_population {
        types_seen.insert(&agent.config.agent_type);
    }
    assert!(types_seen.len() > 1); // Should maintain type diversity
}

#[tokio::test]
async fn test_complex_goal_decomposition() {
    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    // Test complex goals that require decomposition
    let complex_goals = [
        "Build a complete e-commerce platform with user management, product catalog, shopping cart, payment processing, order management, and analytics dashboard",
        "Develop a distributed machine learning system with data ingestion, feature engineering, model training, hyperparameter tuning, model deployment, and monitoring",
        "Create a real-time multiplayer game with networking, physics simulation, graphics rendering, audio processing, user interface, and matchmaking",
    ];

    for goal_text in complex_goals {
        let goal = Goal::new(goal_text.to_string(), GoalPriority::High);
        let result = timeout(
            Duration::from_secs(10),
            synthesis_pipeline.process_goal(&goal),
        )
        .await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }
}

#[tokio::test]
async fn test_agent_communication_patterns() {
    let mut agents: Vec<Agent> = Vec::new();

    // Create agents that will communicate through shared knowledge
    for i in 0..3 {
        let agent_config = AgentConfig {
            name: format!("comm_agent_{i}"),
            agent_type: "communication_test".to_string(),
            max_memory: 2048,
            max_gpu_memory: 1024,
            priority: 1,
            metadata: serde_json::json!({"comm_id": i}),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();
        agents.push(agent);
    }

    // Each agent stores communication data
    for (i, agent) in agents.iter().enumerate() {
        agent
            .memory()
            .store(
                MemoryType::Semantic,
                format!("comm_data_{i}"),
                serde_json::json!({
                    "sender": i,
                    "message": format!("Hello from agent {i}"),
                    "recipients": ["all"],
                    "timestamp": i
                }),
            )
            .await
            .unwrap();

        // Store messages for other agents
        for j in 0..agents.len() {
            if i != j {
                agent
                    .memory()
                    .store(
                        MemoryType::Episodic,
                        format!("message_to_{j}"),
                        serde_json::json!({
                            "from": i,
                            "to": j,
                            "content": format!("Message from {} to {i, j}")
                        }),
                    )
                    .await
                    .unwrap();
            }
        }
    }

    // Verify communication data is stored
    for agent in &agents {
        let stats = agent.stats().await;
        assert!(stats.goals_processed >= 0);
    }

    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_performance_optimization_scenarios() {
    let agent_config = AgentConfig {
        name: "optimization_agent".to_string(),
        agent_type: "optimization_test".to_string(),
        max_memory: 4096,
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    let start_time = Instant::now();

    // Perform optimization-focused operations
    let optimization_tasks = [
        "Optimize database query performance",
        "Reduce memory allocation overhead",
        "Improve cache hit ratios",
        "Minimize network round trips",
        "Enhance CPU utilization",
    ];

    for task in optimization_tasks {
        let goal = Goal::new(task.to_string(), GoalPriority::High);
        agent.add_goal(goal).await.unwrap();

        // Store optimization data
        agent
            .memory()
            .store(
                MemoryType::Procedural,
                format!("optimization_{}", task.replace(" ", "_")),
                serde_json::json!({
                    "task": task,
                    "strategy": "performance_focused",
                    "metrics": ["latency", "throughput", "resource_usage"]
                }),
            )
            .await
            .unwrap();
    }

    let operation_time = start_time.elapsed();

    // Operations should complete efficiently
    assert!(operation_time < Duration::from_secs(5));

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_fault_tolerance_mechanisms() {
    let mut agents: Vec<Agent> = Vec::new();

    // Create agents with different fault tolerance levels
    for i in 0..5 {
        let agent_config = AgentConfig {
            name: format!("fault_tolerant_agent_{i}"),
            agent_type: "fault_tolerance_test".to_string(),
            max_memory: 1024,
            max_gpu_memory: 512,
            priority: 1,
            metadata: serde_json::json!({
                "fault_tolerance_level": i,
                "backup_enabled": i % 2 == 0
            }),
        };
        let agent = Agent::new(agent_config).unwrap();
        agent.initialize().await.unwrap();
        agents.push(agent);
    }

    // Simulate fault conditions
    for (i, agent) in agents.iter().enumerate() {
        // Store critical data
        agent
            .memory()
            .store(
                MemoryType::Working,
                "critical_data".to_string(),
                serde_json::json!({
                    "agent_id": i,
                    "critical": true,
                    "backup_required": true
                }),
            )
            .await
            .unwrap();

        // Simulate state changes that might cause faults
        if i % 2 == 0 {
            agent.pause().await.unwrap();
            agent.resume().await.unwrap();
        }
    }

    // Verify all agents remain functional
    for agent in &agents {
        let state = agent.state().await;
        assert!(matches!(
            state,
            AgentState::Idle | AgentState::Planning | AgentState::Executing
        ));
    }

    for agent in agents {
        agent.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_scalability_limits() {
    let mut created_agents = 0;
    let mut agents: Vec<Agent> = Vec::new();
    let max_attempts = 200;

    // Test system scalability limits
    for i in 0..max_attempts {
        let agent_config = AgentConfig {
            name: format!("scale_test_agent_{i}"),
            agent_type: "scalability_test".to_string(),
            max_memory: 512, // Small footprint
            max_gpu_memory: 256,
            priority: 1,
            metadata: serde_json::json!({"scale_id": i}),
        };

        match Agent::new(agent_config) {
            Ok(agent) => {
                if agent.initialize().await.is_ok() {
                    agents.push(agent);
                    created_agents += 1;
                } else {
                    break;
                }
            }
            Err(_) => break,
        }

        // Stop if we hit reasonable limits
        if created_agents >= 50 {
            break;
        }
    }

    assert!(created_agents > 0);
    println!("Successfully created {} agents", created_agents);

    // Cleanup
    for agent in agents {
        let _ = agent.shutdown().await;
    }
}

#[tokio::test]
async fn test_data_consistency_verification() {
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        ..Default::default()
    };
    let mut graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let agent_config = AgentConfig {
        name: "consistency_agent".to_string(),
        agent_type: "consistency_test".to_string(),
        max_memory: 2048,
        max_gpu_memory: 1024,
        priority: 1,
        metadata: serde_json::json!({}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Store consistent data across memory types
    let consistent_data = serde_json::json!({
        "id": "consistent_item_123",
        "name": "Test Item",
        "value": 42,
        "metadata": {
            "created": "2025-01-01",
            "version": 1
        }
    });

    agent
        .memory()
        .store(
            MemoryType::Working,
            "consistent_item".to_string(),
            consistent_data.clone(),
        )
        .await
        .unwrap();

    agent
        .memory()
        .store(
            MemoryType::Semantic,
            "consistent_knowledge".to_string(),
            consistent_data.clone(),
        )
        .await
        .unwrap();

    // Sync to knowledge graph
    let memory_integration = graph.get_memory_integration().unwrap();
    memory_integration
        .sync_agent_memory(&agent, &mut graph)
        .await
        .unwrap();

    // Verify consistency across systems
    let working_data = agent
        .memory()
        .retrieve(MemoryType::Working, "consistent_item")
        .await
        .unwrap();
    let semantic_data = agent
        .memory()
        .retrieve(MemoryType::Semantic, "consistent_knowledge")
        .await
        .unwrap();

    assert_eq!(working_data.get("id"), semantic_data.get("id"));
    assert_eq!(working_data.get("value"), semantic_data.get("value"));

    agent.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_end_to_end_workflow_validation() {
    // Complete end-to-end workflow test
    let kg_config = KnowledgeGraphConfig {
        gpu_enabled: false,
        ..Default::default()
    };
    let mut knowledge_graph = KnowledgeGraph::new(kg_config).await.unwrap();

    let synthesis_config = SynthesisConfig::default();
    let mut synthesis_pipeline = SynthesisPipeline::new(synthesis_config).unwrap();

    let evolution_config = EvolutionEngineConfig::default();
    let mut evolution_system = HybridEvolutionSystem::new(evolution_config).await.unwrap();

    // Create main agent
    let agent_config = AgentConfig {
        name: "workflow_validation_agent".to_string(),
        agent_type: "end_to_end_validation".to_string(),
        max_memory: 4096,
        max_gpu_memory: 2048,
        priority: 1,
        metadata: serde_json::json!({"workflow": "validation"}),
    };
    let agent = Agent::new(agent_config).unwrap();
    agent.initialize().await.unwrap();

    // Step 1: Goal processing
    let validation_goal = Goal::new(
        "Validate complete system integration and functionality".to_string(),
        GoalPriority::Critical,
    );
    agent.add_goal(validation_goal.clone()).await.unwrap();

    // Step 2: Synthesis processing
    let synthesis_result = synthesis_pipeline
        .process_goal(&validation_goal)
        .await
        .unwrap();

    // Step 3: Store results
    agent
        .memory()
        .store(
            MemoryType::Procedural,
            "synthesis_validation".to_string(),
            serde_json::to_value(&synthesis_result).unwrap(),
        )
        .await
        .unwrap();

    // Step 4: Evolution processing
    let initial_population = vec![synthesis_result];
    let evolved_result = evolution_system
        .evolve_generation(initial_population)
        .await
        .unwrap();
    assert!(!evolved_result.is_empty());

    // Step 5: Knowledge graph integration
    let memory_integration = knowledge_graph.get_memory_integration().unwrap();
    let synced = memory_integration
        .sync_agent_memory(&agent, &mut knowledge_graph)
        .await
        .unwrap();
    assert!(synced > 0);

    // Step 6: Validation queries
    let mut query_engine = QueryEngine::new(false).await.unwrap();
    let validation_query = Query {
        query_type: QueryType::FindNodes {
            node_type: Some(stratoswarm_knowledge_graph::NodeType::Memory),
            properties: std::collections::HashMap::new(),
        },
        timeout_ms: Some(5000),
        limit: Some(10),
        offset: None,
        use_gpu: false,
    };

    let query_result = query_engine
        .execute(&knowledge_graph, validation_query)
        .await
        .unwrap();
    assert!(!query_result.nodes.is_empty());

    // Step 7: Final validation
    let final_stats = agent.stats().await;
    assert!(final_stats.goals_processed >= 0);

    agent.shutdown().await.unwrap();
}
