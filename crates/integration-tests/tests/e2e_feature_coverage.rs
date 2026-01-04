//! Comprehensive end-to-end tests covering all major ExoRust features
//! Validates complete data flow through the system

use std::sync::Arc;
use stratoswarm_agent_core::{Agent, AgentConfig};
use stratoswarm_bootstrap::{BootstrapConfig, BootstrapManager};
use stratoswarm_compliance::{ComplianceFramework, ComplianceManager};
use stratoswarm_cost_optimization::{CostOptimizer, WorkloadScheduler};
use stratoswarm_disaster_recovery::{BackupType, DisasterRecoveryManager};
use stratoswarm_emergency_controls::{
    EmergencyController, KillSwitch, ResourceLimits, SafetyViolationType,
};
use stratoswarm_evolution::{EvolutionEngine, GeneticEvolutionEngine};
use stratoswarm_global_knowledge_graph::{GlobalKnowledgeGraph, QueryRequest};
use stratoswarm_governance::{
    AgentPermissions, CoordinationStrategy, GovernanceEngine, LifecyclePhase, PermissionLevel,
    PolicyManager,
};
use stratoswarm_monitoring::{MetricType, PerformanceMonitor};
use stratoswarm_multi_region::{DataSovereigntyPolicy, MultiRegionController, Region};
use stratoswarm_zero_trust::{DeviceTrust, ZeroTrustController};
use uuid::Uuid;

/// Test complete Bootstrap → Governance → Agent Creation → Safety flow
#[tokio::test]
async fn test_bootstrap_to_agent_creation_flow() {
    // Step 1: Bootstrap system
    let bootstrap_config = BootstrapConfig::default();
    let bootstrap_manager = BootstrapManager::new(bootstrap_config)
        .await
        .expect("Failed to create bootstrap manager");

    let bootstrap_result = bootstrap_manager
        .initialize_system()
        .await
        .expect("Failed to bootstrap system");

    assert!(bootstrap_result.success);
    assert!(!bootstrap_result.system_id.is_empty());

    // Step 2: Initialize governance
    let governance_engine = GovernanceEngine::new()
        .await
        .expect("Failed to create governance engine");

    // Create policy for agent creation
    let policy_id = governance_engine
        .create_policy(
            "agent_creation_policy",
            serde_json::json!({
                "max_agents": 100,
                "required_permissions": ["agent:create"],
                "resource_limits": {
                    "max_memory_gb": 10,
                    "max_gpu_units": 2
                }
            }),
        )
        .await
        .expect("Failed to create policy");

    // Step 3: Register agent permissions
    let agent_permissions = AgentPermissions {
        agent_id: Uuid::new_v4(),
        permissions: vec![PermissionLevel::Create, PermissionLevel::Execute],
        granted_by: bootstrap_result.system_id.clone(),
        expires_at: None,
    };

    governance_engine
        .grant_permissions(agent_permissions.clone())
        .await
        .expect("Failed to grant permissions");

    // Step 4: Create agent with governance validation
    let agent_config = AgentConfig {
        name: "test_agent".to_string(),
        agent_type: "autonomous".to_string(),
        max_memory: 1024 * 1024 * 1024,    // 1GB
        max_gpu_memory: 512 * 1024 * 1024, // 512MB
        priority: 5,
        metadata: serde_json::json!({
            "purpose": "e2e_testing",
            "governance_policy": policy_id
        }),
    };

    // Validate with governance before creation
    let validation_result = governance_engine
        .validate_action(
            &agent_permissions.agent_id,
            "agent:create",
            &serde_json::to_value(&agent_config).unwrap(),
        )
        .await
        .expect("Failed to validate action");

    assert!(validation_result.allowed);

    // Create agent
    let agent = Agent::new(agent_config).expect("Failed to create agent");

    // Step 5: Initialize safety mechanisms
    let emergency_controller = EmergencyController::new()
        .await
        .expect("Failed to create emergency controller");

    // Register agent with safety monitoring
    emergency_controller
        .register_agent(
            agent.id(),
            ResourceLimits {
                max_memory_bytes: 1024 * 1024 * 1024,
                max_cpu_percent: 50.0,
                max_gpu_memory_bytes: 512 * 1024 * 1024,
                max_network_bandwidth_mbps: 100.0,
            },
        )
        .await
        .expect("Failed to register agent with safety");

    // Verify complete flow
    assert!(agent.is_running());
    assert_eq!(
        governance_engine.get_active_agents().await.unwrap().len(),
        1
    );
    assert!(emergency_controller.is_agent_monitored(&agent.id()).await);
}

/// Test multi-region deployment with compliance
#[tokio::test]
async fn test_multi_region_deployment_with_compliance() {
    // Initialize multi-region controller
    let multi_region = Arc::new(
        MultiRegionController::new()
            .await
            .expect("Failed to create multi-region controller"),
    );

    // Initialize compliance manager
    let compliance_manager = Arc::new(
        ComplianceManager::new()
            .await
            .expect("Failed to create compliance manager"),
    );

    // Configure regions with data sovereignty
    let regions = vec![
        Region {
            id: "us-east-1".to_string(),
            name: "US East".to_string(),
            sovereignty_policy: DataSovereigntyPolicy::GDPR,
            active: true,
        },
        Region {
            id: "eu-west-1".to_string(),
            name: "EU West".to_string(),
            sovereignty_policy: DataSovereigntyPolicy::GDPR,
            active: true,
        },
        Region {
            id: "ap-south-1".to_string(),
            name: "Asia Pacific".to_string(),
            sovereignty_policy: DataSovereigntyPolicy::APPI,
            active: true,
        },
    ];

    for region in regions {
        multi_region
            .register_region(region.clone())
            .await
            .expect("Failed to register region");

        // Enable compliance for region
        compliance_manager
            .enable_framework(
                ComplianceFramework::from_sovereignty(region.sovereignty_policy),
                region.id.clone(),
            )
            .await
            .expect("Failed to enable compliance");
    }

    // Deploy agent to specific region with compliance check
    let agent_config = AgentConfig {
        name: "regional_agent".to_string(),
        agent_type: "data_processor".to_string(),
        max_memory: 2 * 1024 * 1024 * 1024,
        max_gpu_memory: 1024 * 1024 * 1024,
        priority: 8,
        metadata: serde_json::json!({
            "target_region": "eu-west-1",
            "data_types": ["personal_data", "financial_data"]
        }),
    };

    // Validate compliance before deployment
    let compliance_result = compliance_manager
        .validate_deployment(&agent_config, "eu-west-1")
        .await
        .expect("Failed to validate compliance");

    assert!(compliance_result.compliant);
    assert!(compliance_result
        .frameworks
        .contains(&ComplianceFramework::GDPR));

    // Deploy to region
    let deployment_result = multi_region
        .deploy_agent(agent_config, "eu-west-1")
        .await
        .expect("Failed to deploy agent");

    assert!(deployment_result.success);
    assert_eq!(deployment_result.region, "eu-west-1");
}

/// Test evolution system with mutations and safety
#[tokio::test]
async fn test_evolution_with_safety_mechanisms() {
    // Initialize evolution engine
    let evolution_engine = Arc::new(GeneticEvolutionEngine::with_defaults(Arc::new(
        SimpleFitnessFunction,
    )));

    // Initialize safety controller
    let emergency_controller = Arc::new(
        EmergencyController::new()
            .await
            .expect("Failed to create emergency controller"),
    );

    // Set evolution safety boundaries
    emergency_controller
        .set_behavior_boundary(
            "evolution_mutation_rate",
            serde_json::json!({
                "max_mutation_rate": 0.3,
                "max_population_size": 1000,
                "forbidden_patterns": ["self_replication", "resource_exhaustion"]
            }),
        )
        .await
        .expect("Failed to set evolution boundaries");

    // Initialize population
    let mut population = evolution_engine
        .initialize_population(100)
        .await
        .expect("Failed to initialize population");

    // Evolve with safety monitoring
    for generation in 0..10 {
        // Check safety before evolution
        let safety_check = emergency_controller
            .check_safety_violation(
                SafetyViolationType::EvolutionAnomaly,
                &serde_json::json!({
                    "generation": generation,
                    "population_size": population.individuals.len(),
                    "mutation_rate": evolution_engine.get_mutation_rate()
                }),
            )
            .await;

        if safety_check.is_err() {
            // Safety violation - activate kill switch
            emergency_controller
                .activate_kill_switch(KillSwitch::EvolutionHalt)
                .await
                .expect("Failed to activate kill switch");
            break;
        }

        // Evolve generation
        evolution_engine
            .evolve_generation(&mut population)
            .await
            .expect("Failed to evolve generation");

        // Log metrics
        println!(
            "Generation {}: avg_fitness={:.3}, best_fitness={:.3}",
            generation,
            population.average_fitness(),
            population.best_fitness()
        );
    }

    assert!(population.generation <= 10);
    assert!(population.individuals.len() <= 1000);
}

/// Test disaster recovery activation flow
#[tokio::test]
async fn test_disaster_recovery_activation() {
    // Initialize DR manager
    let dr_manager = Arc::new(
        DisasterRecoveryManager::new()
            .await
            .expect("Failed to create DR manager"),
    );

    // Create test data
    let test_data = vec![
        ("agent_state", vec![1, 2, 3, 4, 5]),
        ("knowledge_graph", vec![10, 20, 30, 40, 50]),
        ("evolution_data", vec![100, 200, 300, 400, 500]),
    ];

    // Create backup
    let backup_id = dr_manager
        .create_backup(BackupType::Full, test_data.clone())
        .await
        .expect("Failed to create backup");

    // Simulate disaster - data corruption
    let corrupted_data = vec![
        ("agent_state", vec![0, 0, 0, 0, 0]),
        ("knowledge_graph", vec![0, 0, 0, 0, 0]),
        ("evolution_data", vec![0, 0, 0, 0, 0]),
    ];

    // Detect corruption
    let integrity_check = dr_manager
        .verify_data_integrity(&corrupted_data)
        .await
        .expect("Failed to check integrity");

    assert!(!integrity_check.passed);
    assert!(!integrity_check.corrupted_items.is_empty());

    // Activate disaster recovery
    let recovery_result = dr_manager
        .restore_from_backup(backup_id)
        .await
        .expect("Failed to restore from backup");

    assert!(recovery_result.success);
    assert_eq!(recovery_result.restored_items.len(), 3);

    // Verify data restored correctly
    for (key, original_value) in test_data {
        let restored_value = dr_manager
            .get_data(key)
            .await
            .expect("Failed to get restored data");
        assert_eq!(restored_value, original_value);
    }
}

/// Test performance monitoring and regression detection
#[tokio::test]
async fn test_performance_monitoring() {
    // Initialize performance monitor
    let perf_monitor = Arc::new(
        PerformanceMonitor::new()
            .await
            .expect("Failed to create performance monitor"),
    );

    // Simulate workload with varying performance
    let workload_patterns = vec![
        ("normal", 100.0, 50.0),    // name, latency_ms, throughput_ops
        ("degraded", 500.0, 10.0),  // performance degradation
        ("recovered", 120.0, 45.0), // partial recovery
        ("optimal", 80.0, 60.0),    // better than baseline
    ];

    for (pattern_name, latency, throughput) in workload_patterns {
        // Record metrics
        perf_monitor
            .record_metric(MetricType::Latency, latency)
            .await
            .expect("Failed to record latency");

        perf_monitor
            .record_metric(MetricType::Throughput, throughput)
            .await
            .expect("Failed to record throughput");

        // Check for regression
        let regression_detected = perf_monitor
            .detect_regression()
            .await
            .expect("Failed to detect regression");

        if pattern_name == "degraded" {
            assert!(regression_detected.has_regression);
            assert!(regression_detected
                .degraded_metrics
                .contains(&MetricType::Latency));
            assert!(regression_detected
                .degraded_metrics
                .contains(&MetricType::Throughput));
        }

        println!(
            "Pattern '{}': latency={:.1}ms, throughput={:.1}ops/s, regression={}",
            pattern_name, latency, throughput, regression_detected.has_regression
        );
    }
}

/// Test cost optimization decisions
#[tokio::test]
async fn test_cost_optimization_workflow() {
    // Initialize cost optimizer
    let cost_optimizer = Arc::new(
        CostOptimizer::new()
            .await
            .expect("Failed to create cost optimizer"),
    );

    // Initialize workload scheduler
    let scheduler = Arc::new(
        WorkloadScheduler::new()
            .await
            .expect("Failed to create workload scheduler"),
    );

    // Define workloads with different resource requirements
    let workloads = vec![
        ("high_priority_gpu", 8.0, 32.0, 4, 100), // cpu, mem_gb, gpu, priority
        ("batch_processing", 16.0, 64.0, 0, 20),
        ("ml_training", 32.0, 128.0, 8, 80),
        ("web_service", 4.0, 8.0, 0, 90),
    ];

    let mut total_cost = 0.0;

    for (name, cpu, memory, gpu, priority) in workloads {
        // Get cost estimate
        let cost_estimate = cost_optimizer
            .estimate_cost(cpu, memory, gpu)
            .await
            .expect("Failed to estimate cost");

        // Make optimization decision
        let optimization = cost_optimizer
            .optimize_placement(cpu, memory, gpu, priority)
            .await
            .expect("Failed to optimize placement");

        if optimization.should_schedule {
            // Schedule workload
            let schedule_result = scheduler
                .schedule_workload(name, cpu, memory, gpu, priority)
                .await
                .expect("Failed to schedule workload");

            if schedule_result.scheduled {
                total_cost += cost_estimate.hourly_cost;
                println!(
                    "Scheduled '{}': cost=${:.2}/hr, node={}",
                    name, cost_estimate.hourly_cost, schedule_result.assigned_node
                );
            }
        } else {
            println!(
                "Deferred '{}': cost=${:.2}/hr (optimization: {})",
                name, cost_estimate.hourly_cost, optimization.reason
            );
        }
    }

    // Verify cost optimization worked
    assert!(total_cost > 0.0);
    assert!(total_cost < 100.0); // Should be optimized below this threshold
}

/// Test global knowledge graph queries
#[tokio::test]
async fn test_global_knowledge_graph_queries() {
    // Initialize global knowledge graph
    let knowledge_graph = Arc::new(
        GlobalKnowledgeGraph::new()
            .await
            .expect("Failed to create global knowledge graph"),
    );

    // Populate with test data across regions
    let regions = vec!["us-east-1", "eu-west-1", "ap-south-1"];
    let node_types = vec!["Agent", "Goal", "Resource", "Policy"];

    for region in &regions {
        for (i, node_type) in node_types.iter().enumerate() {
            for j in 0..10 {
                let node_id = format!("{}-{}-{}", region, node_type, j);
                knowledge_graph
                    .add_node(
                        node_id.clone(),
                        node_type.to_string(),
                        serde_json::json!({
                            "region": region,
                            "index": j,
                            "created_at": chrono::Utc::now(),
                        }),
                    )
                    .await
                    .expect("Failed to add node");

                // Add relationships
                if i > 0 && j > 0 {
                    let target_id = format!("{}-{}-{}", region, node_types[i - 1], j - 1);
                    knowledge_graph
                        .add_edge(node_id, target_id, "RELATES_TO")
                        .await
                        .expect("Failed to add edge");
                }
            }
        }
    }

    // Test global query
    let query = QueryRequest {
        pattern: "MATCH (a:Agent)-[:RELATES_TO]->(g:Goal) RETURN a, g",
        filters: Some(serde_json::json!({
            "region": {"$in": regions}
        })),
        limit: Some(100),
    };

    let results = knowledge_graph
        .execute_query(query)
        .await
        .expect("Failed to execute query");

    assert!(!results.nodes.is_empty());
    assert!(results.execution_time_ms < 100.0); // Should be fast

    // Test cross-region aggregation
    let aggregation = knowledge_graph
        .aggregate_by_type("Agent")
        .await
        .expect("Failed to aggregate");

    assert_eq!(aggregation.total_count, 30); // 10 per region * 3 regions
    assert_eq!(aggregation.by_region.len(), 3);
}

/// Test zero-trust security enforcement
#[tokio::test]
async fn test_zero_trust_security() {
    // Initialize zero-trust controller
    let zero_trust = Arc::new(
        ZeroTrustController::new()
            .await
            .expect("Failed to create zero-trust controller"),
    );

    // Test device registration and trust
    let device_id = Uuid::new_v4();
    let device_trust = DeviceTrust {
        device_id,
        trust_score: 0.0, // Untrusted initially
        last_verified: None,
        attestation: None,
    };

    // Register device
    zero_trust
        .register_device(device_trust)
        .await
        .expect("Failed to register device");

    // Attempt access without trust
    let access_result = zero_trust
        .verify_access(device_id, "sensitive_resource")
        .await;

    assert!(access_result.is_err()); // Should be denied

    // Perform device attestation
    let attestation_result = zero_trust
        .attest_device(device_id, vec![1, 2, 3, 4, 5]) // Mock attestation data
        .await
        .expect("Failed to attest device");

    assert!(attestation_result.verified);
    assert!(attestation_result.trust_score > 0.5);

    // Retry access with trust
    let access_result = zero_trust
        .verify_access(device_id, "sensitive_resource")
        .await
        .expect("Failed to verify access");

    assert!(access_result.allowed);
    assert!(access_result.trust_score > 0.5);
}

/// Test complete data flow through the system
#[tokio::test]
async fn test_complete_data_flow() {
    println!("=== Starting Complete Data Flow Test ===");

    // 1. Bootstrap
    println!("1. Bootstrapping system...");
    let bootstrap = BootstrapManager::new(BootstrapConfig::default())
        .await
        .expect("Bootstrap failed");
    let system_id = bootstrap.initialize_system().await.unwrap().system_id;

    // 2. Governance setup
    println!("2. Setting up governance...");
    let governance = GovernanceEngine::new().await.unwrap();
    governance.initialize(system_id.clone()).await.unwrap();

    // 3. Safety mechanisms
    println!("3. Initializing safety mechanisms...");
    let safety = EmergencyController::new().await.unwrap();
    safety.activate().await.unwrap();

    // 4. Create agent
    println!("4. Creating agent with governance approval...");
    let agent_config = AgentConfig {
        name: "data_flow_agent".to_string(),
        agent_type: "processor".to_string(),
        max_memory: 1024 * 1024 * 1024,
        max_gpu_memory: 512 * 1024 * 1024,
        priority: 5,
        metadata: serde_json::json!({"system_id": system_id}),
    };

    let agent = Agent::new(agent_config).unwrap();
    governance.register_agent(agent.id()).await.unwrap();
    safety.monitor_agent(agent.id()).await.unwrap();

    // 5. Process data through agent
    println!("5. Processing data through agent...");
    let input_data = vec![1, 2, 3, 4, 5];
    let processed = agent.process(input_data.clone()).await.unwrap();
    assert_ne!(processed, input_data);

    // 6. Store in knowledge graph
    println!("6. Storing in knowledge graph...");
    let kg = GlobalKnowledgeGraph::new().await.unwrap();
    kg.add_node(
        agent.id().to_string(),
        "ProcessedData".to_string(),
        serde_json::json!({
            "input": input_data,
            "output": processed,
            "timestamp": chrono::Utc::now()
        }),
    )
    .await
    .unwrap();

    // 7. Backup data
    println!("7. Creating backup...");
    let dr = DisasterRecoveryManager::new().await.unwrap();
    let backup_id = dr
        .create_backup(
            BackupType::Incremental,
            vec![("agent_data", processed.clone())],
        )
        .await
        .unwrap();

    // 8. Monitor performance
    println!("8. Recording performance metrics...");
    let monitor = PerformanceMonitor::new().await.unwrap();
    monitor
        .record_metric(MetricType::DataProcessed, processed.len() as f64)
        .await
        .unwrap();

    // 9. Optimize costs
    println!("9. Optimizing resource costs...");
    let optimizer = CostOptimizer::new().await.unwrap();
    let cost = optimizer.calculate_agent_cost(&agent).await.unwrap();
    assert!(cost > 0.0);

    // 10. Verify complete flow
    println!("10. Verifying complete flow...");
    assert!(governance.is_agent_compliant(&agent.id()).await.unwrap());
    assert!(!safety.has_violations(&agent.id()).await.unwrap());
    assert!(dr.verify_backup(backup_id).await.unwrap());

    println!("=== Complete Data Flow Test PASSED ===");
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    pub struct SimpleFitnessFunction;

    impl stratoswarm_evolution::FitnessFunction for SimpleFitnessFunction {
        fn evaluate(&self, _individual: &stratoswarm_evolution::Individual) -> f64 {
            rand::random::<f64>()
        }
    }
}
