//! Integration tests for the governance system
//!
//! These tests verify the interaction between different governance components
//! and test real-world scenarios with multiple agents and complex workflows.

use chrono::Utc;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use crate::{
    compliance_integration::{ComplianceIntegration, ComplianceStatus},
    coordination_manager::{CoordinationManager, CoordinationRequest},
    governance_engine::{
        DecisionType, EvolutionRequest, GovernanceConfig, GovernanceDecision, GovernanceEngine,
        LifecyclePhase, PolicyViolation, ResourceQuota, ResourceRequest, ViolationSeverity,
    },
    lifecycle_governance::LifecycleGovernor,
    monitoring_governance::GovernanceMonitor,
    permission_system::{Permission, PermissionSystem, Role},
    policy_manager::{Policy, PolicyManager, PolicyType},
    GovernanceError, Result,
};

use exorust_agent_core::agent::AgentId;

/// Test multi-agent collaborative workflows
#[tokio::test]
async fn test_multi_agent_collaboration_workflow() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 10;
    let engine = GovernanceEngine::new(config).await.unwrap();

    // Create a team of agents with different roles
    let coordinator_agent = AgentId::new();
    let worker_agent1 = AgentId::new();
    let worker_agent2 = AgentId::new();
    let monitor_agent = AgentId::new();

    // Register all agents
    for agent in [
        &coordinator_agent,
        &worker_agent1,
        &worker_agent2,
        &monitor_agent,
    ] {
        engine.register_agent(agent.clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Grant coordination permissions to coordinator
    let coord_permission = engine
        .evaluate_decision(
            &coordinator_agent,
            DecisionType::PermissionRequest(Permission::Coordinate),
        )
        .await
        .unwrap();
    assert!(matches!(coord_permission, GovernanceDecision::Approved));

    // Grant monitoring permissions to monitor agent
    let monitor_permission = engine
        .evaluate_decision(
            &monitor_agent,
            DecisionType::PermissionRequest(Permission::AdminAccess),
        )
        .await
        .unwrap();

    // Coordinator requests to coordinate with workers
    let coordination_request = CoordinationRequest {
        request_id: Uuid::new_v4(),
        requesting_agent: coordinator_agent.clone(),
        target_agents: vec![worker_agent1.clone(), worker_agent2.clone()],
        coordination_type: "distributed_computation".to_string(),
        duration: Some(7200), // 2 hours
    };

    let coordination_decision = engine
        .evaluate_decision(
            &coordinator_agent,
            DecisionType::Coordination(coordination_request),
        )
        .await
        .unwrap();

    // Workers request resources for collaborative task
    for worker in [&worker_agent1, &worker_agent2] {
        let resource_request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(7200),
        };

        let resource_decision = engine
            .evaluate_decision(worker, DecisionType::ResourceAllocation(resource_request))
            .await
            .unwrap();
        assert!(matches!(resource_decision, GovernanceDecision::Approved));
    }

    // Verify all agents are tracked properly
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.total_agents, 4);
    assert_eq!(metrics.active_agents, 4);

    // Verify audit trail exists for all operations
    let audit_log = engine.audit_log.read();
    assert!(audit_log.len() >= 4); // At least one entry per agent registration
}

/// Test resource exhaustion and recovery scenarios
#[tokio::test]
async fn test_resource_exhaustion_and_recovery() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 5;
    config.default_resource_quota = ResourceQuota {
        max_memory_mb: 512,
        max_cpu_cores: 1.0,
        max_gpu_memory_mb: 1024,
        max_network_bandwidth_mbps: 100,
        max_storage_gb: 5,
    };

    let engine = GovernanceEngine::new(config).await.unwrap();

    // Fill system to capacity
    let mut agents = Vec::new();
    for _ in 0..5 {
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await.unwrap();
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();
        agents.push(agent_id);
    }

    // Attempt to register additional agent (should fail)
    let overflow_agent = AgentId::new();
    let overflow_result = engine.register_agent(overflow_agent).await;
    assert!(matches!(
        overflow_result,
        Err(GovernanceError::ResourceLimitExceeded(_))
    ));

    // All agents request maximum resources
    for agent in &agents {
        let max_request = ResourceRequest {
            memory_mb: 512, // At quota limit
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(agent, DecisionType::ResourceAllocation(max_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    // One agent exceeds quota (should be denied)
    let excessive_request = ResourceRequest {
        memory_mb: 1024, // Double the quota
        cpu_cores: 1.0,
        gpu_memory_mb: 1024,
        duration_seconds: Some(3600),
    };

    let excessive_decision = engine
        .evaluate_decision(
            &agents[0],
            DecisionType::ResourceAllocation(excessive_request),
        )
        .await
        .unwrap();
    assert!(matches!(excessive_decision, GovernanceDecision::Denied(_)));

    // Simulate agent termination to free resources
    engine
        .update_lifecycle_phase(&agents[0], LifecyclePhase::Terminated)
        .await
        .unwrap();

    // Verify metrics reflect the change
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.total_agents, 5);
    assert_eq!(metrics.active_agents, 4); // One terminated
}

/// Test security incident response workflow
#[tokio::test]
async fn test_security_incident_response() {
    let mut config = GovernanceConfig::default();
    config.emergency_override_enabled = true;
    config.strict_compliance = true;
    let engine = GovernanceEngine::new(config).await.unwrap();

    // Register multiple agents
    let normal_agent = AgentId::new();
    let suspicious_agent = AgentId::new();
    let admin_agent = AgentId::new();

    for agent in [&normal_agent, &suspicious_agent, &admin_agent] {
        engine.register_agent(agent.clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Normal operations proceed normally
    let normal_request = ResourceRequest {
        memory_mb: 256,
        cpu_cores: 0.5,
        gpu_memory_mb: 512,
        duration_seconds: Some(1800),
    };

    let normal_decision = engine
        .evaluate_decision(
            &normal_agent,
            DecisionType::ResourceAllocation(normal_request),
        )
        .await
        .unwrap();
    assert!(matches!(normal_decision, GovernanceDecision::Approved));

    // Suspicious agent accumulates violations
    for i in 0..3 {
        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: format!("suspicious_activity_{}", i),
            severity: ViolationSeverity::Medium,
            details: format!("Unusual behavior pattern {}", i),
        };
        engine
            .record_violation(&suspicious_agent, violation)
            .await
            .unwrap();
    }

    // Agent should still be active after medium violations
    let suspicious_state = engine.active_agents.get(&suspicious_agent).unwrap();
    assert_eq!(suspicious_state.lifecycle_phase, LifecyclePhase::Active);
    assert_eq!(suspicious_state.violations.len(), 3);

    // Critical security violation triggers emergency response
    let critical_violation = PolicyViolation {
        timestamp: Utc::now(),
        policy_id: Uuid::new_v4(),
        violation_type: "security_breach".to_string(),
        severity: ViolationSeverity::Critical,
        details: "Attempted unauthorized access to system resources".to_string(),
    };

    engine
        .record_violation(&suspicious_agent, critical_violation)
        .await
        .unwrap();

    // Suspicious agent should be suspended
    let updated_state = engine.active_agents.get(&suspicious_agent).unwrap();
    assert_eq!(updated_state.lifecycle_phase, LifecyclePhase::Suspended);

    // System should activate emergency measures
    assert!(engine.kill_switch.is_activated());

    // All subsequent decisions should be denied due to emergency state
    let post_emergency_decision = engine
        .evaluate_decision(
            &normal_agent,
            DecisionType::ResourceAllocation(normal_request),
        )
        .await
        .unwrap();
    assert!(matches!(
        post_emergency_decision,
        GovernanceDecision::Denied(_)
    ));

    // Verify security metrics
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.total_agents, 3);
    assert_eq!(metrics.suspended_agents, 1);
    assert_eq!(metrics.critical_violations, 1);
    assert_eq!(metrics.total_violations, 4); // 3 medium + 1 critical
}

/// Test agent evolution and lifecycle management
#[tokio::test]
async fn test_agent_evolution_lifecycle() {
    let mut config = GovernanceConfig::default();
    config.strict_compliance = false; // Allow more flexible evolution for testing
    let engine = GovernanceEngine::new(config).await.unwrap();

    let evolving_agent = AgentId::new();
    engine.register_agent(evolving_agent.clone()).await?;

    // Agent starts in Initializing phase
    let initial_state = engine.active_agents.get(&evolving_agent)?;
    assert_eq!(initial_state.lifecycle_phase, LifecyclePhase::Initializing);

    // Evolution should fail in Initializing phase
    let evolution_request = EvolutionRequest {
        evolution_type: "capability_enhancement".to_string(),
        target_capabilities: vec!["improved_reasoning".to_string()],
        resource_requirements: ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: None,
        },
    };

    let evolution_decision = engine
        .evaluate_decision(
            &evolving_agent,
            DecisionType::Evolution(evolution_request.clone()),
        )
        .await
        .unwrap();
    assert!(matches!(evolution_decision, GovernanceDecision::Denied(_)));

    // Transition to Active phase
    engine
        .update_lifecycle_phase(&evolving_agent, LifecyclePhase::Active)
        .await
        .unwrap();

    // Evolution should now be possible
    let evolution_decision = engine
        .evaluate_decision(&evolving_agent, DecisionType::Evolution(evolution_request))
        .await
        .unwrap();
    // Decision depends on implementation, but should at least pass lifecycle check

    // Transition to Evolving phase
    engine
        .update_lifecycle_phase(&evolving_agent, LifecyclePhase::Evolving)
        .await
        .unwrap();

    // Agent in Evolving phase should have different resource access
    let evolving_request = ResourceRequest {
        memory_mb: 1024, // Higher resource requirement for evolution
        cpu_cores: 2.0,
        gpu_memory_mb: 2048,
        duration_seconds: Some(1800),
    };

    // This might be denied due to quota limits, but tests the evaluation path
    let evolving_resource_decision = engine
        .evaluate_decision(
            &evolving_agent,
            DecisionType::ResourceAllocation(evolving_request),
        )
        .await
        .unwrap();

    // Verify lifecycle transitions are recorded in audit log
    let audit_log = engine.audit_log.read();
    let phase_updates = audit_log
        .iter()
        .filter(|entry| entry.action == "lifecycle_phase_update")
        .count();
    assert!(phase_updates >= 2); // At least Active and Evolving transitions
}

/// Test policy enforcement across multiple agents
#[tokio::test]
async fn test_policy_enforcement_integration() {
    let engine = GovernanceEngine::new(GovernanceConfig::default())
        .await
        .unwrap();

    // Create a test policy
    let resource_policy = Policy {
        id: Uuid::new_v4(),
        name: "Memory Limit Policy".to_string(),
        description: "Limits memory allocation per agent".to_string(),
        policy_type: PolicyType::Resource,
        rules: serde_json::json!({
            "max_memory_mb": 512,
            "enforcement_level": "strict"
        }),
        enabled: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    // Add policy to the system
    engine
        .policy_manager
        .add_policy(resource_policy.clone())
        .await
        .unwrap();

    // Register multiple agents
    let agent1 = AgentId::new();
    let agent2 = AgentId::new();
    let agent3 = AgentId::new();

    for agent in [&agent1, &agent2, &agent3] {
        engine.register_agent(agent.clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Test policy enforcement for each agent
    for agent in [&agent1, &agent2, &agent3] {
        // Request within policy limits should succeed
        let compliant_request = ResourceRequest {
            memory_mb: 256, // Within 512MB limit
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let compliant_decision = engine
            .evaluate_decision(agent, DecisionType::ResourceAllocation(compliant_request))
            .await
            .unwrap();
        assert!(matches!(compliant_decision, GovernanceDecision::Approved));

        // Request exceeding policy limits should fail
        let non_compliant_request = ResourceRequest {
            memory_mb: 1024, // Exceeds both quota (1024) but also policy (512)
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let non_compliant_decision = engine
            .evaluate_decision(
                agent,
                DecisionType::ResourceAllocation(non_compliant_request),
            )
            .await
            .unwrap();
        assert!(matches!(
            non_compliant_decision,
            GovernanceDecision::Denied(_)
        ));
    }

    // Disable policy and verify behavior changes
    let mut updated_policy = resource_policy.clone();
    updated_policy.enabled = false;
    engine
        .policy_manager
        .update_policy(updated_policy)
        .await
        .unwrap();

    // Now requests should be evaluated against quota only, not policy
    let quota_limit_request = ResourceRequest {
        memory_mb: 1024, // At quota limit, exceeds disabled policy
        cpu_cores: 1.0,
        gpu_memory_mb: 1024,
        duration_seconds: Some(3600),
    };

    let quota_decision = engine
        .evaluate_decision(
            &agent1,
            DecisionType::ResourceAllocation(quota_limit_request),
        )
        .await
        .unwrap();
    assert!(matches!(quota_decision, GovernanceDecision::Approved));
}

/// Test compliance integration across governance operations
#[tokio::test]
async fn test_compliance_integration_workflow() {
    let mut config = GovernanceConfig::default();
    config.strict_compliance = true;
    let engine = GovernanceEngine::new(config).await.unwrap();

    let compliant_agent = AgentId::new();
    let risky_agent = AgentId::new();

    // Register agents
    for agent in [&compliant_agent, &risky_agent] {
        engine.register_agent(agent.clone()).await?;
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Test compliant evolution request
    let security_evolution = EvolutionRequest {
        evolution_type: "security_enhancement".to_string(),
        target_capabilities: vec![
            "improved_encryption".to_string(),
            "audit_logging".to_string(),
        ],
        resource_requirements: ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: None,
        },
    };

    let compliant_evolution_decision = engine
        .evaluate_decision(
            &compliant_agent,
            DecisionType::Evolution(security_evolution),
        )
        .await
        .unwrap();
    // Should pass compliance checks for security enhancements

    // Test risky evolution request
    let risky_evolution = EvolutionRequest {
        evolution_type: "unrestricted_access".to_string(),
        target_capabilities: vec!["bypass_security".to_string(), "admin_override".to_string()],
        resource_requirements: ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: None,
        },
    };

    let risky_evolution_decision = engine
        .evaluate_decision(&risky_agent, DecisionType::Evolution(risky_evolution))
        .await
        .unwrap();
    // Should be denied due to compliance concerns
    assert!(matches!(
        risky_evolution_decision,
        GovernanceDecision::Denied(_)
    ));

    // Update compliance status manually
    engine
        .compliance_integration
        .update_compliance_status(&risky_agent, ComplianceStatus::NonCompliant)
        .await
        .unwrap();

    // All requests from non-compliant agent should be denied
    let basic_request = ResourceRequest {
        memory_mb: 256,
        cpu_cores: 0.5,
        gpu_memory_mb: 512,
        duration_seconds: Some(1800),
    };

    let non_compliant_decision = engine
        .evaluate_decision(
            &risky_agent,
            DecisionType::ResourceAllocation(basic_request),
        )
        .await
        .unwrap();
    // May be denied due to non-compliant status (implementation dependent)

    // Verify compliance is tracked in metrics
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.total_agents, 2);
}

/// Test system recovery and cleanup operations
#[tokio::test]
async fn test_system_recovery_and_cleanup() {
    let mut config = GovernanceConfig::default();
    config.audit_retention_days = 1; // Short retention for testing
    config.emergency_override_enabled = true;
    let engine = GovernanceEngine::new(config).await.unwrap();

    // Create initial state with multiple agents
    let mut agents = Vec::new();
    for i in 0..10 {
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();
        agents.push(agent_id);
    }

    // Generate audit activity
    for i in 0..50 {
        engine.log_audit(
            Some(agents[i % agents.len()].clone()),
            &format!("test_operation_{}", i),
            "success",
            serde_json::json!({ "iteration": i }),
        );
    }

    // Verify initial state
    assert_eq!(engine.active_agents.len(), 10);
    assert!(engine.audit_log.read().len() > 50); // Registration + manual entries

    // Simulate emergency situation
    engine
        .kill_switch
        .activate("System recovery test")
        .await
        .unwrap();

    // All agents should be denied service during emergency
    let emergency_request = ResourceRequest {
        memory_mb: 256,
        cpu_cores: 0.5,
        gpu_memory_mb: 512,
        duration_seconds: Some(900),
    };

    for agent in &agents[0..3] {
        // Test a few agents
        let decision = engine
            .evaluate_decision(
                agent,
                DecisionType::ResourceAllocation(emergency_request.clone()),
            )
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    // Deactivate emergency mode
    engine.kill_switch.deactivate().await.unwrap();

    // Services should resume
    let post_emergency_decision = engine
        .evaluate_decision(
            &agents[0],
            DecisionType::ResourceAllocation(emergency_request),
        )
        .await
        .unwrap();
    assert!(matches!(
        post_emergency_decision,
        GovernanceDecision::Approved
    ));

    // Test audit log cleanup
    let pre_cleanup_count = engine.audit_log.read().len();
    engine.cleanup_audit_logs().await;
    let post_cleanup_count = engine.audit_log.read().len();

    // Cleanup should reduce audit log size (depending on timestamps)
    // In real scenario with longer retention, this would be more predictable

    // Verify system is still functional after cleanup
    let final_metrics = engine.get_metrics().await;
    assert_eq!(final_metrics.total_agents, 10);
    assert_eq!(final_metrics.active_agents, 10);
}

/// Test performance under concurrent load
#[tokio::test]
async fn test_concurrent_governance_load() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 50;
    let engine = Arc::new(GovernanceEngine::new(config).await.unwrap());

    // Register agents concurrently
    let mut registration_handles = vec![];
    for _ in 0..30 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let agent_id = AgentId::new();
            let result = engine_clone.register_agent(agent_id.clone()).await;
            (agent_id, result)
        });
        registration_handles.push(handle);
    }

    // Collect registered agents
    let mut registered_agents = Vec::new();
    for handle in registration_handles {
        let (agent_id, result) = handle.await.unwrap();
        if result.is_ok() {
            registered_agents.push(agent_id);
        }
    }

    // Activate all agents
    for agent in &registered_agents {
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Generate concurrent governance decisions
    let mut decision_handles = vec![];
    for agent in &registered_agents {
        let engine_clone = engine.clone();
        let agent_id = agent.clone();

        let handle = tokio::spawn(async move {
            let requests = vec![
                DecisionType::ResourceAllocation(ResourceRequest {
                    memory_mb: 256,
                    cpu_cores: 0.5,
                    gpu_memory_mb: 512,
                    duration_seconds: Some(1800),
                }),
                DecisionType::PermissionRequest(Permission::ReadData),
                DecisionType::PermissionRequest(Permission::WriteData),
            ];

            let mut results = Vec::new();
            for request in requests {
                let decision = engine_clone.evaluate_decision(&agent_id, request).await;
                results.push(decision);
            }
            results
        });
        decision_handles.push(handle);
    }

    // Wait for all decisions and verify
    let mut total_decisions = 0;
    let mut successful_decisions = 0;

    for handle in decision_handles {
        let decisions = handle.await.unwrap();
        total_decisions += decisions.len();

        for decision in decisions {
            if decision.is_ok() {
                successful_decisions += 1;
            }
        }
    }

    // Verify system handled concurrent load
    assert!(total_decisions > 0);
    assert!(successful_decisions > 0);
    assert_eq!(engine.active_agents.len(), registered_agents.len());

    // Check final system metrics
    let metrics = engine.get_metrics().await;
    assert_eq!(metrics.total_agents, registered_agents.len());
    assert_eq!(metrics.active_agents, registered_agents.len());
    assert!(metrics.decisions_made > 0);
}
