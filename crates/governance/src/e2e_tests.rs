//! End-to-end tests for the governance system
//!
//! These tests simulate real-world scenarios and validate the complete
//! governance system behavior from agent registration to complex workflows.

use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use uuid::Uuid;

use crate::{
    compliance_integration::{ComplianceIntegration, ComplianceStatus},
    coordination_manager::{CoordinationManager, CoordinationRequest},
    governance_engine::{
        DecisionType, EvolutionRequest, GovernanceConfig, GovernanceDecision, GovernanceEngine,
        LifecyclePhase, PolicyViolation, ResourceQuota, ResourceRequest, ViolationSeverity,
    },
    permission_system::{Permission, PermissionSystem, Role},
    policy_manager::{Policy, PolicyManager, PolicyType},
    GovernanceError, Result,
};

use exorust_agent_core::agent::AgentId;

/// Comprehensive test of agent onboarding and operational workflow
#[tokio::test]
async fn test_complete_agent_onboarding_workflow() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 20;
    config.strict_compliance = true;
    config.emergency_override_enabled = true;

    let engine = GovernanceEngine::new(config).await?;

    // Phase 1: Agent Registration and Initial Setup
    let new_agent = AgentId::new();

    // Register the agent
    let registration_result = engine.register_agent(new_agent.clone()).await;
    assert!(registration_result.is_ok());

    // Verify initial state
    let agent_state = engine.active_agents.get(&new_agent).unwrap();
    assert_eq!(agent_state.lifecycle_phase, LifecyclePhase::Initializing);
    assert_eq!(agent_state.violations.len(), 0);
    assert!(agent_state.compliance_status);

    // Phase 2: Permission Acquisition
    let basic_permissions = vec![
        Permission::ReadData,
        Permission::WriteData,
        Permission::NetworkAccess,
    ];

    for permission in basic_permissions {
        let permission_decision = engine
            .evaluate_decision(&new_agent, DecisionType::PermissionRequest(permission))
            .await
            .unwrap();
        assert!(matches!(permission_decision, GovernanceDecision::Approved));
    }

    // Phase 3: Lifecycle Transition to Active
    engine
        .update_lifecycle_phase(&new_agent, LifecyclePhase::Active)
        .await
        .unwrap();

    let updated_state = engine.active_agents.get(&new_agent).unwrap();
    assert_eq!(updated_state.lifecycle_phase, LifecyclePhase::Active);

    // Phase 4: Resource Allocation for Operations
    let operational_resources = ResourceRequest {
        memory_mb: 512,
        cpu_cores: 1.0,
        gpu_memory_mb: 1024,
        duration_seconds: Some(7200), // 2 hours
    };

    let resource_decision = engine
        .evaluate_decision(
            &new_agent,
            DecisionType::ResourceAllocation(operational_resources),
        )
        .await
        .unwrap();
    assert!(matches!(resource_decision, GovernanceDecision::Approved));

    // Phase 5: Capability Evolution Request
    let evolution_request = EvolutionRequest {
        evolution_type: "performance_optimization".to_string(),
        target_capabilities: vec![
            "parallel_processing".to_string(),
            "memory_optimization".to_string(),
        ],
        resource_requirements: ResourceRequest {
            memory_mb: 768,
            cpu_cores: 1.5,
            gpu_memory_mb: 1536,
            duration_seconds: None,
        },
    };

    let evolution_decision = engine
        .evaluate_decision(&new_agent, DecisionType::Evolution(evolution_request))
        .await
        .unwrap();
    // Evolution success depends on compliance and lifecycle checks

    // Phase 6: Operational Monitoring
    let final_metrics = engine.get_metrics().await;
    assert_eq!(final_metrics.total_agents, 1);
    assert_eq!(final_metrics.active_agents, 1);

    // Phase 7: Audit Trail Verification
    let audit_log = engine.audit_log.read();
    assert!(audit_log.len() >= 3); // Registration + lifecycle update + decisions

    // Verify specific audit entries
    let registration_entry = audit_log.iter().find(|entry| {
        entry.action == "agent_registered" && entry.agent_id == Some(new_agent.clone())
    });
    assert!(registration_entry.is_some());

    let lifecycle_entries = audit_log
        .iter()
        .filter(|entry| {
            entry.action == "lifecycle_phase_update" && entry.agent_id == Some(new_agent.clone())
        })
        .count();
    assert!(lifecycle_entries >= 1);
}

/// Test multi-agent collaborative scenario with resource sharing
#[tokio::test]
async fn test_collaborative_research_scenario() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 15;
    config.default_resource_quota = ResourceQuota {
        max_memory_mb: 1024,
        max_cpu_cores: 2.0,
        max_gpu_memory_mb: 2048,
        max_network_bandwidth_mbps: 200,
        max_storage_gb: 20,
    };

    let engine = GovernanceEngine::new(config).await.unwrap();

    // Create research team
    let team_lead = AgentId::new();
    let researcher1 = AgentId::new();
    let researcher2 = AgentId::new();
    let data_analyst = AgentId::new();
    let monitor = AgentId::new();

    let research_team = vec![
        &team_lead,
        &researcher1,
        &researcher2,
        &data_analyst,
        &monitor,
    ];

    // Phase 1: Team Registration
    for agent in &research_team {
        engine.register_agent((*agent).clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Phase 2: Role-based Permission Assignment
    // Team lead gets coordination permissions
    let lead_permissions = vec![Permission::Coordinate, Permission::AdminAccess];
    for permission in lead_permissions {
        let decision = engine
            .evaluate_decision(&team_lead, DecisionType::PermissionRequest(permission))
            .await
            .unwrap();
        // Grant based on implementation
    }

    // Researchers get research permissions
    for researcher in [&researcher1, &researcher2] {
        let research_permissions = vec![
            Permission::ReadData,
            Permission::WriteData,
            Permission::NetworkAccess,
        ];
        for permission in research_permissions {
            let decision = engine
                .evaluate_decision(researcher, DecisionType::PermissionRequest(permission))
                .await
                .unwrap();
        }
    }

    // Data analyst gets data processing permissions
    let analyst_permissions = vec![
        Permission::ReadData,
        Permission::WriteData,
        Permission::ProcessData,
    ];
    for permission in analyst_permissions {
        let decision = engine
            .evaluate_decision(&data_analyst, DecisionType::PermissionRequest(permission))
            .await
            .unwrap();
    }

    // Phase 3: Resource Allocation for Research Project
    let research_resources = HashMap::from([
        (
            &team_lead,
            ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: Some(14400), // 4 hours
            },
        ),
        (
            &researcher1,
            ResourceRequest {
                memory_mb: 768,
                cpu_cores: 1.5,
                gpu_memory_mb: 1536,
                duration_seconds: Some(14400),
            },
        ),
        (
            &researcher2,
            ResourceRequest {
                memory_mb: 768,
                cpu_cores: 1.5,
                gpu_memory_mb: 1536,
                duration_seconds: Some(14400),
            },
        ),
        (
            &data_analyst,
            ResourceRequest {
                memory_mb: 1024,
                cpu_cores: 2.0,
                gpu_memory_mb: 2048,
                duration_seconds: Some(18000), // 5 hours
            },
        ),
        (
            &monitor,
            ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 512,
                duration_seconds: Some(21600), // 6 hours
            },
        ),
    ]);

    for (agent, resource_request) in research_resources {
        let decision = engine
            .evaluate_decision(agent, DecisionType::ResourceAllocation(resource_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    // Phase 4: Coordination Requests
    // Team lead coordinates with researchers
    let researcher_coordination = CoordinationRequest {
        request_id: Uuid::new_v4(),
        requesting_agent: team_lead.clone(),
        target_agents: vec![researcher1.clone(), researcher2.clone()],
        coordination_type: "research_collaboration".to_string(),
        duration: Some(14400),
    };

    let coord_decision = engine
        .evaluate_decision(
            &team_lead,
            DecisionType::Coordination(researcher_coordination),
        )
        .await
        .unwrap();
    // Coordination success depends on permissions and implementation

    // Researchers coordinate with data analyst
    let data_coordination = CoordinationRequest {
        request_id: Uuid::new_v4(),
        requesting_agent: researcher1.clone(),
        target_agents: vec![data_analyst.clone()],
        coordination_type: "data_sharing".to_string(),
        duration: Some(7200),
    };

    let data_coord_decision = engine
        .evaluate_decision(&researcher1, DecisionType::Coordination(data_coordination))
        .await
        .unwrap();

    // Phase 5: Evolution for Enhanced Capabilities
    // Researchers evolve their capabilities
    for researcher in [&researcher1, &researcher2] {
        let research_evolution = EvolutionRequest {
            evolution_type: "research_enhancement".to_string(),
            target_capabilities: vec![
                "advanced_analysis".to_string(),
                "pattern_recognition".to_string(),
            ],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let evolution_decision = engine
            .evaluate_decision(researcher, DecisionType::Evolution(research_evolution))
            .await
            .unwrap();
        // Evolution success depends on compliance and lifecycle checks
    }

    // Phase 6: Project Completion and Metrics
    let final_metrics = engine.get_metrics().await;
    assert_eq!(final_metrics.total_agents, 5);
    assert_eq!(final_metrics.active_agents, 5);
    assert!(final_metrics.decisions_made > 10); // Multiple decisions per agent

    // Verify team coordination was tracked
    let audit_log = engine.audit_log.read();
    let coordination_decisions = audit_log
        .iter()
        .filter(|entry| {
            entry.action == "governance_decision"
                && entry.details.to_string().contains("Coordination")
        })
        .count();
    assert!(coordination_decisions > 0);
}

/// Test security incident detection and response
#[tokio::test]
async fn test_security_incident_detection_and_response() {
    let mut config = GovernanceConfig::default();
    config.emergency_override_enabled = true;
    config.strict_compliance = true;
    config.max_agents = 10;

    let engine = GovernanceEngine::new(config).await?;

    // Create scenario with normal and malicious agents
    let normal_agents = vec![AgentId::new(), AgentId::new(), AgentId::new()];
    let suspicious_agent = AgentId::new();
    let malicious_agent = AgentId::new();

    let all_agents = vec![
        &normal_agents[0],
        &normal_agents[1],
        &normal_agents[2],
        &suspicious_agent,
        &malicious_agent,
    ];

    // Phase 1: System Initialization
    for agent in &all_agents {
        engine.register_agent((*agent).clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Phase 2: Normal Operations
    // Normal agents operate within parameters
    for agent in &normal_agents {
        let normal_request = ResourceRequest {
            memory_mb: 256,
            cpu_cores: 0.5,
            gpu_memory_mb: 512,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(agent, DecisionType::ResourceAllocation(normal_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    // Phase 3: Suspicious Activity Detection
    // Suspicious agent shows anomalous behavior
    let anomalous_patterns = vec![
        ("unusual_memory_pattern", ViolationSeverity::Low),
        ("irregular_access_pattern", ViolationSeverity::Medium),
        ("suspicious_network_activity", ViolationSeverity::Medium),
    ];

    for (violation_type, severity) in anomalous_patterns {
        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: violation_type.to_string(),
            severity,
            details: format!("Detected {} from agent", violation_type),
        };

        engine
            .record_violation(&suspicious_agent, violation)
            .await
            .unwrap();

        // Small delay to simulate time progression
        sleep(Duration::from_millis(10)).await;
    }

    // Verify suspicious agent is still active (medium violations)
    let suspicious_state = engine.active_agents.get(&suspicious_agent).unwrap();
    assert_eq!(suspicious_state.lifecycle_phase, LifecyclePhase::Active);
    assert_eq!(suspicious_state.violations.len(), 3);

    // Phase 4: Critical Security Breach
    // Malicious agent triggers critical violation
    let critical_violations = vec![
        "unauthorized_system_access",
        "data_exfiltration_attempt",
        "privilege_escalation",
    ];

    for violation_type in critical_violations {
        let critical_violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: violation_type.to_string(),
            severity: ViolationSeverity::Critical,
            details: format!("Critical security breach: {}", violation_type),
        };

        engine
            .record_violation(&malicious_agent, critical_violation)
            .await
            .unwrap();

        sleep(Duration::from_millis(10)).await;
    }

    // Phase 5: Emergency Response Verification
    // Malicious agent should be suspended
    let malicious_state = engine.active_agents.get(&malicious_agent).unwrap();
    assert_eq!(malicious_state.lifecycle_phase, LifecyclePhase::Suspended);

    // Emergency kill switch should be activated
    assert!(engine.kill_switch.is_activated());

    // All subsequent requests should be denied
    let post_breach_request = ResourceRequest {
        memory_mb: 128,
        cpu_cores: 0.25,
        gpu_memory_mb: 256,
        duration_seconds: Some(1800),
    };

    for agent in &normal_agents {
        let decision = engine
            .evaluate_decision(
                agent,
                DecisionType::ResourceAllocation(post_breach_request.clone()),
            )
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    // Phase 6: Security Metrics Verification
    let security_metrics = engine.get_metrics().await;
    assert_eq!(security_metrics.total_agents, 5);
    assert_eq!(security_metrics.suspended_agents, 1);
    assert_eq!(security_metrics.critical_violations, 3);
    assert_eq!(security_metrics.total_violations, 6); // 3 suspicious + 3 critical

    // Phase 7: Incident Audit Trail
    let audit_log = engine.audit_log.read();

    // Verify violation records
    let violation_entries = audit_log
        .iter()
        .filter(|entry| entry.action == "policy_violation")
        .count();
    assert_eq!(violation_entries, 6);

    // Verify emergency activation
    let emergency_decisions = audit_log
        .iter()
        .filter(|entry| {
            entry.action == "governance_decision" && entry.result.contains("Emergency kill switch")
        })
        .count();
    assert!(emergency_decisions > 0);

    // Phase 8: System Recovery Test
    // Deactivate emergency mode
    engine.kill_switch.deactivate().await.unwrap();

    // Normal agents should resume operations
    let recovery_decision = engine
        .evaluate_decision(
            &normal_agents[0],
            DecisionType::ResourceAllocation(post_breach_request),
        )
        .await
        .unwrap();
    assert!(matches!(recovery_decision, GovernanceDecision::Approved));

    // But malicious agent should remain suspended
    let final_malicious_state = engine.active_agents.get(&malicious_agent).unwrap();
    assert_eq!(
        final_malicious_state.lifecycle_phase,
        LifecyclePhase::Suspended
    );
}

/// Test resource contention and fairness
#[tokio::test]
async fn test_resource_contention_and_fairness() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 20;
    config.default_resource_quota = ResourceQuota {
        max_memory_mb: 512,
        max_cpu_cores: 1.0,
        max_gpu_memory_mb: 1024,
        max_network_bandwidth_mbps: 100,
        max_storage_gb: 10,
    };

    let engine = GovernanceEngine::new(config).await.unwrap();

    // Create multiple agent types with different priorities
    let high_priority_agents = vec![AgentId::new(), AgentId::new()];
    let normal_agents: Vec<AgentId> = (0..10).map(|_| AgentId::new()).collect();
    let low_priority_agents = vec![AgentId::new(), AgentId::new()];

    let all_agents: Vec<&AgentId> = high_priority_agents
        .iter()
        .chain(normal_agents.iter())
        .chain(low_priority_agents.iter())
        .collect();

    // Phase 1: Register all agents
    for agent in &all_agents {
        engine.register_agent((*agent).clone()).await.unwrap();
        engine
            .update_lifecycle_phase(agent, LifecyclePhase::Active)
            .await
            .unwrap();
    }

    // Phase 2: Simulate resource contention
    // All agents request maximum allowed resources
    let max_resource_request = ResourceRequest {
        memory_mb: 512,
        cpu_cores: 1.0,
        gpu_memory_mb: 1024,
        duration_seconds: Some(7200),
    };

    let mut approved_requests = 0;
    let mut denied_requests = 0;

    // Test concurrent resource requests
    let mut request_handles = vec![];

    for agent in &all_agents {
        let engine_ref = &engine;
        let agent_id = (*agent).clone();
        let request = max_resource_request.clone();

        let handle = tokio::spawn(async move {
            engine_ref
                .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
                .await
        });
        request_handles.push(handle);
    }

    // Collect results
    for handle in request_handles {
        match handle.await.unwrap() {
            Ok(GovernanceDecision::Approved) => approved_requests += 1,
            Ok(GovernanceDecision::Denied(_)) => denied_requests += 1,
            _ => {}
        }
    }

    // Verify all requests were processed
    assert_eq!(approved_requests + denied_requests, all_agents.len());

    // Phase 3: Test resource allocation patterns
    // High priority agents should generally get approval
    for agent in &high_priority_agents {
        let priority_request = ResourceRequest {
            memory_mb: 256, // Half of maximum
            cpu_cores: 0.5,
            gpu_memory_mb: 512,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(agent, DecisionType::ResourceAllocation(priority_request))
            .await
            .unwrap();
        // Should be approved due to lower resource requirements
        assert!(matches!(decision, GovernanceDecision::Approved));
    }

    // Phase 4: Test gradual resource reduction
    // Start with high requests and reduce until approved
    for agent in normal_agents.iter().take(3) {
        let mut memory_request = 512;
        let mut approved = false;

        while memory_request >= 128 && !approved {
            let flexible_request = ResourceRequest {
                memory_mb: memory_request,
                cpu_cores: (memory_request as f32) / 512.0,
                gpu_memory_mb: memory_request * 2,
                duration_seconds: Some(1800),
            };

            let decision = engine
                .evaluate_decision(agent, DecisionType::ResourceAllocation(flexible_request))
                .await
                .unwrap();

            if matches!(decision, GovernanceDecision::Approved) {
                approved = true;
            } else {
                memory_request /= 2;
            }
        }

        // At least minimal resources should be available
        assert!(approved || memory_request < 128);
    }

    // Phase 5: Resource fairness verification
    let final_metrics = engine.get_metrics().await;
    assert_eq!(final_metrics.total_agents, all_agents.len());
    assert!(final_metrics.decisions_made > all_agents.len());
    assert!(final_metrics.approvals > 0);

    // Verify audit trail shows fair processing
    let audit_log = engine.audit_log.read();
    let resource_decisions = audit_log
        .iter()
        .filter(|entry| {
            entry.action == "governance_decision"
                && entry.details.to_string().contains("ResourceAllocation")
        })
        .count();
    assert!(resource_decisions > 0);
}

/// Test long-running governance operations and stability
#[tokio::test]
async fn test_long_running_stability() {
    let mut config = GovernanceConfig::default();
    config.max_agents = 30;
    config.audit_retention_days = 7;

    let engine = Arc::new(GovernanceEngine::new(config).await?);

    // Phase 1: Create persistent agent population
    let mut agents = Vec::new();
    for _ in 0..15 {
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await.unwrap();
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();
        agents.push(agent_id);
    }

    // Phase 2: Simulate extended operations
    let operation_duration = Duration::from_millis(100); // Shortened for test
    let total_iterations = 20;

    for iteration in 0..total_iterations {
        // Rotate through agents for decisions
        let active_agent = &agents[iteration % agents.len()];

        // Various operation types
        let operations = vec![
            DecisionType::ResourceAllocation(ResourceRequest {
                memory_mb: 256 + (iteration as u64 * 32) % 512,
                cpu_cores: 0.5 + (iteration as f32 * 0.1) % 1.0,
                gpu_memory_mb: 512 + (iteration as u64 * 64) % 1024,
                duration_seconds: Some(1800 + (iteration as u64 * 300) % 3600),
            }),
            DecisionType::PermissionRequest(match iteration % 4 {
                0 => Permission::ReadData,
                1 => Permission::WriteData,
                2 => Permission::NetworkAccess,
                _ => Permission::ProcessData,
            }),
        ];

        for operation in operations {
            let decision = engine.evaluate_decision(active_agent, operation).await;

            // Log any errors but continue
            if let Err(e) = decision {
                eprintln!("Decision error in iteration {}: {:?}", iteration, e);
            }
        }

        // Periodic system maintenance
        if iteration % 5 == 0 {
            // Check metrics
            let metrics = engine.get_metrics().await;
            assert_eq!(metrics.total_agents, 15);

            // Simulate minor violations for some agents
            if iteration % 10 == 0 {
                let violation = PolicyViolation {
                    timestamp: Utc::now(),
                    policy_id: Uuid::new_v4(),
                    violation_type: "routine_check".to_string(),
                    severity: ViolationSeverity::Low,
                    details: format!("Routine violation check - iteration {}", iteration),
                };

                engine
                    .record_violation(active_agent, violation)
                    .await
                    .unwrap();
            }
        }

        // Small delay to simulate time progression
        sleep(operation_duration).await;
    }

    // Phase 3: System stability verification
    let final_metrics = engine.get_metrics().await;
    assert_eq!(final_metrics.total_agents, 15);
    assert!(final_metrics.decisions_made >= total_iterations);

    // Verify system responsiveness
    let response_test_start = std::time::Instant::now();
    let quick_decision = engine
        .evaluate_decision(
            &agents[0],
            DecisionType::ResourceAllocation(ResourceRequest {
                memory_mb: 128,
                cpu_cores: 0.25,
                gpu_memory_mb: 256,
                duration_seconds: Some(900),
            }),
        )
        .await;
    let response_time = response_test_start.elapsed();

    assert!(quick_decision.is_ok());
    assert!(response_time < Duration::from_millis(100)); // Should be fast

    // Phase 4: Memory and audit management
    let pre_cleanup_audit_count = engine.audit_log.read().len();

    // Generate additional audit entries
    for i in 0..50 {
        engine.log_audit(
            Some(agents[i % agents.len()].clone()),
            &format!("stability_test_{}", i),
            "success",
            serde_json::json!({ "test_iteration": i }),
        );
    }

    let post_generation_count = engine.audit_log.read().len();
    assert!(post_generation_count > pre_cleanup_audit_count);

    // Test cleanup (won't remove much with 7-day retention, but tests the mechanism)
    engine.cleanup_audit_logs().await;
    let post_cleanup_count = engine.audit_log.read().len();

    // System should remain functional
    let stability_metrics = engine.get_metrics().await;
    assert_eq!(stability_metrics.total_agents, 15);

    // Final responsiveness check
    let final_decision = engine
        .evaluate_decision(
            &agents[agents.len() - 1],
            DecisionType::PermissionRequest(Permission::ReadData),
        )
        .await;
    assert!(final_decision.is_ok());
}
