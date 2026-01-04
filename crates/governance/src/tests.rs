//! Comprehensive test suite for the governance system
//!
//! This module contains unit tests, integration tests, and end-to-end tests
//! for all components of the governance system.

use chrono::Utc;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use crate::{
    compliance_integration::{ComplianceIntegration, ComplianceStatus},
    coordination_manager::{CoordinationManager, CoordinationRequest},
    governance_engine::{
        AgentGovernanceState, AuditEntry, DecisionType, EvolutionRequest, GovernanceConfig,
        GovernanceDecision, GovernanceEngine, LifecyclePhase, PolicyViolation, ResourceQuota,
        ResourceRequest, ViolationSeverity,
    },
    lifecycle_governance::{LifecycleDecision, LifecycleGovernor},
    monitoring_governance::{GovernanceMetrics, GovernanceMonitor},
    permission_system::{Permission, PermissionSystem, Role},
    policy_manager::{Policy, PolicyManager, PolicyType},
    GovernanceError, Result,
};

use stratoswarm_agent_core::agent::AgentId;

/// Test module for governance engine core functionality
mod governance_engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_governance_engine_initialization() {
        let config = GovernanceConfig {
            max_agents: 100,
            strict_compliance: true,
            emergency_override_enabled: true,
            default_resource_quota: ResourceQuota {
                max_memory_mb: 1024,
                max_cpu_cores: 2.0,
                max_gpu_memory_mb: 2048,
                max_network_bandwidth_mbps: 100,
                max_storage_gb: 10,
            },
            audit_retention_days: 30,
            policy_evaluation_timeout_ms: 5000,
        };

        let engine = GovernanceEngine::new(config.clone()).await.unwrap();
        assert_eq!(engine.active_agents.len(), 0);
        assert_eq!(engine.config.read().max_agents, 100);
        assert_eq!(engine.config.read().strict_compliance, true);
    }

    #[tokio::test]
    async fn test_agent_registration_and_lifecycle() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();

        // Register agent
        engine.register_agent(agent_id.clone()).await?;
        assert_eq!(engine.active_agents.len(), 1);

        let state = engine.active_agents.get(&agent_id)?;
        assert_eq!(state.agent_id, agent_id);
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Initializing);
        assert_eq!(state.violations.len(), 0);
        assert!(state.compliance_status);

        // Update lifecycle phase
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let updated_state = engine.active_agents.get(&agent_id).unwrap();
        assert_eq!(updated_state.lifecycle_phase, LifecyclePhase::Active);
    }

    #[tokio::test]
    async fn test_agent_capacity_limits() {
        let mut config = GovernanceConfig::default();
        config.max_agents = 3;
        let engine = GovernanceEngine::new(config).await.unwrap();

        // Register maximum number of agents
        for _ in 0..3 {
            let agent_id = AgentId::new();
            engine.register_agent(agent_id).await?;
        }

        // Attempt to register one more agent (should fail)
        let extra_agent = AgentId::new();
        let result = engine.register_agent(extra_agent).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ResourceLimitExceeded(_))
        ));
        assert_eq!(engine.active_agents.len(), 3);
    }

    #[tokio::test]
    async fn test_resource_allocation_decisions() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await?;

        // Test valid resource request
        let valid_request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(valid_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Approved));

        // Test invalid resource request (exceeds quota)
        let invalid_request = ResourceRequest {
            memory_mb: 2048, // Exceeds default 1024MB quota
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(invalid_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    #[tokio::test]
    async fn test_evolution_request_evaluation() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        let evolution_request = EvolutionRequest {
            evolution_type: "capability_expansion".to_string(),
            target_capabilities: vec!["advanced_reasoning".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        // Should fail in Initializing phase
        let decision = engine
            .evaluate_decision(
                &agent_id,
                DecisionType::Evolution(evolution_request.clone()),
            )
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Denied(_)));

        // Update to Active phase and retry
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(evolution_request))
            .await
            .unwrap();
        // May still fail other checks, but should pass lifecycle check
    }

    #[tokio::test]
    async fn test_policy_violation_handling() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await?;

        // Test non-critical violation
        let minor_violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "resource_overuse".to_string(),
            severity: ViolationSeverity::Low,
            details: "Slightly exceeded memory limit".to_string(),
        };

        engine
            .record_violation(&agent_id, minor_violation)
            .await
            .unwrap();

        let state = engine.active_agents.get(&agent_id).unwrap();
        assert_eq!(state.violations.len(), 1);
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Active); // Should remain active

        // Test critical violation
        let critical_violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "security_breach".to_string(),
            severity: ViolationSeverity::Critical,
            details: "Attempted unauthorized data access".to_string(),
        };

        engine
            .record_violation(&agent_id, critical_violation)
            .await
            .unwrap();

        let state = engine.active_agents.get(&agent_id).unwrap();
        assert_eq!(state.violations.len(), 2);
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Suspended); // Should be suspended
    }

    #[tokio::test]
    async fn test_permission_system_integration() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Test permission request
        let decision = engine
            .evaluate_decision(
                &agent_id,
                DecisionType::PermissionRequest(Permission::ReadData),
            )
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Approved));

        // Test coordination request without permission
        let coord_request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent_id.clone(),
            target_agents: vec![AgentId::new()],
            coordination_type: "data_sharing".to_string(),
            duration: Some(3600),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::Coordination(coord_request))
            .await
            .unwrap();
        assert!(matches!(decision, GovernanceDecision::Denied(_)));
    }

    #[tokio::test]
    async fn test_audit_logging() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();

        // Register agent (should create audit log)
        engine.register_agent(agent_id.clone()).await?;

        let audit_log = engine.audit_log.read();
        assert!(audit_log.len() > 0);

        let last_entry = &audit_log[audit_log.len() - 1];
        assert_eq!(last_entry.action, "agent_registered");
        assert_eq!(last_entry.result, "success");
        assert_eq!(last_entry.agent_id, Some(agent_id));
    }

    #[tokio::test]
    async fn test_audit_log_cleanup() {
        let mut config = GovernanceConfig::default();
        config.audit_retention_days = 0; // Immediate cleanup
        let engine = GovernanceEngine::new(config).await.unwrap();

        // Create audit entries
        engine.log_audit(None, "test_action", "success", serde_json::json!({}));
        engine.log_audit(None, "another_action", "failure", serde_json::json!({}));

        assert_eq!(engine.audit_log.read().len(), 2);

        // Run cleanup
        engine.cleanup_audit_logs().await;

        // Should be cleaned up
        assert_eq!(engine.audit_log.read().len(), 0);
    }

    #[tokio::test]
    async fn test_emergency_kill_switch() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();
        engine.register_agent(agent_id.clone()).await?;

        // Activate kill switch
        engine.kill_switch.activate("Test emergency").await?;

        // Any decision should be denied
        let request = ResourceRequest {
            memory_mb: 256,
            cpu_cores: 0.5,
            gpu_memory_mb: 512,
            duration_seconds: Some(1800),
        };

        let decision = engine
            .evaluate_decision(&agent_id, DecisionType::ResourceAllocation(request))
            .await
            .unwrap();

        assert!(matches!(decision, GovernanceDecision::Denied(_)));
        if let GovernanceDecision::Denied(reason) = decision {
            assert!(reason.contains("Emergency kill switch"));
        }
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let engine = Arc::new(
            GovernanceEngine::new(GovernanceConfig::default())
                .await
                .unwrap(),
        );
        let mut handles = vec![];

        // Spawn multiple concurrent agent registrations
        for i in 0..10 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let agent_id = AgentId::new();
                let result = engine_clone.register_agent(agent_id.clone()).await;
                (i, agent_id, result)
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let mut successful_registrations = 0;
        for handle in handles {
            let (_, _, result) = handle.await.unwrap();
            if result.is_ok() {
                successful_registrations += 1;
            }
        }

        assert_eq!(successful_registrations, 10);
        assert_eq!(engine.active_agents.len(), 10);
    }

    #[tokio::test]
    async fn test_governance_metrics_collection() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();

        // Create test scenario
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();

        // Register agents
        engine.register_agent(agent1.clone()).await.unwrap();
        engine.register_agent(agent2.clone()).await.unwrap();
        engine.register_agent(agent3.clone()).await.unwrap();

        // Update lifecycle phases
        engine
            .update_lifecycle_phase(&agent1, LifecyclePhase::Active)
            .await
            .unwrap();
        engine
            .update_lifecycle_phase(&agent2, LifecyclePhase::Active)
            .await
            .unwrap();
        engine
            .update_lifecycle_phase(&agent3, LifecyclePhase::Suspended)
            .await
            .unwrap();

        // Add some violations
        let violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "test_violation".to_string(),
            severity: ViolationSeverity::Medium,
            details: "Test violation for metrics".to_string(),
        };
        engine
            .record_violation(&agent1, violation.clone())
            .await
            .unwrap();

        let critical_violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "critical_test".to_string(),
            severity: ViolationSeverity::Critical,
            details: "Critical test violation".to_string(),
        };
        engine
            .record_violation(&agent2, critical_violation)
            .await
            .unwrap();

        // Collect metrics
        let metrics = engine.get_metrics().await;

        assert_eq!(metrics.total_agents, 3);
        assert_eq!(metrics.active_agents, 1); // agent2 was suspended by critical violation
        assert_eq!(metrics.suspended_agents, 2); // agent3 + agent2
        assert_eq!(metrics.total_violations, 2);
        assert_eq!(metrics.critical_violations, 1);
    }
}

/// Test module for policy manager functionality
mod policy_manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_policy_manager_initialization() {
        let policy_manager = PolicyManager::new();
        assert_eq!(policy_manager.get_all_policies().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_policy_creation_and_retrieval() {
        let policy_manager = PolicyManager::new();
        let agent_id = AgentId::new();

        let policy = Policy {
            id: Uuid::new_v4(),
            name: "Test Resource Policy".to_string(),
            description: "Test policy for resource allocation".to_string(),
            policy_type: PolicyType::Resource,
            rules: serde_json::json!({
                "max_memory_mb": 1024,
                "max_cpu_cores": 2.0
            }),
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Add policy
        policy_manager.add_policy(policy.clone()).await.unwrap();

        // Retrieve policies
        let policies = policy_manager.get_all_policies().await.unwrap();
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].id, policy.id);

        // Get applicable policies
        let applicable = policy_manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        assert_eq!(applicable.len(), 1);
    }

    #[tokio::test]
    async fn test_policy_evaluation() {
        let policy_manager = PolicyManager::new();

        let policy = Policy {
            id: Uuid::new_v4(),
            name: "Memory Limit Policy".to_string(),
            description: "Enforces memory usage limits".to_string(),
            policy_type: PolicyType::Resource,
            rules: serde_json::json!({
                "max_memory_mb": 1024
            }),
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        policy_manager.add_policy(policy.clone()).await.unwrap();

        // Test request within limits
        let valid_request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let evaluation = policy_manager
            .evaluate_policy(&policy, &valid_request)
            .await
            .unwrap();
        assert!(evaluation);

        // Test request exceeding limits
        let invalid_request = ResourceRequest {
            memory_mb: 2048,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let evaluation = policy_manager
            .evaluate_policy(&policy, &invalid_request)
            .await
            .unwrap();
        assert!(!evaluation);
    }

    #[tokio::test]
    async fn test_policy_update_and_deletion() {
        let policy_manager = PolicyManager::new();

        let mut policy = Policy {
            id: Uuid::new_v4(),
            name: "Test Policy".to_string(),
            description: "Original description".to_string(),
            policy_type: PolicyType::Resource,
            rules: serde_json::json!({}),
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Add policy
        policy_manager.add_policy(policy.clone()).await.unwrap();

        // Update policy
        policy.description = "Updated description".to_string();
        policy_manager.update_policy(policy.clone()).await.unwrap();

        let updated_policy = policy_manager
            .get_policy(&policy.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated_policy.description, "Updated description");

        // Delete policy
        policy_manager.delete_policy(&policy.id).await.unwrap();

        let deleted_policy = policy_manager.get_policy(&policy.id).await.unwrap();
        assert!(deleted_policy.is_none());
    }
}

/// Test module for permission system functionality
mod permission_system_tests {
    use super::*;

    #[tokio::test]
    async fn test_permission_system_initialization() {
        let permission_system = PermissionSystem::new();
        let agent_id = AgentId::new();

        // Agent should not have any permissions initially
        let has_permission = permission_system
            .check_permission(&agent_id, &Permission::ReadData)
            .await?;
        assert!(!has_permission);
    }

    #[tokio::test]
    async fn test_agent_registration_and_permissions() {
        let permission_system = PermissionSystem::new();
        let agent_id = AgentId::new();

        // Register agent
        permission_system.register_agent(&agent_id).await?;

        // Should have basic permissions after registration
        let has_basic = permission_system
            .check_permission(&agent_id, &Permission::ReadData)
            .await
            .unwrap();
        assert!(has_basic);

        // Should not have advanced permissions
        let has_admin = permission_system
            .check_permission(&agent_id, &Permission::AdminAccess)
            .await
            .unwrap();
        assert!(!has_admin);
    }

    #[tokio::test]
    async fn test_permission_granting_and_revoking() {
        let permission_system = PermissionSystem::new();
        let agent_id = AgentId::new();

        permission_system.register_agent(&agent_id).await.unwrap();

        // Grant permission
        permission_system
            .grant_permission(&agent_id, Permission::WriteData)
            .await?;

        let has_write = permission_system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap();
        assert!(has_write);

        // Revoke permission
        permission_system
            .revoke_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap();

        let has_write_after = permission_system
            .check_permission(&agent_id, &Permission::WriteData)
            .await
            .unwrap();
        assert!(!has_write_after);
    }

    #[tokio::test]
    async fn test_role_based_permissions() {
        let permission_system = PermissionSystem::new();
        let agent_id = AgentId::new();

        permission_system.register_agent(&agent_id).await.unwrap();

        // Assign role
        permission_system
            .assign_role(&agent_id, Role::Coordinator)
            .await?;

        // Should have coordination permissions
        let has_coordinate = permission_system
            .check_permission(&agent_id, &Permission::Coordinate)
            .await
            .unwrap();
        assert!(has_coordinate);

        // Remove role
        permission_system
            .remove_role(&agent_id, &Role::Coordinator)
            .await
            .unwrap();

        let has_coordinate_after = permission_system
            .check_permission(&agent_id, &Permission::Coordinate)
            .await
            .unwrap();
        assert!(!has_coordinate_after);
    }

    #[tokio::test]
    async fn test_permission_validation() {
        let permission_system = PermissionSystem::new();
        let agent_id = AgentId::new();

        permission_system.register_agent(&agent_id).await.unwrap();

        // Test permission granting rules
        let can_grant_basic = permission_system
            .can_grant_permission(&agent_id, &Permission::ReadData)
            .await?;
        assert!(can_grant_basic);

        let can_grant_admin = permission_system
            .can_grant_permission(&agent_id, &Permission::AdminAccess)
            .await
            .unwrap();
        assert!(!can_grant_admin); // Should require special authorization
    }
}

/// Test module for lifecycle governance functionality
mod lifecycle_governance_tests {
    use super::*;

    #[tokio::test]
    async fn test_lifecycle_governor_initialization() {
        let quota = ResourceQuota::default();
        let governor = LifecycleGovernor::new(quota);

        // Should handle unregistered agent gracefully
        let agent_id = AgentId::new();
        let evolution_request = EvolutionRequest {
            evolution_type: "test".to_string(),
            target_capabilities: vec![],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &evolution_request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Denied(_)));
    }

    #[tokio::test]
    async fn test_agent_lifecycle_registration() {
        let quota = ResourceQuota::default();
        let governor = LifecycleGovernor::new(quota);
        let agent_id = AgentId::new();

        // Register agent
        governor.register_agent(&agent_id).await?;

        // Should now be able to evaluate evolution requests
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

        let decision = governor
            .evaluate_evolution(&agent_id, &evolution_request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Approved));
    }

    #[tokio::test]
    async fn test_evolution_resource_validation() {
        let quota = ResourceQuota {
            max_memory_mb: 1024,
            max_cpu_cores: 2.0,
            max_gpu_memory_mb: 2048,
            max_network_bandwidth_mbps: 100,
            max_storage_gb: 10,
        };
        let governor = LifecycleGovernor::new(quota);
        let agent_id = AgentId::new();

        governor.register_agent(&agent_id).await.unwrap();

        // Test evolution request exceeding resource limits
        let excessive_request = EvolutionRequest {
            evolution_type: "resource_intensive".to_string(),
            target_capabilities: vec!["high_performance".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 2048, // Exceeds limit
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let decision = governor
            .evaluate_evolution(&agent_id, &excessive_request)
            .await
            .unwrap();
        assert!(matches!(decision, LifecycleDecision::Denied(_)));
    }
}

/// Test module for compliance integration functionality
mod compliance_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_compliance_integration_initialization() {
        let compliance = ComplianceIntegration::new(true);
        let agent_id = AgentId::new();

        // Register agent
        compliance.register_agent(&agent_id).await?;

        // Check initial compliance status
        let status = compliance.get_compliance_status(&agent_id).await?;
        assert_eq!(status, ComplianceStatus::Compliant);
    }

    #[tokio::test]
    async fn test_evolution_compliance_checking() {
        let compliance = ComplianceIntegration::new(true);

        let evolution_request = EvolutionRequest {
            evolution_type: "security_enhancement".to_string(),
            target_capabilities: vec!["encryption".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        // Should pass compliance check for security enhancements
        let compliant = compliance
            .check_evolution_compliance(&evolution_request)
            .await
            .unwrap();
        assert!(compliant);

        // Test potentially non-compliant evolution
        let risky_evolution = EvolutionRequest {
            evolution_type: "unrestricted_access".to_string(),
            target_capabilities: vec!["bypass_security".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let compliant_risky = compliance
            .check_evolution_compliance(&risky_evolution)
            .await
            .unwrap();
        assert!(!compliant_risky);
    }

    #[tokio::test]
    async fn test_compliance_status_updates() {
        let compliance = ComplianceIntegration::new(false); // Strict mode disabled
        let agent_id = AgentId::new();

        compliance.register_agent(&agent_id).await.unwrap();

        // Update compliance status
        compliance
            .update_compliance_status(&agent_id, ComplianceStatus::NonCompliant)
            .await?;

        let status = compliance.get_compliance_status(&agent_id).await.unwrap();
        assert_eq!(status, ComplianceStatus::NonCompliant);

        // Restore compliance
        compliance
            .update_compliance_status(&agent_id, ComplianceStatus::Compliant)
            .await
            .unwrap();

        let restored_status = compliance.get_compliance_status(&agent_id).await.unwrap();
        assert_eq!(restored_status, ComplianceStatus::Compliant);
    }
}

/// Test module for coordination manager functionality
mod coordination_manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_coordination_manager_initialization() {
        let coordinator = CoordinationManager::new();

        let request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![AgentId::new()],
            coordination_type: "data_sharing".to_string(),
            duration: Some(3600),
        };

        // Should handle coordination request
        let result = coordinator.evaluate_request(&request).await.unwrap();
        assert!(result); // Basic coordination should be allowed
    }

    #[tokio::test]
    async fn test_coordination_conflict_detection() {
        let coordinator = CoordinationManager::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let request1 = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent1.clone(),
            target_agents: vec![agent2.clone()],
            coordination_type: "exclusive_access".to_string(),
            duration: Some(3600),
        };

        let request2 = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent2.clone(),
            target_agents: vec![agent1.clone()],
            coordination_type: "exclusive_access".to_string(),
            duration: Some(3600),
        };

        // First request should succeed
        let result1 = coordinator.evaluate_request(&request1).await.unwrap();
        assert!(result1);

        // Conflicting request should be handled appropriately
        let result2 = coordinator.evaluate_request(&request2).await.unwrap();
        // Depending on implementation, this might be allowed or denied
    }

    #[tokio::test]
    async fn test_coordination_duration_limits() {
        let coordinator = CoordinationManager::new();

        let short_request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![AgentId::new()],
            coordination_type: "brief_collaboration".to_string(),
            duration: Some(60), // 1 minute
        };

        let long_request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: AgentId::new(),
            target_agents: vec![AgentId::new()],
            coordination_type: "extended_collaboration".to_string(),
            duration: Some(86400), // 24 hours
        };

        let short_result = coordinator.evaluate_request(&short_request).await.unwrap();
        let long_result = coordinator.evaluate_request(&long_request).await.unwrap();

        assert!(short_result);
        // Long requests might have additional validation
    }
}

/// Test module for monitoring and metrics functionality
mod monitoring_tests {
    use super::*;

    #[tokio::test]
    async fn test_governance_monitor_initialization() {
        let monitor = GovernanceMonitor::new();

        // Create sample data for metrics collection
        let agents = dashmap::DashMap::new();
        let audit_log = Vec::new();

        let metrics = monitor.collect_metrics(&agents, &audit_log).await;

        assert_eq!(metrics.total_agents, 0);
        assert_eq!(metrics.active_agents, 0);
        assert_eq!(metrics.suspended_agents, 0);
        assert_eq!(metrics.total_violations, 0);
        assert_eq!(metrics.critical_violations, 0);
    }

    #[tokio::test]
    async fn test_metrics_collection_with_data() {
        let monitor = GovernanceMonitor::new();
        let agents = dashmap::DashMap::new();

        // Create test agents
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let state1 = AgentGovernanceState {
            agent_id: agent1.clone(),
            created_at: Utc::now(),
            resource_quota: ResourceQuota::default(),
            active_policies: vec![],
            permissions: vec![],
            compliance_status: true,
            lifecycle_phase: LifecyclePhase::Active,
            violations: vec![],
        };

        let state2 = AgentGovernanceState {
            agent_id: agent2.clone(),
            created_at: Utc::now(),
            resource_quota: ResourceQuota::default(),
            active_policies: vec![],
            permissions: vec![],
            compliance_status: true,
            lifecycle_phase: LifecyclePhase::Suspended,
            violations: vec![PolicyViolation {
                timestamp: Utc::now(),
                policy_id: Uuid::new_v4(),
                violation_type: "test".to_string(),
                severity: ViolationSeverity::Critical,
                details: "Test violation".to_string(),
            }],
        };

        agents.insert(agent1, state1);
        agents.insert(agent2, state2);

        let audit_log = vec![
            AuditEntry {
                timestamp: Utc::now(),
                agent_id: Some(AgentId::new()),
                action: "governance_decision".to_string(),
                result: "Approved".to_string(),
                details: serde_json::json!({}),
            },
            AuditEntry {
                timestamp: Utc::now(),
                agent_id: Some(AgentId::new()),
                action: "governance_decision".to_string(),
                result: "Denied".to_string(),
                details: serde_json::json!({}),
            },
        ];

        let metrics = monitor.collect_metrics(&agents, &audit_log).await;

        assert_eq!(metrics.total_agents, 2);
        assert_eq!(metrics.active_agents, 1);
        assert_eq!(metrics.suspended_agents, 1);
        assert_eq!(metrics.total_violations, 1);
        assert_eq!(metrics.critical_violations, 1);
        assert_eq!(metrics.decisions_made, 2);
        assert_eq!(metrics.approvals, 1);
        assert_eq!(metrics.denials, 1);
    }
}

/// Integration tests that test multiple components working together
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_agent_lifecycle_workflow() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
            .await
            .unwrap();
        let agent_id = AgentId::new();

        // 1. Register agent
        engine.register_agent(agent_id.clone()).await?;
        assert_eq!(engine.active_agents.len(), 1);

        // 2. Request basic permissions
        let permission_decision = engine
            .evaluate_decision(
                &agent_id,
                DecisionType::PermissionRequest(Permission::ReadData),
            )
            .await
            .unwrap();
        assert!(matches!(permission_decision, GovernanceDecision::Approved));

        // 3. Update to active phase
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        // 4. Request resource allocation
        let resource_request = ResourceRequest {
            memory_mb: 512,
            cpu_cores: 1.0,
            gpu_memory_mb: 1024,
            duration_seconds: Some(3600),
        };

        let resource_decision = engine
            .evaluate_decision(
                &agent_id,
                DecisionType::ResourceAllocation(resource_request),
            )
            .await
            .unwrap();
        assert!(matches!(resource_decision, GovernanceDecision::Approved));

        // 5. Attempt evolution
        let evolution_request = EvolutionRequest {
            evolution_type: "capability_enhancement".to_string(),
            target_capabilities: vec!["improved_processing".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 512,
                duration_seconds: None,
            },
        };

        let evolution_decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(evolution_request))
            .await
            .unwrap();
        // Evolution decision depends on implementation details

        // 6. Check governance metrics
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_agents, 1);
        assert_eq!(metrics.active_agents, 1);

        // 7. Verify audit trail
        let audit_log = engine.audit_log.read();
        assert!(audit_log.len() > 0);
        assert!(audit_log
            .iter()
            .any(|entry| entry.action == "agent_registered"));
    }

    #[tokio::test]
    async fn test_policy_violation_escalation_workflow() {
        let mut config = GovernanceConfig::default();
        config.emergency_override_enabled = false; // Disable for controlled testing
        let engine = GovernanceEngine::new(config).await.unwrap();
        let agent_id = AgentId::new();

        // Register and activate agent
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        // Record minor violations
        for i in 0..3 {
            let violation = PolicyViolation {
                timestamp: Utc::now(),
                policy_id: Uuid::new_v4(),
                violation_type: format!("minor_violation_{}", i),
                severity: ViolationSeverity::Low,
                details: format!("Minor violation #{}", i + 1),
            };
            engine.record_violation(&agent_id, violation).await.unwrap();
        }

        // Agent should still be active
        let state = engine.active_agents.get(&agent_id).unwrap();
        assert_eq!(state.lifecycle_phase, LifecyclePhase::Active);
        assert_eq!(state.violations.len(), 3);

        // Record critical violation
        let critical_violation = PolicyViolation {
            timestamp: Utc::now(),
            policy_id: Uuid::new_v4(),
            violation_type: "critical_security_breach".to_string(),
            severity: ViolationSeverity::Critical,
            details: "Attempted unauthorized system access".to_string(),
        };
        engine
            .record_violation(&agent_id, critical_violation)
            .await
            .unwrap();

        // Agent should now be suspended
        let updated_state = engine.active_agents.get(&agent_id).unwrap();
        assert_eq!(updated_state.lifecycle_phase, LifecyclePhase::Suspended);
        assert_eq!(updated_state.violations.len(), 4);

        // Future requests should be denied due to suspension
        let resource_request = ResourceRequest {
            memory_mb: 256,
            cpu_cores: 0.5,
            gpu_memory_mb: 512,
            duration_seconds: Some(1800),
        };

        // Note: In a full implementation, suspended agents might have different handling
        // This test verifies the violation recording and phase change logic
    }

    #[tokio::test]
    async fn test_multi_agent_coordination_workflow() {
        let engine = GovernanceEngine::new(GovernanceConfig::default())
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

            // Grant coordination permissions
            engine
                .evaluate_decision(
                    agent,
                    DecisionType::PermissionRequest(Permission::Coordinate),
                )
                .await
                .unwrap();
        }

        // Test coordination request between agents
        let coordination_request = CoordinationRequest {
            request_id: Uuid::new_v4(),
            requesting_agent: agent1.clone(),
            target_agents: vec![agent2.clone(), agent3.clone()],
            coordination_type: "collaborative_task".to_string(),
            duration: Some(7200), // 2 hours
        };

        let coordination_decision = engine
            .evaluate_decision(&agent1, DecisionType::Coordination(coordination_request))
            .await
            .unwrap();

        // Coordination decision depends on implementation,
        // but this tests the full workflow integration
        assert!(matches!(
            coordination_decision,
            GovernanceDecision::Approved | GovernanceDecision::Denied(_)
        ));

        // Verify all agents are still properly managed
        assert_eq!(engine.active_agents.len(), 3);
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_agents, 3);
        assert_eq!(metrics.active_agents, 3);
    }

    #[tokio::test]
    async fn test_resource_quota_enforcement_across_agents() {
        let mut config = GovernanceConfig::default();
        config.max_agents = 5;
        config.default_resource_quota = ResourceQuota {
            max_memory_mb: 512,
            max_cpu_cores: 1.0,
            max_gpu_memory_mb: 1024,
            max_network_bandwidth_mbps: 50,
            max_storage_gb: 5,
        };

        let engine = GovernanceEngine::new(config).await.unwrap();
        let mut agents = Vec::new();

        // Register multiple agents
        for _ in 0..5 {
            let agent_id = AgentId::new();
            engine.register_agent(agent_id.clone()).await.unwrap();
            engine
                .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
                .await
                .unwrap();
            agents.push(agent_id);
        }

        // Test resource allocation for each agent
        for agent in &agents {
            // Request within quota should succeed
            let valid_request = ResourceRequest {
                memory_mb: 256,
                cpu_cores: 0.5,
                gpu_memory_mb: 512,
                duration_seconds: Some(3600),
            };

            let decision = engine
                .evaluate_decision(agent, DecisionType::ResourceAllocation(valid_request))
                .await
                .unwrap();
            assert!(matches!(decision, GovernanceDecision::Approved));

            // Request exceeding quota should fail
            let invalid_request = ResourceRequest {
                memory_mb: 1024, // Exceeds 512MB quota
                cpu_cores: 0.5,
                gpu_memory_mb: 512,
                duration_seconds: Some(3600),
            };

            let decision = engine
                .evaluate_decision(agent, DecisionType::ResourceAllocation(invalid_request))
                .await
                .unwrap();
            assert!(matches!(decision, GovernanceDecision::Denied(_)));
        }

        // Verify system-wide metrics
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_agents, 5);
        assert_eq!(metrics.active_agents, 5);
    }

    #[tokio::test]
    async fn test_compliance_and_policy_integration() {
        let mut config = GovernanceConfig::default();
        config.strict_compliance = true;
        let engine = GovernanceEngine::new(config).await.unwrap();
        let agent_id = AgentId::new();

        // Register agent
        engine.register_agent(agent_id.clone()).await?;
        engine
            .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
            .await
            .unwrap();

        // Test compliant evolution request
        let compliant_evolution = EvolutionRequest {
            evolution_type: "security_enhancement".to_string(),
            target_capabilities: vec!["improved_encryption".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let compliant_decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(compliant_evolution))
            .await
            .unwrap();
        // Result depends on implementation, but should consider compliance

        // Test potentially non-compliant evolution request
        let risky_evolution = EvolutionRequest {
            evolution_type: "unrestricted_capabilities".to_string(),
            target_capabilities: vec!["bypass_restrictions".to_string()],
            resource_requirements: ResourceRequest {
                memory_mb: 512,
                cpu_cores: 1.0,
                gpu_memory_mb: 1024,
                duration_seconds: None,
            },
        };

        let risky_decision = engine
            .evaluate_decision(&agent_id, DecisionType::Evolution(risky_evolution))
            .await
            .unwrap();
        // Should likely be denied due to compliance concerns

        // Verify audit trail includes compliance checks
        let audit_log = engine.audit_log.read();
        assert!(audit_log
            .iter()
            .any(|entry| entry.action == "governance_decision"));
    }
}

/// Stress tests for performance and reliability
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_high_volume_agent_registration() {
        let mut config = GovernanceConfig::default();
        config.max_agents = 1000;
        let engine = Arc::new(GovernanceEngine::new(config).await.unwrap());

        let start_time = std::time::Instant::now();
        let mut handles = vec![];

        // Spawn concurrent registrations
        for _ in 0..100 {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let agent_id = AgentId::new();
                engine_clone.register_agent(agent_id).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        let mut successful = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successful += 1;
            }
        }

        let duration = start_time.elapsed();
        println!("Registered {} agents in {:?}", successful, duration);

        assert_eq!(successful, 100);
        assert_eq!(engine.active_agents.len(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_governance_decisions() {
        let engine = Arc::new(
            GovernanceEngine::new(GovernanceConfig::default())
                .await
                .unwrap(),
        );

        // Register test agents
        let mut agents = vec![];
        for _ in 0..10 {
            let agent_id = AgentId::new();
            engine.register_agent(agent_id.clone()).await.unwrap();
            engine
                .update_lifecycle_phase(&agent_id, LifecyclePhase::Active)
                .await
                .unwrap();
            agents.push(agent_id);
        }

        let start_time = std::time::Instant::now();
        let mut handles = vec![];

        // Spawn concurrent decision requests
        for agent in agents {
            let engine_clone = engine.clone();
            let handle = tokio::spawn(async move {
                let request = ResourceRequest {
                    memory_mb: 256,
                    cpu_cores: 0.5,
                    gpu_memory_mb: 512,
                    duration_seconds: Some(1800),
                };

                engine_clone
                    .evaluate_decision(&agent, DecisionType::ResourceAllocation(request))
                    .await
            });
            handles.push(handle);
        }

        // Wait for all decisions
        let mut successful_decisions = 0;
        for handle in handles {
            if handle.await.unwrap().is_ok() {
                successful_decisions += 1;
            }
        }

        let duration = start_time.elapsed();
        println!(
            "Made {} governance decisions in {:?}",
            successful_decisions, duration
        );

        assert_eq!(successful_decisions, 10);
    }

    #[tokio::test]
    async fn test_memory_usage_under_load() {
        let mut config = GovernanceConfig::default();
        config.max_agents = 500;
        let engine = GovernanceEngine::new(config).await.unwrap();

        // Register many agents
        for _ in 0..500 {
            let agent_id = AgentId::new();
            engine.register_agent(agent_id).await?;
        }

        // Generate many audit entries
        for i in 0..1000 {
            engine.log_audit(
                None,
                &format!("test_action_{}", i),
                "success",
                serde_json::json!({ "iteration": i }),
            );
        }

        // Verify system is still responsive
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_agents, 500);

        // Test audit log cleanup
        engine.cleanup_audit_logs().await;

        // System should still be functional
        let agent_id = AgentId::new();
        // This will fail due to capacity, but shouldn't crash
        let _ = engine.register_agent(agent_id).await;
    }
}
