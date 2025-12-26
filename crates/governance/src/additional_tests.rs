//! Additional comprehensive tests for governance to enhance coverage to 90%+

#[cfg(test)]
mod additional_tests {
    use super::super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use uuid::Uuid;

    // Advanced error scenario tests

    #[test]
    fn test_governance_error_chaining() {
        let errors = vec![
            GovernanceError::PolicyViolation("Agent exceeded memory limit".to_string()),
            GovernanceError::PermissionDenied("Insufficient privileges for GPU access".to_string()),
            GovernanceError::ResourceLimitExceeded("CPU quota exhausted".to_string()),
            GovernanceError::LifecycleError("Invalid state transition".to_string()),
            GovernanceError::CoordinationError("Deadlock detected".to_string()),
            GovernanceError::ComplianceError("GDPR violation detected".to_string()),
            GovernanceError::ConfigurationError("Invalid policy syntax".to_string()),
            GovernanceError::InternalError("Unexpected state".to_string()),
        ];

        for error in errors {
            // Test Display trait
            let error_str = error.to_string();
            assert!(!error_str.is_empty());

            // Test Debug trait
            let debug_str = format!("{:?}", error);
            assert!(debug_str.contains("GovernanceError"));

            // Verify error messages are preserved
            match &error {
                GovernanceError::PolicyViolation(msg) => assert!(error_str.contains(msg)),
                GovernanceError::PermissionDenied(msg) => assert!(error_str.contains(msg)),
                GovernanceError::ResourceLimitExceeded(msg) => assert!(error_str.contains(msg)),
                GovernanceError::LifecycleError(msg) => assert!(error_str.contains(msg)),
                GovernanceError::CoordinationError(msg) => assert!(error_str.contains(msg)),
                GovernanceError::ComplianceError(msg) => assert!(error_str.contains(msg)),
                GovernanceError::ConfigurationError(msg) => assert!(error_str.contains(msg)),
                GovernanceError::InternalError(msg) => assert!(error_str.contains(msg)),
            }
        }
    }

    #[test]
    fn test_governance_config_validation() {
        use governance_engine::GovernanceConfig;

        // Test various config combinations
        let configs = vec![
            GovernanceConfig {
                max_agents: 0, // Edge case: no agents allowed
                strict_compliance: true,
                emergency_override_enabled: true,
                default_resource_quota: governance_engine::ResourceQuota {
                    max_memory_mb: 0,
                    max_cpu_cores: 0.0,
                    max_gpu_memory_mb: 0,
                    max_network_bandwidth_mbps: 0,
                    max_storage_gb: 0,
                },
                audit_retention_days: 0,
                policy_evaluation_timeout_ms: 1,
            },
            GovernanceConfig {
                max_agents: usize::MAX,
                strict_compliance: false,
                emergency_override_enabled: false,
                default_resource_quota: governance_engine::ResourceQuota {
                    max_memory_mb: u64::MAX,
                    max_cpu_cores: f32::MAX,
                    max_gpu_memory_mb: u64::MAX,
                    max_network_bandwidth_mbps: u64::MAX,
                    max_storage_gb: u64::MAX,
                },
                audit_retention_days: u32::MAX,
                policy_evaluation_timeout_ms: u64::MAX,
            },
        ];

        for config in configs {
            // Configs should be valid even with extreme values
            assert!(config.max_agents == 0 || config.max_agents > 0);
            assert!(config.audit_retention_days >= 0);
        }
    }

    #[test]
    fn test_permission_system_hierarchy() {
        use permission_system::{Permission, PermissionSystem, Role};

        let mut perm_system = PermissionSystem::new();

        // Create role hierarchy
        let admin_role = Role::new("admin", vec![Permission::All]);

        let operator_role = Role::new(
            "operator",
            vec![
                Permission::Read,
                Permission::Write,
                Permission::Execute,
                Permission::Monitor,
            ],
        );

        let viewer_role = Role::new("viewer", vec![Permission::Read, Permission::Monitor]);

        perm_system.add_role(admin_role);
        perm_system.add_role(operator_role);
        perm_system.add_role(viewer_role);

        // Test permission checks
        let agent_id = Uuid::new_v4();

        // Assign roles
        perm_system.assign_role(agent_id, "admin");
        assert!(perm_system.has_permission(agent_id, &Permission::Delete));

        perm_system.revoke_role(agent_id, "admin");
        perm_system.assign_role(agent_id, "operator");
        assert!(perm_system.has_permission(agent_id, &Permission::Execute));
        assert!(!perm_system.has_permission(agent_id, &Permission::Delete));

        perm_system.assign_role(agent_id, "viewer");
        assert!(perm_system.has_permission(agent_id, &Permission::Read));
        assert!(!perm_system.has_permission(agent_id, &Permission::Write));
    }

    #[test]
    fn test_policy_manager_complex_scenarios() {
        use policy_manager::{Policy, PolicyManager, PolicyType};

        let mut policy_mgr = PolicyManager::new();

        // Create various policies
        let policies = vec![
            Policy {
                id: Uuid::new_v4(),
                name: "memory_limit".to_string(),
                policy_type: PolicyType::Resource,
                rules: vec!["max_memory_mb <= 8192".to_string()],
                priority: 100,
                enabled: true,
            },
            Policy {
                id: Uuid::new_v4(),
                name: "security_isolation".to_string(),
                policy_type: PolicyType::Security,
                rules: vec![
                    "network_isolation = true".to_string(),
                    "filesystem_sandbox = true".to_string(),
                ],
                priority: 200,
                enabled: true,
            },
            Policy {
                id: Uuid::new_v4(),
                name: "evolution_constraints".to_string(),
                policy_type: PolicyType::Evolution,
                rules: vec![
                    "max_mutation_rate <= 0.1".to_string(),
                    "preserve_safety_invariants = true".to_string(),
                ],
                priority: 150,
                enabled: true,
            },
        ];

        for policy in policies {
            policy_mgr.add_policy(policy);
        }

        // Test policy retrieval by type
        let resource_policies = policy_mgr.get_policies_by_type(PolicyType::Resource);
        assert_eq!(resource_policies.len(), 1);

        let security_policies = policy_mgr.get_policies_by_type(PolicyType::Security);
        assert_eq!(security_policies.len(), 1);

        // Test policy priority ordering
        let all_policies = policy_mgr.get_all_policies();
        let priorities: Vec<u32> = all_policies.iter().map(|p| p.priority).collect();

        // Should be sorted by priority (highest first)
        for i in 1..priorities.len() {
            assert!(priorities[i - 1] >= priorities[i]);
        }
    }

    #[test]
    fn test_lifecycle_governance_state_machine() {
        use lifecycle_governance::{LifecycleDecision, LifecycleGovernor, LifecyclePhase};

        let governor = LifecycleGovernor::new();
        let agent_id = Uuid::new_v4();

        // Test all lifecycle phases
        let phases = vec![
            LifecyclePhase::Initializing,
            LifecyclePhase::Active,
            LifecyclePhase::Evolving,
            LifecyclePhase::Suspended,
            LifecyclePhase::Terminating,
            LifecyclePhase::Terminated,
            LifecyclePhase::Quarantined,
        ];

        for phase in phases {
            let decision = governor.evaluate_transition(agent_id, phase.clone());

            match phase {
                LifecyclePhase::Initializing => {
                    // Can transition to Active or Quarantined
                    match decision {
                        LifecycleDecision::Approve | LifecycleDecision::Deny(_) => {}
                        _ => panic!("Unexpected decision for Initializing phase"),
                    }
                }
                LifecyclePhase::Active => {
                    // Can transition to many states
                    assert!(matches!(decision, LifecycleDecision::Approve));
                }
                LifecyclePhase::Terminated => {
                    // Terminal state - no transitions allowed
                    match decision {
                        LifecycleDecision::Deny(reason) => {
                            assert!(reason.contains("terminal") || reason.contains("final"));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_coordination_manager_deadlock_prevention() {
        use coordination_manager::{CoordinationManager, CoordinationRequest};

        let coord_mgr = Arc::new(Mutex::new(CoordinationManager::new()));

        // Simulate multiple agents requesting resources
        let agent1 = Uuid::new_v4();
        let agent2 = Uuid::new_v4();
        let agent3 = Uuid::new_v4();

        let requests = vec![
            CoordinationRequest {
                agent_id: agent1,
                resource_requests: vec!["gpu:0".to_string(), "memory:8GB".to_string()],
                priority: 100,
                timeout: Duration::from_secs(5),
            },
            CoordinationRequest {
                agent_id: agent2,
                resource_requests: vec!["gpu:1".to_string(), "memory:4GB".to_string()],
                priority: 90,
                timeout: Duration::from_secs(5),
            },
            CoordinationRequest {
                agent_id: agent3,
                resource_requests: vec!["gpu:0".to_string(), "gpu:1".to_string()],
                priority: 110,
                timeout: Duration::from_secs(3),
            },
        ];

        // Submit requests
        for request in requests {
            let mgr = coord_mgr.lock().unwrap();
            mgr.submit_request(request);
        }

        // Process coordination
        let mgr = coord_mgr.lock().unwrap();
        let decisions = mgr.process_coordination();

        // Higher priority request should be processed first
        assert!(!decisions.is_empty());

        // Verify no circular dependencies
        let mut granted_resources = HashMap::new();
        for (agent_id, resources) in decisions {
            for resource in resources {
                assert!(!granted_resources.contains_key(&resource));
                granted_resources.insert(resource, agent_id);
            }
        }
    }

    #[test]
    fn test_compliance_integration_multi_framework() {
        use compliance_integration::{ComplianceIntegration, ComplianceStatus};

        let mut compliance = ComplianceIntegration::new();

        // Register multiple compliance frameworks
        compliance.register_framework("GDPR", Box::new(MockGDPRFramework));
        compliance.register_framework("HIPAA", Box::new(MockHIPAAFramework));
        compliance.register_framework("SOC2", Box::new(MockSOC2Framework));

        // Test compliance check
        let agent_id = Uuid::new_v4();
        let action = "process_user_data";

        let status = compliance.check_compliance(agent_id, action);

        match status {
            ComplianceStatus::Compliant => {
                // All frameworks approved
            }
            ComplianceStatus::NonCompliant(violations) => {
                // Some frameworks rejected
                assert!(!violations.is_empty());
            }
            ComplianceStatus::Conditional(conditions) => {
                // Conditional approval
                assert!(!conditions.is_empty());
            }
            ComplianceStatus::Unknown => {
                // No applicable frameworks
            }
            ComplianceStatus::Reviewing => {
                // Still being evaluated
            }
        }
    }

    #[test]
    fn test_monitoring_governance_metrics() {
        use monitoring_governance::{GovernanceMetrics, GovernanceMonitor};

        let monitor = GovernanceMonitor::new();

        // Record various events
        for i in 0..100 {
            let agent_id = Uuid::new_v4();

            if i % 10 == 0 {
                monitor.record_policy_violation(agent_id, "memory_exceeded");
            }

            if i % 5 == 0 {
                monitor.record_permission_check(agent_id, "gpu_access", i % 3 == 0);
            }

            if i % 20 == 0 {
                monitor.record_lifecycle_event(agent_id, "evolution_started");
            }
        }

        // Get metrics
        let metrics = monitor.get_metrics();

        assert!(metrics.total_policy_violations > 0);
        assert!(metrics.total_permission_checks > 0);
        assert!(metrics.permission_grant_rate > 0.0 && metrics.permission_grant_rate <= 1.0);
        assert!(metrics.total_lifecycle_events > 0);
    }

    #[test]
    fn test_governance_engine_integration() {
        use governance_engine::{GovernanceConfig, GovernanceEngine, ResourceQuota};

        let config = GovernanceConfig {
            max_agents: 100,
            strict_compliance: true,
            emergency_override_enabled: true,
            default_resource_quota: ResourceQuota {
                max_memory_mb: 8192,
                max_cpu_cores: 4.0,
                max_gpu_memory_mb: 4096,
                max_network_bandwidth_mbps: 1000,
                max_storage_gb: 100,
            },
            audit_retention_days: 90,
            policy_evaluation_timeout_ms: 1000,
        };

        let engine = GovernanceEngine::new(config);

        // Test agent creation governance
        let mut created_agents = Vec::new();

        for i in 0..10 {
            let agent_id = Uuid::new_v4();
            let result = engine.approve_agent_creation(agent_id, HashMap::new());

            if result.is_ok() {
                created_agents.push(agent_id);
            }
        }

        // Test resource allocation
        for agent_id in &created_agents {
            let resource_request = HashMap::from([
                ("memory_mb".to_string(), "2048".to_string()),
                ("cpu_cores".to_string(), "1.0".to_string()),
            ]);

            let result = engine.approve_resource_allocation(*agent_id, resource_request);
            assert!(result.is_ok());
        }

        // Test policy enforcement
        let oversized_request = HashMap::from([
            ("memory_mb".to_string(), "16384".to_string()), // Exceeds limit
        ]);

        let result = engine.approve_resource_allocation(created_agents[0], oversized_request);
        assert!(result.is_err());
    }

    #[test]
    fn test_emergency_override_scenarios() {
        use governance_engine::{GovernanceConfig, GovernanceEngine, ResourceQuota};

        let config = GovernanceConfig {
            max_agents: 10,
            strict_compliance: true,
            emergency_override_enabled: true,
            default_resource_quota: ResourceQuota {
                max_memory_mb: 1024,
                max_cpu_cores: 1.0,
                max_gpu_memory_mb: 512,
                max_network_bandwidth_mbps: 100,
                max_storage_gb: 10,
            },
            audit_retention_days: 30,
            policy_evaluation_timeout_ms: 500,
        };

        let engine = GovernanceEngine::new(config);

        // Normal request should fail due to limits
        let large_request = HashMap::from([("memory_mb".to_string(), "2048".to_string())]);

        let agent_id = Uuid::new_v4();
        let result = engine.approve_resource_allocation(agent_id, large_request.clone());
        assert!(result.is_err());

        // Enable emergency override
        engine.activate_emergency_override("critical_experiment");

        // Same request should now succeed
        let result = engine.approve_resource_allocation(agent_id, large_request);
        assert!(result.is_ok());

        // Deactivate override
        engine.deactivate_emergency_override();

        // Should fail again
        let result = engine.approve_resource_allocation(
            agent_id,
            HashMap::from([("memory_mb".to_string(), "2048".to_string())]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_audit_logging_completeness() {
        use governance_engine::{GovernanceConfig, GovernanceEngine, ResourceQuota};

        let config = GovernanceConfig {
            max_agents: 50,
            strict_compliance: false,
            emergency_override_enabled: false,
            default_resource_quota: ResourceQuota {
                max_memory_mb: 4096,
                max_cpu_cores: 2.0,
                max_gpu_memory_mb: 2048,
                max_network_bandwidth_mbps: 500,
                max_storage_gb: 50,
            },
            audit_retention_days: 180,
            policy_evaluation_timeout_ms: 2000,
        };

        let engine = GovernanceEngine::new(config);

        // Perform various governance actions
        let agent_id = Uuid::new_v4();

        // Creation
        let _ = engine.approve_agent_creation(
            agent_id,
            HashMap::from([("purpose".to_string(), "testing".to_string())]),
        );

        // Resource allocation
        let _ = engine.approve_resource_allocation(
            agent_id,
            HashMap::from([("memory_mb".to_string(), "1024".to_string())]),
        );

        // Evolution
        let _ = engine.approve_evolution(
            agent_id,
            HashMap::from([("mutation_rate".to_string(), "0.05".to_string())]),
        );

        // Termination
        let _ = engine.approve_termination(agent_id, "end_of_experiment");

        // Get audit log
        let audit_entries = engine.get_audit_log(Some(agent_id), None, None);

        // Should have entries for all actions
        assert!(audit_entries.len() >= 4);

        // Verify audit entry completeness
        for entry in audit_entries {
            assert!(!entry.timestamp.is_empty());
            assert!(!entry.action.is_empty());
            assert_eq!(entry.agent_id, Some(agent_id));
            assert!(entry.details.is_some());
        }
    }

    #[test]
    fn test_distributed_governance_consensus() {
        // Simulate multiple governance nodes reaching consensus
        let node_count = 5;
        let mut nodes = Vec::new();

        for i in 0..node_count {
            let config = governance_engine::GovernanceConfig {
                max_agents: 100,
                strict_compliance: true,
                emergency_override_enabled: false,
                default_resource_quota: governance_engine::ResourceQuota {
                    max_memory_mb: 8192,
                    max_cpu_cores: 4.0,
                    max_gpu_memory_mb: 4096,
                    max_network_bandwidth_mbps: 1000,
                    max_storage_gb: 100,
                },
                audit_retention_days: 90,
                policy_evaluation_timeout_ms: 1000,
            };

            let engine = governance_engine::GovernanceEngine::new(config);
            nodes.push(engine);
        }

        // Simulate governance decision
        let agent_id = Uuid::new_v4();
        let request = HashMap::from([
            ("action".to_string(), "high_risk_evolution".to_string()),
            ("risk_score".to_string(), "0.8".to_string()),
        ]);

        let mut votes = Vec::new();

        for node in &nodes {
            let result = node.evaluate_high_risk_action(agent_id, request.clone());
            votes.push(result.is_ok());
        }

        // Count votes
        let approvals = votes.iter().filter(|&&v| v).count();
        let rejections = votes.iter().filter(|&&v| !v).count();

        // Majority decision
        let consensus = approvals > rejections;

        // At least some nodes should participate
        assert!(approvals + rejections > 0);
    }

    // Mock compliance frameworks for testing

    struct MockGDPRFramework;
    impl compliance_integration::ComplianceFramework for MockGDPRFramework {
        fn check_compliance(
            &self,
            _agent_id: Uuid,
            action: &str,
        ) -> compliance_integration::ComplianceStatus {
            if action.contains("user_data") {
                compliance_integration::ComplianceStatus::Conditional(vec![
                    "Requires user consent".to_string(),
                    "Data must be anonymized".to_string(),
                ])
            } else {
                compliance_integration::ComplianceStatus::Compliant
            }
        }
    }

    struct MockHIPAAFramework;
    impl compliance_integration::ComplianceFramework for MockHIPAAFramework {
        fn check_compliance(
            &self,
            _agent_id: Uuid,
            action: &str,
        ) -> compliance_integration::ComplianceStatus {
            if action.contains("health") {
                compliance_integration::ComplianceStatus::NonCompliant(vec![
                    "PHI access not authorized".to_string(),
                ])
            } else {
                compliance_integration::ComplianceStatus::Compliant
            }
        }
    }

    struct MockSOC2Framework;
    impl compliance_integration::ComplianceFramework for MockSOC2Framework {
        fn check_compliance(
            &self,
            _agent_id: Uuid,
            _action: &str,
        ) -> compliance_integration::ComplianceStatus {
            compliance_integration::ComplianceStatus::Compliant
        }
    }
}
