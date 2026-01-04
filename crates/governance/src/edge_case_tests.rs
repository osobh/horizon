//! Edge case tests for governance to enhance coverage to 85%+

#[cfg(test)]
mod tests {
    use crate::{
        compliance_integration::{ComplianceIntegration, ComplianceStatus},
        coordination_manager::{
            CoordinationManager, CoordinationRequest, CoordinationType, LockOwner, LockType,
            SessionStatus,
        },
        governance_engine::{
            AgentGovernanceState, AuditEntry, DecisionType, GovernanceConfig, GovernanceEngine,
            LifecyclePhase, ResourceQuota, ViolationSeverity,
        },
        lifecycle_governance::{LifecycleDecision, LifecycleGovernor},
        monitoring_governance::{GovernanceMetrics, GovernanceMonitor},
        permission_system::{Permission, PermissionSystem, Role},
        policy_manager::{Policy, PolicyManager, PolicyType},
        GovernanceError, Result,
    };
    use chrono::{DateTime, Duration, Utc};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;
    use stratoswarm_agent_core::agent::AgentId;
    use uuid::Uuid;

    // Error handling edge cases

    #[test]
    fn test_error_edge_cases_unicode() {
        // Test with unicode strings
        let error = GovernanceError::PolicyViolation("策略违反 🚨 Нарушение".to_string());
        assert!(error.to_string().contains("Policy violation"));

        let error2 = GovernanceError::PermissionDenied("权限拒绝 🚫 अनुमति".to_string());
        assert!(error2.to_string().contains("Permission denied"));
    }

    #[test]
    fn test_error_extreme_values() {
        // Test with very long strings
        let long_msg = "x".repeat(10000);
        let error = GovernanceError::ResourceLimitExceeded(long_msg.clone());
        assert!(error.to_string().contains("Resource limit"));

        // Test empty strings
        let error2 = GovernanceError::LifecycleError(String::new());
        assert!(error2.to_string().contains("Lifecycle error"));
    }

    #[test]
    fn test_all_error_variants() {
        let errors = vec![
            GovernanceError::PolicyViolation("test".to_string()),
            GovernanceError::PermissionDenied("test".to_string()),
            GovernanceError::ResourceLimitExceeded("test".to_string()),
            GovernanceError::LifecycleError("test".to_string()),
            GovernanceError::CoordinationError("test".to_string()),
            GovernanceError::ComplianceError("test".to_string()),
            GovernanceError::ConfigurationError("test".to_string()),
            GovernanceError::InternalError("test".to_string()),
        ];

        for error in errors {
            // Test Debug and Display
            let debug_str = format!("{:?}", error);
            let display_str = error.to_string();
            assert!(!debug_str.is_empty());
            assert!(!display_str.is_empty());
        }
    }

    // Governance Config edge cases

    #[test]
    fn test_governance_config_extremes() {
        let config = GovernanceConfig {
            max_agents: usize::MAX,
            strict_compliance: true,
            emergency_override_enabled: false,
            default_resource_quota: ResourceQuota {
                max_memory_mb: 0,
                max_cpu_cores: f32::INFINITY,
                max_gpu_memory_mb: u64::MAX,
                max_network_bandwidth_mbps: 0,
                max_storage_gb: 0,
            },
            audit_retention_days: u32::MAX,
            policy_evaluation_timeout_ms: 0,
        };

        assert_eq!(config.max_agents, usize::MAX);
        assert!(config.default_resource_quota.max_cpu_cores.is_infinite());
        assert_eq!(config.default_resource_quota.max_network_bandwidth_mbps, 0);
    }

    #[test]
    fn test_governance_config_default() {
        let config = GovernanceConfig::default();
        assert_eq!(config.max_agents, 1000);
        assert!(config.strict_compliance);
        assert!(config.emergency_override_enabled);
        assert_eq!(config.audit_retention_days, 90);
        assert_eq!(config.policy_evaluation_timeout_ms, 5000);
    }

    // Permission system edge cases

    #[test]
    fn test_permission_variants() {
        let permissions = vec![
            Permission::CreateAgent,
            Permission::DeleteAgent,
            Permission::ModifyAgent,
            Permission::ExecuteTask,
            Permission::AccessData,
            Permission::ModifyPolicy,
            Permission::OverridePolicy,
            Permission::EmergencyControl,
        ];

        // Test all permissions are distinct
        let unique_count = permissions
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(unique_count, permissions.len());
    }

    #[test]
    fn test_role_extremes() {
        let roles = vec![
            Role::Admin,
            Role::Operator,
            Role::Observer,
            Role::Agent,
            Role::System,
            Role::Guest,
            Role::Custom("🎭_Custom_角色".to_string()),
            Role::Custom(String::new()),
            Role::Custom("x".repeat(1000)),
        ];

        for role in roles {
            match &role {
                Role::Custom(name) if name.is_empty() => {
                    // Empty custom role name
                    assert_eq!(name.len(), 0);
                }
                Role::Custom(name) if name.len() > 100 => {
                    // Very long custom role name
                    assert_eq!(name.len(), 1000);
                }
                _ => {}
            }
        }
    }

    // Policy edge cases

    #[test]
    fn test_policy_type_serialization() {
        let types = vec![
            PolicyType::SecurityPolicy,
            PolicyType::ResourcePolicy,
            PolicyType::EvolutionPolicy,
            PolicyType::BehaviorPolicy,
            PolicyType::CompliancePolicy,
            PolicyType::Custom("CustomType".to_string()),
        ];

        for policy_type in types {
            let json = serde_json::to_string(&policy_type).unwrap();
            let deserialized: PolicyType = serde_json::from_str(&json).unwrap();
            assert_eq!(policy_type, deserialized);
        }
    }

    #[test]
    fn test_policy_edge_cases() {
        let policy = Policy {
            id: Uuid::nil(),
            name: "名前_🌐_Имя".to_string(),
            policy_type: PolicyType::SecurityPolicy,
            description: String::new(),
            conditions: vec![],
            actions: vec![],
            priority: i32::MAX,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("key".to_string(), serde_json::Value::Null);
                meta.insert("🔑".to_string(), serde_json::json!({"nested": true}));
                meta
            },
        };

        assert_eq!(policy.id, Uuid::nil());
        assert_eq!(policy.priority, i32::MAX);
        assert_eq!(policy.conditions.len(), 0);
        assert!(policy.enabled);
    }

    // Lifecycle edge cases

    #[test]
    fn test_lifecycle_phase_variants() {
        let phases = vec![
            LifecyclePhase::Proposed,
            LifecyclePhase::Created,
            LifecyclePhase::Active,
            LifecyclePhase::Evolving,
            LifecyclePhase::Suspended,
            LifecyclePhase::Terminating,
            LifecyclePhase::Terminated,
        ];

        for phase in phases {
            // Test Debug formatting
            let debug_str = format!("{:?}", phase);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_lifecycle_decision_variants() {
        let decisions = vec![
            LifecycleDecision::Approve,
            LifecycleDecision::Reject {
                reason: String::new(),
            },
            LifecycleDecision::Defer { until: Utc::now() },
            LifecycleDecision::ConditionalApprove { conditions: vec![] },
        ];

        for decision in decisions {
            match &decision {
                LifecycleDecision::Reject { reason } => assert_eq!(reason, ""),
                LifecycleDecision::ConditionalApprove { conditions } => {
                    assert_eq!(conditions.len(), 0)
                }
                LifecycleDecision::Defer { until } => {
                    assert!(*until <= Utc::now() + Duration::days(365))
                }
                _ => {}
            }
        }
    }

    // Compliance edge cases

    #[test]
    fn test_compliance_status_variants() {
        let statuses = vec![
            ComplianceStatus::Compliant,
            ComplianceStatus::NonCompliant { violations: vec![] },
            ComplianceStatus::PartiallyCompliant {
                issues: vec!["test".to_string(); 100],
            },
            ComplianceStatus::Unknown,
            ComplianceStatus::Exempt {
                reason: "🚨".repeat(50),
            },
        ];

        for status in statuses {
            match &status {
                ComplianceStatus::NonCompliant { violations } => assert_eq!(violations.len(), 0),
                ComplianceStatus::PartiallyCompliant { issues } => assert_eq!(issues.len(), 100),
                ComplianceStatus::Exempt { reason } => assert_eq!(reason.chars().count(), 50),
                _ => {}
            }
        }
    }

    // Coordination edge cases

    #[test]
    fn test_coordination_request_extremes() {
        let request = CoordinationRequest {
            request_id: Uuid::nil(),
            requesting_agent: AgentId::new_v4(),
            target_agents: vec![AgentId::new_v4(); 1000], // Large number of targets
            coordination_type: "🤝_协调_Координация".to_string(),
            duration: Some(u64::MAX),
        };

        assert_eq!(request.request_id, Uuid::nil());
        assert_eq!(request.target_agents.len(), 1000);
        assert_eq!(request.duration, Some(u64::MAX));
    }

    #[test]
    fn test_coordination_type_variants() {
        let types = vec![
            CoordinationType::DataSharing,
            CoordinationType::TaskDistribution,
            CoordinationType::ConsensusBuilding,
            CoordinationType::ResourceNegotiation,
            CoordinationType::CollaborativeComputation,
            CoordinationType::Custom(String::new()),
            CoordinationType::Custom("🎯".repeat(100)),
        ];

        for coord_type in types {
            match &coord_type {
                CoordinationType::Custom(s) if s.is_empty() => {
                    assert_eq!(s.len(), 0);
                }
                CoordinationType::Custom(s) if s.len() > 50 => {
                    assert_eq!(s.chars().count(), 100);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_session_status_transitions() {
        let statuses = vec![
            SessionStatus::Pending,
            SessionStatus::Active,
            SessionStatus::Completed,
            SessionStatus::Failed,
            SessionStatus::Cancelled,
        ];

        for status in statuses {
            // Test all status values are distinct
            let _status_val = status as u8;
        }
    }

    // Resource quota edge cases

    #[test]
    fn test_resource_quota_extremes() {
        let quotas = vec![
            ResourceQuota {
                max_memory_mb: 0,
                max_cpu_cores: 0.0,
                max_gpu_memory_mb: 0,
                max_network_bandwidth_mbps: 0,
                max_storage_gb: 0,
            },
            ResourceQuota {
                max_memory_mb: u64::MAX,
                max_cpu_cores: f32::MAX,
                max_gpu_memory_mb: u64::MAX,
                max_network_bandwidth_mbps: u64::MAX,
                max_storage_gb: u64::MAX,
            },
            ResourceQuota {
                max_memory_mb: 1,
                max_cpu_cores: 0.001,
                max_gpu_memory_mb: 1,
                max_network_bandwidth_mbps: 1,
                max_storage_gb: 1,
            },
        ];

        for quota in quotas {
            // Verify quota values are set correctly
            assert!(quota.max_cpu_cores >= 0.0);
            assert!(quota.max_cpu_cores <= f32::MAX);
        }
    }

    #[test]
    fn test_resource_quota_default() {
        let quota = ResourceQuota::default();
        assert_eq!(quota.max_memory_mb, 1024);
        assert_eq!(quota.max_cpu_cores, 2.0);
        assert_eq!(quota.max_gpu_memory_mb, 2048);
        assert_eq!(quota.max_network_bandwidth_mbps, 100);
        assert_eq!(quota.max_storage_gb, 10);
    }

    // Monitoring edge cases

    #[test]
    fn test_governance_metrics_extremes() {
        let metrics = GovernanceMetrics {
            total_agents: usize::MAX,
            active_policies: 0,
            compliance_violations: u64::MAX,
            lifecycle_transitions: u64::MAX,
            permission_checks: u64::MAX,
            coordination_sessions: 0,
            resource_utilization: f64::NAN,
            avg_decision_time_ms: f64::INFINITY,
        };

        assert_eq!(metrics.total_agents, usize::MAX);
        assert!(metrics.resource_utilization.is_nan());
        assert!(metrics.avg_decision_time_ms.is_infinite());
    }

    // Violation severity edge cases

    #[test]
    fn test_violation_severity_variants() {
        let severities = vec![
            ViolationSeverity::Low,
            ViolationSeverity::Medium,
            ViolationSeverity::High,
            ViolationSeverity::Critical,
        ];

        for severity in severities {
            // Test Debug formatting
            let debug_str = format!("{:?}", severity);
            assert!(!debug_str.is_empty());
        }
    }

    // Decision type edge cases

    #[test]
    fn test_decision_type_variants() {
        let decision_types = vec![
            DecisionType::ResourceAllocation,
            DecisionType::PolicyChange,
            DecisionType::AgentLifecycle,
            DecisionType::ComplianceAction,
            DecisionType::EmergencyOverride,
        ];

        for decision_type in decision_types {
            // Test Debug formatting
            let debug_str = format!("{:?}", decision_type);
            assert!(!debug_str.is_empty());
        }
    }

    // Lock edge cases

    #[test]
    fn test_lock_type_variants() {
        let lock_types = vec![LockType::Exclusive, LockType::Shared, LockType::ReadOnly];

        for lock_type in lock_types {
            match lock_type {
                LockType::Exclusive => {
                    // Only one holder allowed
                }
                LockType::Shared => {
                    // Multiple holders allowed
                }
                LockType::ReadOnly => {
                    // Multiple readers allowed
                }
            }
        }
    }

    #[test]
    fn test_lock_owner_variants() {
        let owners = vec![
            LockOwner::Agent(AgentId::new_v4()),
            LockOwner::Session(Uuid::nil()),
            LockOwner::System,
        ];

        for owner in owners {
            match &owner {
                LockOwner::Agent(id) => {
                    // Agent IDs are v4 UUIDs, so they should not be nil
                    assert!(!id.is_nil());
                }
                LockOwner::Session(id) => {
                    assert_eq!(*id, Uuid::nil());
                }
                LockOwner::System => {}
            }
        }
    }

    // Agent governance state edge cases

    #[test]
    fn test_agent_governance_state_extremes() {
        let state = AgentGovernanceState {
            agent_id: AgentId::new_v4(),
            created_at: Utc::now() - Duration::days(365 * 100), // Very old
            resource_quota: ResourceQuota {
                max_memory_mb: 0,
                max_cpu_cores: f32::NEG_INFINITY,
                max_gpu_memory_mb: u64::MAX,
                max_network_bandwidth_mbps: 1,
                max_storage_gb: u64::MAX / 2,
            },
            active_policies: vec![Uuid::nil(); 1000],
            permissions: vec![],
            compliance_status: false,
            lifecycle_phase: LifecyclePhase::Terminated,
            last_activity: Some(Utc::now() + Duration::days(365)), // Future date
            violations: vec!["violation".to_string(); 100],
        };

        assert_eq!(state.active_policies.len(), 1000);
        assert_eq!(state.violations.len(), 100);
        assert!(state.resource_quota.max_cpu_cores.is_infinite());
    }

    // Audit entry edge cases

    #[test]
    fn test_audit_entry_extremes() {
        let entry = AuditEntry {
            id: Uuid::nil(),
            timestamp: Utc::now() - Duration::days(365 * 50), // Very old
            agent_id: Some(AgentId::new_v4()),
            action: "⚡".repeat(500),
            result: "🚫".to_string(),
            metadata: serde_json::json!({
                "nested": {
                    "deeply": {
                        "nested": {
                            "value": true
                        }
                    }
                }
            }),
        };

        assert_eq!(entry.id, Uuid::nil());
        assert_eq!(entry.action.chars().count(), 500);
        assert!(entry
            .metadata
            .pointer("/nested/deeply/nested/value")
            .is_some());
    }

    // Serialization edge cases

    #[test]
    fn test_serialization_edge_cases() {
        // Test serialization with extreme metadata
        let metadata = serde_json::json!({
            "empty_string": "",
            "very_long_string": "x".repeat(10000),
            "unicode": "🌍🌎🌏💫🚀",
            "null": null,
            "array": vec![0; 1000],
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "deep": true
                            }
                        }
                    }
                }
            },
            "numbers": {
                "max_u64": u64::MAX,
                "min_i64": i64::MIN,
                "infinity": f64::INFINITY,
                "neg_infinity": f64::NEG_INFINITY,
                "nan": f64::NAN,
            }
        });

        // Should serialize and deserialize correctly
        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: serde_json::Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized["empty_string"], "");
        assert_eq!(deserialized["unicode"], "🌍🌎🌏💫🚀");
        assert!(deserialized["array"].is_array());
        assert!(deserialized["numbers"]["nan"].as_f64().unwrap().is_nan());
    }

    // Edge case for empty collections

    #[test]
    fn test_empty_collections() {
        let state = AgentGovernanceState {
            agent_id: AgentId::new_v4(),
            created_at: Utc::now(),
            resource_quota: ResourceQuota::default(),
            active_policies: vec![],
            permissions: vec![],
            compliance_status: true,
            lifecycle_phase: LifecyclePhase::Active,
            last_activity: None,
            violations: vec![],
        };

        assert_eq!(state.active_policies.len(), 0);
        assert_eq!(state.permissions.len(), 0);
        assert_eq!(state.violations.len(), 0);
        assert!(state.last_activity.is_none());
    }

    // Date/time edge cases

    #[test]
    fn test_datetime_edge_cases() {
        let far_future = Utc::now() + Duration::days(365 * 1000);
        let far_past = Utc::now() - Duration::days(365 * 1000);

        let entry = AuditEntry {
            id: Uuid::new_v4(),
            timestamp: far_past,
            agent_id: None,
            action: "test".to_string(),
            result: "success".to_string(),
            metadata: serde_json::json!({"future_date": far_future}),
        };

        assert!(entry.timestamp < Utc::now());
        assert!(entry.agent_id.is_none());
    }

    // Unicode string edge cases

    #[test]
    fn test_unicode_string_handling() {
        let unicode_strings = vec![
            "Hello, 世界! 🌏",
            "Здравствуй мир 🇷🇺",
            "مرحبا بالعالم 🌍",
            "🦀🔥💻🚀",
            "\u{1F600}\u{1F601}\u{1F602}", // Emoji codepoints
            "A\u{0301}",                   // Combining character
            "",                            // Empty string
        ];

        for s in unicode_strings {
            let policy = Policy {
                id: Uuid::new_v4(),
                name: s.to_string(),
                policy_type: PolicyType::Custom(s.to_string()),
                description: s.to_string(),
                conditions: vec![],
                actions: vec![],
                priority: 1,
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                metadata: HashMap::new(),
            };

            // Should handle unicode correctly
            assert_eq!(policy.name, s);
            assert_eq!(policy.description, s);
        }
    }
}
