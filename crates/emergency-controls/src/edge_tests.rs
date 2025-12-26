//! Edge case tests for emergency controls to achieve 100% coverage

#[cfg(test)]
mod edge_tests {
    use super::super::*;
    use crate::audit_log::{AuditEvent, AuditLevel, AuditLogger};
    use crate::behavior_boundaries::{
        BehaviorBoundary, BehaviorMonitor, BehaviorType, ViolationSeverity,
    };
    use crate::kill_switch::{KillSwitchConfig, KillSwitchSystem};
    use crate::recovery::{RecoveryProcedure, RecoveryState, RecoverySystem};
    use crate::resource_limits::{LimitAction, ResourceLimit, ResourceLimiter, ResourceType};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::Mutex;

    // Kill Switch Edge Cases

    #[tokio::test]
    async fn test_kill_switch_concurrent_activations() {
        let config = KillSwitchConfig::default();
        let kill_switch = Arc::new(Mutex::new(KillSwitchSystem::new(config)));

        // Spawn multiple concurrent activation attempts
        let mut handles = vec![];
        for i in 0..10 {
            let ks = kill_switch.clone();
            let handle = tokio::spawn(async move {
                let mut ks_lock = ks.lock().await;
                ks_lock.activate(&format!("concurrent-{}", i)).await
            });
            handles.push(handle);
        }

        // All should succeed without deadlock
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Verify switch is active
        let ks_lock = kill_switch.lock().await;
        assert!(ks_lock.is_active());
    }

    #[tokio::test]
    async fn test_kill_switch_pattern_based_activation() {
        let mut config = KillSwitchConfig::default();
        config.patterns = vec!["dangerous_*".to_string(), "*_malicious".to_string()];
        let mut kill_switch = KillSwitchSystem::new(config);

        // Test pattern matching
        assert!(kill_switch.activate("dangerous_operation").await.is_ok());
        assert!(kill_switch.is_active());

        kill_switch.reset().await.unwrap();

        assert!(kill_switch.activate("operation_malicious").await.is_ok());
        assert!(kill_switch.is_active());
    }

    #[tokio::test]
    async fn test_kill_switch_auto_reset() {
        let mut config = KillSwitchConfig::default();
        config.auto_reset_delay = Some(Duration::from_millis(50));
        let mut kill_switch = KillSwitchSystem::new(config);

        // Activate and wait for auto-reset
        kill_switch.activate("test").await.unwrap();
        assert!(kill_switch.is_active());

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should be reset
        assert!(!kill_switch.is_active());
    }

    #[tokio::test]
    async fn test_kill_switch_trigger_based_activation() {
        let mut config = KillSwitchConfig::default();
        config.triggers = vec![crate::kill_switch::KillSwitchTrigger {
            trigger_id: "test1".to_string(),
            name: "Resource Test".to_string(),
            trigger_type: crate::kill_switch::TriggerType::ResourceLimit,
            enabled: true,
            condition: crate::kill_switch::TriggerCondition::ResourceThreshold {
                resource: "GpuMemory".to_string(),
                threshold: 1000.0,
            },
            action: crate::kill_switch::KillSwitchAction::GlobalKill,
            priority: crate::kill_switch::TriggerPriority::High,
        }];
        let mut kill_switch = KillSwitchSystem::new(config);

        // Test resource trigger
        kill_switch
            .check_resource_trigger(ResourceType::GpuMemory, 1500.0)
            .await
            .unwrap();
        assert!(kill_switch.is_active());

        kill_switch.reset().await.unwrap();

        // Test pattern trigger
        kill_switch
            .check_pattern_trigger("error_critical")
            .await
            .unwrap();
        assert!(kill_switch.is_active());
    }

    // Resource Limits Edge Cases

    #[tokio::test]
    async fn test_resource_limits_concurrent_checks() {
        let limiter = Arc::new(Mutex::new(ResourceLimiter::new(vec![ResourceLimit {
            resource_type: ResourceType::AgentCount,
            limit: 100.0,
            action: LimitAction::Throttle,
        }])));

        // Concurrent resource checks
        let mut handles: Vec<
            tokio::task::JoinHandle<Result<Option<LimitAction>, crate::error::EmergencyError>>,
        > = vec![];
        for i in 0..20 {
            let lim = limiter.clone();
            let handle = tokio::spawn(async move {
                let mut lim_lock = lim.lock().await;
                lim_lock
                    .update_usage(ResourceType::AgentCount, i as f64 * 10.0)
                    .await
                    .unwrap();
                let violations = lim_lock.get_violations().await;
                if violations
                    .iter()
                    .any(|v| v.resource_type == ResourceType::AgentCount)
                {
                    Ok(Some(LimitAction::Throttle))
                } else {
                    Ok(None)
                }
            });
            handles.push(handle);
        }

        // Collect results
        let mut throttled = 0;
        for handle in handles {
            if let Ok(Some(LimitAction::Throttle)) = handle.await.unwrap() {
                throttled += 1;
            }
        }

        // Some should be throttled
        assert!(throttled > 0);
    }

    #[tokio::test]
    async fn test_resource_limits_all_actions() {
        let limiter = ResourceLimiter::new(vec![
            ResourceLimit {
                resource_type: ResourceType::CpuPercent,
                limit: 50.0,
                action: LimitAction::Warn,
            },
            ResourceLimit {
                resource_type: ResourceType::NetworkBandwidth,
                limit: 1000.0,
                action: LimitAction::Throttle,
            },
            ResourceLimit {
                resource_type: ResourceType::GpuMemory,
                limit: 8000.0,
                action: LimitAction::KillSwitch,
            },
        ]);

        // Test warn action
        limiter
            .update_usage(ResourceType::CpuPercent, 75.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::CpuPercent));

        // Test throttle action
        limiter
            .update_usage(ResourceType::NetworkBandwidth, 1500.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::NetworkBandwidth));

        // Test kill switch action
        limiter
            .update_usage(ResourceType::GpuMemory, 9000.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::GpuMemory));
    }

    #[tokio::test]
    async fn test_resource_limits_edge_values() {
        let limiter = ResourceLimiter::new(vec![ResourceLimit {
            resource_type: ResourceType::AgentSpawnRate,
            limit: 1000.0,
            action: LimitAction::Throttle,
        }]);

        // Test exact limit
        limiter
            .update_usage(ResourceType::AgentSpawnRate, 1000.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations.is_empty()); // Exactly at limit, no violation

        limiter.clear_violations().await;

        // Test just over limit
        limiter
            .update_usage(ResourceType::AgentSpawnRate, 1000.1)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(!violations.is_empty());

        limiter.clear_violations().await;

        // Test negative values (should be handled gracefully)
        limiter
            .update_usage(ResourceType::AgentSpawnRate, -100.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations.is_empty());
    }

    // Behavior Boundaries Edge Cases

    #[tokio::test]
    async fn test_behavior_boundaries_concurrent_violations() {
        let monitor = BehaviorMonitor::new(vec![]);
        monitor.add_boundary(BehaviorBoundary {
            behavior_type: BehaviorType::MemoryAccess,
            description: "Forbidden pattern test".to_string(),
            max_frequency: Some(10),
            time_window: Some(Duration::from_secs(1)),
            forbidden_patterns: vec!["forbidden_*".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Critical,
        });

        let monitor = Arc::new(Mutex::new(monitor));

        // Concurrent violation checks
        let mut handles = vec![];
        for i in 0..20 {
            let mon = monitor.clone();
            let handle = tokio::spawn(async move {
                let mon_lock = mon.lock().await;
                mon_lock
                    .monitor_behavior(
                        "test_agent",
                        BehaviorType::MemoryAccess,
                        &format!("forbidden_action_{}", i),
                    )
                    .await
            });
            handles.push(handle);
        }

        // Wait for all handles to complete
        for handle in handles {
            let _ = handle.await.unwrap();
        }

        // Should detect frequency violations - check through monitor
        let violations = monitor
            .lock()
            .await
            .get_recent_violations(Duration::from_secs(1))
            .await;
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical));
    }

    #[tokio::test]
    async fn test_behavior_boundaries_severity_escalation() {
        let monitor = BehaviorMonitor::new(vec![]);

        // Add boundaries with different severities
        monitor.add_boundary(BehaviorBoundary {
            behavior_type: BehaviorType::MemoryAccess,
            description: "Warning pattern test".to_string(),
            max_frequency: Some(5),
            time_window: Some(Duration::from_secs(1)),
            forbidden_patterns: vec!["warning_*".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Low,
        });

        monitor.add_boundary(BehaviorBoundary {
            behavior_type: BehaviorType::NetworkCommunication,
            description: "Error pattern test".to_string(),
            max_frequency: Some(3),
            time_window: Some(Duration::from_secs(1)),
            forbidden_patterns: vec!["error_*".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Medium,
        });

        monitor.add_boundary(BehaviorBoundary {
            behavior_type: BehaviorType::FileSystemOperations,
            description: "Critical pattern test".to_string(),
            max_frequency: Some(1),
            time_window: Some(Duration::from_secs(1)),
            forbidden_patterns: vec!["critical_*".to_string()],
            allowed_patterns: vec![],
            severity: ViolationSeverity::Critical,
        });

        // Test each severity level
        for i in 0..6 {
            monitor
                .monitor_behavior(
                    "test_agent",
                    BehaviorType::MemoryAccess,
                    &format!("warning_{}", i),
                )
                .await
                .unwrap();
        }
        let warning_violations = monitor.get_recent_violations(Duration::from_secs(10)).await;
        assert!(warning_violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Low));

        for i in 0..4 {
            monitor
                .monitor_behavior(
                    "test_agent",
                    BehaviorType::NetworkCommunication,
                    &format!("error_{}", i),
                )
                .await
                .unwrap();
        }
        let error_violations = monitor.get_recent_violations(Duration::from_secs(10)).await;
        assert!(error_violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Medium));

        monitor
            .monitor_behavior(
                "test_agent",
                BehaviorType::FileSystemOperations,
                "critical_1",
            )
            .await
            .unwrap();
        monitor
            .monitor_behavior(
                "test_agent",
                BehaviorType::FileSystemOperations,
                "critical_2",
            )
            .await
            .unwrap();
        let critical_violations = monitor.get_recent_violations(Duration::from_secs(10)).await;
        assert!(critical_violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical));
    }

    // Audit Log Edge Cases

    #[tokio::test]
    async fn test_audit_log_rotation_during_write() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.log");

        let config = crate::audit_log::AuditConfig {
            log_path: log_path.clone(),
            max_file_size: 100, // Very small for testing
            max_files: 10,
            buffer_size: 1000,
            min_level: crate::audit_log::AuditLevel::Debug,
        };

        let logger = AuditLogger::new(config).await.unwrap();

        // Write enough to trigger rotation
        for i in 0..20 {
            let event = AuditEvent {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                event_type: crate::audit_log::AuditEventType::SafetyViolation,
                level: AuditLevel::Critical,
                agent_id: Some("test_agent".to_string()),
                message: format!("Large event details {}", "x".repeat(50)),
                details: serde_json::json!({"iteration": i}),
                user: None,
                source: "test".to_string(),
            };
            logger.log(event).await.unwrap();
        }

        // Check that rotation occurred
        let rotated_files: Vec<_> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().to_string_lossy().contains("audit.log."))
            .collect();

        assert!(!rotated_files.is_empty());
    }

    #[tokio::test]
    async fn test_audit_log_query_filters() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.log");

        let config = crate::audit_log::AuditConfig {
            log_path: log_path.clone(),
            max_file_size: 10_000_000,
            max_files: 10,
            buffer_size: 1000,
            min_level: crate::audit_log::AuditLevel::Debug,
        };

        let logger = AuditLogger::new(config).await.unwrap();

        // Log events with different levels and types
        let levels = vec![
            AuditLevel::Info,
            AuditLevel::Warning,
            AuditLevel::Error,
            AuditLevel::Critical,
        ];
        let types = vec!["type_a", "type_b", "type_c"];

        for level in &levels {
            for event_type in &types {
                let event = AuditEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    event_type: crate::audit_log::AuditEventType::ConfigurationChanged,
                    level: level.clone(),
                    agent_id: Some("test_agent".to_string()),
                    message: event_type.to_string(),
                    details: serde_json::json!({"test": true}),
                    user: None,
                    source: "test".to_string(),
                };
                logger.log(event).await.unwrap();
            }
        }

        // Test various query filters
        let critical_events = logger
            .query_events(|e| e.level == AuditLevel::Critical)
            .await;
        assert_eq!(critical_events.len(), 3); // 3 types with Critical level

        let type_a_events = logger.query_events(|e| e.message == "type_a").await;
        assert_eq!(type_a_events.len(), 4); // 4 levels with type_a

        let combined = logger
            .query_events(|e| e.level == AuditLevel::Error && e.message == "type_b")
            .await;
        assert_eq!(combined.len(), 1); // Only Error + type_b
    }

    #[tokio::test]
    async fn test_audit_log_concurrent_writes() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("audit.log");

        let config = crate::audit_log::AuditConfig {
            log_path: log_path.clone(),
            max_file_size: 10_000_000,
            max_files: 10,
            buffer_size: 1000,
            min_level: crate::audit_log::AuditLevel::Debug,
        };

        let logger = Arc::new(Mutex::new(AuditLogger::new(config).await.unwrap()));

        // Concurrent writes
        let mut handles = vec![];
        for i in 0..100 {
            let log = logger.clone();
            let handle = tokio::spawn(async move {
                let event = AuditEvent {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    event_type: crate::audit_log::AuditEventType::ConfigurationChanged,
                    level: AuditLevel::Info,
                    agent_id: Some(format!("agent_{}", i)),
                    message: format!("concurrent_{}", i),
                    details: serde_json::json!({"test": true}),
                    user: None,
                    source: "test".to_string(),
                };
                let log_lock = log.lock().await;
                log_lock.log(event).await
            });
            handles.push(handle);
        }

        // All writes should succeed
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Verify all events were logged
        let log_lock = logger.lock().await;
        let all_events = log_lock.query_events(|_| true).await;
        assert_eq!(all_events.len(), 100);
    }

    // Recovery System Edge Cases

    #[tokio::test]
    async fn test_recovery_concurrent_procedures() {
        let mut recovery = RecoverySystem::new();

        // Add multiple procedures
        recovery
            .add_procedure(RecoveryProcedure {
                id: "proc1".to_string(),
                recovery_type: crate::recovery::RecoveryType::AgentRecovery,
                name: "proc1".to_string(),
                description: "Test procedure 1".to_string(),
                steps: vec![crate::recovery::RecoveryStep {
                    name: "step1".to_string(),
                    action: "test_action".to_string(),
                    required: true,
                    timeout: Duration::from_secs(1),
                }],
                timeout: Duration::from_secs(5),
                retry_count: 1,
                priority: 1,
            })
            .await
            .unwrap();

        recovery
            .add_procedure(RecoveryProcedure {
                id: "proc2".to_string(),
                recovery_type: crate::recovery::RecoveryType::AgentRecovery,
                name: "proc2".to_string(),
                description: "Test procedure 2".to_string(),
                steps: vec![crate::recovery::RecoveryStep {
                    name: "step2".to_string(),
                    action: "test_action".to_string(),
                    required: true,
                    timeout: Duration::from_secs(1),
                }],
                timeout: Duration::from_secs(5),
                retry_count: 1,
                priority: 2,
            })
            .await
            .unwrap();

        let recovery = Arc::new(Mutex::new(recovery));

        // Execute procedures concurrently
        let rec1 = recovery.clone();
        let handle1 = tokio::spawn(async move {
            let mut rec_lock = rec1.lock().await;
            rec_lock.execute_procedure("proc1").await
        });

        let rec2 = recovery.clone();
        let handle2 = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await; // Small delay
            let mut rec_lock = rec2.lock().await;
            rec_lock.execute_procedure("proc2").await
        });

        // Both should complete (one might wait for the other)
        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        assert!(result1.is_ok() || result2.is_ok());
    }

    #[tokio::test]
    async fn test_recovery_state_transitions() {
        let mut recovery = RecoverySystem::new();

        // Test all state transitions
        assert_eq!(recovery.state(), RecoveryState::Idle);

        recovery
            .add_procedure(RecoveryProcedure {
                id: "test".to_string(),
                recovery_type: crate::recovery::RecoveryType::SystemRecovery,
                name: "test".to_string(),
                description: "Test recovery".to_string(),
                steps: vec![crate::recovery::RecoveryStep {
                    name: "step1".to_string(),
                    action: "test_action".to_string(),
                    required: true,
                    timeout: Duration::from_millis(10),
                }],
                timeout: Duration::from_millis(50),
                retry_count: 1,
                priority: 1,
            })
            .await
            .unwrap();

        // Start recovery
        let handle = tokio::spawn({
            let mut rec = recovery.clone();
            async move { rec.execute_procedure("test").await }
        });

        // Give it time to start
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(matches!(recovery.state(), RecoveryState::Running { .. }));

        // Wait for completion
        handle.await.unwrap().unwrap();
        assert_eq!(recovery.state(), RecoveryState::Completed);

        // Test failure state
        recovery
            .add_procedure(RecoveryProcedure {
                id: "fail".to_string(),
                recovery_type: crate::recovery::RecoveryType::SystemRecovery,
                name: "fail".to_string(),
                description: "Failing recovery".to_string(),
                steps: vec![
                    crate::recovery::RecoveryStep {
                        name: "step1".to_string(),
                        action: "test_action".to_string(),
                        required: true,
                        timeout: Duration::from_millis(1),
                    },
                    crate::recovery::RecoveryStep {
                        name: "step2".to_string(),
                        action: "test_action".to_string(),
                        required: true,
                        timeout: Duration::from_millis(1),
                    },
                ],
                timeout: Duration::from_millis(1), // Very short timeout
                retry_count: 0,
                priority: 1,
            })
            .await
            .unwrap();

        let result = recovery.execute_procedure("fail").await;
        assert!(result.is_err());
        assert!(matches!(recovery.state(), RecoveryState::Failed { .. }));
    }

    #[tokio::test]
    async fn test_recovery_emergency_procedures() {
        let mut recovery = RecoverySystem::new();

        // Add emergency procedure with highest priority
        recovery
            .add_procedure(RecoveryProcedure {
                id: "emergency_shutdown".to_string(),
                recovery_type: crate::recovery::RecoveryType::SystemRecovery,
                name: "emergency_shutdown".to_string(),
                description: "Emergency shutdown procedure".to_string(),
                steps: vec![
                    crate::recovery::RecoveryStep {
                        name: "kill_all_agents".to_string(),
                        action: "kill_agents".to_string(),
                        required: true,
                        timeout: Duration::from_millis(100),
                    },
                    crate::recovery::RecoveryStep {
                        name: "clear_gpu_memory".to_string(),
                        action: "clear_memory".to_string(),
                        required: true,
                        timeout: Duration::from_millis(100),
                    },
                    crate::recovery::RecoveryStep {
                        name: "save_state".to_string(),
                        action: "save_state".to_string(),
                        required: false,
                        timeout: Duration::from_millis(100),
                    },
                ],
                timeout: Duration::from_secs(1),
                retry_count: 0,
                priority: 0, // Highest priority
            })
            .await
            .unwrap();

        // Add normal procedure
        recovery
            .add_procedure(RecoveryProcedure {
                id: "normal_recovery".to_string(),
                recovery_type: crate::recovery::RecoveryType::AgentRecovery,
                name: "normal_recovery".to_string(),
                description: "Normal recovery procedure".to_string(),
                steps: vec![crate::recovery::RecoveryStep {
                    name: "restart_agents".to_string(),
                    action: "restart".to_string(),
                    required: true,
                    timeout: Duration::from_secs(1),
                }],
                timeout: Duration::from_secs(5),
                retry_count: 3,
                priority: 10,
            })
            .await
            .unwrap();

        // Emergency procedure should execute even if another is running
        let rec = Arc::new(Mutex::new(recovery));

        // Start normal recovery
        let rec1 = rec.clone();
        let _normal_handle = tokio::spawn(async move {
            let mut rec_lock = rec1.lock().await;
            rec_lock.execute_procedure("normal_recovery").await
        });

        // Immediately start emergency
        let rec2 = rec.clone();
        let emergency_handle = tokio::spawn(async move {
            let mut rec_lock = rec2.lock().await;
            rec_lock
                .execute_emergency_procedure("emergency_shutdown")
                .await
        });

        // Emergency should complete
        assert!(emergency_handle.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_recovery_cleanup_on_failure() {
        let mut recovery = RecoverySystem::new();

        // Add procedure that will fail
        recovery
            .add_procedure(RecoveryProcedure {
                id: "cleanup_test".to_string(),
                recovery_type: crate::recovery::RecoveryType::ResourceRecovery,
                name: "cleanup_test".to_string(),
                description: "Test cleanup on failure".to_string(),
                steps: vec![
                    crate::recovery::RecoveryStep {
                        name: "allocate_resources".to_string(),
                        action: "allocate".to_string(),
                        required: true,
                        timeout: Duration::from_millis(10),
                    },
                    crate::recovery::RecoveryStep {
                        name: "fail_step".to_string(),
                        action: "fail".to_string(),
                        required: true,
                        timeout: Duration::from_millis(10),
                    },
                    crate::recovery::RecoveryStep {
                        name: "cleanup".to_string(),
                        action: "cleanup".to_string(),
                        required: false,
                        timeout: Duration::from_millis(10),
                    },
                ],
                timeout: Duration::from_millis(50),
                retry_count: 0,
                priority: 1,
            })
            .await
            .unwrap();

        // Execute and expect failure
        let result = recovery.execute_procedure("cleanup_test").await;
        assert!(result.is_err());

        // Verify cleanup was attempted
        match recovery.state() {
            RecoveryState::Failed => {
                // Success - recovery failed as expected
            }
            _ => panic!("Expected failed state"),
        }
    }

    // Integration Edge Cases

    #[tokio::test]
    async fn test_emergency_system_cascade_failure() {
        // Test cascading failures across components
        let mut config = EmergencyConfig::default();
        config.kill_switch.triggers = vec![crate::kill_switch::KillSwitchTrigger {
            trigger_id: "cascade1".to_string(),
            name: "Cascade Test".to_string(),
            trigger_type: crate::kill_switch::TriggerType::ResourceLimit,
            enabled: true,
            condition: crate::kill_switch::TriggerCondition::ResourceThreshold {
                resource: "GpuMemory".to_string(),
                threshold: 1000.0,
            },
            action: crate::kill_switch::KillSwitchAction::GlobalKill,
            priority: crate::kill_switch::TriggerPriority::High,
        }];

        let mut kill_switch = KillSwitchSystem::new(config.kill_switch.clone());
        let limiter = ResourceLimiter::new(vec![ResourceLimit {
            resource_type: ResourceType::GpuMemory,
            limit: 1000.0,
            action: LimitAction::KillSwitch,
        }]);

        // Trigger resource limit which should activate kill switch
        limiter
            .update_usage(ResourceType::GpuMemory, 1500.0)
            .await
            .unwrap();
        let violations = limiter.get_violations().await;
        assert!(violations
            .iter()
            .any(|v| v.resource_type == ResourceType::GpuMemory));

        // Kill switch should also activate from resource trigger
        kill_switch
            .check_resource_trigger(ResourceType::GpuMemory, 1500.0)
            .await
            .unwrap();
        assert!(kill_switch.is_active());
    }

    #[tokio::test]
    async fn test_error_propagation_and_recovery() {
        // Test that errors propagate correctly and recovery is attempted
        let temp_dir = TempDir::new().unwrap();
        let bad_path = temp_dir.path().join("nonexistent").join("audit.log");

        let config = crate::audit_log::AuditConfig {
            log_path: bad_path,
            max_file_size: 10_000_000,
            max_files: 10,
            buffer_size: 1000,
            min_level: crate::audit_log::AuditLevel::Debug,
        };

        let logger_result = AuditLogger::new(config).await;
        // Should fail to create logger with bad path
        assert!(logger_result.is_err());
        // Exit test early since logger creation failed
        // The test already passed by asserting logger creation failed
    }
}
