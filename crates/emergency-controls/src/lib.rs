//! Emergency Control System for ExoRust
//!
//! Provides critical safety mechanisms including:
//! - Kill switches for runaway agents
//! - Resource consumption limits and monitoring
//! - Safety violation detection and enforcement
//! - Audit logging for compliance
//! - Recovery procedures for system restoration

pub mod audit_log;
pub mod behavior_boundaries;
pub mod error;
pub mod kill_switch;
pub mod recovery;
pub mod resource_limits;

#[cfg(test)]
mod edge_tests;

pub use audit_log::{AuditEvent, AuditEventType, AuditLevel, AuditLogger};
pub use behavior_boundaries::{
    BehaviorBoundary, BehaviorMonitor, BehaviorType, SafetyViolation, ViolationSeverity,
};
pub use error::{EmergencyError, EmergencyResult};
pub use kill_switch::{KillSwitchConfig, KillSwitchEvent, KillSwitchSystem};
pub use recovery::{RecoveryProcedure, RecoveryState, RecoverySystem};
pub use resource_limits::{ResourceLimit, ResourceLimiter, ResourceType};

/// Emergency control system configuration
#[derive(Debug, Clone)]
pub struct EmergencyConfig {
    /// Kill switch configuration
    pub kill_switch: KillSwitchConfig,
    /// Resource limits configuration
    pub resource_limits: Vec<ResourceLimit>,
    /// Behavior boundaries configuration
    pub behavior_boundaries: Vec<BehaviorBoundary>,
    /// Audit log configuration
    pub audit_path: String,
    /// Recovery procedures configuration
    pub recovery_enabled: bool,
}

impl Default for EmergencyConfig {
    fn default() -> Self {
        Self {
            kill_switch: KillSwitchConfig::default(),
            resource_limits: vec![
                ResourceLimit {
                    resource_type: ResourceType::GpuMemory,
                    limit: 16_384.0, // 16GB
                    action: resource_limits::LimitAction::KillSwitch,
                },
                ResourceLimit {
                    resource_type: ResourceType::CpuPercent,
                    limit: 90.0,
                    action: resource_limits::LimitAction::Throttle,
                },
            ],
            behavior_boundaries: vec![],
            audit_path: "/var/log/exorust/emergency.log".to_string(),
            recovery_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_limits::LimitAction;

    #[test]
    fn test_emergency_config_default() {
        let config = EmergencyConfig::default();
        assert_eq!(config.resource_limits.len(), 2);
        assert!(config.recovery_enabled);
        assert!(config.audit_path.contains("emergency.log"));
    }

    #[test]
    fn test_module_exports() {
        // Verify all modules are properly exported
        let _ = EmergencyError::ConfigurationError("test".to_string());
        let _ = KillSwitchConfig::default();
        // These will exist after we create the modules
    }

    #[test]
    fn test_emergency_config_resource_limits() {
        let config = EmergencyConfig::default();

        // Verify first limit is GPU memory
        assert_eq!(
            config.resource_limits[0].resource_type,
            ResourceType::GpuMemory
        );
        assert_eq!(config.resource_limits[0].limit, 16_384.0);
        assert_eq!(config.resource_limits[0].action, LimitAction::KillSwitch);

        // Verify second limit is CPU percent
        assert_eq!(
            config.resource_limits[1].resource_type,
            ResourceType::CpuPercent
        );
        assert_eq!(config.resource_limits[1].limit, 90.0);
        assert_eq!(config.resource_limits[1].action, LimitAction::Throttle);
    }

    #[test]
    fn test_emergency_config_custom() {
        let mut config = EmergencyConfig::default();

        // Customize configuration
        config.kill_switch = KillSwitchConfig {
            global_kill_enabled: false,
            agent_auto_reset_duration: Some(std::time::Duration::from_secs(300)),
            max_concurrent_kills: 5,
            confirmation_required: true,
            audit_enabled: true,
            notify_on_global_kill: false,
            patterns: vec!["experimental-*".to_string()],
            auto_reset_delay: Some(std::time::Duration::from_secs(60)),
            triggers: vec![],
        };

        config.resource_limits.push(ResourceLimit {
            resource_type: ResourceType::NetworkBandwidth,
            limit: 1_000_000_000.0, // 1 GB/s
            action: LimitAction::Warn,
        });

        config.behavior_boundaries.push(BehaviorBoundary {
            behavior_type: BehaviorType::AgentSpawning,
            description: "Experimental agent spawning limits".to_string(),
            max_frequency: Some(100),
            time_window: Some(std::time::Duration::from_secs(60)),
            forbidden_patterns: vec![],
            allowed_patterns: vec!["experimental.*".to_string()],
            severity: ViolationSeverity::Medium,
        });

        config.audit_path = "/custom/audit/path.log".to_string();
        config.recovery_enabled = false;

        // Verify customizations
        assert!(!config.kill_switch.global_kill_enabled);
        assert_eq!(config.resource_limits.len(), 3);
        assert_eq!(config.behavior_boundaries.len(), 1);
        assert_eq!(config.audit_path, "/custom/audit/path.log");
        assert!(!config.recovery_enabled);
    }

    #[test]
    fn test_emergency_config_clone() {
        let original = EmergencyConfig::default();
        let cloned = original.clone();

        assert_eq!(original.resource_limits.len(), cloned.resource_limits.len());
        assert_eq!(original.recovery_enabled, cloned.recovery_enabled);
        assert_eq!(original.audit_path, cloned.audit_path);
    }

    #[test]
    fn test_emergency_config_debug() {
        let config = EmergencyConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("EmergencyConfig"));
        assert!(debug_str.contains("kill_switch"));
        assert!(debug_str.contains("resource_limits"));
        assert!(debug_str.contains("behavior_boundaries"));
        assert!(debug_str.contains("audit_path"));
        assert!(debug_str.contains("recovery_enabled"));
    }

    #[test]
    fn test_emergency_config_empty_limits() {
        let mut config = EmergencyConfig::default();
        config.resource_limits.clear();
        config.behavior_boundaries.clear();

        assert!(config.resource_limits.is_empty());
        assert!(config.behavior_boundaries.is_empty());
    }

    #[test]
    fn test_emergency_config_extreme_values() {
        let mut config = EmergencyConfig::default();

        config.resource_limits.push(ResourceLimit {
            resource_type: ResourceType::SystemMemory,
            limit: f64::MAX,
            action: LimitAction::Custom,
        });

        config.resource_limits.push(ResourceLimit {
            resource_type: ResourceType::DiskIO,
            limit: 0.0,
            action: LimitAction::Suspend,
        });

        assert_eq!(config.resource_limits[2].limit, f64::MAX);
        assert_eq!(config.resource_limits[3].limit, 0.0);
    }

    #[test]
    fn test_emergency_config_all_resource_types() {
        let mut config = EmergencyConfig::default();
        config.resource_limits.clear();

        // Add limits for all resource types
        let resource_types = vec![
            ResourceType::GpuMemory,
            ResourceType::GpuUtilization,
            ResourceType::CpuPercent,
            ResourceType::SystemMemory,
            ResourceType::NetworkBandwidth,
            ResourceType::DiskIO,
            ResourceType::AgentCount,
            ResourceType::AgentSpawnRate,
        ];

        for (i, resource_type) in resource_types.iter().enumerate() {
            config.resource_limits.push(ResourceLimit {
                resource_type: *resource_type,
                limit: (i + 1) as f64 * 100.0,
                action: match i % 5 {
                    0 => LimitAction::Warn,
                    1 => LimitAction::Throttle,
                    2 => LimitAction::Suspend,
                    3 => LimitAction::KillSwitch,
                    _ => LimitAction::Custom,
                },
            });
        }

        assert_eq!(config.resource_limits.len(), 8);
        for (i, limit) in config.resource_limits.iter().enumerate() {
            assert_eq!(limit.limit, (i + 1) as f64 * 100.0);
        }
    }

    #[test]
    fn test_emergency_config_multiple_behavior_boundaries() {
        let mut config = EmergencyConfig::default();

        config.behavior_boundaries.extend(vec![
            BehaviorBoundary {
                behavior_type: BehaviorType::CodeExecution,
                description: "Consensus agent code execution limits".to_string(),
                max_frequency: Some(10),
                time_window: Some(std::time::Duration::from_secs(60)),
                forbidden_patterns: vec!["unsafe.*".to_string()],
                allowed_patterns: vec!["consensus.*".to_string()],
                severity: ViolationSeverity::High,
            },
            BehaviorBoundary {
                behavior_type: BehaviorType::MemoryAccess,
                description: "Evolution agent memory access limits".to_string(),
                max_frequency: Some(1000),
                time_window: Some(std::time::Duration::from_secs(60)),
                forbidden_patterns: vec![],
                allowed_patterns: vec!["evolution.*".to_string()],
                severity: ViolationSeverity::Medium,
            },
            BehaviorBoundary {
                behavior_type: BehaviorType::NetworkCommunication,
                description: "Synthesis agent network communication limits".to_string(),
                max_frequency: Some(500),
                time_window: Some(std::time::Duration::from_secs(60)),
                forbidden_patterns: vec!["external.*".to_string()],
                allowed_patterns: vec!["synthesis.*".to_string()],
                severity: ViolationSeverity::Low,
            },
        ]);

        assert_eq!(config.behavior_boundaries.len(), 3);
        assert_eq!(
            config.behavior_boundaries[0].behavior_type,
            BehaviorType::CodeExecution
        );
        assert_eq!(
            config.behavior_boundaries[1].behavior_type,
            BehaviorType::MemoryAccess
        );
        assert_eq!(
            config.behavior_boundaries[2].behavior_type,
            BehaviorType::NetworkCommunication
        );
    }

    #[test]
    fn test_emergency_config_unicode_paths() {
        let mut config = EmergencyConfig::default();
        config.audit_path = "/日本語/ログ/審査.log".to_string();

        assert!(config.audit_path.contains("日本語"));
        assert!(config.audit_path.contains("ログ"));
        assert!(config.audit_path.contains("審査"));
    }

    #[test]
    fn test_emergency_config_special_characters_path() {
        let mut config = EmergencyConfig::default();
        config.audit_path = "/path/with spaces/and-special_chars!@#$.log".to_string();

        assert!(config.audit_path.contains("with spaces"));
        assert!(config.audit_path.contains("and-special_chars!@#$"));
    }

    #[test]
    fn test_re_exports_availability() {
        // Test that all re-exported types are available
        let _audit_event = AuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            event_type: AuditEventType::SystemStartup,
            level: AuditLevel::Critical,
            agent_id: None,
            message: "test".to_string(),
            details: serde_json::json!({}),
            user: None,
            source: "test".to_string(),
        };

        let _behavior_boundary = BehaviorBoundary {
            behavior_type: BehaviorType::MemoryAccess,
            description: "Test behavior boundary".to_string(),
            max_frequency: Some(100),
            time_window: Some(std::time::Duration::from_secs(60)),
            forbidden_patterns: vec![],
            allowed_patterns: vec!["test.*".to_string()],
            severity: ViolationSeverity::Medium,
        };

        let _resource_limit = ResourceLimit {
            resource_type: ResourceType::GpuMemory,
            limit: 1000.0,
            action: LimitAction::Warn,
        };

        // Verify all types are accessible
        assert!(true);
    }
}
