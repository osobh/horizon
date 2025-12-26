//! Core types and data structures for replay functionality
//!
//! This module contains all the primary data types used throughout the replay system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Replay session containing execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySession {
    pub session_id: Uuid,
    pub snapshot_id: Uuid,
    pub container_id: Uuid,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub status: ReplayStatus,
    pub config: ReplayConfig,
    pub results: Option<ReplayResults>,
}

/// Replay execution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplayStatus {
    Pending,
    Initializing,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Replay configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    pub enable_debugging: bool,
    pub capture_intermediate_states: bool,
    pub step_mode: bool,
    pub max_execution_time_ms: u64,
    pub memory_comparison: bool,
    pub kernel_profiling: bool,
    pub breakpoints: Vec<ReplayBreakpoint>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enable_debugging: true,
            capture_intermediate_states: false,
            step_mode: false,
            max_execution_time_ms: 30000, // 30 seconds
            memory_comparison: true,
            kernel_profiling: false,
            breakpoints: Vec::new(),
        }
    }
}

/// Breakpoint for stepping through replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBreakpoint {
    pub breakpoint_id: Uuid,
    pub condition: BreakpointCondition,
    pub actions: Vec<BreakpointAction>,
    pub enabled: bool,
}

/// Conditions that can trigger breakpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointCondition {
    KernelLaunch { kernel_name: String },
    MemoryAccess { address_range: (u64, u64) },
    ExecutionTime { threshold_ms: u64 },
    CustomCondition { expression: String },
}

/// Actions to take when breakpoint is hit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakpointAction {
    Pause,
    CaptureMemory,
    LogState,
    RunCallback { callback_id: String },
}

/// Results from replay execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResults {
    pub execution_time_ms: u64,
    pub memory_changes: MemoryDiff,
    pub kernel_metrics: KernelMetrics,
    pub intermediate_states: Vec<IntermediateState>,
    pub breakpoints_hit: Vec<BreakpointHit>,
    pub error_log: Vec<String>,
}

/// Memory differences between original and replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDiff {
    pub host_memory_changes: Vec<MemoryChange>,
    pub device_memory_changes: Vec<MemoryChange>,
    pub total_differences: usize,
    pub similarity_percent: f64,
}

/// Individual memory change record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChange {
    pub offset: u64,
    pub original_value: Vec<u8>,
    pub replay_value: Vec<u8>,
    pub change_type: MemoryChangeType,
}

/// Type of memory change detected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryChangeType {
    ValueChanged,
    Added,
    Removed,
    Corrupted,
}

/// Kernel execution metrics during replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetrics {
    pub launch_count: u32,
    pub total_execution_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub memory_bandwidth_gb_s: f64,
    pub occupancy_percent: f64,
    pub error_count: u32,
}

/// Intermediate execution state capture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateState {
    pub timestamp: u64,
    pub kernel_id: String,
    pub memory_snapshot: Vec<u8>,
    pub register_state: HashMap<String, u64>,
    pub execution_metrics: HashMap<String, f64>,
}

/// Breakpoint hit record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakpointHit {
    pub breakpoint_id: Uuid,
    pub timestamp: u64,
    pub condition_met: String,
    pub actions_taken: Vec<String>,
    pub execution_context: String,
}

/// Configuration for replay manager
#[derive(Debug, Clone)]
pub struct ReplayManagerConfig {
    pub max_concurrent_replays: usize,
    pub default_timeout_ms: u64,
    pub auto_cleanup_completed: bool,
    pub cleanup_interval_seconds: u64,
}

impl Default for ReplayManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_replays: 10,
            default_timeout_ms: 60000, // 1 minute
            auto_cleanup_completed: true,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Statistics about replay operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayStats {
    pub total_sessions: usize,
    pub running_sessions: usize,
    pub paused_sessions: usize,
    pub completed_sessions: usize,
    pub failed_sessions: usize,
    pub average_execution_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert!(config.enable_debugging);
        assert!(!config.capture_intermediate_states);
        assert!(!config.step_mode);
        assert_eq!(config.max_execution_time_ms, 30000);
        assert!(config.memory_comparison);
        assert!(!config.kernel_profiling);
        assert!(config.breakpoints.is_empty());
    }

    #[test]
    fn test_replay_session_creation() {
        let session = ReplaySession {
            session_id: Uuid::new_v4(),
            snapshot_id: Uuid::new_v4(),
            container_id: Uuid::new_v4(),
            start_time: 1234567890,
            end_time: None,
            status: ReplayStatus::Pending,
            config: ReplayConfig::default(),
            results: None,
        };

        assert_eq!(session.status, ReplayStatus::Pending);
        assert!(session.end_time.is_none());
        assert!(session.results.is_none());
    }

    #[test]
    fn test_breakpoint_conditions() {
        let breakpoint = ReplayBreakpoint {
            breakpoint_id: Uuid::new_v4(),
            condition: BreakpointCondition::KernelLaunch {
                kernel_name: "test_kernel".to_string(),
            },
            actions: vec![BreakpointAction::Pause, BreakpointAction::CaptureMemory],
            enabled: true,
        };

        assert!(breakpoint.enabled);
        assert_eq!(breakpoint.actions.len(), 2);
        assert!(matches!(
            breakpoint.condition,
            BreakpointCondition::KernelLaunch { .. }
        ));
    }

    #[test]
    fn test_memory_diff_creation() {
        let change = MemoryChange {
            offset: 0x1000,
            original_value: vec![0x00, 0x01],
            replay_value: vec![0x00, 0x02],
            change_type: MemoryChangeType::ValueChanged,
        };

        let diff = MemoryDiff {
            host_memory_changes: vec![change],
            device_memory_changes: vec![],
            total_differences: 1,
            similarity_percent: 99.9,
        };

        assert_eq!(diff.total_differences, 1);
        assert_eq!(diff.host_memory_changes.len(), 1);
        assert_eq!(diff.device_memory_changes.len(), 0);
        assert!(diff.similarity_percent > 99.0);
    }

    #[test]
    fn test_kernel_metrics() {
        let metrics = KernelMetrics {
            launch_count: 5,
            total_execution_time_ms: 500.0,
            average_execution_time_ms: 100.0,
            memory_bandwidth_gb_s: 450.0,
            occupancy_percent: 85.0,
            error_count: 0,
        };

        assert_eq!(metrics.launch_count, 5);
        assert_eq!(metrics.average_execution_time_ms, 100.0);
        assert_eq!(metrics.error_count, 0);
    }

    #[test]
    fn test_replay_status_variants() {
        let statuses = vec![
            ReplayStatus::Pending,
            ReplayStatus::Initializing,
            ReplayStatus::Running,
            ReplayStatus::Paused,
            ReplayStatus::Completed,
            ReplayStatus::Failed,
            ReplayStatus::Cancelled,
        ];

        for status in statuses {
            match status {
                ReplayStatus::Pending => assert_eq!(format!("{:?}", status), "Pending"),
                ReplayStatus::Initializing => assert_eq!(format!("{:?}", status), "Initializing"),
                ReplayStatus::Running => assert_eq!(format!("{:?}", status), "Running"),
                ReplayStatus::Paused => assert_eq!(format!("{:?}", status), "Paused"),
                ReplayStatus::Completed => assert_eq!(format!("{:?}", status), "Completed"),
                ReplayStatus::Failed => assert_eq!(format!("{:?}", status), "Failed"),
                ReplayStatus::Cancelled => assert_eq!(format!("{:?}", status), "Cancelled"),
            }
        }
    }

    #[test]
    fn test_replay_config_custom() {
        let mut breakpoints = vec![];
        breakpoints.push(ReplayBreakpoint {
            breakpoint_id: Uuid::new_v4(),
            condition: BreakpointCondition::ExecutionTime { threshold_ms: 1000 },
            actions: vec![BreakpointAction::Pause],
            enabled: true,
        });

        let config = ReplayConfig {
            enable_debugging: false,
            capture_intermediate_states: true,
            step_mode: true,
            max_execution_time_ms: 60000,
            memory_comparison: false,
            kernel_profiling: true,
            breakpoints,
        };

        assert!(!config.enable_debugging);
        assert!(config.capture_intermediate_states);
        assert!(config.step_mode);
        assert_eq!(config.max_execution_time_ms, 60000);
        assert!(!config.memory_comparison);
        assert!(config.kernel_profiling);
        assert_eq!(config.breakpoints.len(), 1);
    }

    #[test]
    fn test_breakpoint_conditions_all_variants() {
        let conditions = vec![
            BreakpointCondition::KernelLaunch {
                kernel_name: "matmul".to_string(),
            },
            BreakpointCondition::MemoryAccess {
                address_range: (0x1000, 0x2000),
            },
            BreakpointCondition::ExecutionTime { threshold_ms: 5000 },
            BreakpointCondition::CustomCondition {
                expression: "memory[0x100] == 0xFF".to_string(),
            },
        ];

        for condition in conditions {
            match condition {
                BreakpointCondition::KernelLaunch { kernel_name } => {
                    assert!(!kernel_name.is_empty());
                }
                BreakpointCondition::MemoryAccess { address_range } => {
                    assert!(address_range.0 < address_range.1);
                }
                BreakpointCondition::ExecutionTime { threshold_ms } => {
                    assert!(threshold_ms > 0);
                }
                BreakpointCondition::CustomCondition { expression } => {
                    assert!(!expression.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_breakpoint_actions_all_variants() {
        let actions = vec![
            BreakpointAction::Pause,
            BreakpointAction::CaptureMemory,
            BreakpointAction::LogState,
            BreakpointAction::RunCallback {
                callback_id: "test_callback".to_string(),
            },
        ];

        for action in actions {
            match action {
                BreakpointAction::Pause => assert!(true),
                BreakpointAction::CaptureMemory => assert!(true),
                BreakpointAction::LogState => assert!(true),
                BreakpointAction::RunCallback { callback_id } => {
                    assert!(!callback_id.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_memory_change_types() {
        let types = vec![
            MemoryChangeType::ValueChanged,
            MemoryChangeType::Added,
            MemoryChangeType::Removed,
            MemoryChangeType::Corrupted,
        ];

        for change_type in types {
            match change_type {
                MemoryChangeType::ValueChanged => assert!(true),
                MemoryChangeType::Added => assert!(true),
                MemoryChangeType::Removed => assert!(true),
                MemoryChangeType::Corrupted => assert!(true),
            }
        }
    }

    #[test]
    fn test_memory_change_creation() {
        let change = MemoryChange {
            offset: 0x1000,
            original_value: vec![0x00, 0x01, 0x02, 0x03],
            replay_value: vec![0x00, 0x01, 0x02, 0x04],
            change_type: MemoryChangeType::ValueChanged,
        };

        assert_eq!(change.offset, 0x1000);
        assert_eq!(change.original_value.len(), 4);
        assert_eq!(change.replay_value.len(), 4);
        assert_ne!(change.original_value[3], change.replay_value[3]);
    }

    #[test]
    fn test_replay_session_serialization() {
        let session = ReplaySession {
            session_id: Uuid::new_v4(),
            snapshot_id: Uuid::new_v4(),
            container_id: Uuid::new_v4(),
            start_time: 1234567890,
            end_time: Some(1234567900),
            status: ReplayStatus::Completed,
            config: ReplayConfig::default(),
            results: None,
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ReplaySession = serde_json::from_str(&json).unwrap();

        assert_eq!(session.session_id, deserialized.session_id);
        assert_eq!(session.status, deserialized.status);
        assert_eq!(session.start_time, deserialized.start_time);
    }

    #[test]
    fn test_replay_manager_config() {
        let config = ReplayManagerConfig {
            max_concurrent_replays: 5,
            default_timeout_ms: 120000,
            auto_cleanup_completed: false,
            cleanup_interval_seconds: 300,
        };

        assert_eq!(config.max_concurrent_replays, 5);
        assert_eq!(config.default_timeout_ms, 120000);
        assert!(!config.auto_cleanup_completed);
        assert_eq!(config.cleanup_interval_seconds, 300);
    }

    #[test]
    fn test_replay_manager_config_default() {
        let config = ReplayManagerConfig::default();

        assert_eq!(config.max_concurrent_replays, 10);
        assert_eq!(config.default_timeout_ms, 60000);
        assert!(config.auto_cleanup_completed);
        assert_eq!(config.cleanup_interval_seconds, 300);
    }
}
