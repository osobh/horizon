//! ExoRust GPU Container Runtime
//!
//! This crate provides GPU container runtime for isolated agent execution
//! in the ExoRust agent-first operating system.

use anyhow::Result;

pub mod container;
pub mod error;
pub mod isolation;
pub mod lifecycle;
pub mod personality;
pub mod secure_runtime;

#[cfg(test)]
mod test_helpers;

pub use container::{
    ContainerConfig, ContainerStats, EvolutionState, GpuContainer, HardwareAffinity, MemoryTier,
};
pub use error::RuntimeError;
pub use isolation::{
    ExecutionSandbox, IsolationContext, IsolationManager, IsolationResult, IsolationStats,
    KernelSignature, MemoryFence, MemoryPermissions, QuotaViolation, ResourceQuota, ViolationType,
};
pub use lifecycle::{ContainerLifecycle, ContainerState};
pub use personality::{
    AgentPersonality, AlgorithmChoice, CommunicationPattern, Decision, OptimizationTarget, Outcome,
    PersonalityError, PersonalityInfluence, PersonalityType, ResourceAllocation,
};
pub use secure_runtime::SecureContainerRuntime;

/// GPU container runtime for managing agent containers
#[async_trait::async_trait]
pub trait ContainerRuntime: Send + Sync {
    /// Create a new container
    async fn create_container(&self, config: ContainerConfig)
        -> Result<GpuContainer, RuntimeError>;

    /// Start a container
    async fn start_container(&self, container_id: &str) -> Result<(), RuntimeError>;

    /// Stop a container
    async fn stop_container(&self, container_id: &str) -> Result<(), RuntimeError>;

    /// Remove a container
    async fn remove_container(&self, container_id: &str) -> Result<(), RuntimeError>;

    /// Get container statistics
    async fn container_stats(&self, container_id: &str) -> Result<ContainerStats, RuntimeError>;

    /// List all containers
    async fn list_containers(&self) -> Result<Vec<String>, RuntimeError>;
}

/// Runtime statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RuntimeStats {
    pub total_containers: usize,
    pub running_containers: usize,
    pub stopped_containers: usize,
    pub gpu_utilization_percent: f32,
    pub memory_usage_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_stats_creation() {
        let stats = RuntimeStats {
            total_containers: 10,
            running_containers: 5,
            stopped_containers: 5,
            gpu_utilization_percent: 75.5,
            memory_usage_bytes: 1024 * 1024 * 512, // 512 MB
        };

        assert_eq!(stats.total_containers, 10);
        assert_eq!(stats.running_containers, 5);
        assert_eq!(stats.stopped_containers, 5);
        assert_eq!(stats.gpu_utilization_percent, 75.5);
        assert_eq!(stats.memory_usage_bytes, 536870912);
    }

    #[test]
    fn test_runtime_stats_clone() {
        let stats = RuntimeStats {
            total_containers: 20,
            running_containers: 15,
            stopped_containers: 5,
            gpu_utilization_percent: 90.0,
            memory_usage_bytes: 1_000_000_000,
        };

        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_runtime_stats_serialization() {
        let stats = RuntimeStats {
            total_containers: 100,
            running_containers: 80,
            stopped_containers: 20,
            gpu_utilization_percent: 85.5,
            memory_usage_bytes: 2_147_483_648, // 2 GB
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: RuntimeStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_runtime_stats_debug() {
        let stats = RuntimeStats {
            total_containers: 5,
            running_containers: 3,
            stopped_containers: 2,
            gpu_utilization_percent: 60.0,
            memory_usage_bytes: 1024,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("total_containers: 5"));
        assert!(debug_str.contains("running_containers: 3"));
        assert!(debug_str.contains("stopped_containers: 2"));
        assert!(debug_str.contains("gpu_utilization_percent: 60.0"));
        assert!(debug_str.contains("memory_usage_bytes: 1024"));
    }

    #[test]
    fn test_runtime_stats_edge_cases() {
        // Zero stats
        let zero_stats = RuntimeStats {
            total_containers: 0,
            running_containers: 0,
            stopped_containers: 0,
            gpu_utilization_percent: 0.0,
            memory_usage_bytes: 0,
        };
        assert_eq!(zero_stats.total_containers, 0);
        assert_eq!(zero_stats.gpu_utilization_percent, 0.0);

        // Max values
        let max_stats = RuntimeStats {
            total_containers: usize::MAX,
            running_containers: usize::MAX,
            stopped_containers: usize::MAX,
            gpu_utilization_percent: 100.0,
            memory_usage_bytes: usize::MAX,
        };
        assert_eq!(max_stats.total_containers, usize::MAX);
        assert_eq!(max_stats.gpu_utilization_percent, 100.0);
    }

    #[test]
    fn test_runtime_stats_special_float_values() {
        let stats = RuntimeStats {
            total_containers: 1,
            running_containers: 1,
            stopped_containers: 0,
            gpu_utilization_percent: f32::INFINITY,
            memory_usage_bytes: 1024,
        };
        assert!(stats.gpu_utilization_percent.is_infinite());

        let stats_nan = RuntimeStats {
            total_containers: 1,
            running_containers: 1,
            stopped_containers: 0,
            gpu_utilization_percent: f32::NAN,
            memory_usage_bytes: 1024,
        };
        assert!(stats_nan.gpu_utilization_percent.is_nan());
    }

    #[test]
    fn test_runtime_stats_consistency() {
        // Test that running + stopped = total
        let stats = RuntimeStats {
            total_containers: 50,
            running_containers: 30,
            stopped_containers: 20,
            gpu_utilization_percent: 70.0,
            memory_usage_bytes: 1_000_000,
        };

        assert_eq!(
            stats.running_containers + stats.stopped_containers,
            stats.total_containers
        );
    }

    #[test]
    fn test_runtime_stats_json_field_names() {
        let stats = RuntimeStats {
            total_containers: 7,
            running_containers: 4,
            stopped_containers: 3,
            gpu_utilization_percent: 55.5,
            memory_usage_bytes: 2048,
        };

        let json = serde_json::to_string(&stats).unwrap();

        // Verify JSON field names
        assert!(json.contains("\"total_containers\":7"));
        assert!(json.contains("\"running_containers\":4"));
        assert!(json.contains("\"stopped_containers\":3"));
        assert!(json.contains("\"gpu_utilization_percent\":55.5"));
        assert!(json.contains("\"memory_usage_bytes\":2048"));
    }

    #[test]
    fn test_module_reexports() {
        // Verify that all public modules and types are accessible
        let _config = ContainerConfig::default();
        let _state = ContainerState::Created;
        let _error = RuntimeError::ContainerNotFound {
            id: "test".to_string(),
        };
    }

    #[test]
    fn test_container_runtime_trait_object_safety() {
        // This test verifies that ContainerRuntime trait is object-safe
        fn _accepts_runtime_trait(_runtime: Box<dyn ContainerRuntime>) {
            // If this compiles, the trait is object-safe
        }
    }

    #[test]
    fn test_runtime_stats_memory_calculations() {
        let kb = 1024_usize;
        let mb = kb * 1024;
        let gb = mb * 1024;

        let test_cases = vec![
            (0, "0 bytes"),
            (kb, "1 KB"),
            (mb, "1 MB"),
            (gb, "1 GB"),
            (100 * gb, "100 GB"),
        ];

        for (bytes, _description) in test_cases {
            let stats = RuntimeStats {
                total_containers: 1,
                running_containers: 1,
                stopped_containers: 0,
                gpu_utilization_percent: 50.0,
                memory_usage_bytes: bytes,
            };
            assert_eq!(stats.memory_usage_bytes, bytes);
        }
    }

    #[test]
    fn test_runtime_stats_utilization_ranges() {
        let utilization_values = vec![0.0, 25.0, 50.0, 75.0, 99.9, 100.0];

        for util in utilization_values {
            let stats = RuntimeStats {
                total_containers: 1,
                running_containers: 1,
                stopped_containers: 0,
                gpu_utilization_percent: util,
                memory_usage_bytes: 1024,
            };
            assert_eq!(stats.gpu_utilization_percent, util);
        }
    }

    #[test]
    fn test_runtime_stats_large_container_counts() {
        let large_count = 1_000_000;
        let stats = RuntimeStats {
            total_containers: large_count,
            running_containers: large_count / 2,
            stopped_containers: large_count / 2,
            gpu_utilization_percent: 80.0,
            memory_usage_bytes: large_count * 1024, // 1KB per container
        };

        assert_eq!(stats.total_containers, large_count);
        assert_eq!(stats.running_containers, 500_000);
        assert_eq!(stats.stopped_containers, 500_000);
    }

    #[test]
    fn test_runtime_stats_partial_update() {
        let mut stats = RuntimeStats {
            total_containers: 10,
            running_containers: 5,
            stopped_containers: 5,
            gpu_utilization_percent: 50.0,
            memory_usage_bytes: 1_000_000,
        };

        // Simulate container state changes
        stats.running_containers += 2;
        stats.stopped_containers -= 2;
        assert_eq!(stats.running_containers, 7);
        assert_eq!(stats.stopped_containers, 3);
        assert_eq!(stats.total_containers, 10); // Total remains same

        // Update GPU utilization
        stats.gpu_utilization_percent = 75.0;
        assert_eq!(stats.gpu_utilization_percent, 75.0);
    }

    #[test]
    fn test_runtime_stats_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RuntimeStats>();
    }
}
