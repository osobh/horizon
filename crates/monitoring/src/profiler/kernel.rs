//! Kernel profile data structures

use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Profile data for a GPU kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    pub kernel_id: String,
    pub container_id: Uuid,
    pub timestamp: u64,
    pub duration: Duration,
    pub gpu_time_ns: u64,
    pub memory_throughput_gb_s: f64,
    pub compute_throughput_gflops: f64,
    pub occupancy_percent: f64,
    pub register_usage: u32,
    pub shared_memory_bytes: u64,
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
}

impl Default for KernelProfile {
    fn default() -> Self {
        Self {
            kernel_id: String::new(),
            container_id: Uuid::nil(),
            timestamp: 0,
            duration: Duration::from_secs(0),
            gpu_time_ns: 0,
            memory_throughput_gb_s: 0.0,
            compute_throughput_gflops: 0.0,
            occupancy_percent: 0.0,
            register_usage: 0,
            shared_memory_bytes: 0,
            grid_size: (1, 1, 1),
            block_size: (1, 1, 1),
        }
    }
}

impl KernelProfile {
    /// Create a new kernel profile
    pub fn new(kernel_id: String, container_id: Uuid) -> Self {
        Self {
            kernel_id,
            container_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ..Default::default()
        }
    }

    /// Set performance metrics
    pub fn with_metrics(
        mut self,
        gpu_time_ns: u64,
        memory_throughput_gb_s: f64,
        compute_throughput_gflops: f64,
    ) -> Self {
        self.gpu_time_ns = gpu_time_ns;
        self.memory_throughput_gb_s = memory_throughput_gb_s;
        self.compute_throughput_gflops = compute_throughput_gflops;
        self.duration = Duration::from_nanos(gpu_time_ns);
        self
    }

    /// Set occupancy information
    pub fn with_occupancy(mut self, occupancy_percent: f64, register_usage: u32) -> Self {
        self.occupancy_percent = occupancy_percent;
        self.register_usage = register_usage;
        self
    }

    /// Set kernel configuration
    pub fn with_config(
        mut self,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        shared_memory_bytes: u64,
    ) -> Self {
        self.grid_size = grid_size;
        self.block_size = block_size;
        self.shared_memory_bytes = shared_memory_bytes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_profile_creation() {
        let container_id = Uuid::new_v4();
        let profile = KernelProfile::new("test_kernel".to_string(), container_id);

        assert_eq!(profile.kernel_id, "test_kernel");
        assert_eq!(profile.container_id, container_id);
        assert!(profile.timestamp > 0);
    }

    #[test]
    fn test_kernel_profile_with_metrics() {
        let profile = KernelProfile::default().with_metrics(1_000_000, 250.5, 1500.75);

        assert_eq!(profile.gpu_time_ns, 1_000_000);
        assert_eq!(profile.memory_throughput_gb_s, 250.5);
        assert_eq!(profile.compute_throughput_gflops, 1500.75);
        assert_eq!(profile.duration, Duration::from_nanos(1_000_000));
    }

    #[test]
    fn test_kernel_profile_with_occupancy() {
        let profile = KernelProfile::default().with_occupancy(85.5, 32);

        assert_eq!(profile.occupancy_percent, 85.5);
        assert_eq!(profile.register_usage, 32);
    }

    #[test]
    fn test_kernel_profile_with_config() {
        let profile = KernelProfile::default().with_config((256, 1, 1), (32, 32, 1), 16384);

        assert_eq!(profile.grid_size, (256, 1, 1));
        assert_eq!(profile.block_size, (32, 32, 1));
        assert_eq!(profile.shared_memory_bytes, 16384);
    }

    #[test]
    fn test_kernel_profile_builder_chain() {
        let container_id = Uuid::new_v4();
        let profile = KernelProfile::new("matmul".to_string(), container_id)
            .with_metrics(2_500_000, 450.0, 3200.0)
            .with_occupancy(92.0, 48)
            .with_config((512, 512, 1), (16, 16, 1), 32768);

        assert_eq!(profile.kernel_id, "matmul");
        assert_eq!(profile.gpu_time_ns, 2_500_000);
        assert_eq!(profile.occupancy_percent, 92.0);
        assert_eq!(profile.grid_size, (512, 512, 1));
    }
}
