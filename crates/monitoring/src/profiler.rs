//! GPU profiling implementation for kernel performance analysis

pub mod config;
pub mod kernel;
pub mod manager;
pub mod mock;
pub mod nsight;
pub mod traits;

pub use config::ProfileConfig;
pub use kernel::KernelProfile;
pub use manager::ProfilingManager;
pub use mock::MockProfiler;
pub use nsight::NsightComputeProfiler;
pub use traits::Profiler;

// Re-export common types
pub use crate::MonitoringError;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_profiling_manager_basic() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        // Add profilers
        let mock_profiler = Arc::new(MockProfiler::default());
        manager.add_profiler(mock_profiler.clone());

        // Start profile session
        let container_id = Uuid::new_v4();
        let kernel_id = "test_kernel";

        manager
            .start_profile(container_id, kernel_id)
            .await
            .unwrap();

        // Stop profile
        let result = manager.stop_profile(container_id, kernel_id).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].kernel_id, kernel_id);
    }

    #[tokio::test]
    async fn test_multiple_profilers() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        // Add multiple profilers
        let mock1 = Arc::new(MockProfiler::default());
        let mock2 = Arc::new(MockProfiler::default());

        manager.add_profiler(mock1);
        manager.add_profiler(mock2);

        let container_id = Uuid::new_v4();
        let kernel_id = "test_kernel";

        manager
            .start_profile(container_id, kernel_id)
            .await
            .unwrap();

        let result = manager.stop_profile(container_id, kernel_id).await.unwrap();

        // Should get profiles from both profilers
        assert_eq!(result.len(), 2);
    }
}
