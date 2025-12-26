//! Mock profiler for testing

use super::kernel::KernelProfile;
use super::traits::Profiler;
use crate::MonitoringError;
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Mock profiler for testing
pub struct MockProfiler {
    profiles: Arc<RwLock<Vec<KernelProfile>>>,
    active_profiles: Arc<DashMap<(Uuid, String), u64>>,
}

impl MockProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(Vec::new())),
            active_profiles: Arc::new(DashMap::new()),
        }
    }

    pub fn get_profiles(&self) -> Vec<KernelProfile> {
        self.profiles.read().unwrap().clone()
    }

    pub fn clear_profiles(&self) {
        self.profiles.write().unwrap().clear();
        self.active_profiles.clear();
    }
}

impl Default for MockProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Profiler for MockProfiler {
    async fn start_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<(), MonitoringError> {
        let key = (container_id, kernel_id.to_string());

        if self.active_profiles.contains_key(&key) {
            return Err(MonitoringError::ProfilerFailed {
                reason: "Profile already active".to_string(),
            });
        }

        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.active_profiles.insert(key, start_time);
        Ok(())
    }

    async fn stop_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<KernelProfile, MonitoringError> {
        let key = (container_id, kernel_id.to_string());

        let start_time = self
            .active_profiles
            .remove(&key)
            .map(|(_, v)| v)
            .ok_or_else(|| MonitoringError::ProfilerFailed {
                reason: "No active profile found".to_string(),
            })?;

        let end_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let duration_ns = end_time - start_time;

        // Create mock profile with synthetic data
        let profile = KernelProfile::new(kernel_id.to_string(), container_id)
            .with_metrics(
                duration_ns,
                150.0 + (duration_ns as f64 / 1_000_000.0), // Mock throughput
                1000.0 + (duration_ns as f64 / 1_000_000.0), // Mock GFLOPS
            )
            .with_occupancy(75.0, 32)
            .with_config((256, 1, 1), (32, 32, 1), 16384);

        self.profiles.write().unwrap().push(profile.clone());

        Ok(profile)
    }

    fn name(&self) -> &str {
        "MockProfiler"
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn export_profile(
        &self,
        _profile: &KernelProfile,
        _output_file: &std::path::Path,
    ) -> Result<(), MonitoringError> {
        // Mock export - just return success
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_mock_profiler_basic() {
        let profiler = MockProfiler::new();
        let container_id = Uuid::new_v4();

        // Start profile
        profiler
            .start_profile(container_id, "test_kernel")
            .await
            .unwrap();

        // Add some delay
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Stop profile
        let profile = profiler
            .stop_profile(container_id, "test_kernel")
            .await
            .unwrap();

        assert_eq!(profile.kernel_id, "test_kernel");
        assert_eq!(profile.container_id, container_id);
        assert!(profile.gpu_time_ns > 0);

        // Check profile was stored
        let profiles = profiler.get_profiles();
        assert_eq!(profiles.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_profiler_duplicate_start() {
        let profiler = MockProfiler::new();
        let container_id = Uuid::new_v4();

        profiler
            .start_profile(container_id, "kernel1")
            .await
            .unwrap();

        // Try to start again
        let result = profiler.start_profile(container_id, "kernel1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_profiler_stop_without_start() {
        let profiler = MockProfiler::new();
        let container_id = Uuid::new_v4();

        let result = profiler.stop_profile(container_id, "kernel1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_profiler_clear() {
        let profiler = MockProfiler::new();
        let container_id = Uuid::new_v4();

        // Create some profiles
        for i in 0..3 {
            let kernel_id = format!("kernel_{i}");
            profiler
                .start_profile(container_id, &kernel_id)
                .await
                .unwrap();
            profiler
                .stop_profile(container_id, &kernel_id)
                .await
                .unwrap();
        }

        assert_eq!(profiler.get_profiles().len(), 3);

        // Clear profiles
        profiler.clear_profiles();
        assert_eq!(profiler.get_profiles().len(), 0);
    }

    #[tokio::test]
    async fn test_mock_profiler_availability() {
        let profiler = MockProfiler::new();
        assert!(profiler.is_available().await);
        assert_eq!(profiler.name(), "MockProfiler");
    }
}
