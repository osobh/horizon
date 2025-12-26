//! Profiling manager for coordinating multiple profilers

use super::config::ProfileConfig;
use super::kernel::KernelProfile;
use super::traits::Profiler;
use crate::MonitoringError;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Manager for coordinating multiple profilers
pub struct ProfilingManager {
    profilers: RwLock<Vec<Arc<dyn Profiler + Send + Sync>>>,
    #[allow(dead_code)]
    config: ProfileConfig,
}

impl ProfilingManager {
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            profilers: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Add a profiler
    pub async fn add_profiler(&self, profiler: Arc<dyn Profiler + Send + Sync>) {
        self.profilers.write().await.push(profiler);
    }

    /// Start profiling on all available profilers
    pub async fn start_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<(), MonitoringError> {
        let profilers = self.profilers.read().await;
        let mut errors = Vec::new();

        for profiler in profilers.iter() {
            if profiler.is_available().await {
                if let Err(e) = profiler.start_profile(container_id, kernel_id).await {
                    errors.push(format!("{}: {}", profiler.name(), e));
                }
            }
        }

        if !errors.is_empty() && errors.len() == profilers.len() {
            return Err(MonitoringError::ProfilerFailed {
                reason: format!("All profilers failed: {}", errors.join(", ")),
            });
        }

        Ok(())
    }

    /// Stop profiling and collect results from all profilers
    pub async fn stop_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<Vec<KernelProfile>, MonitoringError> {
        let profilers = self.profilers.read().await;
        let mut profiles = Vec::new();
        let mut errors = Vec::new();

        for profiler in profilers.iter() {
            match profiler.stop_profile(container_id, kernel_id).await {
                Ok(profile) => profiles.push(profile),
                Err(e) => errors.push(format!("{}: {}", profiler.name(), e)),
            }
        }

        if profiles.is_empty() && !errors.is_empty() {
            return Err(MonitoringError::ProfilerFailed {
                reason: format!("No profiles collected: {}", errors.join(", ")),
            });
        }

        Ok(profiles)
    }

    /// Get available profilers
    pub async fn available_profilers(&self) -> Vec<String> {
        let profilers = self.profilers.read().await;
        let mut available = Vec::new();

        for profiler in profilers.iter() {
            if profiler.is_available().await {
                available.push(profiler.name().to_string());
            }
        }

        available
    }

    /// Clear all profilers
    pub async fn clear_profilers(&self) {
        self.profilers.write().await.clear();
    }

    /// Get profiler count
    pub async fn profiler_count(&self) -> usize {
        self.profilers.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::mock::MockProfiler;

    #[tokio::test]
    async fn test_profiling_manager_basic() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        // Add mock profiler
        let mock = Arc::new(MockProfiler::default());
        manager.add_profiler(mock.clone());

        assert_eq!(manager.profiler_count(), 1);

        let container_id = Uuid::new_v4();

        // Start profiling
        manager
            .start_profile(container_id, "test_kernel")
            .await
            .unwrap();

        // Stop and get results
        let profiles = manager
            .stop_profile(container_id, "test_kernel")
            .await
            .unwrap();

        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].kernel_id, "test_kernel");
    }

    #[tokio::test]
    async fn test_profiling_manager_multiple_profilers() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        // Add multiple mock profilers
        for i in 0..3 {
            let mut mock = MockProfiler::default();
            manager.add_profiler(Arc::new(mock));
        }

        assert_eq!(manager.profiler_count(), 3);

        let container_id = Uuid::new_v4();

        manager
            .start_profile(container_id, "kernel1")
            .await
            .unwrap();

        let profiles = manager.stop_profile(container_id, "kernel1").await.unwrap();

        assert_eq!(profiles.len(), 3);
    }

    #[tokio::test]
    async fn test_profiling_manager_available_profilers() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        manager.add_profiler(Arc::new(MockProfiler::default()));

        let available = manager.available_profilers().await;
        assert_eq!(available.len(), 1);
        assert_eq!(available[0], "MockProfiler");
    }

    #[tokio::test]
    async fn test_profiling_manager_clear() {
        let manager = ProfilingManager::new(ProfileConfig::default());

        manager.add_profiler(Arc::new(MockProfiler::default()));
        manager.add_profiler(Arc::new(MockProfiler::default()));

        assert_eq!(manager.profiler_count(), 2);

        manager.clear_profilers();
        assert_eq!(manager.profiler_count(), 0);
    }

    #[tokio::test]
    async fn test_profiling_manager_no_profilers() {
        let manager = ProfilingManager::new(ProfileConfig::default());
        let container_id = Uuid::new_v4();

        // Should succeed even with no profilers
        manager
            .start_profile(container_id, "kernel1")
            .await
            .unwrap();

        let profiles = manager.stop_profile(container_id, "kernel1").await.unwrap();

        assert_eq!(profiles.len(), 0);
    }
}
