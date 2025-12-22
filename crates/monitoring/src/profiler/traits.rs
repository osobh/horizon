//! Profiler traits

use super::kernel::KernelProfile;
use crate::MonitoringError;
use async_trait::async_trait;
use std::path::Path;
use uuid::Uuid;

/// Trait for GPU profilers
#[async_trait]
pub trait Profiler: Send + Sync {
    /// Start profiling a kernel
    async fn start_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<(), MonitoringError>;

    /// Stop profiling and get results
    async fn stop_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<KernelProfile, MonitoringError>;

    /// Get profiler name
    fn name(&self) -> &str;

    /// Check if profiler is available
    async fn is_available(&self) -> bool;

    /// Export profile data
    async fn export_profile(
        &self,
        profile: &KernelProfile,
        output_file: &Path,
    ) -> Result<(), MonitoringError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Test implementation
    struct TestProfiler {
        name: String,
        available: bool,
    }

    #[async_trait]
    impl Profiler for TestProfiler {
        async fn start_profile(
            &self,
            _container_id: Uuid,
            _kernel_id: &str,
        ) -> Result<(), MonitoringError> {
            if self.available {
                Ok(())
            } else {
                Err(MonitoringError::ProfilerFailed {
                    reason: "Not available".to_string(),
                })
            }
        }

        async fn stop_profile(
            &self,
            container_id: Uuid,
            kernel_id: &str,
        ) -> Result<KernelProfile, MonitoringError> {
            if self.available {
                Ok(KernelProfile::new(kernel_id.to_string(), container_id))
            } else {
                Err(MonitoringError::ProfilerFailed {
                    reason: "Not available".to_string(),
                })
            }
        }

        fn name(&self) -> &str {
            &self.name
        }

        async fn is_available(&self) -> bool {
            self.available
        }

        async fn export_profile(
            &self,
            _profile: &KernelProfile,
            _output_file: &PathBuf,
        ) -> Result<(), MonitoringError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_profiler_trait_implementation() {
        let profiler = TestProfiler {
            name: "test".to_string(),
            available: true,
        };

        assert_eq!(profiler.name(), "test");
        assert!(profiler.is_available().await);

        let container_id = Uuid::new_v4();
        profiler
            .start_profile(container_id, "kernel1")
            .await
            .unwrap();

        let profile = profiler
            .stop_profile(container_id, "kernel1")
            .await
            .unwrap();
        assert_eq!(profile.kernel_id, "kernel1");
    }

    #[tokio::test]
    async fn test_profiler_unavailable() {
        let profiler = TestProfiler {
            name: "unavailable".to_string(),
            available: false,
        };

        assert!(!profiler.is_available().await);

        let container_id = Uuid::new_v4();
        let result = profiler.start_profile(container_id, "kernel1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_profiler_as_trait_object() {
        let profiler: Arc<dyn Profiler> = Arc::new(TestProfiler {
            name: "dynamic".to_string(),
            available: true,
        });

        assert_eq!(profiler.name(), "dynamic");
        assert!(profiler.is_available().await);
    }
}
