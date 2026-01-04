//! Nsight Compute profiler integration

use super::config::ProfileConfig;
use super::kernel::KernelProfile;
use super::traits::Profiler;
use crate::MonitoringError;
use async_trait::async_trait;
use dashmap::DashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use uuid::Uuid;

// Internal types for managing profile sessions
#[derive(Clone)]
#[allow(dead_code)]
struct ProfileSession {
    container_id: Uuid,
    kernel_id: String,
    start_time: u64,
    output_file: PathBuf,
    process_id: Option<u32>,
}

/// NVIDIA Nsight Compute profiler integration
pub struct NsightComputeProfiler {
    config: ProfileConfig,
    active_sessions: Arc<DashMap<(Uuid, String), ProfileSession>>,
}

impl NsightComputeProfiler {
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            config,
            active_sessions: Arc::new(DashMap::new()),
        }
    }

    /// Check if ncu is available
    fn check_ncu_available() -> bool {
        Command::new("ncu")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Generate output filename
    fn generate_output_file(&self, container_id: Uuid, kernel_id: &str) -> PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.config
            .output_dir
            .join(format!("profile_{container_id}_{kernel_id}_{timestamp}"))
    }

    /// Parse Nsight Compute output
    #[allow(dead_code)]
    fn parse_ncu_output(&self, output: &str) -> Result<KernelProfile, MonitoringError> {
        // This is a simplified parser - real implementation would parse ncu CSV/JSON output
        let profile = KernelProfile::default();

        for line in output.lines() {
            if line.contains("Duration") {
                // Parse duration
            } else if line.contains("Memory Throughput") {
                // Parse memory throughput
            } else if line.contains("Compute Throughput") {
                // Parse compute throughput
            }
        }

        Ok(profile)
    }
}

#[async_trait]
impl Profiler for NsightComputeProfiler {
    async fn start_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<(), MonitoringError> {
        if !self.is_available().await {
            return Err(MonitoringError::ProfilerFailed {
                reason: "Nsight Compute not available".to_string(),
            });
        }

        let key = (container_id, kernel_id.to_string());

        if self.active_sessions.contains_key(&key) {
            return Err(MonitoringError::ProfilerFailed {
                reason: "Profile session already active".to_string(),
            });
        }

        let session = ProfileSession {
            container_id,
            kernel_id: kernel_id.to_string(),
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            output_file: self.generate_output_file(container_id, kernel_id),
            process_id: None,
        };

        self.active_sessions.insert(key, session);

        Ok(())
    }

    async fn stop_profile(
        &self,
        container_id: Uuid,
        kernel_id: &str,
    ) -> Result<KernelProfile, MonitoringError> {
        let key = (container_id, kernel_id.to_string());

        let (_, session) =
            self.active_sessions
                .remove(&key)
                .ok_or_else(|| MonitoringError::ProfilerFailed {
                    reason: "No active profile session found".to_string(),
                })?;

        // In a real implementation, we would:
        // 1. Stop the ncu process
        // 2. Wait for it to complete
        // 3. Parse the output file
        // 4. Convert to KernelProfile

        // For now, return a mock profile
        let duration = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - session.start_time;

        Ok(KernelProfile::new(kernel_id.to_string(), container_id)
            .with_metrics(
                duration * 1_000_000_000, // Convert to nanoseconds
                200.0,                    // Mock throughput
                1500.0,                   // Mock GFLOPS
            )
            .with_occupancy(80.0, 48))
    }

    fn name(&self) -> &str {
        "NsightComputeProfiler"
    }

    async fn is_available(&self) -> bool {
        Self::check_ncu_available()
    }

    async fn export_profile(
        &self,
        profile: &KernelProfile,
        output_file: &std::path::Path,
    ) -> Result<(), MonitoringError> {
        // In a real implementation, export to ncu report format
        let json =
            serde_json::to_string_pretty(profile).map_err(|e| MonitoringError::ProfilerFailed {
                reason: format!("Failed to serialize profile: {e}"),
            })?;

        tokio::fs::write(output_file, json)
            .await
            .map_err(|e| MonitoringError::ProfilerFailed {
                reason: format!("Failed to write profile: {e}"),
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nsight_profiler_creation() {
        let config = ProfileConfig::default();
        let profiler = NsightComputeProfiler::new(config);

        assert_eq!(profiler.name(), "NsightComputeProfiler");
    }

    #[tokio::test]
    async fn test_nsight_profiler_output_file_generation() {
        let config = ProfileConfig::default();
        let profiler = NsightComputeProfiler::new(config);

        let container_id = Uuid::new_v4();
        let output_file = profiler.generate_output_file(container_id, "test_kernel");

        assert!(output_file
            .to_string_lossy()
            .contains(&container_id.to_string()));
        assert!(output_file.to_string_lossy().contains("test_kernel"));
    }

    #[tokio::test]
    async fn test_nsight_profiler_session_management() {
        let config = ProfileConfig::default();
        let profiler = NsightComputeProfiler::new(config);
        let container_id = Uuid::new_v4();

        // Mock availability check
        if !profiler.is_available().await {
            // Skip test if ncu not available
            return;
        }

        // Start profile
        profiler
            .start_profile(container_id, "kernel1")
            .await
            .unwrap();

        // Check duplicate start fails
        let result = profiler.start_profile(container_id, "kernel1").await;
        assert!(result.is_err());

        // Stop profile
        let profile = profiler
            .stop_profile(container_id, "kernel1")
            .await
            .unwrap();

        assert_eq!(profile.kernel_id, "kernel1");
    }

    #[tokio::test]
    async fn test_nsight_profiler_export() {
        let config = ProfileConfig::default();
        let profiler = NsightComputeProfiler::new(config);

        let profile = KernelProfile::new("test".to_string(), Uuid::new_v4());
        let output_file = PathBuf::from("/tmp/test_profile.json");

        profiler
            .export_profile(&profile, &output_file)
            .await
            .unwrap();

        // Verify file exists
        assert!(tokio::fs::metadata(&output_file).await.is_ok());

        // Clean up
        let _ = tokio::fs::remove_file(&output_file).await;
    }
}
