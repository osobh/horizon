//! Mock build backend for integration tests
//!
//! This provides a mock implementation of the build backend that doesn't
//! require Docker or kernel modules, allowing tests to run anywhere.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use swarmlet::build_backend::{BackendCapabilities, BackendType, BuildBackend, BuildContext};
use swarmlet::build_job::{BuildArtifact, BuildResourceUsage, BuildResult, CargoCommand};
use swarmlet::{Result, SwarmletError};

/// Configuration for mock build behavior
#[derive(Debug, Clone)]
pub struct MockBuildConfig {
    /// Simulated build duration
    pub duration: Duration,
    /// Exit code to return (0 = success)
    pub exit_code: i32,
    /// Whether to simulate timeout
    pub simulate_timeout: bool,
    /// Resource usage to report
    pub resource_usage: BuildResourceUsage,
    /// Artifacts to return
    pub artifacts: Vec<BuildArtifact>,
}

impl Default for MockBuildConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_millis(100),
            exit_code: 0,
            simulate_timeout: false,
            resource_usage: BuildResourceUsage {
                cpu_seconds: 10.0,
                peak_memory_mb: 256.0,
                disk_read_bytes: 10 * 1024 * 1024,
                disk_write_bytes: 20 * 1024 * 1024,
                compile_time_seconds: 10.0,
                crates_compiled: 15,
                cache_hits: 5,
                cache_misses: 10,
            },
            artifacts: vec![],
        }
    }
}

impl MockBuildConfig {
    /// Create a successful build config
    pub fn success() -> Self {
        Self::default()
    }

    /// Create a failing build config
    pub fn failure(exit_code: i32) -> Self {
        Self {
            exit_code,
            ..Default::default()
        }
    }

    /// Create a timeout config
    pub fn timeout() -> Self {
        Self {
            simulate_timeout: true,
            exit_code: -1,
            ..Default::default()
        }
    }

    /// Set custom duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set custom resource usage
    pub fn with_resource_usage(mut self, usage: BuildResourceUsage) -> Self {
        self.resource_usage = usage;
        self
    }
}

/// Mock build backend for testing
pub struct MockBuildBackend {
    /// Build configurations keyed by job ID pattern
    configs: Arc<RwLock<HashMap<String, MockBuildConfig>>>,
    /// Default configuration
    default_config: MockBuildConfig,
    /// Active workspaces
    workspaces: Arc<RwLock<HashMap<String, PathBuf>>>,
    /// Build execution counter
    build_count: AtomicU32,
    /// Record of executed builds
    executed_builds: Arc<RwLock<Vec<ExecutedBuild>>>,
}

/// Record of an executed build for verification
#[derive(Debug, Clone)]
pub struct ExecutedBuild {
    pub job_id: String,
    pub command: CargoCommand,
    pub cargo_args: Vec<String>,
    pub environment: HashMap<String, String>,
    pub result: BuildResult,
}

impl MockBuildBackend {
    /// Create a new mock backend with default success behavior
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            default_config: MockBuildConfig::default(),
            workspaces: Arc::new(RwLock::new(HashMap::new())),
            build_count: AtomicU32::new(0),
            executed_builds: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a mock backend that always fails
    pub fn failing(exit_code: i32) -> Self {
        Self {
            default_config: MockBuildConfig::failure(exit_code),
            ..Self::new()
        }
    }

    /// Set configuration for a specific job ID
    pub async fn set_config(&self, job_id: &str, config: MockBuildConfig) {
        self.configs
            .write()
            .await
            .insert(job_id.to_string(), config);
    }

    /// Get the number of builds executed
    pub fn build_count(&self) -> u32 {
        self.build_count.load(Ordering::SeqCst)
    }

    /// Get all executed builds
    pub async fn get_executed_builds(&self) -> Vec<ExecutedBuild> {
        self.executed_builds.read().await.clone()
    }

    /// Clear execution history
    pub async fn clear_history(&self) {
        self.executed_builds.write().await.clear();
        self.build_count.store(0, Ordering::SeqCst);
    }

    /// Get config for a job
    async fn get_config(&self, job_id: &str) -> MockBuildConfig {
        self.configs
            .read()
            .await
            .get(job_id)
            .cloned()
            .unwrap_or_else(|| self.default_config.clone())
    }
}

impl Default for MockBuildBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BuildBackend for MockBuildBackend {
    async fn execute_cargo(
        &self,
        command: &CargoCommand,
        context: &BuildContext,
    ) -> Result<BuildResult> {
        // Get job ID from workspace path
        let job_id = context
            .workspace
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let config = self.get_config(&job_id).await;

        // Simulate build duration
        if !config.simulate_timeout {
            tokio::time::sleep(config.duration).await;
        }

        let result = BuildResult {
            exit_code: config.exit_code,
            resource_usage: config.resource_usage.clone(),
            duration: config.duration,
            artifacts: config.artifacts.clone(),
        };

        // Record the execution
        let executed = ExecutedBuild {
            job_id: job_id.clone(),
            command: command.clone(),
            cargo_args: context.cargo_args.clone(),
            environment: context.environment.clone(),
            result: result.clone(),
        };

        self.executed_builds.write().await.push(executed);
        self.build_count.fetch_add(1, Ordering::SeqCst);

        if config.simulate_timeout {
            return Err(SwarmletError::WorkloadExecution(
                "Build timed out".to_string(),
            ));
        }

        Ok(result)
    }

    async fn create_workspace(&self, job_id: &str) -> Result<PathBuf> {
        let workspace = std::env::temp_dir()
            .join("swarmlet-mock-builds")
            .join(job_id);

        tokio::fs::create_dir_all(&workspace).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create workspace: {}", e),
            ))
        })?;

        self.workspaces
            .write()
            .await
            .insert(job_id.to_string(), workspace.clone());

        Ok(workspace)
    }

    async fn cleanup_workspace(&self, workspace: &PathBuf) -> Result<()> {
        if workspace.exists() {
            tokio::fs::remove_dir_all(workspace).await.map_err(|e| {
                SwarmletError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to cleanup workspace: {}", e),
                ))
            })?;
        }

        // Remove from tracking
        let job_id = workspace.file_name().and_then(|n| n.to_str()).unwrap_or("");
        self.workspaces.write().await.remove(job_id);

        Ok(())
    }

    async fn is_available(&self) -> bool {
        true // Mock is always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            full_namespace_isolation: false,
            cgroups_v2: false,
            gpu_passthrough: false,
            seccomp: false,
            user_namespace: false,
            max_containers: 100,
        }
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Docker // Pretend to be Docker for compatibility
    }

    fn name(&self) -> &str {
        "mock"
    }
}

/// Helper to create a mock backend wrapped in Arc for sharing
pub fn create_mock_backend() -> Arc<MockBuildBackend> {
    Arc::new(MockBuildBackend::new())
}

/// Helper to create a failing mock backend
pub fn create_failing_mock_backend(exit_code: i32) -> Arc<MockBuildBackend> {
    Arc::new(MockBuildBackend::failing(exit_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_backend_success() {
        let backend = MockBuildBackend::new();
        let workspace = backend.create_workspace("test-job").await.unwrap();

        let context = BuildContext::new(workspace.clone(), PathBuf::from("/usr/bin"));
        let result = backend
            .execute_cargo(&CargoCommand::Build, &context)
            .await
            .unwrap();

        assert_eq!(result.exit_code, 0);
        assert_eq!(backend.build_count(), 1);

        backend.cleanup_workspace(&workspace).await.unwrap();
    }

    #[tokio::test]
    async fn test_mock_backend_failure() {
        let backend = MockBuildBackend::failing(1);
        let workspace = backend.create_workspace("test-job").await.unwrap();

        let context = BuildContext::new(workspace.clone(), PathBuf::from("/usr/bin"));
        let result = backend
            .execute_cargo(&CargoCommand::Build, &context)
            .await
            .unwrap();

        assert_eq!(result.exit_code, 1);

        backend.cleanup_workspace(&workspace).await.unwrap();
    }

    #[tokio::test]
    async fn test_mock_backend_custom_config() {
        let backend = MockBuildBackend::new();
        backend
            .set_config("custom-job", MockBuildConfig::failure(42))
            .await;

        let workspace = backend.create_workspace("custom-job").await.unwrap();
        let context = BuildContext::new(workspace.clone(), PathBuf::from("/usr/bin"));

        let result = backend
            .execute_cargo(&CargoCommand::Build, &context)
            .await
            .unwrap();

        assert_eq!(result.exit_code, 42);

        backend.cleanup_workspace(&workspace).await.unwrap();
    }
}
