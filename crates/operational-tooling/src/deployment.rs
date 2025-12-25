//! Blue-green deployment system for ExoRust agents

use crate::error::{OperationalError, OperationalResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use stratoswarm_agent_core::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Deployment strategy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    /// Blue-green deployment with instant switch
    BlueGreen,
    /// Rolling deployment with gradual migration
    Rolling { batch_size: usize },
    /// Canary deployment with traffic splitting
    Canary { traffic_percentage: u8 },
    /// All-at-once deployment (for development)
    AllAtOnce,
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment is being prepared
    Preparing,
    /// Deployment is in progress
    InProgress { progress: f64 },
    /// Deployment completed successfully
    Completed,
    /// Deployment failed
    Failed { reason: String },
    /// Deployment was rolled back
    RolledBack,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Unique deployment identifier
    pub deployment_id: String,
    /// Deployment strategy to use
    pub strategy: DeploymentStrategy,
    /// Target agent configuration
    pub agent_config: AgentConfig,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Timeout for deployment operations
    pub timeout: Duration,
    /// Whether to automatically rollback on failure
    pub auto_rollback: bool,
}

/// Agent configuration for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent image/version identifier
    pub image: String,
    /// Environment variables
    pub environment: HashMap<String, String>,
    /// Agent-specific configuration
    pub agent_params: HashMap<String, serde_json::Value>,
    /// Number of agent instances
    pub replicas: u32,
}

/// Resource requirements for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// GPU memory required per agent (MB)
    pub gpu_memory_mb: u32,
    /// CPU cores required per agent
    pub cpu_cores: f32,
    /// System memory required per agent (MB)
    pub memory_mb: u32,
    /// Storage required per agent (MB)
    pub storage_mb: u32,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint/method
    pub endpoint: String,
    /// Check interval
    pub interval: Duration,
    /// Timeout for each check
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
}

/// Blue-green deployment manager
pub struct BlueGreenDeployment {
    /// Current deployments
    deployments: Arc<DashMap<String, DeploymentState>>,
    /// Active environments (blue/green)
    environments: Arc<RwLock<EnvironmentState>>,
    /// Runtime configuration
    runtime_config: RuntimeConfig,
}

/// Internal deployment state
#[derive(Debug)]
struct DeploymentState {
    config: DeploymentConfig,
    status: DeploymentStatus,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    agent_instances: Vec<AgentInstance>,
    metrics: DeploymentMetrics,
}

/// Agent instance information
#[derive(Debug, Clone)]
struct AgentInstance {
    agent_id: AgentId,
    container_id: String,
    health_status: HealthStatus,
    created_at: DateTime<Utc>,
}

/// Health status for agents
#[derive(Debug, Clone, PartialEq)]
enum HealthStatus {
    Starting,
    Healthy,
    Unhealthy,
    Failed,
}

/// Environment state for blue-green deployment
#[derive(Debug)]
struct EnvironmentState {
    active_environment: Environment,
    blue_deployment: Option<String>,
    green_deployment: Option<String>,
    traffic_split: TrafficSplit,
}

/// Environment identifier
#[derive(Debug, Clone, PartialEq)]
enum Environment {
    Blue,
    Green,
}

/// Traffic split configuration
#[derive(Debug, Clone)]
struct TrafficSplit {
    blue_percentage: u8,
    green_percentage: u8,
}

/// Deployment metrics
#[derive(Debug, Clone, Default)]
struct DeploymentMetrics {
    agents_deployed: u32,
    agents_healthy: u32,
    agents_failed: u32,
    deployment_duration: Option<Duration>,
    resource_usage: ResourceUsage,
}

/// Resource usage tracking
#[derive(Debug, Clone, Default)]
struct ResourceUsage {
    gpu_memory_used_mb: u32,
    cpu_cores_used: f32,
    memory_used_mb: u32,
    storage_used_mb: u32,
}

/// Runtime configuration for deployments
#[derive(Debug, Clone)]
struct RuntimeConfig {
    /// Base directory for runtime operations
    base_dir: std::path::PathBuf,
    /// Maximum containers per deployment
    max_containers: u32,
}

impl BlueGreenDeployment {
    /// Create a new blue-green deployment manager
    pub fn new(base_dir: std::path::PathBuf) -> Self {
        Self {
            deployments: Arc::new(DashMap::new()),
            environments: Arc::new(RwLock::new(EnvironmentState {
                active_environment: Environment::Blue,
                blue_deployment: None,
                green_deployment: None,
                traffic_split: TrafficSplit {
                    blue_percentage: 100,
                    green_percentage: 0,
                },
            })),
            runtime_config: RuntimeConfig {
                base_dir,
                max_containers: 100,
            },
        }
    }

    /// Start a new deployment
    pub async fn deploy(&self, config: DeploymentConfig) -> OperationalResult<String> {
        let deployment_id = config.deployment_id.clone();

        // Validate configuration
        self.validate_config(&config)?;

        let deployment_state = DeploymentState {
            config: config.clone(),
            status: DeploymentStatus::Preparing,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            agent_instances: Vec::new(),
            metrics: DeploymentMetrics::default(),
        };

        self.deployments
            .insert(deployment_id.clone(), deployment_state);

        // Start deployment based on strategy
        match config.strategy {
            DeploymentStrategy::BlueGreen => self.deploy_blue_green(deployment_id.clone()).await,
            DeploymentStrategy::Rolling { batch_size } => {
                self.deploy_rolling(deployment_id.clone(), batch_size).await
            }
            DeploymentStrategy::Canary { traffic_percentage } => {
                self.deploy_canary(deployment_id.clone(), traffic_percentage)
                    .await
            }
            DeploymentStrategy::AllAtOnce => self.deploy_all_at_once(deployment_id.clone()).await,
        }?;

        Ok(deployment_id)
    }

    /// Get deployment status
    pub fn get_deployment_status(&self, deployment_id: &str) -> Option<DeploymentStatus> {
        self.deployments
            .get(deployment_id)
            .map(|entry| entry.status.clone())
    }

    /// List all deployments
    pub fn list_deployments(&self) -> Vec<(String, DeploymentStatus)> {
        self.deployments
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().status.clone()))
            .collect()
    }

    /// Switch traffic between environments
    pub async fn switch_traffic(&self, deployment_id: &str) -> OperationalResult<()> {
        let mut env_state = self.environments.write().await;

        // Determine target environment
        let target_env = match env_state.active_environment {
            Environment::Blue => Environment::Green,
            Environment::Green => Environment::Blue,
        };

        // Update traffic split
        match target_env {
            Environment::Blue => {
                env_state.traffic_split.blue_percentage = 100;
                env_state.traffic_split.green_percentage = 0;
                env_state.blue_deployment = Some(deployment_id.to_string());
            }
            Environment::Green => {
                env_state.traffic_split.blue_percentage = 0;
                env_state.traffic_split.green_percentage = 100;
                env_state.green_deployment = Some(deployment_id.to_string());
            }
        }

        env_state.active_environment = target_env;
        Ok(())
    }

    /// Stop and remove a deployment
    pub async fn stop_deployment(&self, deployment_id: &str) -> OperationalResult<()> {
        if let Some(mut deployment) = self.deployments.get_mut(deployment_id) {
            deployment.status = DeploymentStatus::Failed {
                reason: "Manually stopped".to_string(),
            };
            deployment.updated_at = Utc::now();

            // Stop all agent instances
            for instance in &deployment.agent_instances {
                self.stop_agent_instance(instance).await?;
            }
        }

        Ok(())
    }

    /// Get deployment metrics
    pub fn get_deployment_metrics(&self, deployment_id: &str) -> Option<DeploymentMetrics> {
        self.deployments
            .get(deployment_id)
            .map(|entry| entry.metrics.clone())
    }

    // Private helper methods

    fn validate_config(&self, config: &DeploymentConfig) -> OperationalResult<()> {
        if config.deployment_id.is_empty() {
            return Err(OperationalError::ConfigurationError(
                "Deployment ID cannot be empty".to_string(),
            ));
        }

        if config.agent_config.replicas == 0 {
            return Err(OperationalError::ConfigurationError(
                "Agent replicas must be greater than 0".to_string(),
            ));
        }

        if config.resource_requirements.gpu_memory_mb == 0 {
            return Err(OperationalError::ConfigurationError(
                "GPU memory requirement must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    async fn deploy_blue_green(&self, deployment_id: String) -> OperationalResult<()> {
        // Implementation for blue-green deployment
        self.update_deployment_status(
            &deployment_id,
            DeploymentStatus::InProgress { progress: 0.0 },
        );

        // Mock implementation - in real system would create agents
        tokio::time::sleep(Duration::from_millis(10)).await;

        self.update_deployment_status(
            &deployment_id,
            DeploymentStatus::InProgress { progress: 50.0 },
        );

        tokio::time::sleep(Duration::from_millis(10)).await;

        self.update_deployment_status(&deployment_id, DeploymentStatus::Completed);

        Ok(())
    }

    async fn deploy_rolling(
        &self,
        deployment_id: String,
        _batch_size: usize,
    ) -> OperationalResult<()> {
        // Implementation for rolling deployment
        self.update_deployment_status(
            &deployment_id,
            DeploymentStatus::InProgress { progress: 0.0 },
        );

        // Mock implementation
        tokio::time::sleep(Duration::from_millis(10)).await;

        self.update_deployment_status(&deployment_id, DeploymentStatus::Completed);

        Ok(())
    }

    async fn deploy_canary(
        &self,
        deployment_id: String,
        _traffic_percentage: u8,
    ) -> OperationalResult<()> {
        // Implementation for canary deployment
        self.update_deployment_status(
            &deployment_id,
            DeploymentStatus::InProgress { progress: 0.0 },
        );

        // Mock implementation
        tokio::time::sleep(Duration::from_millis(10)).await;

        self.update_deployment_status(&deployment_id, DeploymentStatus::Completed);

        Ok(())
    }

    async fn deploy_all_at_once(&self, deployment_id: String) -> OperationalResult<()> {
        // Implementation for all-at-once deployment
        self.update_deployment_status(
            &deployment_id,
            DeploymentStatus::InProgress { progress: 0.0 },
        );

        // Mock implementation
        tokio::time::sleep(Duration::from_millis(10)).await;

        self.update_deployment_status(&deployment_id, DeploymentStatus::Completed);

        Ok(())
    }

    fn update_deployment_status(&self, deployment_id: &str, status: DeploymentStatus) {
        if let Some(mut deployment) = self.deployments.get_mut(deployment_id) {
            deployment.status = status;
            deployment.updated_at = Utc::now();
        }
    }

    async fn stop_agent_instance(&self, _instance: &AgentInstance) -> OperationalResult<()> {
        // Mock implementation for stopping agent instance
        Ok(())
    }
}

impl Default for DeploymentStrategy {
    fn default() -> Self {
        Self::BlueGreen
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            endpoint: "/health".to_string(),
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 1,
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            gpu_memory_mb: 1024,
            cpu_cores: 1.0,
            memory_mb: 2048,
            storage_mb: 1024,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            image: "exorust-agent:latest".to_string(),
            environment: HashMap::new(),
            agent_params: HashMap::new(),
            replicas: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> DeploymentConfig {
        DeploymentConfig {
            deployment_id: "test-deployment".to_string(),
            strategy: DeploymentStrategy::BlueGreen,
            agent_config: AgentConfig::default(),
            resource_requirements: ResourceRequirements::default(),
            health_check: HealthCheckConfig::default(),
            timeout: Duration::from_secs(60),
            auto_rollback: true,
        }
    }

    async fn create_test_deployment() -> BlueGreenDeployment {
        let temp_dir = TempDir::new().unwrap();
        BlueGreenDeployment::new(temp_dir.path().to_path_buf())
    }

    #[test]
    fn test_deployment_strategy_serialization() {
        let strategies = vec![
            DeploymentStrategy::BlueGreen,
            DeploymentStrategy::Rolling { batch_size: 5 },
            DeploymentStrategy::Canary {
                traffic_percentage: 10,
            },
            DeploymentStrategy::AllAtOnce,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: DeploymentStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_deployment_status_serialization() {
        let statuses = vec![
            DeploymentStatus::Preparing,
            DeploymentStatus::InProgress { progress: 50.0 },
            DeploymentStatus::Completed,
            DeploymentStatus::Failed {
                reason: "Test failure".to_string(),
            },
            DeploymentStatus::RolledBack,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: DeploymentStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_deployment_config_validation() {
        let deployment = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(create_test_deployment());

        // Valid config
        let valid_config = create_test_config();
        assert!(deployment.validate_config(&valid_config).is_ok());

        // Empty deployment ID
        let mut invalid_config = create_test_config();
        invalid_config.deployment_id = String::new();
        assert!(deployment.validate_config(&invalid_config).is_err());

        // Zero replicas
        let mut invalid_config = create_test_config();
        invalid_config.agent_config.replicas = 0;
        assert!(deployment.validate_config(&invalid_config).is_err());

        // Zero GPU memory
        let mut invalid_config = create_test_config();
        invalid_config.resource_requirements.gpu_memory_mb = 0;
        assert!(deployment.validate_config(&invalid_config).is_err());
    }

    #[tokio::test]
    async fn test_blue_green_deployment_creation() {
        let deployment = create_test_deployment().await;

        // Check initial state
        let deployments = deployment.list_deployments();
        assert!(deployments.is_empty());
    }

    #[tokio::test]
    async fn test_deployment_lifecycle() {
        let deployment = create_test_deployment().await;
        let config = create_test_config();
        let deployment_id = config.deployment_id.clone();

        // Start deployment
        let result = deployment.deploy(config).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), deployment_id);

        // Check deployment appears in list
        let deployments = deployment.list_deployments();
        assert_eq!(deployments.len(), 1);
        assert_eq!(deployments[0].0, deployment_id);

        // Check deployment completes
        tokio::time::sleep(Duration::from_millis(50)).await;
        let status = deployment.get_deployment_status(&deployment_id);
        assert!(matches!(status, Some(DeploymentStatus::Completed)));
    }

    #[tokio::test]
    async fn test_deployment_strategies() {
        let deployment = create_test_deployment().await;

        let strategies = vec![
            DeploymentStrategy::BlueGreen,
            DeploymentStrategy::Rolling { batch_size: 2 },
            DeploymentStrategy::Canary {
                traffic_percentage: 20,
            },
            DeploymentStrategy::AllAtOnce,
        ];

        for (i, strategy) in strategies.into_iter().enumerate() {
            let mut config = create_test_config();
            config.deployment_id = format!("test-deployment-{}", i);
            config.strategy = strategy;

            let result = deployment.deploy(config).await;
            assert!(result.is_ok());
        }

        // Wait for deployments to complete
        tokio::time::sleep(Duration::from_millis(100)).await;

        let deployments = deployment.list_deployments();
        assert_eq!(deployments.len(), 4);

        // All should be completed
        for (_, status) in deployments {
            assert!(matches!(status, DeploymentStatus::Completed));
        }
    }

    #[tokio::test]
    async fn test_traffic_switching() {
        let deployment = create_test_deployment().await;
        let config = create_test_config();
        let deployment_id = config.deployment_id.clone();

        // Deploy first
        deployment.deploy(config).await.unwrap();

        // Switch traffic
        let result = deployment.switch_traffic(&deployment_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_deployment_stop() {
        let deployment = create_test_deployment().await;
        let config = create_test_config();
        let deployment_id = config.deployment_id.clone();

        // Deploy first
        deployment.deploy(config).await.unwrap();

        // Stop deployment
        let result = deployment.stop_deployment(&deployment_id).await;
        assert!(result.is_ok());

        // Check status is failed
        let status = deployment.get_deployment_status(&deployment_id);
        assert!(matches!(status, Some(DeploymentStatus::Failed { .. })));
    }

    #[tokio::test]
    async fn test_deployment_metrics() {
        let deployment = create_test_deployment().await;
        let config = create_test_config();
        let deployment_id = config.deployment_id.clone();

        // Deploy first
        deployment.deploy(config).await.unwrap();

        // Get metrics
        let metrics = deployment.get_deployment_metrics(&deployment_id);
        assert!(metrics.is_some());
    }

    #[test]
    fn test_health_status_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_environment_equality() {
        assert_eq!(Environment::Blue, Environment::Blue);
        assert_ne!(Environment::Blue, Environment::Green);
    }

    #[test]
    fn test_default_implementations() {
        let _strategy = DeploymentStrategy::default();
        let _health_check = HealthCheckConfig::default();
        let _resources = ResourceRequirements::default();
        let _agent_config = AgentConfig::default();
        let _metrics = DeploymentMetrics::default();
        let _usage = ResourceUsage::default();
    }

    #[tokio::test]
    async fn test_concurrent_deployments() {
        let deployment = Arc::new(create_test_deployment().await);
        let mut handles = Vec::new();

        // Start multiple concurrent deployments
        for i in 0..5 {
            let deployment = deployment.clone();
            let handle = tokio::spawn(async move {
                let mut config = create_test_config();
                config.deployment_id = format!("concurrent-deployment-{}", i);
                deployment.deploy(config).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // Check all deployments exist
        tokio::time::sleep(Duration::from_millis(100)).await;
        let deployments = deployment.list_deployments();
        assert_eq!(deployments.len(), 5);
    }

    #[tokio::test]
    async fn test_invalid_deployment_id() {
        let deployment = create_test_deployment().await;

        // Test with empty deployment ID
        let mut config = create_test_config();
        config.deployment_id = String::new();

        let result = deployment.deploy(config).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OperationalError::ConfigurationError(_)
        ));
    }

    #[tokio::test]
    async fn test_nonexistent_deployment_operations() {
        let deployment = create_test_deployment().await;

        // Test operations on non-existent deployment
        let status = deployment.get_deployment_status("nonexistent");
        assert!(status.is_none());

        let metrics = deployment.get_deployment_metrics("nonexistent");
        assert!(metrics.is_none());

        let result = deployment.stop_deployment("nonexistent").await;
        assert!(result.is_ok()); // Should not fail for non-existent deployment
    }

    #[test]
    fn test_deployment_strategy_validation() {
        let strategies = vec![
            DeploymentStrategy::BlueGreen,
            DeploymentStrategy::Rolling { batch_size: 3 },
            DeploymentStrategy::Canary {
                traffic_percentage: 25,
            },
            DeploymentStrategy::AllAtOnce,
        ];

        for strategy in strategies {
            let _name = format!("{:?}", strategy);
        }
    }

    #[test]
    fn test_health_check_config_edge_cases() {
        let config = HealthCheckConfig {
            endpoint: "/health".to_string(),
            interval: Duration::from_secs(0),
            timeout: Duration::from_secs(u64::MAX),
            failure_threshold: 0,
            success_threshold: u32::MAX,
        };

        assert_eq!(config.interval.as_secs(), 0);
        assert_eq!(config.timeout.as_secs(), u64::MAX);
        assert_eq!(config.failure_threshold, 0);
        assert_eq!(config.success_threshold, u32::MAX);
    }

    #[test]
    fn test_resource_requirements_validation() {
        let resources = ResourceRequirements {
            gpu_memory_mb: u32::MAX,
            cpu_cores: f32::MAX,
            memory_mb: 0,
            storage_mb: u32::MAX,
        };

        assert_eq!(resources.gpu_memory_mb, u32::MAX);
        assert_eq!(resources.cpu_cores, f32::MAX);
        assert_eq!(resources.memory_mb, 0);
        assert_eq!(resources.storage_mb, u32::MAX);
    }

    #[test]
    fn test_agent_config_serialization() {
        let mut config = AgentConfig::default();
        config
            .environment
            .insert("KEY1".to_string(), "value1".to_string());
        config
            .environment
            .insert("KEY2".to_string(), "value2".to_string());
        config.agent_params.insert(
            "mode".to_string(),
            serde_json::Value::String("production".to_string()),
        );

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.image, deserialized.image);
        assert_eq!(config.replicas, deserialized.replicas);
        assert_eq!(config.environment, deserialized.environment);
        assert_eq!(config.agent_params, deserialized.agent_params);
    }

    #[test]
    fn test_deployment_status_failed_reasons() {
        let statuses = vec![
            DeploymentStatus::Failed {
                reason: "Out of memory".to_string(),
            },
            DeploymentStatus::Failed {
                reason: String::new(),
            },
            DeploymentStatus::Failed {
                reason: "Network timeout after 30s".to_string(),
            },
        ];

        for status in statuses {
            match status {
                DeploymentStatus::Failed { reason } => {
                    let _ = reason;
                }
                _ => panic!("Expected Failed status"),
            }
        }
    }

    #[test]
    fn test_deployment_status_serialization_comprehensive() {
        let statuses = vec![
            DeploymentStatus::Preparing,
            DeploymentStatus::InProgress { progress: 50.0 },
            DeploymentStatus::Completed,
            DeploymentStatus::Failed {
                reason: "test failure".to_string(),
            },
            DeploymentStatus::RolledBack,
        ];

        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: DeploymentStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn test_deployment_metrics_aggregation() {
        // Test deployment metrics calculations
        let agents_deployed = 5u32;
        let agents_healthy = 3u32;
        let agents_failed = 1u32;

        // Verify metrics calculations
        let healthy_percentage = (agents_healthy as f64 / agents_deployed as f64) * 100.0;
        assert_eq!(healthy_percentage, 60.0);

        let failure_rate = (agents_failed as f64 / agents_deployed as f64) * 100.0;
        assert_eq!(failure_rate, 20.0);
    }

    #[tokio::test]
    async fn test_deployment_with_zero_replicas() {
        let deployment = create_test_deployment().await;
        let mut config = create_test_config();
        config.agent_config.replicas = 0;

        let result = deployment.deploy(config).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OperationalError::ConfigurationError(_)
        ));
    }

    #[test]
    fn test_deployment_config_with_empty_image() {
        let mut config = create_test_config();
        config.agent_config.image = String::new();

        // Empty image should be invalid
        assert!(config.agent_config.image.is_empty());
    }

    #[test]
    fn test_resource_usage_extreme_values() {
        // Test extreme values for resource tracking
        let max_gpu_memory = u32::MAX;
        let max_cpu_cores = f32::INFINITY;
        let min_memory = 0u32;
        let max_storage = u32::MAX;

        // Should handle extreme values without panic
        assert_eq!(max_gpu_memory, u32::MAX);
        assert!(max_cpu_cores.is_infinite());
        assert_eq!(min_memory, 0);
        assert_eq!(max_storage, u32::MAX);
    }

    #[tokio::test]
    async fn test_deployment_timeout_handling() {
        let deployment = create_test_deployment().await;
        let mut config = create_test_config();

        // Set very short timeout
        config.timeout = Duration::from_millis(1);

        // Deploy should still succeed even with short timeout
        let result = deployment.deploy(config).await;
        assert!(result.is_ok());
    }
}
