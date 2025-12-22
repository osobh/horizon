//! Configuration management for swarmlet

use crate::{defaults, Result, SwarmletError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

/// Swarmlet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Node configuration
    pub node: NodeConfig,

    /// Network configuration
    pub network: NetworkConfig,

    /// Storage configuration
    pub storage: StorageConfig,

    /// Resource limits
    pub resources: ResourceConfig,

    /// Security configuration
    pub security: SecurityConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    // Additional fields for agent operation
    #[serde(skip)]
    pub data_dir: PathBuf,

    #[serde(skip)]
    pub api_port: Option<u16>,
}

/// Node-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node name (defaults to hostname)
    pub name: Option<String>,

    /// Node labels for scheduling
    pub labels: std::collections::HashMap<String, String>,

    /// Maximum number of concurrent workloads
    pub max_workloads: Option<u32>,

    /// Node capabilities to advertise
    pub capabilities: Vec<String>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Discovery port for cluster mesh
    pub discovery_port: u16,

    /// API port for local HTTP server
    pub api_port: u16,

    /// Metrics port for monitoring
    pub metrics_port: u16,

    /// Bind interface (e.g., "eth0", "0.0.0.0")
    pub bind_interface: String,

    /// Enable TLS for cluster communication
    pub tls_enabled: bool,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Data directory for persistent storage
    pub data_dir: PathBuf,

    /// Workload directory for container volumes
    pub workload_dir: PathBuf,

    /// Log directory
    pub log_dir: PathBuf,

    /// Maximum storage usage in GB
    pub max_storage_gb: Option<f32>,

    /// Cleanup policy for old data
    pub cleanup_policy: CleanupPolicy,
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum CPU cores to allocate to workloads
    pub max_cpu_cores: Option<f32>,

    /// Maximum memory in GB to allocate to workloads
    pub max_memory_gb: Option<f32>,

    /// Maximum disk space in GB for workloads
    pub max_disk_gb: Option<f32>,

    /// CPU reservation (always keep this much free)
    pub cpu_reservation_percent: f32,

    /// Memory reservation (always keep this much free)
    pub memory_reservation_percent: f32,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable authentication for local API
    pub api_auth_enabled: bool,

    /// API token for local access
    pub api_token: Option<String>,

    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,

    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,

    /// Trusted CA certificates path
    pub ca_cert_path: Option<PathBuf>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Log to file instead of stdout
    pub log_to_file: bool,

    /// Log file path
    pub log_file: Option<PathBuf>,

    /// Maximum log file size in MB
    pub max_log_size_mb: u32,

    /// Number of log files to keep
    pub log_file_count: u32,

    /// Export logs to cluster
    pub export_to_cluster: bool,
}

/// Cleanup policy for old data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupPolicy {
    /// Never cleanup (manual cleanup required)
    Never,

    /// Cleanup after specified days
    AfterDays(u32),

    /// Cleanup when disk usage exceeds percentage
    WhenDiskUsageExceeds(f32),

    /// Cleanup oldest files when storage limit reached
    LeastRecentlyUsed,
}

impl Config {
    /// Load configuration from file
    pub async fn load(config_path: &str) -> Result<Self> {
        let config_content = fs::read_to_string(config_path).await?;
        let config: Config = toml::from_str(&config_content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save(&self, config_path: &str) -> Result<()> {
        let config_content = toml::to_string_pretty(self)
            .map_err(|e| SwarmletError::Configuration(format!("Serialization error: {e}")))?;

        fs::write(config_path, config_content).await?;
        Ok(())
    }

    /// Create default configuration
    pub fn with_defaults() -> Self {
        let data_dir = PathBuf::from(defaults::DATA_DIR);

        Self {
            node: NodeConfig {
                name: None,
                labels: std::collections::HashMap::new(),
                max_workloads: None,
                capabilities: vec![
                    "workload_execution".to_string(),
                    "health_reporting".to_string(),
                    "metrics_collection".to_string(),
                ],
            },
            network: NetworkConfig {
                discovery_port: defaults::DISCOVERY_PORT,
                api_port: defaults::API_PORT,
                metrics_port: defaults::METRICS_PORT,
                bind_interface: "0.0.0.0".to_string(),
                tls_enabled: false,
            },
            storage: StorageConfig {
                data_dir: data_dir.clone(),
                workload_dir: data_dir.join("workloads"),
                log_dir: data_dir.join("logs"),
                max_storage_gb: None,
                cleanup_policy: CleanupPolicy::AfterDays(30),
            },
            resources: ResourceConfig {
                max_cpu_cores: None,
                max_memory_gb: None,
                max_disk_gb: None,
                cpu_reservation_percent: 10.0,
                memory_reservation_percent: 20.0,
            },
            security: SecurityConfig {
                api_auth_enabled: false,
                api_token: None,
                tls_cert_path: None,
                tls_key_path: None,
                ca_cert_path: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                log_to_file: false,
                log_file: None,
                max_log_size_mb: 100,
                log_file_count: 5,
                export_to_cluster: true,
            },
            data_dir,
            api_port: Some(defaults::API_PORT),
        }
    }

    /// Create default configuration with custom data directory
    pub fn default_with_data_dir(data_dir: PathBuf) -> Self {
        let base_config = Self::with_defaults();
        Config {
            data_dir: data_dir.clone(),
            storage: StorageConfig {
                data_dir: data_dir.clone(),
                workload_dir: data_dir.join("workloads"),
                log_dir: data_dir.join("logs"),
                ..base_config.storage
            },
            ..base_config
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate ports
        if self.network.discovery_port == 0 || self.network.api_port == 0 {
            return Err(SwarmletError::Configuration(
                "Invalid port configuration".to_string(),
            ));
        }

        // Validate resource limits
        if let Some(cpu_cores) = self.resources.max_cpu_cores {
            if cpu_cores <= 0.0 {
                return Err(SwarmletError::Configuration(
                    "Invalid CPU core limit".to_string(),
                ));
            }
        }

        if let Some(memory_gb) = self.resources.max_memory_gb {
            if memory_gb <= 0.0 {
                return Err(SwarmletError::Configuration(
                    "Invalid memory limit".to_string(),
                ));
            }
        }

        // Validate reservation percentages
        if self.resources.cpu_reservation_percent < 0.0
            || self.resources.cpu_reservation_percent > 100.0
        {
            return Err(SwarmletError::Configuration(
                "Invalid CPU reservation percentage".to_string(),
            ));
        }

        if self.resources.memory_reservation_percent < 0.0
            || self.resources.memory_reservation_percent > 100.0
        {
            return Err(SwarmletError::Configuration(
                "Invalid memory reservation percentage".to_string(),
            ));
        }

        // Validate log level
        match self.logging.level.to_lowercase().as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => {
                return Err(SwarmletError::Configuration(
                    "Invalid log level".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Create directories specified in configuration
    pub async fn create_directories(&self) -> Result<()> {
        let dirs = [
            &self.storage.data_dir,
            &self.storage.workload_dir,
            &self.storage.log_dir,
        ];

        for dir in &dirs {
            if !dir.exists() {
                fs::create_dir_all(dir).await.map_err(|e| {
                    SwarmletError::Configuration(format!(
                        "Failed to create directory {dir:?}: {e}"
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Get effective resource limits based on system capabilities
    pub async fn get_effective_limits(&self) -> Result<EffectiveLimits> {
        use sysinfo::System;

        let mut system = System::new_all();
        system.refresh_all();

        let total_cpu_cores = num_cpus::get() as f32;
        let total_memory_gb = system.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

        // Calculate effective limits
        let cpu_reservation = total_cpu_cores * (self.resources.cpu_reservation_percent / 100.0);
        let memory_reservation =
            total_memory_gb * (self.resources.memory_reservation_percent / 100.0);

        let max_cpu = self
            .resources
            .max_cpu_cores
            .unwrap_or(total_cpu_cores - cpu_reservation)
            .min(total_cpu_cores - cpu_reservation);

        let max_memory = self
            .resources
            .max_memory_gb
            .unwrap_or(total_memory_gb - memory_reservation)
            .min(total_memory_gb - memory_reservation);

        let max_disk = self.resources.max_disk_gb.unwrap_or(100.0); // Default 100GB if not specified

        Ok(EffectiveLimits {
            cpu_cores: max_cpu,
            memory_gb: max_memory,
            disk_gb: max_disk,
        })
    }

    /// Generate example configuration file content
    pub fn example_toml() -> String {
        let config = Self::default();
        toml::to_string_pretty(&config)
            .unwrap_or_else(|_| "# Failed to generate example config".to_string())
    }
}

/// Effective resource limits after considering system constraints
#[derive(Debug, Clone)]
pub struct EffectiveLimits {
    pub cpu_cores: f32,
    pub memory_gb: f32,
    pub disk_gb: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config_creation() {
        let config = Config::with_defaults();
        assert_eq!(config.network.discovery_port, defaults::DISCOVERY_PORT);
        assert_eq!(config.network.api_port, defaults::API_PORT);
        assert!(!config.node.capabilities.is_empty());
    }

    #[test]
    fn test_config_validation() {
        let config = Config::with_defaults();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.resources.cpu_reservation_percent = 150.0;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_custom_data_dir() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();

        let config = Config::default_with_data_dir(data_dir.clone());
        assert_eq!(config.data_dir, data_dir);
        assert_eq!(config.storage.data_dir, data_dir);
    }

    #[tokio::test]
    async fn test_effective_limits_calculation() {
        let config = Config::with_defaults();
        let limits = config.get_effective_limits().await.unwrap();

        assert!(limits.cpu_cores > 0.0);
        assert!(limits.memory_gb > 0.0);
        assert!(limits.disk_gb > 0.0);
    }

    #[test]
    fn test_example_toml_generation() {
        let toml_content = Config::example_toml();
        assert!(toml_content.contains("[node]"));
        assert!(toml_content.contains("[network]"));
        assert!(toml_content.contains("[storage]"));
    }
}
