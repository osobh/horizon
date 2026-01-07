//! Application configuration management
//!
//! Handles loading and saving configuration from ~/.hpc/config.toml

use crate::core::profile::{Profile, ProfileSettings};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Default profile to use
    #[serde(default = "default_profile")]
    pub default_profile: String,

    /// Default environment
    #[serde(default)]
    pub default_environment: String,

    /// TUI settings
    #[serde(default)]
    pub tui: TuiSettings,

    /// Deployment settings
    #[serde(default)]
    pub deploy: DeploySettings,

    /// Environment-specific settings
    #[serde(default)]
    pub profiles: ProfilesConfig,
}

fn default_profile() -> String {
    "default".to_string()
}

/// TUI-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiSettings {
    /// Update interval in milliseconds
    #[serde(default = "default_tick_rate")]
    pub tick_rate_ms: u64,

    /// Maximum log entries to display
    #[serde(default = "default_max_logs")]
    pub max_log_entries: usize,

    /// Color theme (dark, light)
    #[serde(default = "default_theme")]
    pub theme: String,

    /// Enable mouse support
    #[serde(default = "default_true")]
    pub mouse_support: bool,

    /// Show GPU metrics in dashboard
    #[serde(default = "default_true")]
    pub show_gpu_metrics: bool,
}

fn default_tick_rate() -> u64 {
    250
}
fn default_max_logs() -> usize {
    1000
}
fn default_theme() -> String {
    "dark".to_string()
}
fn default_true() -> bool {
    true
}

impl Default for TuiSettings {
    fn default() -> Self {
        Self {
            tick_rate_ms: default_tick_rate(),
            max_log_entries: default_max_logs(),
            theme: default_theme(),
            mouse_support: true,
            show_gpu_metrics: true,
        }
    }
}

/// Deployment-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploySettings {
    /// Default components for local deployment
    #[serde(default)]
    pub local_defaults: Vec<String>,

    /// Container registry
    #[serde(default)]
    pub registry: Option<String>,

    /// Default namespace for cluster deployments
    #[serde(default = "default_namespace")]
    pub default_namespace: String,

    /// Timeout for deployments in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

fn default_namespace() -> String {
    "hpc-ai".to_string()
}
fn default_timeout() -> u64 {
    300
}

impl Default for DeploySettings {
    fn default() -> Self {
        Self {
            local_defaults: vec!["stratoswarm".to_string(), "argus".to_string()],
            registry: None,
            default_namespace: default_namespace(),
            timeout_seconds: default_timeout(),
        }
    }
}

/// Environment profiles configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilesConfig {
    /// Dev environment settings
    #[serde(default)]
    pub dev: ProfileSettings,

    /// Staging environment settings
    #[serde(default)]
    pub staging: ProfileSettings,

    /// Production environment settings
    #[serde(default)]
    pub prod: ProfileSettings,

    /// Custom profiles
    #[serde(default)]
    pub custom: HashMap<String, Profile>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            default_profile: default_profile(),
            default_environment: "dev".to_string(),
            tui: TuiSettings::default(),
            deploy: DeploySettings::default(),
            profiles: ProfilesConfig::default(),
        }
    }
}

impl AppConfig {
    /// Get the configuration directory path
    pub fn config_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hpc")
    }

    /// Get the configuration file path
    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.toml")
    }

    /// Get the stacks directory path
    pub fn stacks_dir() -> PathBuf {
        Self::config_dir().join("stacks")
    }

    /// Get the state directory path
    pub fn state_dir() -> PathBuf {
        Self::config_dir().join("state")
    }

    /// Load configuration from file or return default
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

            toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {:?}", config_path))
        } else {
            Ok(Self::default())
        }
    }

    /// Load configuration from a specific path
    pub fn load_from(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;

        toml::from_str(&content).with_context(|| format!("Failed to parse config file: {}", path))
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_dir = Self::config_dir();
        std::fs::create_dir_all(&config_dir)
            .with_context(|| format!("Failed to create config directory: {:?}", config_dir))?;

        let config_path = Self::config_path();
        let content = toml::to_string_pretty(self)
            .with_context(|| "Failed to serialize configuration")?;

        std::fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        Ok(())
    }

    /// Initialize configuration directories
    pub fn init_dirs() -> Result<()> {
        std::fs::create_dir_all(Self::config_dir())?;
        std::fs::create_dir_all(Self::stacks_dir())?;
        std::fs::create_dir_all(Self::state_dir())?;
        Ok(())
    }

    /// Get a profile by name
    pub fn get_profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.custom.get(name)
    }
}

/// Generate example configuration file content
pub fn example_config() -> &'static str {
    r#"# HPC-AI CLI Configuration
# Location: ~/.hpc/config.toml

# Default profile to use
default_profile = "default"

# Default environment (dev, staging, prod)
default_environment = "dev"

# TUI Settings
[tui]
tick_rate_ms = 250
max_log_entries = 1000
theme = "dark"
mouse_support = true
show_gpu_metrics = true

# Deployment Settings
[deploy]
local_defaults = ["stratoswarm", "argus"]
default_namespace = "hpc-ai"
timeout_seconds = 300
# registry = "ghcr.io/hpc-ai"

# Dev Environment
[profiles.dev]
replicas = 1
gpu_resources = 0
# stratoswarm_endpoint = "http://localhost:8080"
# argus_endpoint = "http://localhost:9090"

# Staging Environment
[profiles.staging]
replicas = 2
gpu_resources = 1
# stratoswarm_endpoint = "https://staging.stratoswarm.local:8080"
# argus_endpoint = "https://staging.argus.local:9090"

# Production Environment
[profiles.prod]
replicas = 4
gpu_resources = 4
# stratoswarm_endpoint = "https://prod.stratoswarm.local:8080"
# argus_endpoint = "https://prod.argus.local:9090"
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.default_profile, "default");
        assert_eq!(config.tui.tick_rate_ms, 250);
    }

    #[test]
    fn test_example_config_parses() {
        let config: AppConfig = toml::from_str(example_config()).unwrap();
        assert_eq!(config.tui.theme, "dark");
    }
}
