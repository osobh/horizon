//! Configuration management for StratoSwarm CLI

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CLI configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default namespace
    pub default_namespace: String,

    /// API endpoint
    pub api_endpoint: String,

    /// Authentication token
    pub auth_token: Option<String>,

    /// Output format
    pub default_format: String,

    /// Color output
    pub color: bool,
}

impl CliConfig {
    /// Load configuration from default locations
    pub fn load() -> Result<Self> {
        // Try to load from config file
        if let Some(config_path) = Self::config_path() {
            if config_path.exists() {
                let content = std::fs::read_to_string(config_path)?;
                return Ok(toml::from_str(&content)?);
            }
        }

        // Return default config
        Ok(Self::default())
    }

    /// Get the default config path
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("stratoswarm").join("config.toml"))
    }

    /// Save configuration
    pub fn save(&self) -> Result<()> {
        if let Some(config_path) = Self::config_path() {
            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let content = toml::to_string_pretty(self)?;
            std::fs::write(config_path, content)?;
        }
        Ok(())
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_namespace: "default".to_string(),
            api_endpoint: "http://localhost:8080".to_string(),
            auth_token: None,
            default_format: "table".to_string(),
            color: true,
        }
    }
}
