//! Rust toolchain management
//!
//! This module handles Rust toolchain provisioning using rustup,
//! with support for multiple versions and download-on-demand.

use crate::build_job::RustToolchain;
use crate::{Result, SwarmletError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Manages Rust toolchains with download-on-demand
pub struct ToolchainManager {
    /// Base directory for toolchains
    toolchains_dir: PathBuf,
    /// Cache of installed toolchains
    installed: Arc<RwLock<HashMap<String, ToolchainInfo>>>,
}

/// Information about an installed toolchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolchainInfo {
    /// Toolchain identifier (e.g., "stable", "1.76.0", "nightly-2024-01-15")
    pub id: String,
    /// Path to the toolchain
    pub path: PathBuf,
    /// Rust version
    pub version: String,
    /// When the toolchain was installed
    pub installed_at: DateTime<Utc>,
    /// When the toolchain was last used
    pub last_used: DateTime<Utc>,
    /// Installed components
    pub components: Vec<String>,
    /// Installed targets
    pub targets: Vec<String>,
}

impl ToolchainManager {
    /// Create a new toolchain manager
    pub async fn new(toolchains_dir: PathBuf) -> Result<Self> {
        // Create the toolchains directory if it doesn't exist
        tokio::fs::create_dir_all(&toolchains_dir).await.map_err(|e| {
            SwarmletError::Configuration(format!(
                "Failed to create toolchains directory: {e}"
            ))
        })?;

        // Scan for existing toolchains
        let installed = Self::scan_toolchains(&toolchains_dir).await?;

        info!(
            "Toolchain manager initialized with {} existing toolchains",
            installed.len()
        );

        Ok(Self {
            toolchains_dir,
            installed: Arc::new(RwLock::new(installed)),
        })
    }

    /// Scan for existing toolchains in the directory
    async fn scan_toolchains(dir: &PathBuf) -> Result<HashMap<String, ToolchainInfo>> {
        let mut toolchains = HashMap::new();

        if !dir.exists() {
            return Ok(toolchains);
        }

        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read toolchains directory: {e}"),
            ))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            SwarmletError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read directory entry: {e}"),
            ))
        })? {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                let info_path = path.join("toolchain.json");

                if info_path.exists() {
                    match tokio::fs::read_to_string(&info_path).await {
                        Ok(content) => match serde_json::from_str::<ToolchainInfo>(&content) {
                            Ok(info) => {
                                debug!("Found existing toolchain: {}", name);
                                toolchains.insert(name, info);
                            }
                            Err(e) => {
                                warn!("Failed to parse toolchain info for {}: {}", name, e);
                            }
                        },
                        Err(e) => {
                            warn!("Failed to read toolchain info for {}: {}", name, e);
                        }
                    }
                }
            }
        }

        Ok(toolchains)
    }

    /// Ensure a toolchain is available, downloading if necessary
    pub async fn ensure_toolchain(&self, spec: &RustToolchain) -> Result<PathBuf> {
        let toolchain_key = self.toolchain_key(spec);

        // Check if already installed
        {
            let installed = self.installed.read().await;
            if let Some(info) = installed.get(&toolchain_key) {
                // Verify components
                if self.has_all_components(info, &spec.components) {
                    debug!("Using existing toolchain: {}", toolchain_key);
                    return Ok(info.path.clone());
                }
                debug!(
                    "Toolchain {} exists but missing components, reinstalling",
                    toolchain_key
                );
            }
        }

        // Download and install
        self.install_toolchain(spec).await
    }

    /// Install a toolchain using rustup
    async fn install_toolchain(&self, spec: &RustToolchain) -> Result<PathBuf> {
        let toolchain_key = self.toolchain_key(spec);
        let toolchain_path = self.toolchains_dir.join(&toolchain_key);

        info!("Installing Rust toolchain: {}", toolchain_key);

        // Create isolated RUSTUP_HOME for this toolchain
        let rustup_home = toolchain_path.join("rustup");
        let cargo_home = toolchain_path.join("cargo");
        tokio::fs::create_dir_all(&rustup_home).await?;
        tokio::fs::create_dir_all(&cargo_home).await?;

        // Build toolchain string
        let toolchain_str = spec.toolchain_string();

        // Run rustup to install
        let output = tokio::process::Command::new("rustup")
            .env("RUSTUP_HOME", &rustup_home)
            .env("CARGO_HOME", &cargo_home)
            .args(["toolchain", "install", &toolchain_str, "--profile", "minimal"])
            .output()
            .await
            .map_err(|e| {
                SwarmletError::WorkloadExecution(format!("Failed to run rustup: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SwarmletError::WorkloadExecution(format!(
                "Toolchain installation failed: {}",
                stderr
            )));
        }

        // Install additional components
        for component in &spec.components {
            info!("Installing component: {}", component);
            let result = tokio::process::Command::new("rustup")
                .env("RUSTUP_HOME", &rustup_home)
                .env("CARGO_HOME", &cargo_home)
                .args([
                    "component",
                    "add",
                    component,
                    "--toolchain",
                    &toolchain_str,
                ])
                .output()
                .await;

            if let Err(e) = result {
                warn!("Failed to install component {}: {}", component, e);
            }
        }

        // Install targets
        for target in &spec.targets {
            info!("Installing target: {}", target);
            let result = tokio::process::Command::new("rustup")
                .env("RUSTUP_HOME", &rustup_home)
                .env("CARGO_HOME", &cargo_home)
                .args(["target", "add", target, "--toolchain", &toolchain_str])
                .output()
                .await;

            if let Err(e) = result {
                warn!("Failed to install target {}: {}", target, e);
            }
        }

        // Get installed version
        let version_output = tokio::process::Command::new("rustup")
            .env("RUSTUP_HOME", &rustup_home)
            .env("CARGO_HOME", &cargo_home)
            .args(["run", &toolchain_str, "rustc", "--version"])
            .output()
            .await;

        let version = version_output
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Create toolchain info
        let info = ToolchainInfo {
            id: toolchain_key.clone(),
            path: toolchain_path.clone(),
            version,
            installed_at: Utc::now(),
            last_used: Utc::now(),
            components: spec.components.clone(),
            targets: spec.targets.clone(),
        };

        // Save toolchain info
        let info_path = toolchain_path.join("toolchain.json");
        let info_json = serde_json::to_string_pretty(&info).map_err(|e| {
            SwarmletError::Serialization(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to serialize toolchain info: {e}"),
            )))
        })?;
        tokio::fs::write(&info_path, info_json).await?;

        // Update installed cache
        {
            let mut installed = self.installed.write().await;
            installed.insert(toolchain_key.clone(), info);
        }

        info!("Toolchain {} installed successfully", toolchain_key);
        Ok(toolchain_path)
    }

    /// Generate a unique key for a toolchain specification
    fn toolchain_key(&self, spec: &RustToolchain) -> String {
        if let Some(date) = &spec.date {
            format!("{}-{}", spec.channel, date)
        } else {
            spec.channel.clone()
        }
    }

    /// Check if a toolchain has all required components
    fn has_all_components(&self, info: &ToolchainInfo, required: &[String]) -> bool {
        required.iter().all(|c| info.components.contains(c))
    }

    /// List all installed toolchains
    pub async fn list(&self) -> Vec<ToolchainInfo> {
        let installed = self.installed.read().await;
        installed.values().cloned().collect()
    }

    /// Remove unused toolchains
    pub async fn gc(&self, keep: &[&str]) -> Result<Vec<String>> {
        let mut removed = Vec::new();
        let mut installed = self.installed.write().await;

        let to_remove: Vec<String> = installed
            .keys()
            .filter(|k| !keep.contains(&k.as_str()))
            .cloned()
            .collect();

        for key in to_remove {
            if let Some(info) = installed.remove(&key) {
                if let Err(e) = tokio::fs::remove_dir_all(&info.path).await {
                    warn!("Failed to remove toolchain {}: {}", key, e);
                } else {
                    info!("Removed toolchain: {}", key);
                    removed.push(key);
                }
            }
        }

        Ok(removed)
    }

    /// Get the path to a toolchain if installed
    pub async fn get_path(&self, spec: &RustToolchain) -> Option<PathBuf> {
        let key = self.toolchain_key(spec);
        let installed = self.installed.read().await;
        installed.get(&key).map(|info| info.path.clone())
    }

    /// Update last used timestamp for a toolchain
    pub async fn mark_used(&self, spec: &RustToolchain) -> Result<()> {
        let key = self.toolchain_key(spec);
        let mut installed = self.installed.write().await;

        if let Some(info) = installed.get_mut(&key) {
            info.last_used = Utc::now();

            // Update the info file
            let info_path = info.path.join("toolchain.json");
            let info_json = serde_json::to_string_pretty(&info)?;
            tokio::fs::write(&info_path, info_json).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toolchain_key() {
        let manager = ToolchainManager {
            toolchains_dir: PathBuf::from("/tmp"),
            installed: Arc::new(RwLock::new(HashMap::new())),
        };

        let stable = RustToolchain::stable();
        assert_eq!(manager.toolchain_key(&stable), "stable");

        let nightly = RustToolchain::nightly();
        assert_eq!(manager.toolchain_key(&nightly), "nightly");

        let dated = RustToolchain {
            channel: "nightly".to_string(),
            date: Some("2024-01-15".to_string()),
            ..Default::default()
        };
        assert_eq!(manager.toolchain_key(&dated), "nightly-2024-01-15");

        let version = RustToolchain::version("1.76.0");
        assert_eq!(manager.toolchain_key(&version), "1.76.0");
    }

    #[test]
    fn test_has_all_components() {
        let manager = ToolchainManager {
            toolchains_dir: PathBuf::from("/tmp"),
            installed: Arc::new(RwLock::new(HashMap::new())),
        };

        let info = ToolchainInfo {
            id: "stable".to_string(),
            path: PathBuf::from("/tmp/stable"),
            version: "1.76.0".to_string(),
            installed_at: Utc::now(),
            last_used: Utc::now(),
            components: vec!["rustfmt".to_string(), "clippy".to_string()],
            targets: vec![],
        };

        assert!(manager.has_all_components(&info, &["rustfmt".to_string()]));
        assert!(manager.has_all_components(
            &info,
            &["rustfmt".to_string(), "clippy".to_string()]
        ));
        assert!(!manager.has_all_components(&info, &["rust-src".to_string()]));
    }
}
