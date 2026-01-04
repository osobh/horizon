//! Saved agent state for persistence across restarts

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use crate::{join::JoinResult, Result, SwarmletError};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedAgentState {
    /// The join result containing cluster membership info
    pub join_result: JoinResult,
    /// WireGuard private key (base64 encoded)
    pub wireguard_private_key: String,
    /// WireGuard public key (base64 encoded)
    pub wireguard_public_key: String,
    /// Cluster's public key for signature verification (base64, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_public_key: Option<String>,
    /// Timestamp when state was saved
    pub saved_at: chrono::DateTime<chrono::Utc>,
    /// State file format version for migration
    pub version: u32,
}

impl SavedAgentState {
    /// Current state file format version
    pub const CURRENT_VERSION: u32 = 1;

    /// State file name
    pub const STATE_FILE: &'static str = "swarmlet_state.json";

    /// Get the state file path for a given data directory
    pub fn state_path(data_dir: &Path) -> PathBuf {
        data_dir.join(Self::STATE_FILE)
    }

    /// Load saved state from disk
    pub async fn load(data_dir: &Path) -> Result<Self> {
        let state_path = Self::state_path(data_dir);

        if !state_path.exists() {
            return Err(SwarmletError::Configuration(format!(
                "No saved state found at {}",
                state_path.display()
            )));
        }

        let content = tokio::fs::read_to_string(&state_path).await.map_err(|e| {
            SwarmletError::Configuration(format!("Failed to read state file: {}", e))
        })?;

        let state: SavedAgentState = serde_json::from_str(&content).map_err(|e| {
            SwarmletError::Configuration(format!("Failed to parse state file: {}", e))
        })?;

        // Check version and migrate if needed
        if state.version > Self::CURRENT_VERSION {
            return Err(SwarmletError::Configuration(format!(
                "State file version {} is newer than supported version {}",
                state.version,
                Self::CURRENT_VERSION
            )));
        }

        info!("Loaded saved agent state from {}", state_path.display());
        Ok(state)
    }

    /// Save state to disk atomically
    pub async fn save(&self, data_dir: &Path) -> Result<()> {
        let state_path = Self::state_path(data_dir);
        let temp_path = state_path.with_extension("json.tmp");

        // Ensure data directory exists
        if !data_dir.exists() {
            tokio::fs::create_dir_all(data_dir).await.map_err(|e| {
                SwarmletError::Configuration(format!("Failed to create data directory: {}", e))
            })?;
        }

        // Serialize state
        let content = serde_json::to_string_pretty(&self).map_err(|e| {
            SwarmletError::Configuration(format!("Failed to serialize state: {}", e))
        })?;

        // Write to temp file
        tokio::fs::write(&temp_path, content).await.map_err(|e| {
            SwarmletError::Configuration(format!("Failed to write temp state file: {}", e))
        })?;

        // Atomically rename to final path
        tokio::fs::rename(&temp_path, &state_path)
            .await
            .map_err(|e| {
                SwarmletError::Configuration(format!("Failed to rename state file: {}", e))
            })?;

        debug!("Saved agent state to {}", state_path.display());
        Ok(())
    }
}
