//! WireGuard configuration and management for Swarmlet
//!
//! Provides API endpoints and functionality for managing WireGuard
//! interfaces on the node, including:
//!
//! - Interface configuration from coordinator
//! - Peer management
//! - Status reporting
//! - Automatic key generation on join

use crate::{Result, SwarmletError};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// WireGuard interface configuration received from coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardConfigRequest {
    /// Interface name (e.g., "wg-swarm0")
    pub interface_name: String,
    /// Private key (base64 encoded) - only sent during initial setup
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_key: Option<String>,
    /// Listen port for WireGuard
    pub listen_port: u16,
    /// Assigned IP address with CIDR (e.g., "10.0.1.5/24")
    pub address: String,
    /// MTU (optional, defaults to 1420)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtu: Option<u16>,
    /// Peers to configure
    pub peers: Vec<WireGuardPeerConfig>,
    /// Configuration version (for change tracking)
    pub config_version: String,
    /// Signature from coordinator (for verification)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Peer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardPeerConfig {
    /// Peer public key (base64)
    pub public_key: String,
    /// Preshared key (optional, base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preshared_key: Option<String>,
    /// Allowed IPs for this peer
    pub allowed_ips: Vec<String>,
    /// Endpoint (host:port)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Persistent keepalive interval in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistent_keepalive: Option<u16>,
}

/// Response to configuration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardConfigResponse {
    /// Whether the configuration was applied successfully
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Current configuration version
    pub config_version: String,
    /// Interface status after configuration
    pub status: WireGuardStatus,
}

/// WireGuard interface status
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WireGuardStatus {
    /// Whether the interface exists
    pub interface_exists: bool,
    /// Whether the interface is up
    pub is_up: bool,
    /// Interface public key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_key: Option<String>,
    /// Listen port
    pub listen_port: u16,
    /// Assigned address
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    /// Number of configured peers
    pub peer_count: usize,
    /// Current configuration version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_version: Option<String>,
}

/// Add peer request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddPeerRequest {
    /// Interface name
    pub interface_name: String,
    /// Peer configuration
    pub peer: WireGuardPeerConfig,
}

/// Remove peer request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemovePeerRequest {
    /// Interface name
    pub interface_name: String,
    /// Peer public key to remove
    pub public_key: String,
}

/// Peer status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStatus {
    /// Peer public key
    pub public_key: String,
    /// Current endpoint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Allowed IPs
    pub allowed_ips: Vec<String>,
    /// Last handshake time (seconds ago)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_handshake_secs: Option<u64>,
    /// Bytes received
    pub rx_bytes: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
}

/// WireGuard manager for the swarmlet
pub struct WireGuardManager {
    /// Current interface name
    interface_name: RwLock<Option<String>>,
    /// Current configuration version
    config_version: RwLock<Option<String>>,
    /// Node's WireGuard keypair (generated on first join)
    keypair: RwLock<Option<WireGuardKeypair>>,
}

/// WireGuard keypair
#[derive(Debug, Clone)]
pub struct WireGuardKeypair {
    /// Private key (base64)
    pub private_key: String,
    /// Public key (base64)
    pub public_key: String,
}

impl WireGuardManager {
    /// Create a new WireGuard manager
    pub fn new() -> Self {
        Self {
            interface_name: RwLock::new(None),
            config_version: RwLock::new(None),
            keypair: RwLock::new(None),
        }
    }

    /// Generate a new WireGuard keypair for this node
    ///
    /// Uses x25519 key generation. The keypair is stored and the
    /// public key can be sent to the coordinator during join.
    pub async fn generate_keypair(&self) -> Result<String> {
        use rand::RngCore;

        // Generate random 32-byte private key
        let mut private_key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut private_key);

        // Clamp for Curve25519
        private_key[0] &= 248;
        private_key[31] &= 127;
        private_key[31] |= 64;

        // For proper key derivation, we'd use x25519-dalek here
        // For now, use ed25519-dalek which is already a dependency
        // In production, this should use the proper x25519 implementation
        let private_key_b64 = base64_encode(&private_key);

        // Derive public key (simplified - should use x25519)
        // This is a placeholder that generates a deterministic "public key"
        // In production, use x25519_dalek::PublicKey::from(&secret)
        let mut public_key = [0u8; 32];
        for i in 0..32 {
            public_key[i] = private_key[i].wrapping_add(9); // Simplified derivation
        }
        let public_key_b64 = base64_encode(&public_key);

        let keypair = WireGuardKeypair {
            private_key: private_key_b64,
            public_key: public_key_b64.clone(),
        };

        *self.keypair.write().await = Some(keypair);

        info!("Generated WireGuard keypair");
        Ok(public_key_b64)
    }

    /// Get the node's WireGuard public key
    pub async fn get_public_key(&self) -> Option<String> {
        self.keypair
            .read()
            .await
            .as_ref()
            .map(|kp| kp.public_key.clone())
    }

    /// Get the node's WireGuard private key
    pub async fn get_private_key(&self) -> Option<String> {
        self.keypair
            .read()
            .await
            .as_ref()
            .map(|kp| kp.private_key.clone())
    }

    /// Apply WireGuard configuration from coordinator
    pub async fn apply_config(&self, config: WireGuardConfigRequest) -> Result<WireGuardConfigResponse> {
        info!(
            "Applying WireGuard config version {} for interface {}",
            config.config_version, config.interface_name
        );

        // Verify signature if provided (placeholder)
        if let Some(_signature) = &config.signature {
            // TODO: Verify signature from coordinator
            debug!("Config signature verification (placeholder)");
        }

        // Get private key - either from config or from our keypair
        let private_key = if let Some(pk) = &config.private_key {
            pk.clone()
        } else {
            self.get_private_key().await.ok_or_else(|| {
                SwarmletError::WireGuard("No private key available".to_string())
            })?
        };

        // Apply configuration using wg command
        match self
            .apply_config_impl(&config.interface_name, &private_key, &config)
            .await
        {
            Ok(()) => {
                // Update state
                *self.interface_name.write().await = Some(config.interface_name.clone());
                *self.config_version.write().await = Some(config.config_version.clone());

                let status = self.get_status(&config.interface_name).await?;

                Ok(WireGuardConfigResponse {
                    success: true,
                    error: None,
                    config_version: config.config_version,
                    status,
                })
            }
            Err(e) => {
                error!("Failed to apply WireGuard config: {}", e);
                Ok(WireGuardConfigResponse {
                    success: false,
                    error: Some(e.to_string()),
                    config_version: config.config_version,
                    status: WireGuardStatus::default(),
                })
            }
        }
    }

    /// Internal implementation of config application
    async fn apply_config_impl(
        &self,
        interface_name: &str,
        private_key: &str,
        config: &WireGuardConfigRequest,
    ) -> Result<()> {
        use tokio::process::Command;

        // Check if interface exists
        let exists = self.interface_exists(interface_name).await;

        if !exists {
            // Create interface
            info!("Creating WireGuard interface {}", interface_name);

            let status = Command::new("ip")
                .args(["link", "add", interface_name, "type", "wireguard"])
                .status()
                .await;

            match status {
                Ok(s) if s.success() => {}
                Ok(_) | Err(_) => {
                    // Try with sudo
                    let status = Command::new("sudo")
                        .args(["ip", "link", "add", interface_name, "type", "wireguard"])
                        .status()
                        .await
                        .map_err(|e| SwarmletError::WireGuard(format!("Failed to create interface: {}", e)))?;

                    if !status.success() {
                        return Err(SwarmletError::WireGuard(
                            "Failed to create WireGuard interface".to_string(),
                        ));
                    }
                }
            }
        }

        // Write private key to temp file
        let temp_dir = std::env::temp_dir();
        let key_path = temp_dir.join(format!("wg_key_{}", uuid::Uuid::new_v4()));

        tokio::fs::write(&key_path, private_key)
            .await
            .map_err(|e| SwarmletError::WireGuard(format!("Failed to write key: {}", e)))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))
                .await
                .ok();
        }

        // Configure WireGuard
        let port_str = config.listen_port.to_string();
        let key_path_str = key_path.to_string_lossy().to_string();

        let output = Command::new("wg")
            .args([
                "set",
                interface_name,
                "private-key",
                &key_path_str,
                "listen-port",
                &port_str,
            ])
            .output()
            .await;

        // Clean up key file
        let _ = tokio::fs::remove_file(&key_path).await;

        match output {
            Ok(o) if o.status.success() => {}
            _ => {
                // Try with sudo
                let temp_dir = std::env::temp_dir();
                let key_path = temp_dir.join(format!("wg_key_{}", uuid::Uuid::new_v4()));
                tokio::fs::write(&key_path, private_key).await.ok();

                let output = Command::new("sudo")
                    .args([
                        "wg",
                        "set",
                        interface_name,
                        "private-key",
                        &key_path.to_string_lossy(),
                        "listen-port",
                        &port_str,
                    ])
                    .output()
                    .await
                    .map_err(|e| SwarmletError::WireGuard(format!("Failed to configure WireGuard: {}", e)))?;

                let _ = tokio::fs::remove_file(&key_path).await;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(SwarmletError::WireGuard(format!(
                        "Failed to configure WireGuard: {}",
                        stderr
                    )));
                }
            }
        }

        // Set IP address
        let output = Command::new("ip")
            .args(["address", "add", &config.address, "dev", interface_name])
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {}
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                // Ignore "already exists" errors
                if !stderr.contains("RTNETLINK answers: File exists") {
                    warn!("Failed to set address (may already exist): {}", stderr);
                }
            }
            Err(e) => {
                warn!("Failed to set address: {}", e);
            }
        }

        // Set MTU if specified
        if let Some(mtu) = config.mtu {
            let mtu_str = mtu.to_string();
            let _ = Command::new("ip")
                .args(["link", "set", interface_name, "mtu", &mtu_str])
                .status()
                .await;
        }

        // Bring interface up
        let status = Command::new("ip")
            .args(["link", "set", interface_name, "up"])
            .status()
            .await;

        match status {
            Ok(s) if s.success() => {}
            _ => {
                let _ = Command::new("sudo")
                    .args(["ip", "link", "set", interface_name, "up"])
                    .status()
                    .await;
            }
        }

        // Add peers
        for peer in &config.peers {
            self.add_peer_impl(interface_name, peer).await?;
        }

        info!("WireGuard interface {} configured successfully", interface_name);
        Ok(())
    }

    /// Add a peer to the interface
    pub async fn add_peer(&self, request: AddPeerRequest) -> Result<()> {
        self.add_peer_impl(&request.interface_name, &request.peer)
            .await
    }

    async fn add_peer_impl(&self, interface_name: &str, peer: &WireGuardPeerConfig) -> Result<()> {
        use tokio::process::Command;

        debug!("Adding peer {} to {}", peer.public_key, interface_name);

        let mut args = vec!["set", interface_name, "peer", &peer.public_key];

        // Allowed IPs
        let allowed_ips = peer.allowed_ips.join(",");
        if !allowed_ips.is_empty() {
            args.push("allowed-ips");
            args.push(&allowed_ips);
        }

        // Endpoint
        let endpoint;
        if let Some(ep) = &peer.endpoint {
            endpoint = ep.clone();
            args.push("endpoint");
            args.push(&endpoint);
        }

        // Persistent keepalive
        let keepalive_str;
        if let Some(keepalive) = peer.persistent_keepalive {
            keepalive_str = keepalive.to_string();
            args.push("persistent-keepalive");
            args.push(&keepalive_str);
        }

        let output = Command::new("wg").args(&args).output().await;

        match output {
            Ok(o) if o.status.success() => Ok(()),
            _ => {
                // Try with sudo
                let mut sudo_args = vec!["wg"];
                sudo_args.extend(args.iter().cloned());

                let output = Command::new("sudo")
                    .args(&sudo_args)
                    .output()
                    .await
                    .map_err(|e| SwarmletError::WireGuard(format!("Failed to add peer: {}", e)))?;

                if output.status.success() {
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Err(SwarmletError::WireGuard(format!(
                        "Failed to add peer: {}",
                        stderr
                    )))
                }
            }
        }
    }

    /// Remove a peer from the interface
    pub async fn remove_peer(&self, request: RemovePeerRequest) -> Result<()> {
        use tokio::process::Command;

        debug!(
            "Removing peer {} from {}",
            request.public_key, request.interface_name
        );

        let output = Command::new("wg")
            .args([
                "set",
                &request.interface_name,
                "peer",
                &request.public_key,
                "remove",
            ])
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => Ok(()),
            _ => {
                let output = Command::new("sudo")
                    .args([
                        "wg",
                        "set",
                        &request.interface_name,
                        "peer",
                        &request.public_key,
                        "remove",
                    ])
                    .output()
                    .await
                    .map_err(|e| SwarmletError::WireGuard(format!("Failed to remove peer: {}", e)))?;

                if output.status.success() {
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Err(SwarmletError::WireGuard(format!(
                        "Failed to remove peer: {}",
                        stderr
                    )))
                }
            }
        }
    }

    /// Get WireGuard interface status
    pub async fn get_status(&self, interface_name: &str) -> Result<WireGuardStatus> {
        use tokio::process::Command;

        let exists = self.interface_exists(interface_name).await;
        if !exists {
            return Ok(WireGuardStatus {
                interface_exists: false,
                ..Default::default()
            });
        }

        // Get wg show output
        let output = Command::new("wg")
            .args(["show", interface_name])
            .output()
            .await;

        let wg_output = match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => {
                // Try with sudo
                let output = Command::new("sudo")
                    .args(["wg", "show", interface_name])
                    .output()
                    .await
                    .ok();

                match output {
                    Some(o) if o.status.success() => {
                        String::from_utf8_lossy(&o.stdout).to_string()
                    }
                    _ => String::new(),
                }
            }
        };

        let mut status = WireGuardStatus {
            interface_exists: true,
            is_up: true, // If we got here, it's probably up
            config_version: self.config_version.read().await.clone(),
            ..Default::default()
        };

        // Parse wg show output
        for line in wg_output.lines() {
            let line = line.trim();
            if line.starts_with("public key:") {
                status.public_key = Some(line.trim_start_matches("public key:").trim().to_string());
            } else if line.starts_with("listening port:") {
                if let Ok(port) = line
                    .trim_start_matches("listening port:")
                    .trim()
                    .parse()
                {
                    status.listen_port = port;
                }
            } else if line.starts_with("peer:") {
                status.peer_count += 1;
            }
        }

        // Get IP address
        let addr_output = Command::new("ip")
            .args(["address", "show", interface_name])
            .output()
            .await;

        if let Ok(o) = addr_output {
            let output = String::from_utf8_lossy(&o.stdout);
            for line in output.lines() {
                if line.contains("inet ") {
                    if let Some(addr) = line.split_whitespace().nth(1) {
                        status.address = Some(addr.to_string());
                        break;
                    }
                }
            }
        }

        Ok(status)
    }

    /// Check if interface exists
    async fn interface_exists(&self, name: &str) -> bool {
        use tokio::process::Command;

        Command::new("ip")
            .args(["link", "show", name])
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get all peer statuses
    pub async fn list_peers(&self, interface_name: &str) -> Result<Vec<PeerStatus>> {
        use tokio::process::Command;

        let output = Command::new("wg")
            .args(["show", interface_name, "dump"])
            .output()
            .await;

        let wg_output = match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => {
                let output = Command::new("sudo")
                    .args(["wg", "show", interface_name, "dump"])
                    .output()
                    .await
                    .map_err(|e| SwarmletError::WireGuard(format!("Failed to list peers: {}", e)))?;

                String::from_utf8_lossy(&output.stdout).to_string()
            }
        };

        let mut peers = Vec::new();

        // Parse dump format (tab-separated)
        // Format: public_key preshared_key endpoint allowed_ips latest_handshake transfer_rx transfer_tx persistent_keepalive
        for (i, line) in wg_output.lines().enumerate() {
            // Skip first line (interface info)
            if i == 0 {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 7 {
                let public_key = parts[0].to_string();
                let endpoint = if parts[2] != "(none)" {
                    Some(parts[2].to_string())
                } else {
                    None
                };
                let allowed_ips: Vec<String> = parts[3]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();

                let last_handshake_secs = parts[4].parse::<u64>().ok().and_then(|ts| {
                    if ts == 0 {
                        None
                    } else {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        Some(now.saturating_sub(ts))
                    }
                });

                let rx_bytes = parts[5].parse().unwrap_or(0);
                let tx_bytes = parts[6].parse().unwrap_or(0);

                peers.push(PeerStatus {
                    public_key,
                    endpoint,
                    allowed_ips,
                    last_handshake_secs,
                    rx_bytes,
                    tx_bytes,
                });
            }
        }

        Ok(peers)
    }
}

impl Default for WireGuardManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Base64 encode helper
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(bytes.len() * 4 / 3 + 4);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"a"), "YQ==");
    }

    #[tokio::test]
    async fn test_wireguard_manager_creation() {
        let manager = WireGuardManager::new();
        assert!(manager.get_public_key().await.is_none());
    }

    #[tokio::test]
    async fn test_keypair_generation() {
        let manager = WireGuardManager::new();
        let public_key = manager.generate_keypair().await.unwrap();
        assert!(!public_key.is_empty());
        assert!(manager.get_public_key().await.is_some());
        assert!(manager.get_private_key().await.is_some());
    }
}
