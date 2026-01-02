//! WireGuard configuration and management for Swarmlet
//!
//! Provides API endpoints and functionality for managing WireGuard
//! interfaces on the node, including:
//!
//! - Interface configuration from coordinator
//! - Peer management
//! - Status reporting
//! - Automatic key generation on join

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use crate::{Result, SwarmletError};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use x25519_dalek::{PublicKey, StaticSecret};
use rand_core::OsRng;

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
    /// Cluster's public key for verifying config signatures
    cluster_public_key: RwLock<Option<[u8; 32]>>,
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
            cluster_public_key: RwLock::new(None),
        }
    }

    /// Set the cluster's public key for signature verification
    pub async fn set_cluster_public_key(&self, public_key: [u8; 32]) {
        *self.cluster_public_key.write().await = Some(public_key);
        info!("Cluster public key set for config signature verification");
    }

    /// Set the cluster's public key from base64-encoded string
    pub async fn set_cluster_public_key_b64(&self, public_key_b64: &str) -> Result<()> {
        let bytes = BASE64_STANDARD
            .decode(public_key_b64)
            .map_err(|e| SwarmletError::WireGuard(format!("Invalid public key encoding: {}", e)))?;

        if bytes.len() != 32 {
            return Err(SwarmletError::WireGuard(format!(
                "Invalid public key length: expected 32, got {}",
                bytes.len()
            )));
        }

        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        self.set_cluster_public_key(key).await;
        Ok(())
    }

    /// Get the cluster's public key if set
    pub async fn get_cluster_public_key(&self) -> Option<[u8; 32]> {
        *self.cluster_public_key.read().await
    }

    /// Verify a WireGuard configuration signature
    ///
    /// The signature should cover: interface_name + config_version + address + peers_hash
    fn verify_config_signature(&self, config: &WireGuardConfigRequest, cluster_key: &[u8; 32]) -> Result<()> {
        let signature_b64 = config.signature.as_ref().ok_or_else(|| {
            SwarmletError::WireGuard("No signature provided for verification".to_string())
        })?;

        // Decode signature from base64
        let signature_bytes = BASE64_STANDARD
            .decode(signature_b64)
            .map_err(|e| SwarmletError::WireGuard(format!("Invalid signature encoding: {}", e)))?;

        if signature_bytes.len() != 64 {
            return Err(SwarmletError::WireGuard(format!(
                "Invalid signature length: expected 64, got {}",
                signature_bytes.len()
            )));
        }

        let signature_array: [u8; 64] = signature_bytes.try_into()
            .map_err(|_| SwarmletError::WireGuard("Invalid signature format".to_string()))?;

        // Reconstruct the message that was signed
        // Format: interface_name:config_version:address:peer_public_keys_sorted
        let mut peer_keys: Vec<&str> = config.peers.iter().map(|p| p.public_key.as_str()).collect();
        peer_keys.sort();
        let peers_str = peer_keys.join(",");

        let message = format!(
            "{}:{}:{}:{}",
            config.interface_name,
            config.config_version,
            config.address,
            peers_str
        );

        // Create signature and verifying key
        let signature = Signature::from_bytes(&signature_array);
        let verifying_key = VerifyingKey::from_bytes(cluster_key)
            .map_err(|e| SwarmletError::WireGuard(format!("Invalid cluster public key: {}", e)))?;

        // Verify the signature
        verifying_key.verify(message.as_bytes(), &signature)
            .map_err(|_| SwarmletError::WireGuard("Config signature verification failed".to_string()))?;

        debug!("Config signature verified successfully");
        Ok(())
    }

    /// Generate a new WireGuard keypair for this node
    ///
    /// Uses x25519 (Curve25519) key generation via x25519-dalek.
    /// The keypair is stored and the public key can be sent to
    /// the coordinator during join.
    pub async fn generate_keypair(&self) -> Result<String> {
        // Generate a new X25519 secret key using cryptographically secure RNG
        let secret = StaticSecret::random_from_rng(OsRng);

        // Derive the public key from the secret
        let public = PublicKey::from(&secret);

        // Encode both keys as base64 (WireGuard format)
        let private_key_b64 = BASE64_STANDARD.encode(secret.as_bytes());
        let public_key_b64 = BASE64_STANDARD.encode(public.as_bytes());

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

    /// Restore a keypair from saved state
    ///
    /// Used when loading agent state from disk after a restart.
    pub async fn restore_keypair(&self, private_key: String, public_key: String) {
        let keypair = WireGuardKeypair {
            private_key,
            public_key,
        };
        *self.keypair.write().await = Some(keypair);
        info!("Restored WireGuard keypair from saved state");
    }

    /// Apply WireGuard configuration from coordinator
    pub async fn apply_config(&self, config: WireGuardConfigRequest) -> Result<WireGuardConfigResponse> {
        info!(
            "Applying WireGuard config version {} for interface {}",
            config.config_version, config.interface_name
        );

        // Verify signature if cluster public key is set and signature is provided
        if config.signature.is_some() {
            if let Some(cluster_key) = *self.cluster_public_key.read().await {
                // Verify the config signature using ed25519
                if let Err(e) = self.verify_config_signature(&config, &cluster_key) {
                    error!("Config signature verification failed: {}", e);
                    return Ok(WireGuardConfigResponse {
                        success: false,
                        error: Some(format!("Signature verification failed: {}", e)),
                        config_version: config.config_version,
                        status: WireGuardStatus::default(),
                    });
                }
                debug!("Config signature verified successfully");
            } else {
                // No cluster key set - log warning but continue (for backwards compatibility)
                warn!("Config has signature but no cluster public key set for verification");
            }
        } else {
            // No signature provided - log for auditing
            debug!("Config received without signature");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_encode() {
        assert_eq!(BASE64_STANDARD.encode(b"hello"), "aGVsbG8=");
        assert_eq!(BASE64_STANDARD.encode(b""), "");
        assert_eq!(BASE64_STANDARD.encode(b"a"), "YQ==");
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

    #[tokio::test]
    async fn test_keypair_produces_valid_x25519_keys() {
        let manager = WireGuardManager::new();
        let public_key_b64 = manager.generate_keypair().await.unwrap();

        // Decode and verify the public key is 32 bytes
        let public_key_bytes = BASE64_STANDARD.decode(&public_key_b64).unwrap();
        assert_eq!(public_key_bytes.len(), 32);

        // Verify private key is also 32 bytes
        let private_key_b64 = manager.get_private_key().await.unwrap();
        let private_key_bytes = BASE64_STANDARD.decode(&private_key_b64).unwrap();
        assert_eq!(private_key_bytes.len(), 32);
    }

    #[tokio::test]
    async fn test_set_cluster_public_key() {
        let manager = WireGuardManager::new();

        // Set a valid 32-byte key
        let test_key = [42u8; 32];
        manager.set_cluster_public_key(test_key).await;

        // Verify it's set
        let stored_key = manager.cluster_public_key.read().await;
        assert!(stored_key.is_some());
        assert_eq!(stored_key.unwrap(), test_key);
    }

    #[tokio::test]
    async fn test_set_cluster_public_key_b64() {
        let manager = WireGuardManager::new();

        // Create a valid base64-encoded 32-byte key
        let test_key = [42u8; 32];
        let test_key_b64 = BASE64_STANDARD.encode(&test_key);

        manager.set_cluster_public_key_b64(&test_key_b64).await.unwrap();

        // Verify it's set correctly
        let stored_key = manager.cluster_public_key.read().await;
        assert!(stored_key.is_some());
        assert_eq!(stored_key.unwrap(), test_key);
    }

    #[tokio::test]
    async fn test_set_cluster_public_key_b64_invalid_length() {
        let manager = WireGuardManager::new();

        // Try to set an invalid length key (16 bytes instead of 32)
        let invalid_key = [42u8; 16];
        let invalid_key_b64 = BASE64_STANDARD.encode(&invalid_key);

        let result = manager.set_cluster_public_key_b64(&invalid_key_b64).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_config_signature_verification() {
        use ed25519_dalek::{SigningKey, Signer};
        use rand_core::RngCore;

        // Generate a signing keypair for testing using random bytes
        let mut secret_bytes = [0u8; 32];
        rand_core::OsRng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();

        // Create a test config
        let config = WireGuardConfigRequest {
            interface_name: "wg-test".to_string(),
            private_key: None,
            listen_port: 51820,
            address: "10.0.0.1/24".to_string(),
            mtu: None,
            peers: vec![
                WireGuardPeerConfig {
                    public_key: "peer1key".to_string(),
                    preshared_key: None,
                    allowed_ips: vec!["10.0.0.2/32".to_string()],
                    endpoint: None,
                    persistent_keepalive: None,
                },
            ],
            config_version: "v1".to_string(),
            signature: None, // We'll set this after signing
        };

        // Construct the message to sign (same format as verify_config_signature)
        let mut peer_keys: Vec<&str> = config.peers.iter().map(|p| p.public_key.as_str()).collect();
        peer_keys.sort();
        let peers_str = peer_keys.join(",");
        let message = format!(
            "{}:{}:{}:{}",
            config.interface_name,
            config.config_version,
            config.address,
            peers_str
        );

        // Sign the message
        let signature = signing_key.sign(message.as_bytes());
        let signature_b64 = BASE64_STANDARD.encode(signature.to_bytes());

        // Create config with signature
        let signed_config = WireGuardConfigRequest {
            signature: Some(signature_b64),
            ..config
        };

        // Verify the signature
        let manager = WireGuardManager::new();
        let cluster_key: [u8; 32] = verifying_key.to_bytes();
        let result = manager.verify_config_signature(&signed_config, &cluster_key);
        assert!(result.is_ok(), "Signature verification should succeed");
    }

    #[test]
    fn test_config_signature_verification_fails_with_wrong_key() {
        use ed25519_dalek::{SigningKey, Signer};
        use rand_core::RngCore;

        // Generate a signing keypair for testing
        let mut secret_bytes = [0u8; 32];
        rand_core::OsRng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);

        // Generate a different key for verification (should fail)
        let mut wrong_secret_bytes = [0u8; 32];
        rand_core::OsRng.fill_bytes(&mut wrong_secret_bytes);
        let wrong_verifying_key = SigningKey::from_bytes(&wrong_secret_bytes).verifying_key();

        // Create a test config
        let config = WireGuardConfigRequest {
            interface_name: "wg-test".to_string(),
            private_key: None,
            listen_port: 51820,
            address: "10.0.0.1/24".to_string(),
            mtu: None,
            peers: vec![],
            config_version: "v1".to_string(),
            signature: None,
        };

        // Sign with one key
        let message = format!("{}:{}:{}:", config.interface_name, config.config_version, config.address);
        let signature = signing_key.sign(message.as_bytes());
        let signature_b64 = BASE64_STANDARD.encode(signature.to_bytes());

        let signed_config = WireGuardConfigRequest {
            signature: Some(signature_b64),
            ..config
        };

        // Try to verify with wrong key
        let manager = WireGuardManager::new();
        let wrong_key: [u8; 32] = wrong_verifying_key.to_bytes();
        let result = manager.verify_config_signature(&signed_config, &wrong_key);
        assert!(result.is_err(), "Signature verification should fail with wrong key");
    }
}
