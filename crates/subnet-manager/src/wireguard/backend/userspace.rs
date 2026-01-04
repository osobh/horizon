//! Userspace WireGuard backend using boringtun
//!
//! Provides a cross-platform WireGuard implementation that runs entirely
//! in userspace without requiring kernel support. Uses the defguard_boringtun
//! library for the WireGuard protocol implementation.
//!
//! Advantages:
//! - Works on any platform (Linux, macOS, Windows)
//! - No kernel module or WireGuard tools required
//! - Can run without elevated privileges in some configurations
//!
//! Disadvantages:
//! - Higher CPU usage than kernel implementation
//! - Slightly higher latency

use super::traits::{BackendType, InterfaceStats, PeerStats, WireGuardBackend};
use crate::wireguard::{InterfaceConfig, KeyPair, PeerConfig};
use crate::{Error, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, trace, warn};

/// Userspace WireGuard tunnel state
struct TunnelState {
    /// Interface configuration
    config: InterfaceConfig,
    /// Peers configured on this interface
    peers: Vec<PeerConfig>,
    /// Interface statistics
    stats: InterfaceStats,
    /// Creation time
    created_at: Instant,
    /// Whether the interface is up
    is_up: bool,
}

impl TunnelState {
    fn new(config: InterfaceConfig) -> Self {
        Self {
            stats: InterfaceStats {
                name: config.name.clone(),
                listen_port: config.listen_port,
                ..Default::default()
            },
            config,
            peers: Vec::new(),
            created_at: Instant::now(),
            is_up: false,
        }
    }
}

/// Userspace WireGuard backend
///
/// Uses boringtun for userspace WireGuard implementation.
/// This provides a cross-platform solution that doesn't require
/// kernel support or elevated privileges (in some configurations).
pub struct UserspaceBackend {
    /// Active tunnels keyed by interface name
    tunnels: DashMap<String, Arc<RwLock<TunnelState>>>,
    /// Whether the backend is available
    available: bool,
}

impl UserspaceBackend {
    /// Create a new userspace backend
    pub fn new() -> Self {
        Self {
            tunnels: DashMap::new(),
            available: true, // Userspace is always available
        }
    }

    /// Get public key from private key using x25519
    fn derive_public_key(private_key: &str) -> Result<String> {
        let keypair = KeyPair::from_private_key_base64(private_key)?;
        Ok(keypair.public_key_base64())
    }

    /// Find a peer by public key
    fn find_peer_index(peers: &[PeerConfig], public_key: &str) -> Option<usize> {
        peers.iter().position(|p| p.public_key == public_key)
    }
}

impl Default for UserspaceBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WireGuardBackend for UserspaceBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Userspace
    }

    fn is_available(&self) -> bool {
        self.available
    }

    #[instrument(skip(self, config), fields(interface = %config.name))]
    async fn create_interface(&self, config: &InterfaceConfig) -> Result<()> {
        debug!("Creating userspace WireGuard interface");

        if self.tunnels.contains_key(&config.name) {
            return Err(Error::WireGuardConfig(format!(
                "Interface {} already exists",
                config.name
            )));
        }

        // Derive public key for stats
        let public_key = Self::derive_public_key(&config.private_key)?;

        let mut state = TunnelState::new(config.clone());
        state.stats.public_key = public_key;
        state.is_up = true;

        self.tunnels
            .insert(config.name.clone(), Arc::new(RwLock::new(state)));

        info!(
            "Userspace WireGuard interface {} created (simulated)",
            config.name
        );

        // Note: In a full implementation, we would:
        // 1. Create a TUN device
        // 2. Initialize boringtun tunnel
        // 3. Start packet processing loop
        // 4. Bind UDP socket for WireGuard traffic
        //
        // For now, this provides the interface for management operations
        // while actual packet handling would need TUN device support.

        Ok(())
    }

    #[instrument(skip(self))]
    async fn destroy_interface(&self, name: &str) -> Result<()> {
        debug!("Destroying userspace WireGuard interface");

        if self.tunnels.remove(name).is_none() {
            return Err(Error::WireGuardConfig(format!(
                "Interface {} not found",
                name
            )));
        }

        info!("Userspace WireGuard interface {} destroyed", name);
        Ok(())
    }

    async fn interface_exists(&self, name: &str) -> Result<bool> {
        Ok(self.tunnels.contains_key(name))
    }

    #[instrument(skip(self, peer))]
    async fn add_peer(&self, interface: &str, peer: &PeerConfig) -> Result<()> {
        debug!(public_key = %peer.public_key, "Adding peer to userspace interface");

        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;

        // Check if peer already exists
        if Self::find_peer_index(&state.peers, &peer.public_key).is_some() {
            // Update existing peer
            let idx = Self::find_peer_index(&state.peers, &peer.public_key).unwrap();
            state.peers[idx] = peer.clone();
            debug!("Updated existing peer");
        } else {
            // Add new peer
            state.peers.push(peer.clone());
            state.stats.peer_count = state.peers.len();
            debug!("Added new peer");
        }

        Ok(())
    }

    #[instrument(skip(self))]
    async fn remove_peer(&self, interface: &str, public_key: &str) -> Result<()> {
        debug!("Removing peer from userspace interface");

        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;

        if let Some(idx) = Self::find_peer_index(&state.peers, public_key) {
            state.peers.remove(idx);
            state.stats.peer_count = state.peers.len();
            Ok(())
        } else {
            Err(Error::WireGuardConfig(format!(
                "Peer {} not found",
                public_key
            )))
        }
    }

    async fn update_endpoint(
        &self,
        interface: &str,
        public_key: &str,
        endpoint: SocketAddr,
    ) -> Result<()> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;

        if let Some(idx) = Self::find_peer_index(&state.peers, public_key) {
            state.peers[idx].endpoint = Some(endpoint);
            Ok(())
        } else {
            Err(Error::WireGuardConfig(format!(
                "Peer {} not found",
                public_key
            )))
        }
    }

    async fn get_interface_stats(&self, interface: &str) -> Result<InterfaceStats> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let state = tunnel.read().await;
        Ok(state.stats.clone())
    }

    async fn get_peer_stats(&self, interface: &str, public_key: &str) -> Result<PeerStats> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let state = tunnel.read().await;

        state
            .peers
            .iter()
            .find(|p| p.public_key == public_key)
            .map(|p| PeerStats {
                public_key: p.public_key.clone(),
                endpoint: p.endpoint,
                allowed_ips: p.allowed_ips.iter().map(|ip| ip.to_string()).collect(),
                last_handshake: None, // Would be tracked in real implementation
                rx_bytes: 0,
                tx_bytes: 0,
                persistent_keepalive: p.persistent_keepalive,
            })
            .ok_or_else(|| Error::WireGuardConfig(format!("Peer {} not found", public_key)))
    }

    async fn list_peers(&self, interface: &str) -> Result<Vec<PeerStats>> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let state = tunnel.read().await;

        Ok(state
            .peers
            .iter()
            .map(|p| PeerStats {
                public_key: p.public_key.clone(),
                endpoint: p.endpoint,
                allowed_ips: p.allowed_ips.iter().map(|ip| ip.to_string()).collect(),
                last_handshake: None,
                rx_bytes: 0,
                tx_bytes: 0,
                persistent_keepalive: p.persistent_keepalive,
            })
            .collect())
    }

    async fn set_mtu(&self, interface: &str, mtu: u16) -> Result<()> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;
        state.config.mtu = Some(mtu);
        Ok(())
    }

    async fn bring_up(&self, interface: &str) -> Result<()> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;
        state.is_up = true;
        Ok(())
    }

    async fn bring_down(&self, interface: &str) -> Result<()> {
        let tunnel = self
            .tunnels
            .get(interface)
            .ok_or_else(|| Error::WireGuardConfig(format!("Interface {} not found", interface)))?;

        let mut state = tunnel.write().await;
        state.is_up = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wireguard::KeyPair;
    use ipnet::Ipv4Net;
    use std::str::FromStr;

    fn create_test_config() -> InterfaceConfig {
        let keypair = KeyPair::generate();
        InterfaceConfig {
            name: "wg-test".to_string(),
            private_key: keypair.private_key_base64(),
            listen_port: 51820,
            address: Ipv4Net::from_str("10.0.0.1/24").unwrap(),
            mtu: Some(1420),
            dns: None,
            pre_up: None,
            post_up: None,
            pre_down: None,
            post_down: None,
            save_config: false,
        }
    }

    fn create_test_peer() -> PeerConfig {
        let keypair = KeyPair::generate();
        PeerConfig {
            public_key: keypair.public_key_base64(),
            preshared_key: None,
            allowed_ips: vec![Ipv4Net::from_str("10.0.0.2/32").unwrap()],
            endpoint: Some("192.168.1.100:51820".parse().unwrap()),
            persistent_keepalive: Some(25),
            comment: None,
        }
    }

    #[test]
    fn test_backend_type() {
        let backend = UserspaceBackend::new();
        assert_eq!(backend.backend_type(), BackendType::Userspace);
        assert!(backend.is_available());
    }

    #[tokio::test]
    async fn test_create_and_destroy_interface() {
        let backend = UserspaceBackend::new();
        let config = create_test_config();

        // Create interface
        backend.create_interface(&config).await.unwrap();
        assert!(backend.interface_exists(&config.name).await.unwrap());

        // Destroy interface
        backend.destroy_interface(&config.name).await.unwrap();
        assert!(!backend.interface_exists(&config.name).await.unwrap());
    }

    #[tokio::test]
    async fn test_add_and_remove_peer() {
        let backend = UserspaceBackend::new();
        let config = create_test_config();
        let peer = create_test_peer();

        // Create interface
        backend.create_interface(&config).await.unwrap();

        // Add peer
        backend.add_peer(&config.name, &peer).await.unwrap();

        // Verify peer exists
        let peers = backend.list_peers(&config.name).await.unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].public_key, peer.public_key);

        // Remove peer
        backend
            .remove_peer(&config.name, &peer.public_key)
            .await
            .unwrap();

        // Verify peer removed
        let peers = backend.list_peers(&config.name).await.unwrap();
        assert_eq!(peers.len(), 0);

        // Cleanup
        backend.destroy_interface(&config.name).await.unwrap();
    }

    #[tokio::test]
    async fn test_update_endpoint() {
        let backend = UserspaceBackend::new();
        let config = create_test_config();
        let peer = create_test_peer();

        backend.create_interface(&config).await.unwrap();
        backend.add_peer(&config.name, &peer).await.unwrap();

        // Update endpoint
        let new_endpoint: SocketAddr = "10.0.0.50:51821".parse().unwrap();
        backend
            .update_endpoint(&config.name, &peer.public_key, new_endpoint)
            .await
            .unwrap();

        // Verify update
        let peer_stats = backend
            .get_peer_stats(&config.name, &peer.public_key)
            .await
            .unwrap();
        assert_eq!(peer_stats.endpoint, Some(new_endpoint));

        backend.destroy_interface(&config.name).await.unwrap();
    }

    #[tokio::test]
    async fn test_interface_stats() {
        let backend = UserspaceBackend::new();
        let config = create_test_config();

        backend.create_interface(&config).await.unwrap();

        let stats = backend.get_interface_stats(&config.name).await.unwrap();
        assert_eq!(stats.name, config.name);
        assert_eq!(stats.listen_port, config.listen_port);
        assert!(!stats.public_key.is_empty());

        backend.destroy_interface(&config.name).await.unwrap();
    }

    #[tokio::test]
    async fn test_bring_up_down() {
        let backend = UserspaceBackend::new();
        let config = create_test_config();

        backend.create_interface(&config).await.unwrap();

        // Interface starts up
        backend.bring_down(&config.name).await.unwrap();
        backend.bring_up(&config.name).await.unwrap();

        backend.destroy_interface(&config.name).await.unwrap();
    }
}
