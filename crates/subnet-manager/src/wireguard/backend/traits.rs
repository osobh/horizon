//! WireGuard backend trait definition
//!
//! Defines the interface that all WireGuard backends must implement,
//! allowing for multiple implementations (command-line, netlink, userspace).

use crate::wireguard::{InterfaceConfig, PeerConfig};
use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

/// Type of WireGuard backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// Shell out to `wg` and `ip` commands (most portable)
    Command,
    /// Use Linux netlink for direct kernel interface (fastest, Linux only)
    Netlink,
    /// Use boringtun for userspace WireGuard (cross-platform)
    Userspace,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Command => write!(f, "command"),
            BackendType::Netlink => write!(f, "netlink"),
            BackendType::Userspace => write!(f, "userspace"),
        }
    }
}

/// Statistics for a WireGuard interface
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterfaceStats {
    /// Interface name
    pub name: String,
    /// Public key of this interface
    pub public_key: String,
    /// Listen port
    pub listen_port: u16,
    /// Number of peers
    pub peer_count: usize,
    /// Total bytes received
    pub rx_bytes: u64,
    /// Total bytes transmitted
    pub tx_bytes: u64,
}

/// Statistics for a WireGuard peer
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerStats {
    /// Peer public key
    pub public_key: String,
    /// Peer endpoint (if known)
    pub endpoint: Option<SocketAddr>,
    /// Allowed IPs for this peer
    pub allowed_ips: Vec<String>,
    /// Time since last handshake
    pub last_handshake: Option<Duration>,
    /// Bytes received from peer
    pub rx_bytes: u64,
    /// Bytes transmitted to peer
    pub tx_bytes: u64,
    /// Persistent keepalive interval
    pub persistent_keepalive: Option<u16>,
}

/// WireGuard backend trait
///
/// All WireGuard backend implementations must implement this trait.
/// This allows the subnet-manager to work with different WireGuard
/// implementations transparently.
#[async_trait]
pub trait WireGuardBackend: Send + Sync {
    /// Get the backend type
    fn backend_type(&self) -> BackendType;

    /// Check if this backend is available on the current system
    fn is_available(&self) -> bool;

    /// Create a new WireGuard interface
    ///
    /// This creates the interface, sets the private key and listen port,
    /// assigns the IP address, and brings the interface up.
    async fn create_interface(&self, config: &InterfaceConfig) -> Result<()>;

    /// Destroy a WireGuard interface
    ///
    /// Removes the interface from the system.
    async fn destroy_interface(&self, name: &str) -> Result<()>;

    /// Check if an interface exists
    async fn interface_exists(&self, name: &str) -> Result<bool>;

    /// Add a peer to an interface
    async fn add_peer(&self, interface: &str, peer: &PeerConfig) -> Result<()>;

    /// Remove a peer from an interface
    async fn remove_peer(&self, interface: &str, public_key: &str) -> Result<()>;

    /// Update a peer's endpoint (for roaming support)
    async fn update_endpoint(
        &self,
        interface: &str,
        public_key: &str,
        endpoint: SocketAddr,
    ) -> Result<()>;

    /// Get interface statistics
    async fn get_interface_stats(&self, interface: &str) -> Result<InterfaceStats>;

    /// Get peer statistics
    async fn get_peer_stats(&self, interface: &str, public_key: &str) -> Result<PeerStats>;

    /// List all peers on an interface
    async fn list_peers(&self, interface: &str) -> Result<Vec<PeerStats>>;

    /// Set the interface MTU
    async fn set_mtu(&self, interface: &str, mtu: u16) -> Result<()>;

    /// Bring interface up
    async fn bring_up(&self, interface: &str) -> Result<()>;

    /// Bring interface down
    async fn bring_down(&self, interface: &str) -> Result<()>;
}

/// Backend capability flags
#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    /// Can create interfaces
    pub can_create_interface: bool,
    /// Can set MTU
    pub can_set_mtu: bool,
    /// Can get detailed statistics
    pub can_get_stats: bool,
    /// Supports hot peer updates
    pub supports_hot_update: bool,
    /// Supports persistent keepalive
    pub supports_keepalive: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            can_create_interface: true,
            can_set_mtu: true,
            can_get_stats: true,
            supports_hot_update: true,
            supports_keepalive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Command.to_string(), "command");
        assert_eq!(BackendType::Netlink.to_string(), "netlink");
        assert_eq!(BackendType::Userspace.to_string(), "userspace");
    }

    #[test]
    fn test_interface_stats_default() {
        let stats = InterfaceStats::default();
        assert!(stats.name.is_empty());
        assert_eq!(stats.peer_count, 0);
    }
}
