//! Subnet-aware WireGuard layer
//!
//! Provides multi-subnet WireGuard routing with automatic peer discovery
//! and cross-subnet connectivity management.

use super::{InterfaceConfig, PeerConfig, WireGuardConfigGenerator};
use crate::events::{SubnetEventPublisher, SubnetMessage, PeerConfigUpdatedEvent};
use crate::models::{CrossSubnetRoute, RouteDirection, Subnet, SubnetAssignment};
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// Errors for subnet-aware WireGuard operations
#[derive(Debug, Error)]
pub enum SubnetWireGuardError {
    #[error("Subnet not found: {0}")]
    SubnetNotFound(Uuid),

    #[error("Peer not found: {0}")]
    PeerNotFound(String),

    #[error("No route between subnets {0} and {1}")]
    NoRoute(Uuid, Uuid),

    #[error("Interface error: {0}")]
    InterfaceError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// A peer with subnet context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetPeer {
    /// Node ID
    pub node_id: Uuid,
    /// WireGuard public key
    pub public_key: String,
    /// Endpoint address (if known)
    pub endpoint: Option<SocketAddr>,
    /// Assigned IP within subnet
    pub assigned_ip: Ipv4Addr,
    /// Subnet ID this peer belongs to
    pub subnet_id: Uuid,
    /// Allowed IPs for this peer
    pub allowed_ips: Vec<String>,
    /// Persistent keepalive interval
    pub persistent_keepalive: Option<u16>,
    /// Last handshake time (if available)
    pub last_handshake: Option<chrono::DateTime<Utc>>,
}

impl SubnetPeer {
    /// Create a new subnet peer
    pub fn new(
        node_id: Uuid,
        public_key: &str,
        assigned_ip: Ipv4Addr,
        subnet_id: Uuid,
    ) -> Self {
        Self {
            node_id,
            public_key: public_key.to_string(),
            endpoint: None,
            assigned_ip,
            subnet_id,
            allowed_ips: vec![format!("{}/32", assigned_ip)],
            persistent_keepalive: Some(25),
            last_handshake: None,
        }
    }

    /// Set the endpoint
    pub fn with_endpoint(mut self, endpoint: SocketAddr) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Add additional allowed IPs
    pub fn with_allowed_ips(mut self, ips: Vec<String>) -> Self {
        self.allowed_ips.extend(ips);
        self
    }
}

/// Interface state for a subnet
#[derive(Debug)]
struct SubnetInterface {
    subnet_id: Uuid,
    interface_name: String,
    listen_port: u16,
    private_key: String,
    public_key: String,
    peers: DashMap<String, SubnetPeer>, // keyed by public_key
}

/// Subnet-aware WireGuard layer
///
/// Manages multiple WireGuard interfaces, one per subnet, with
/// cross-subnet routing based on configured routes.
pub struct SubnetAwareWireGuard {
    /// Interfaces by subnet ID
    interfaces: DashMap<Uuid, Arc<SubnetInterface>>,
    /// Cross-subnet routes
    routes: RwLock<Vec<CrossSubnetRoute>>,
    /// Event publisher for config updates
    event_publisher: Option<Arc<SubnetEventPublisher>>,
    /// Config generator
    config_generator: WireGuardConfigGenerator,
    /// Address map for routing: IP -> (subnet_id, public_key)
    address_map: DashMap<Ipv4Addr, (Uuid, String)>,
}

impl SubnetAwareWireGuard {
    /// Create a new subnet-aware WireGuard layer
    pub fn new() -> Self {
        Self {
            interfaces: DashMap::new(),
            routes: RwLock::new(Vec::new()),
            event_publisher: None,
            config_generator: WireGuardConfigGenerator::new(),
            address_map: DashMap::new(),
        }
    }

    /// Set the event publisher
    pub fn with_event_publisher(mut self, publisher: Arc<SubnetEventPublisher>) -> Self {
        self.event_publisher = Some(publisher);
        self
    }

    /// Create an interface for a subnet
    #[instrument(skip(self, subnet), fields(subnet_id = %subnet.id, interface = %subnet.wg_interface))]
    pub fn create_interface(
        &self,
        subnet: &Subnet,
        private_key: &str,
        public_key: &str,
    ) -> Result<(), SubnetWireGuardError> {
        if self.interfaces.contains_key(&subnet.id) {
            warn!("Interface already exists for subnet");
            return Ok(());
        }

        let interface = SubnetInterface {
            subnet_id: subnet.id,
            interface_name: subnet.wg_interface.clone(),
            listen_port: subnet.wg_listen_port,
            private_key: private_key.to_string(),
            public_key: public_key.to_string(),
            peers: DashMap::new(),
        };

        self.interfaces.insert(subnet.id, Arc::new(interface));
        info!("Created WireGuard interface for subnet");

        Ok(())
    }

    /// Remove an interface for a subnet
    pub fn remove_interface(&self, subnet_id: Uuid) -> Result<(), SubnetWireGuardError> {
        // Remove all peers from address map
        if let Some((_, interface)) = self.interfaces.remove(&subnet_id) {
            for peer in interface.peers.iter() {
                self.address_map.remove(&peer.assigned_ip);
            }
        }
        Ok(())
    }

    /// Add a peer to a subnet
    #[instrument(skip(self), fields(
        subnet_id = %peer.subnet_id,
        node_id = %peer.node_id,
        ip = %peer.assigned_ip
    ))]
    pub async fn add_peer(&self, peer: SubnetPeer) -> Result<(), SubnetWireGuardError> {
        let interface = self
            .interfaces
            .get(&peer.subnet_id)
            .ok_or(SubnetWireGuardError::SubnetNotFound(peer.subnet_id))?;

        // Add to address map
        self.address_map.insert(
            peer.assigned_ip,
            (peer.subnet_id, peer.public_key.clone()),
        );

        // Add to interface peers
        interface.peers.insert(peer.public_key.clone(), peer.clone());

        debug!("Added peer to subnet interface");

        // Publish event
        if let Some(publisher) = &self.event_publisher {
            let _ = publisher
                .publish(SubnetMessage::PeerConfigUpdated(PeerConfigUpdatedEvent {
                    subnet_id: peer.subnet_id,
                    node_id: peer.node_id,
                    public_key: peer.public_key.clone(),
                    endpoint: peer.endpoint.map(|e| e.to_string()),
                    allowed_ips: peer.allowed_ips.clone(),
                    updated_at: Utc::now(),
                }))
                .await;
        }

        Ok(())
    }

    /// Remove a peer from a subnet
    pub fn remove_peer(
        &self,
        subnet_id: Uuid,
        public_key: &str,
    ) -> Result<Option<SubnetPeer>, SubnetWireGuardError> {
        let interface = self
            .interfaces
            .get(&subnet_id)
            .ok_or(SubnetWireGuardError::SubnetNotFound(subnet_id))?;

        if let Some((_, peer)) = interface.peers.remove(public_key) {
            self.address_map.remove(&peer.assigned_ip);
            Ok(Some(peer))
        } else {
            Ok(None)
        }
    }

    /// Update a peer's endpoint
    #[instrument(skip(self), fields(subnet_id = %subnet_id))]
    pub async fn update_peer_endpoint(
        &self,
        subnet_id: Uuid,
        public_key: &str,
        endpoint: SocketAddr,
    ) -> Result<(), SubnetWireGuardError> {
        let interface = self
            .interfaces
            .get(&subnet_id)
            .ok_or(SubnetWireGuardError::SubnetNotFound(subnet_id))?;

        let result = {
            if let Some(mut peer) = interface.peers.get_mut(public_key) {
                peer.endpoint = Some(endpoint);
                debug!(endpoint = %endpoint, "Updated peer endpoint");
                Ok(())
            } else {
                Err(SubnetWireGuardError::PeerNotFound(public_key.to_string()))
            }
        };
        result
    }

    /// Add a cross-subnet route
    pub fn add_route(&self, route: CrossSubnetRoute) {
        let mut routes = self.routes.write();
        routes.push(route);
    }

    /// Remove a cross-subnet route
    pub fn remove_route(&self, source_id: Uuid, dest_id: Uuid) {
        let mut routes = self.routes.write();
        routes.retain(|r| !(r.source_subnet_id == source_id && r.destination_subnet_id == dest_id));
    }

    /// Check if a route exists between two subnets
    pub fn has_route(&self, source_id: Uuid, dest_id: Uuid) -> bool {
        let routes = self.routes.read();
        routes.iter().any(|r| {
            (r.source_subnet_id == source_id && r.destination_subnet_id == dest_id)
                || (r.direction == RouteDirection::Bidirectional
                    && r.source_subnet_id == dest_id
                    && r.destination_subnet_id == source_id)
        })
    }

    /// Get the best route to reach a destination IP
    pub fn route_to(&self, dest_ip: Ipv4Addr) -> Option<(Uuid, String)> {
        self.address_map.get(&dest_ip).map(|r| r.clone())
    }

    /// Get all peers for a subnet
    pub fn get_subnet_peers(&self, subnet_id: Uuid) -> Vec<SubnetPeer> {
        self.interfaces
            .get(&subnet_id)
            .map(|iface| iface.peers.iter().map(|p| p.clone()).collect())
            .unwrap_or_default()
    }

    /// Get all peers across all subnets
    pub fn get_all_peers(&self) -> Vec<SubnetPeer> {
        self.interfaces
            .iter()
            .flat_map(|iface| iface.peers.iter().map(|p| p.clone()).collect::<Vec<_>>())
            .collect()
    }

    /// Generate WireGuard config for a subnet
    pub fn generate_config(&self, subnet_id: Uuid) -> Result<String, SubnetWireGuardError> {
        let interface = self
            .interfaces
            .get(&subnet_id)
            .ok_or(SubnetWireGuardError::SubnetNotFound(subnet_id))?;

        let mut config = format!(
            "[Interface]\nPrivateKey = {}\nListenPort = {}\n\n",
            interface.private_key, interface.listen_port
        );

        for peer in interface.peers.iter() {
            config.push_str(&format!("[Peer]\nPublicKey = {}\n", peer.public_key));
            if let Some(endpoint) = &peer.endpoint {
                config.push_str(&format!("Endpoint = {}\n", endpoint));
            }
            config.push_str(&format!("AllowedIPs = {}\n", peer.allowed_ips.join(", ")));
            if let Some(keepalive) = peer.persistent_keepalive {
                config.push_str(&format!("PersistentKeepalive = {}\n", keepalive));
            }
            config.push('\n');
        }

        Ok(config)
    }

    /// Get interface info for a subnet
    pub fn get_interface_info(&self, subnet_id: Uuid) -> Option<InterfaceInfo> {
        self.interfaces.get(&subnet_id).map(|iface| InterfaceInfo {
            subnet_id: iface.subnet_id,
            interface_name: iface.interface_name.clone(),
            listen_port: iface.listen_port,
            public_key: iface.public_key.clone(),
            peer_count: iface.peers.len(),
        })
    }

    /// Get all interface infos
    pub fn get_all_interfaces(&self) -> Vec<InterfaceInfo> {
        self.interfaces
            .iter()
            .map(|iface| InterfaceInfo {
                subnet_id: iface.subnet_id,
                interface_name: iface.interface_name.clone(),
                listen_port: iface.listen_port,
                public_key: iface.public_key.clone(),
                peer_count: iface.peers.len(),
            })
            .collect()
    }
}

impl Default for SubnetAwareWireGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Interface information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceInfo {
    pub subnet_id: Uuid,
    pub interface_name: String,
    pub listen_port: u16,
    pub public_key: String,
    pub peer_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SubnetPurpose;
    use ipnet::Ipv4Net;
    use std::str::FromStr;

    fn create_test_subnet(id: Uuid, name: &str, port: u16) -> Subnet {
        let mut subnet = Subnet::new(
            name,
            Ipv4Net::from_str("10.100.0.0/24").unwrap(),
            SubnetPurpose::Tenant,
            port,
        );
        subnet.id = id;
        subnet
    }

    #[test]
    fn test_create_interface() {
        let wg = SubnetAwareWireGuard::new();
        let subnet_id = Uuid::new_v4();
        let subnet = create_test_subnet(subnet_id, "test", 51820);

        wg.create_interface(&subnet, "private_key", "public_key")
            .unwrap();

        assert!(wg.interfaces.contains_key(&subnet_id));
    }

    #[tokio::test]
    async fn test_add_peer() {
        let wg = SubnetAwareWireGuard::new();
        let subnet_id = Uuid::new_v4();
        let subnet = create_test_subnet(subnet_id, "test", 51820);

        wg.create_interface(&subnet, "private_key", "public_key")
            .unwrap();

        let peer = SubnetPeer::new(
            Uuid::new_v4(),
            "peer_public_key",
            Ipv4Addr::new(10, 100, 0, 5),
            subnet_id,
        );

        wg.add_peer(peer.clone()).await.unwrap();

        let peers = wg.get_subnet_peers(subnet_id);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].public_key, "peer_public_key");
    }

    #[test]
    fn test_route_lookup() {
        let wg = SubnetAwareWireGuard::new();
        let subnet_id = Uuid::new_v4();
        let ip = Ipv4Addr::new(10, 100, 0, 5);

        wg.address_map
            .insert(ip, (subnet_id, "public_key".to_string()));

        let route = wg.route_to(ip);
        assert!(route.is_some());
        assert_eq!(route.unwrap().0, subnet_id);
    }

    #[test]
    fn test_cross_subnet_routing() {
        let wg = SubnetAwareWireGuard::new();
        let subnet1 = Uuid::new_v4();
        let subnet2 = Uuid::new_v4();

        // No route initially
        assert!(!wg.has_route(subnet1, subnet2));

        // Add bidirectional route
        wg.add_route(CrossSubnetRoute::new(subnet1, subnet2));

        // Should have route in both directions
        assert!(wg.has_route(subnet1, subnet2));
        assert!(wg.has_route(subnet2, subnet1));

        // Remove route
        wg.remove_route(subnet1, subnet2);
        assert!(!wg.has_route(subnet1, subnet2));
    }
}
