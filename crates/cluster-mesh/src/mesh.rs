//! Mesh topology and connection management
//!
//! This module handles the formation and maintenance of the cluster mesh,
//! including NAT traversal, secure connections, and adaptive topology.

use crate::{ClusterMeshError, ClusterNode, NodeClass, Result};
use chrono::{DateTime, Duration, Utc};
use dashmap::{DashMap, DashSet};
use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::{Certificate, PrivateKey};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Mesh topology types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeshTopology {
    /// Full mesh - all nodes connected to all
    FullMesh,
    /// Star topology - all nodes connect to central coordinators
    Star { coordinators: Vec<Uuid> },
    /// Hierarchical - tiered connections based on node class
    Hierarchical,
    /// Hybrid - adaptive topology based on network conditions
    Hybrid {
        core_nodes: Vec<Uuid>,
        edge_strategy: Box<MeshTopology>,
    },
}

/// Node connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConnection {
    pub node_id: Uuid,
    pub endpoint: SocketAddr,
    pub connection_type: ConnectionType,
    pub established_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub latency_ms: f32,
    pub bandwidth_mbps: f32,
    pub packet_loss: f32,
}

/// Connection type between nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    Direct,
    NatTraversal,
    Relay(Uuid), // Relay node ID
    Tunnel,
}

/// Mesh manager
pub struct MeshManager {
    topology: RwLock<MeshTopology>,
    connections: Arc<DashMap<Uuid, NodeConnection>>,
    endpoint: Arc<RwLock<Option<Endpoint>>>,
    node_registry: Arc<DashMap<Uuid, NodeEndpoint>>,
    stun_servers: Vec<String>,
    relay_nodes: Arc<DashSet<Uuid>>,
}

/// Node endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NodeEndpoint {
    node_id: Uuid,
    public_addr: Option<SocketAddr>,
    private_addr: SocketAddr,
    nat_type: crate::discovery::NatType,
    last_updated: DateTime<Utc>,
}

impl MeshManager {
    /// Create a new mesh manager
    pub async fn new() -> Result<Self> {
        Ok(Self {
            topology: RwLock::new(MeshTopology::Hierarchical),
            connections: Arc::new(DashMap::new()),
            endpoint: Arc::new(RwLock::new(None)),
            node_registry: Arc::new(DashMap::new()),
            stun_servers: vec![
                "stun.l.google.com:19302".to_string(),
                "stun1.l.google.com:19302".to_string(),
            ],
            relay_nodes: Arc::new(DashSet::new()),
        })
    }

    /// Start mesh formation
    pub async fn start_mesh_formation(&self) -> Result<()> {
        // Initialize QUIC endpoint
        self.initialize_endpoint().await?;

        // Start connection manager
        tokio::spawn(self.clone().connection_manager());

        // Start topology optimizer
        tokio::spawn(self.clone().topology_optimizer());

        Ok(())
    }

    /// Clone for async spawning
    fn clone(&self) -> Self {
        Self {
            topology: RwLock::new(MeshTopology::Hierarchical),
            connections: Arc::clone(&self.connections),
            endpoint: Arc::clone(&self.endpoint),
            node_registry: Arc::clone(&self.node_registry),
            stun_servers: self.stun_servers.clone(),
            relay_nodes: Arc::clone(&self.relay_nodes),
        }
    }

    /// Initialize QUIC endpoint
    async fn initialize_endpoint(&self) -> Result<()> {
        let (cert, key) = self.generate_self_signed_cert()?;

        // Server configuration
        let mut server_config = ServerConfig::with_single_cert(vec![cert.clone()], key.clone())
            .map_err(|e| ClusterMeshError::Certificate(e.to_string()))?;

        // Configure transport
        let transport_config = Arc::get_mut(&mut server_config.transport).unwrap();
        transport_config.max_concurrent_uni_streams(0_u8.into());
        transport_config.max_idle_timeout(Some(
            Duration::seconds(60).to_std().unwrap().try_into().unwrap(),
        ));

        // Client configuration
        let client_crypto = self.create_client_crypto_config()?;
        let _client_config = ClientConfig::new(Arc::new(client_crypto));

        // Create endpoint
        let addr = "0.0.0.0:0".parse::<SocketAddr>().unwrap();
        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| ClusterMeshError::Network(e.to_string()))?;

        *self.endpoint.write().await = Some(endpoint);

        Ok(())
    }

    /// Generate self-signed certificate
    fn generate_self_signed_cert(&self) -> Result<(Certificate, PrivateKey)> {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| ClusterMeshError::Certificate(e.to_string()))?;

        let cert_der = cert
            .serialize_der()
            .map_err(|e| ClusterMeshError::Certificate(e.to_string()))?;
        let key_der = cert.serialize_private_key_der();

        Ok((Certificate(cert_der), PrivateKey(key_der)))
    }

    /// Create client crypto configuration
    fn create_client_crypto_config(&self) -> Result<rustls::ClientConfig> {
        let config = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCertVerifier))
            .with_no_client_auth();

        Ok(config)
    }

    /// Connection manager loop
    async fn connection_manager(self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            if let Err(e) = self.maintain_connections().await {
                tracing::error!("Connection maintenance error: {}", e);
            }
        }
    }

    /// Topology optimizer loop
    async fn topology_optimizer(self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));

        loop {
            interval.tick().await;

            if let Err(e) = self.optimize_topology().await {
                tracing::error!("Topology optimization error: {}", e);
            }
        }
    }

    /// Add a node to the mesh
    pub async fn add_node(&self, node: &ClusterNode) -> Result<()> {
        // Register node endpoint
        let endpoint = NodeEndpoint {
            node_id: node.id,
            public_addr: None, // Would be discovered via STUN
            private_addr: "127.0.0.1:0".parse().unwrap(), // Placeholder
            nat_type: node.network.nat_type.clone(),
            last_updated: Utc::now(),
        };

        self.node_registry.insert(node.id, endpoint);

        // Determine if node can be a relay
        if matches!(node.class, NodeClass::DataCenter { .. }) {
            self.relay_nodes.insert(node.id);
        }

        // Initiate connections based on topology
        self.establish_connections(node).await?;

        Ok(())
    }

    /// Remove a node from the mesh
    pub async fn remove_node(&self, node: &ClusterNode) -> Result<()> {
        // Remove from registry
        self.node_registry.remove(&node.id);

        // Remove from relay nodes
        self.relay_nodes.remove(&node.id);

        // Close connections
        self.connections.remove(&node.id);

        // Reestablish affected connections
        self.handle_node_departure(node.id).await?;

        Ok(())
    }

    /// Establish connections for a node
    async fn establish_connections(&self, node: &ClusterNode) -> Result<()> {
        let topology = self.topology.read().await;

        match *topology {
            MeshTopology::FullMesh => {
                self.establish_full_mesh_connections(node).await?;
            }
            MeshTopology::Star { ref coordinators } => {
                self.establish_star_connections(node, coordinators).await?;
            }
            MeshTopology::Hierarchical => {
                self.establish_hierarchical_connections(node).await?;
            }
            MeshTopology::Hybrid { core_nodes: _, edge_strategy: _ } => {
                self.establish_hybrid_connections(node).await?;
            }
        }

        Ok(())
    }

    /// Establish full mesh connections
    async fn establish_full_mesh_connections(&self, node: &ClusterNode) -> Result<()> {
        for entry in self.node_registry.iter() {
            let peer_id = *entry.key();
            if peer_id != node.id {
                self.connect_to_peer(node.id, peer_id, entry.value()).await?;
            }
        }

        Ok(())
    }

    /// Establish star topology connections
    async fn establish_star_connections(
        &self,
        node: &ClusterNode,
        coordinators: &[Uuid],
    ) -> Result<()> {
        // Connect to coordinators
        for coordinator_id in coordinators {
            if *coordinator_id != node.id {
                if let Some(endpoint) = self.node_registry.get(coordinator_id) {
                    self.connect_to_peer(node.id, *coordinator_id, endpoint.value())
                        .await?;
                }
            }
        }

        Ok(())
    }

    /// Establish hierarchical connections
    async fn establish_hierarchical_connections(&self, node: &ClusterNode) -> Result<()> {
        match &node.class {
            NodeClass::DataCenter { .. } => {
                // Connect to all datacenter nodes
                for entry in self.node_registry.iter() {
                    let peer_id = *entry.key();
                    if peer_id != node.id {
                        self.connect_to_peer(node.id, peer_id, entry.value()).await?;
                    }
                }
            }
            NodeClass::Workstation { .. } => {
                // Connect to datacenter nodes
                self.connect_to_node_class(
                    node.id,
                    NodeClass::DataCenter {
                        gpus: vec![],
                        bandwidth: crate::classification::Bandwidth::Gigabit(1.0),
                    },
                )
                .await?;
            }
            NodeClass::Laptop { .. } | NodeClass::Edge { .. } => {
                // Connect via relay nodes
                self.connect_via_relay(node.id).await?;
            }
        }

        Ok(())
    }

    /// Establish hybrid connections
    async fn establish_hybrid_connections(&self, node: &ClusterNode) -> Result<()> {
        // Adaptive strategy based on node characteristics
        match node.network.nat_type {
            crate::discovery::NatType::None => {
                // Direct connections possible
                self.establish_hierarchical_connections(node).await?;
            }
            crate::discovery::NatType::FullCone | crate::discovery::NatType::RestrictedCone => {
                // STUN should work
                self.connect_with_nat_traversal(node).await?;
            }
            _ => {
                // Need relay
                self.connect_via_relay(node.id).await?;
            }
        }

        Ok(())
    }

    /// Connect to a specific peer
    async fn connect_to_peer(
        &self,
        _from_id: Uuid,
        to_id: Uuid,
        endpoint: &NodeEndpoint,
    ) -> Result<()> {
        // This would establish actual QUIC connection
        let connection = NodeConnection {
            node_id: to_id,
            endpoint: endpoint.private_addr, // Would use public addr after STUN
            connection_type: ConnectionType::Direct,
            established_at: Utc::now(),
            last_heartbeat: Utc::now(),
            latency_ms: 0.0,
            bandwidth_mbps: 0.0,
            packet_loss: 0.0,
        };

        self.connections.insert(to_id, connection);

        Ok(())
    }

    /// Connect to nodes of a specific class
    async fn connect_to_node_class(&self, _from_id: Uuid, _class: NodeClass) -> Result<()> {
        // Would filter nodes by class and connect
        Ok(())
    }

    /// Connect with NAT traversal
    async fn connect_with_nat_traversal(&self, _node: &ClusterNode) -> Result<()> {
        // Would perform STUN/TURN operations
        Ok(())
    }

    /// Connect via relay node
    async fn connect_via_relay(&self, node_id: Uuid) -> Result<()> {
        if let Some(relay_id) = self.relay_nodes.iter().next() {
            // Would establish relayed connection
            tracing::info!("Node {} connecting via relay {}", node_id, *relay_id);
        }

        Ok(())
    }

    /// Maintain existing connections
    async fn maintain_connections(&self) -> Result<()> {
        let now = Utc::now();
        let timeout = Duration::seconds(120);

        // Check for stale connections
        let stale_connections: Vec<Uuid> = self
            .connections
            .iter()
            .filter(|entry| now - entry.value().last_heartbeat > timeout)
            .map(|entry| *entry.key())
            .collect();

        for node_id in stale_connections {
            tracing::warn!("Connection to node {} timed out", node_id);
            self.connections.remove(&node_id);
        }

        Ok(())
    }

    /// Optimize mesh topology
    async fn optimize_topology(&self) -> Result<()> {
        // Calculate average metrics
        let mut total_latency = 0.0;
        let mut total_bandwidth = 0.0;
        let mut count = 0;

        for entry in self.connections.iter() {
            let conn = entry.value();
            total_latency += conn.latency_ms;
            total_bandwidth += conn.bandwidth_mbps;
            count += 1;
        }

        if count > 0 {
            let avg_latency = total_latency / count as f32;
            let avg_bandwidth = total_bandwidth / count as f32;

            tracing::info!(
                "Mesh metrics - Avg latency: {:.2}ms, Avg bandwidth: {:.2}Mbps",
                avg_latency,
                avg_bandwidth
            );
        }

        Ok(())
    }

    /// Handle node departure
    async fn handle_node_departure(&self, departed_id: Uuid) -> Result<()> {
        // Find connections that were relayed through departed node
        let affected: Vec<Uuid> = self
            .connections
            .iter()
            .filter(|entry| {
                matches!(entry.value().connection_type, ConnectionType::Relay(relay_id) if relay_id == departed_id)
            })
            .map(|entry| *entry.key())
            .collect();

        // Reestablish affected connections
        for node_id in affected {
            tracing::info!(
                "Reestablishing connection for node {} after relay {} departed",
                node_id,
                departed_id
            );
            self.connect_via_relay(node_id).await?;
        }

        Ok(())
    }

    /// Get mesh statistics
    pub async fn get_statistics(&self) -> MeshStatistics {
        let mut stats = MeshStatistics {
            total_nodes: self.node_registry.len(),
            active_connections: self.connections.len(),
            relay_nodes: self.relay_nodes.len(),
            ..Default::default()
        };

        for entry in self.connections.iter() {
            match entry.value().connection_type {
                ConnectionType::Direct => stats.direct_connections += 1,
                ConnectionType::NatTraversal => stats.nat_traversal_connections += 1,
                ConnectionType::Relay(_) => stats.relayed_connections += 1,
                ConnectionType::Tunnel => stats.tunnel_connections += 1,
            }
        }

        stats
    }
}

/// Custom certificate verifier that accepts any certificate
/// (For development/testing only!)
struct AcceptAnyCertVerifier;

impl rustls::client::ServerCertVerifier for AcceptAnyCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &Certificate,
        _intermediates: &[Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> std::result::Result<rustls::client::ServerCertVerified, rustls::Error> {
        // This is a dummy implementation for testing
        // In production, proper certificate verification should be implemented
        Ok(rustls::client::ServerCertVerified::assertion())
    }
}

/// Mesh statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshStatistics {
    pub total_nodes: usize,
    pub active_connections: usize,
    pub direct_connections: usize,
    pub nat_traversal_connections: usize,
    pub relayed_connections: usize,
    pub tunnel_connections: usize,
    pub relay_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::{Bandwidth, NodeClass};

    #[tokio::test]
    async fn test_mesh_manager_creation() {
        let manager = MeshManager::new().await.unwrap();
        let topology = manager.topology.read().await;
        assert_eq!(*topology, MeshTopology::Hierarchical);
    }

    #[tokio::test]
    async fn test_node_registration() {
        let manager = MeshManager::new().await.unwrap();

        let node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "test-node".to_string(),
            class: NodeClass::DataCenter {
                gpus: vec![],
                bandwidth: Bandwidth::TenGigabit(10.0),
            },
            hardware: Default::default(),
            network: crate::NetworkCharacteristics {
                bandwidth_mbps: 10000.0,
                latency_ms: 1.0,
                jitter_ms: 0.1,
                packet_loss: 0.0,
                nat_type: crate::discovery::NatType::None,
            },
            status: crate::NodeStatus::Online,
            capabilities: Default::default(),
            last_heartbeat: Utc::now(),
        };

        manager.add_node(&node).await.unwrap();

        // Check node is registered
        assert!(manager.node_registry.contains_key(&node.id));

        // Check datacenter nodes are added as relays
        assert!(manager.relay_nodes.contains(&node.id));
    }

    #[tokio::test]
    async fn test_topology_types() {
        let topologies = vec![
            MeshTopology::FullMesh,
            MeshTopology::Star {
                coordinators: vec![Uuid::new_v4()],
            },
            MeshTopology::Hierarchical,
            MeshTopology::Hybrid {
                core_nodes: vec![Uuid::new_v4(), Uuid::new_v4()],
                edge_strategy: Box::new(MeshTopology::Star {
                    coordinators: vec![Uuid::new_v4()],
                }),
            },
        ];

        for topology in topologies {
            let serialized = serde_json::to_string(&topology).unwrap();
            let deserialized: MeshTopology = serde_json::from_str(&serialized).unwrap();
            // Can't use direct equality with Box, so check variant
            match (&topology, &deserialized) {
                (MeshTopology::FullMesh, MeshTopology::FullMesh) => {}
                (
                    MeshTopology::Star { coordinators: c1 },
                    MeshTopology::Star { coordinators: c2 },
                ) => {
                    assert_eq!(c1, c2);
                }
                (MeshTopology::Hierarchical, MeshTopology::Hierarchical) => {}
                (
                    MeshTopology::Hybrid {
                        core_nodes: cn1, ..
                    },
                    MeshTopology::Hybrid {
                        core_nodes: cn2, ..
                    },
                ) => {
                    assert_eq!(cn1, cn2);
                }
                _ => panic!("Topology mismatch"),
            }
        }
    }

    #[tokio::test]
    async fn test_connection_types() {
        let types = vec![
            ConnectionType::Direct,
            ConnectionType::NatTraversal,
            ConnectionType::Relay(Uuid::new_v4()),
            ConnectionType::Tunnel,
        ];

        for conn_type in types {
            let serialized = serde_json::to_string(&conn_type).unwrap();
            let deserialized: ConnectionType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(conn_type, deserialized);
        }
    }

    #[tokio::test]
    async fn test_self_signed_cert_generation() {
        let manager = MeshManager::new().await.unwrap();
        let (cert, key) = manager.generate_self_signed_cert().unwrap();

        assert!(!cert.0.is_empty());
        assert!(!key.0.is_empty());
    }

    #[tokio::test]
    async fn test_mesh_statistics() {
        let manager = MeshManager::new().await.unwrap();

        // Add some mock connections
        manager.connections.insert(
            Uuid::new_v4(),
            NodeConnection {
                node_id: Uuid::new_v4(),
                endpoint: "127.0.0.1:8080".parse().unwrap(),
                connection_type: ConnectionType::Direct,
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: 1.0,
                bandwidth_mbps: 1000.0,
                packet_loss: 0.0,
            },
        );

        manager.connections.insert(
            Uuid::new_v4(),
            NodeConnection {
                node_id: Uuid::new_v4(),
                endpoint: "127.0.0.1:8081".parse().unwrap(),
                connection_type: ConnectionType::NatTraversal,
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: 10.0,
                bandwidth_mbps: 100.0,
                packet_loss: 0.1,
            },
        );

        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_connections, 2);
        assert_eq!(stats.direct_connections, 1);
        assert_eq!(stats.nat_traversal_connections, 1);
    }
}

#[cfg(test)]
#[path = "mesh_tests.rs"]
mod mesh_coverage_tests;

#[cfg(test)]
#[path = "mesh_edge_tests.rs"]
mod mesh_edge_case_tests;
