//! WireGuard configuration generation
//!
//! Generates WireGuard configuration files for subnets and nodes.

use crate::models::{CrossSubnetRoute, Subnet, SubnetAssignment};
use crate::{Error, Result};
use ipnet::Ipv4Net;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::net::{Ipv4Addr, SocketAddr};
use uuid::Uuid;

/// WireGuard interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    /// Interface name (e.g., "wg-tenant-acme")
    pub name: String,
    /// Private key (base64)
    pub private_key: String,
    /// Listen port
    pub listen_port: u16,
    /// Interface address with CIDR
    pub address: Ipv4Net,
    /// MTU (default 1420 for WireGuard)
    pub mtu: Option<u16>,
    /// DNS servers to push
    pub dns: Option<Vec<Ipv4Addr>>,
    /// Pre/Post up commands
    pub pre_up: Option<String>,
    pub post_up: Option<String>,
    pub pre_down: Option<String>,
    pub post_down: Option<String>,
    /// Save config on shutdown
    pub save_config: bool,
}

impl InterfaceConfig {
    /// Create a new interface config for a subnet
    pub fn for_subnet(subnet: &Subnet, gateway_ip: Ipv4Addr) -> Result<Self> {
        let private_key = subnet
            .wg_private_key
            .clone()
            .ok_or_else(|| Error::WireGuardConfig("Subnet missing private key".to_string()))?;

        // Create address with /32 for the gateway
        let address = Ipv4Net::new(gateway_ip, 32)
            .map_err(|e| Error::WireGuardConfig(format!("Invalid address: {}", e)))?;

        Ok(Self {
            name: subnet.wg_interface.clone(),
            private_key,
            listen_port: subnet.wg_listen_port,
            address,
            mtu: Some(1420),
            dns: None,
            pre_up: None,
            post_up: None,
            pre_down: None,
            post_down: None,
            save_config: false,
        })
    }

    /// Generate WireGuard config file content
    pub fn to_config_string(&self) -> String {
        let mut config = String::new();

        writeln!(config, "[Interface]").unwrap();
        writeln!(config, "PrivateKey = {}", self.private_key).unwrap();
        writeln!(config, "ListenPort = {}", self.listen_port).unwrap();
        writeln!(config, "Address = {}", self.address).unwrap();

        if let Some(mtu) = self.mtu {
            writeln!(config, "MTU = {}", mtu).unwrap();
        }

        if let Some(ref dns) = self.dns {
            let dns_str: Vec<_> = dns.iter().map(|d| d.to_string()).collect();
            writeln!(config, "DNS = {}", dns_str.join(", ")).unwrap();
        }

        if let Some(ref cmd) = self.pre_up {
            writeln!(config, "PreUp = {}", cmd).unwrap();
        }
        if let Some(ref cmd) = self.post_up {
            writeln!(config, "PostUp = {}", cmd).unwrap();
        }
        if let Some(ref cmd) = self.pre_down {
            writeln!(config, "PreDown = {}", cmd).unwrap();
        }
        if let Some(ref cmd) = self.post_down {
            writeln!(config, "PostDown = {}", cmd).unwrap();
        }

        if self.save_config {
            writeln!(config, "SaveConfig = true").unwrap();
        }

        config
    }
}

/// WireGuard peer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Peer public key (base64)
    pub public_key: String,
    /// Preshared key (optional, base64)
    pub preshared_key: Option<String>,
    /// Allowed IP ranges
    pub allowed_ips: Vec<Ipv4Net>,
    /// Endpoint address
    pub endpoint: Option<SocketAddr>,
    /// Persistent keepalive interval (seconds)
    pub persistent_keepalive: Option<u16>,
    /// Comment/description
    pub comment: Option<String>,
}

impl PeerConfig {
    /// Create a peer config from a subnet assignment
    pub fn from_assignment(assignment: &SubnetAssignment, endpoint: Option<SocketAddr>) -> Self {
        // Single IP allowed for this peer
        let allowed_ip = Ipv4Net::new(assignment.assigned_ip, 32).unwrap();

        Self {
            public_key: assignment.wg_public_key.clone(),
            preshared_key: None,
            allowed_ips: vec![allowed_ip],
            endpoint,
            persistent_keepalive: Some(25),
            comment: Some(format!("Node: {}", assignment.node_id)),
        }
    }

    /// Create a peer config for cross-subnet routing
    pub fn for_subnet_route(remote_subnet: &Subnet, _route: &CrossSubnetRoute) -> Result<Self> {
        let public_key = remote_subnet.wg_public_key.clone().ok_or_else(|| {
            Error::WireGuardConfig("Remote subnet missing public key".to_string())
        })?;

        Ok(Self {
            public_key,
            preshared_key: None,
            allowed_ips: vec![remote_subnet.cidr],
            endpoint: None, // Will be resolved dynamically
            persistent_keepalive: Some(25),
            comment: Some(format!(
                "Subnet: {} ({})",
                remote_subnet.name, remote_subnet.id
            )),
        })
    }

    /// Generate WireGuard config section for this peer
    pub fn to_config_string(&self) -> String {
        let mut config = String::new();

        writeln!(config).unwrap();
        if let Some(ref comment) = self.comment {
            writeln!(config, "# {}", comment).unwrap();
        }
        writeln!(config, "[Peer]").unwrap();
        writeln!(config, "PublicKey = {}", self.public_key).unwrap();

        if let Some(ref psk) = self.preshared_key {
            writeln!(config, "PresharedKey = {}", psk).unwrap();
        }

        let allowed_ips: Vec<_> = self.allowed_ips.iter().map(|ip| ip.to_string()).collect();
        writeln!(config, "AllowedIPs = {}", allowed_ips.join(", ")).unwrap();

        if let Some(endpoint) = self.endpoint {
            writeln!(config, "Endpoint = {}", endpoint).unwrap();
        }

        if let Some(keepalive) = self.persistent_keepalive {
            writeln!(config, "PersistentKeepalive = {}", keepalive).unwrap();
        }

        config
    }
}

/// Complete WireGuard subnet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardSubnetConfig {
    /// Subnet ID
    pub subnet_id: Uuid,
    /// Subnet name
    pub subnet_name: String,
    /// Interface configuration
    pub interface: InterfaceConfig,
    /// Peer configurations
    pub peers: Vec<PeerConfig>,
    /// Cross-subnet route peers
    pub route_peers: Vec<PeerConfig>,
}

impl WireGuardSubnetConfig {
    /// Generate complete WireGuard config file
    pub fn to_config_string(&self) -> String {
        let mut config = String::new();

        // Header comment
        writeln!(
            config,
            "# WireGuard configuration for subnet: {}",
            self.subnet_name
        )
        .unwrap();
        writeln!(config, "# Subnet ID: {}", self.subnet_id).unwrap();
        writeln!(
            config,
            "# Generated by subnet-manager at {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )
        .unwrap();
        writeln!(config).unwrap();

        // Interface section
        config.push_str(&self.interface.to_config_string());

        // Node peers
        if !self.peers.is_empty() {
            writeln!(config).unwrap();
            writeln!(config, "# Node Peers ({} nodes)", self.peers.len()).unwrap();
            for peer in &self.peers {
                config.push_str(&peer.to_config_string());
            }
        }

        // Route peers
        if !self.route_peers.is_empty() {
            writeln!(config).unwrap();
            writeln!(
                config,
                "# Cross-Subnet Routes ({} routes)",
                self.route_peers.len()
            )
            .unwrap();
            for peer in &self.route_peers {
                config.push_str(&peer.to_config_string());
            }
        }

        config
    }
}

/// WireGuard configuration generator
pub struct WireGuardConfigGenerator {
    /// Default MTU
    default_mtu: u16,
    /// Default persistent keepalive
    default_keepalive: u16,
    /// Enable preshared keys
    use_preshared_keys: bool,
}

impl WireGuardConfigGenerator {
    /// Create a new config generator
    pub fn new() -> Self {
        Self {
            default_mtu: 1420,
            default_keepalive: 25,
            use_preshared_keys: false,
        }
    }

    /// Set default MTU
    pub fn with_mtu(mut self, mtu: u16) -> Self {
        self.default_mtu = mtu;
        self
    }

    /// Set default keepalive
    pub fn with_keepalive(mut self, keepalive: u16) -> Self {
        self.default_keepalive = keepalive;
        self
    }

    /// Enable preshared keys
    pub fn with_preshared_keys(mut self, enabled: bool) -> Self {
        self.use_preshared_keys = enabled;
        self
    }

    /// Generate configuration for a subnet
    pub fn generate_subnet_config(
        &self,
        subnet: &Subnet,
        assignments: &[SubnetAssignment],
        routes: &[(CrossSubnetRoute, Subnet)],
        node_endpoints: &HashMap<Uuid, SocketAddr>,
    ) -> Result<WireGuardSubnetConfig> {
        // Get gateway IP
        let gateway_ip = subnet
            .gateway_ip()
            .ok_or_else(|| Error::WireGuardConfig("Subnet has no gateway IP".to_string()))?;

        // Create interface config
        let mut interface = InterfaceConfig::for_subnet(subnet, gateway_ip)?;
        interface.mtu = Some(self.default_mtu);

        // Create peer configs for assigned nodes
        let peers: Vec<PeerConfig> = assignments
            .iter()
            .map(|assignment| {
                let endpoint = node_endpoints.get(&assignment.node_id).copied();
                let mut peer = PeerConfig::from_assignment(assignment, endpoint);
                peer.persistent_keepalive = Some(self.default_keepalive);
                peer
            })
            .collect();

        // Create route peer configs
        let route_peers: Vec<PeerConfig> = routes
            .iter()
            .filter_map(|(route, remote_subnet)| {
                PeerConfig::for_subnet_route(remote_subnet, route).ok()
            })
            .collect();

        Ok(WireGuardSubnetConfig {
            subnet_id: subnet.id,
            subnet_name: subnet.name.clone(),
            interface,
            peers,
            route_peers,
        })
    }

    /// Generate client configuration for a node
    pub fn generate_node_config(
        &self,
        node_private_key: &str,
        assignment: &SubnetAssignment,
        subnet: &Subnet,
        subnet_endpoint: Option<SocketAddr>,
    ) -> Result<String> {
        let mut config = String::new();

        // Header
        writeln!(
            config,
            "# WireGuard configuration for node: {}",
            assignment.node_id
        )
        .unwrap();
        writeln!(config, "# Subnet: {} ({})", subnet.name, subnet.id).unwrap();
        writeln!(
            config,
            "# Generated at {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )
        .unwrap();
        writeln!(config).unwrap();

        // Interface
        writeln!(config, "[Interface]").unwrap();
        writeln!(config, "PrivateKey = {}", node_private_key).unwrap();

        let address = Ipv4Net::new(assignment.assigned_ip, subnet.cidr.prefix_len())
            .map_err(|e| Error::WireGuardConfig(format!("Invalid address: {}", e)))?;
        writeln!(config, "Address = {}", address).unwrap();
        writeln!(config, "MTU = {}", self.default_mtu).unwrap();

        // Subnet server as peer
        let subnet_pubkey = subnet
            .wg_public_key
            .as_ref()
            .ok_or_else(|| Error::WireGuardConfig("Subnet missing public key".to_string()))?;

        writeln!(config).unwrap();
        writeln!(config, "# Subnet Gateway").unwrap();
        writeln!(config, "[Peer]").unwrap();
        writeln!(config, "PublicKey = {}", subnet_pubkey).unwrap();
        writeln!(config, "AllowedIPs = {}", subnet.cidr).unwrap();

        if let Some(endpoint) = subnet_endpoint {
            writeln!(config, "Endpoint = {}", endpoint).unwrap();
        }

        writeln!(config, "PersistentKeepalive = {}", self.default_keepalive).unwrap();

        Ok(config)
    }

    /// Generate iptables rules for cross-subnet routing
    pub fn generate_routing_rules(
        &self,
        subnet: &Subnet,
        routes: &[CrossSubnetRoute],
        remote_subnets: &HashMap<Uuid, Subnet>,
    ) -> Vec<String> {
        let mut rules = Vec::new();

        for route in routes {
            let remote_id = if route.source_subnet_id == subnet.id {
                route.destination_subnet_id
            } else {
                route.source_subnet_id
            };

            if let Some(remote) = remote_subnets.get(&remote_id) {
                // Forward rule
                rules.push(format!(
                    "iptables -A FORWARD -i {} -o {} -j ACCEPT",
                    subnet.wg_interface, remote.wg_interface
                ));

                // Port restrictions if any
                if let Some(ref ports) = route.allowed_ports {
                    for port_range in ports {
                        let port_spec = if port_range.start == port_range.end {
                            format!("{}", port_range.start)
                        } else {
                            format!("{}:{}", port_range.start, port_range.end)
                        };

                        // TCP
                        rules.push(format!(
                            "iptables -A FORWARD -i {} -o {} -p tcp --dport {} -j ACCEPT",
                            subnet.wg_interface, remote.wg_interface, port_spec
                        ));
                        // UDP
                        rules.push(format!(
                            "iptables -A FORWARD -i {} -o {} -p udp --dport {} -j ACCEPT",
                            subnet.wg_interface, remote.wg_interface, port_spec
                        ));
                    }
                }
            }
        }

        rules
    }
}

impl Default for WireGuardConfigGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SubnetPurpose, SubnetStatus};
    use crate::wireguard::keys::KeyPair;
    use std::str::FromStr;

    fn create_test_subnet() -> Subnet {
        let kp = KeyPair::generate();
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut subnet = Subnet::new("Test Subnet", cidr, SubnetPurpose::Tenant, 51820);
        subnet.wg_private_key = Some(kp.private_key_base64());
        subnet.wg_public_key = Some(kp.public_key_base64());
        subnet.status = SubnetStatus::Active;
        subnet
    }

    #[test]
    fn test_interface_config() {
        let subnet = create_test_subnet();
        let gateway = subnet.gateway_ip().unwrap();
        let config = InterfaceConfig::for_subnet(&subnet, gateway).unwrap();

        assert_eq!(config.name, subnet.wg_interface);
        assert_eq!(config.listen_port, 51820);

        let config_str = config.to_config_string();
        assert!(config_str.contains("[Interface]"));
        assert!(config_str.contains("PrivateKey ="));
        assert!(config_str.contains("ListenPort = 51820"));
    }

    #[test]
    fn test_peer_config() {
        let assignment = SubnetAssignment {
            id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            subnet_id: Uuid::new_v4(),
            assigned_ip: Ipv4Addr::new(10, 100, 0, 5),
            wg_public_key: KeyPair::generate().public_key_base64(),
            assigned_at: chrono::Utc::now(),
            assignment_method: "manual".to_string(),
            policy_id: None,
            is_migration_temp: false,
        };

        let peer = PeerConfig::from_assignment(&assignment, None);

        assert_eq!(peer.public_key, assignment.wg_public_key);
        assert_eq!(peer.allowed_ips.len(), 1);
        assert_eq!(peer.allowed_ips[0].addr(), assignment.assigned_ip);

        let config_str = peer.to_config_string();
        assert!(config_str.contains("[Peer]"));
        assert!(config_str.contains("PublicKey ="));
        assert!(config_str.contains("AllowedIPs ="));
    }

    #[test]
    fn test_subnet_config_generation() {
        let subnet = create_test_subnet();
        let generator = WireGuardConfigGenerator::new();

        let config = generator
            .generate_subnet_config(&subnet, &[], &[], &HashMap::new())
            .unwrap();

        assert_eq!(config.subnet_id, subnet.id);
        assert!(config.peers.is_empty());

        let config_str = config.to_config_string();
        assert!(config_str.contains("# WireGuard configuration"));
        assert!(config_str.contains("[Interface]"));
    }

    #[test]
    fn test_node_config_generation() {
        let subnet = create_test_subnet();
        let node_kp = KeyPair::generate();
        let assignment = SubnetAssignment {
            id: Uuid::new_v4(),
            node_id: Uuid::new_v4(),
            subnet_id: subnet.id,
            assigned_ip: Ipv4Addr::new(10, 100, 0, 5),
            wg_public_key: node_kp.public_key_base64(),
            assigned_at: chrono::Utc::now(),
            assignment_method: "manual".to_string(),
            policy_id: None,
            is_migration_temp: false,
        };

        let generator = WireGuardConfigGenerator::new();
        let config = generator
            .generate_node_config(
                &node_kp.private_key_base64(),
                &assignment,
                &subnet,
                Some("1.2.3.4:51820".parse().unwrap()),
            )
            .unwrap();

        assert!(config.contains("[Interface]"));
        assert!(config.contains("[Peer]"));
        assert!(config.contains("Endpoint = 1.2.3.4:51820"));
    }

    #[test]
    fn test_config_with_peers() {
        let subnet = create_test_subnet();
        let assignments: Vec<_> = (2..5)
            .map(|i| SubnetAssignment {
                id: Uuid::new_v4(),
                node_id: Uuid::new_v4(),
                subnet_id: subnet.id,
                assigned_ip: Ipv4Addr::new(10, 100, 0, i),
                wg_public_key: KeyPair::generate().public_key_base64(),
                assigned_at: chrono::Utc::now(),
                assignment_method: "manual".to_string(),
                policy_id: None,
                is_migration_temp: false,
            })
            .collect();

        let generator = WireGuardConfigGenerator::new();
        let config = generator
            .generate_subnet_config(&subnet, &assignments, &[], &HashMap::new())
            .unwrap();

        assert_eq!(config.peers.len(), 3);

        let config_str = config.to_config_string();
        assert!(config_str.contains("# Node Peers (3 nodes)"));
        assert!(config_str.matches("[Peer]").count() >= 3);
    }
}
