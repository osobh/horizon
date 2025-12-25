//! Subnet and related models

use chrono::{DateTime, Utc};
use ipnet::Ipv4Net;
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use uuid::Uuid;

/// Purpose/dimension of a subnet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubnetPurpose {
    /// Tenant isolation - each organization gets an isolated subnet
    Tenant,
    /// Node type segregation - separate subnets for different node types
    NodeType,
    /// Geographic/regional - subnets per datacenter/region/AZ
    Geographic,
    /// Resource pool isolation - each pool gets isolated network
    ResourcePool,
}

impl SubnetPurpose {
    /// Get the address space for this purpose
    pub fn address_space(&self) -> Ipv4Net {
        match self {
            SubnetPurpose::Tenant => crate::address_space::TENANT_SPACE,
            SubnetPurpose::NodeType => crate::address_space::NODE_TYPE_SPACE,
            SubnetPurpose::Geographic => crate::address_space::GEOGRAPHIC_SPACE,
            SubnetPurpose::ResourcePool => crate::address_space::RESOURCE_POOL_SPACE,
        }
    }

    /// Get the default prefix length for new subnets of this purpose
    pub fn default_prefix(&self) -> u8 {
        match self {
            SubnetPurpose::Tenant => crate::address_space::TENANT_SUBNET_PREFIX,
            SubnetPurpose::NodeType => 16, // /16 for node types
            SubnetPurpose::Geographic => crate::address_space::GEOGRAPHIC_SUBNET_PREFIX,
            SubnetPurpose::ResourcePool => crate::address_space::RESOURCE_POOL_SUBNET_PREFIX,
        }
    }
}

/// Status of a subnet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubnetStatus {
    /// Subnet is being provisioned
    Provisioning,
    /// Subnet is active and accepting nodes
    Active,
    /// Subnet is draining (no new nodes, existing nodes migrating out)
    Draining,
    /// Subnet is archived (no nodes, preserved for audit)
    Archived,
}

/// Node type for node type segregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    /// High-performance datacenter nodes (always-on, high reliability)
    DataCenter,
    /// Desktop workstations (regular hours, medium reliability)
    Workstation,
    /// Laptop computers (variable connectivity, lower reliability)
    Laptop,
    /// Edge devices (IoT, embedded, intermittent connectivity)
    Edge,
}

impl NodeType {
    /// Get the dedicated address space for this node type
    pub fn address_space(&self) -> Ipv4Net {
        match self {
            NodeType::DataCenter => crate::address_space::DATACENTER_SPACE,
            NodeType::Workstation => crate::address_space::WORKSTATION_SPACE,
            NodeType::Laptop => crate::address_space::LAPTOP_SPACE,
            NodeType::Edge => crate::address_space::EDGE_SPACE,
        }
    }
}

/// Geographic region for regional subnets
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Region {
    /// Region code (e.g., "us-east-1", "eu-west-2")
    pub code: String,
    /// Human-readable name
    pub name: String,
    /// Parent region (for hierarchical regions)
    pub parent: Option<String>,
    /// Availability zone within region
    pub availability_zone: Option<String>,
}

/// A WireGuard subnet for node isolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subnet {
    /// Unique subnet identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// CIDR block (e.g., "10.100.0.0/20")
    pub cidr: Ipv4Net,
    /// Purpose/dimension of this subnet
    pub purpose: SubnetPurpose,
    /// Current status
    pub status: SubnetStatus,

    // Dimension-specific fields (one populated based on purpose)
    /// Tenant ID (for Tenant purpose)
    pub tenant_id: Option<Uuid>,
    /// Node type (for NodeType purpose)
    pub node_type: Option<NodeType>,
    /// Region (for Geographic purpose)
    pub region: Option<Region>,
    /// Resource pool ID (for ResourcePool purpose)
    pub resource_pool_id: Option<Uuid>,

    // WireGuard configuration
    /// WireGuard interface name (e.g., "wg-tenant-acme")
    pub wg_interface: String,
    /// WireGuard listen port
    pub wg_listen_port: u16,
    /// Server public key
    pub wg_public_key: Option<String>,
    /// Server private key (encrypted at rest)
    #[serde(skip_serializing)]
    pub wg_private_key: Option<String>,

    // Capacity
    /// Maximum nodes allowed (None = unlimited within CIDR)
    pub max_nodes: Option<i32>,
    /// Current node count
    pub current_nodes: i32,

    // Template
    /// Template this subnet was created from
    pub template_id: Option<Uuid>,

    // Metadata
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Created by user/system
    pub created_by: Option<Uuid>,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

impl Subnet {
    /// Create a new subnet with the given parameters
    pub fn new(
        name: impl Into<String>,
        cidr: Ipv4Net,
        purpose: SubnetPurpose,
        wg_listen_port: u16,
    ) -> Self {
        let name = name.into();
        let wg_interface = format!("wg-{}", name.to_lowercase().replace(' ', "-"));

        Self {
            id: Uuid::new_v4(),
            name,
            description: None,
            cidr,
            purpose,
            status: SubnetStatus::Provisioning,
            tenant_id: None,
            node_type: None,
            region: None,
            resource_pool_id: None,
            wg_interface,
            wg_listen_port,
            wg_public_key: None,
            wg_private_key: None,
            max_nodes: None,
            current_nodes: 0,
            template_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
            metadata: None,
        }
    }

    /// Check if the subnet can accept more nodes
    pub fn can_accept_nodes(&self) -> bool {
        if self.status != SubnetStatus::Active {
            return false;
        }

        if let Some(max) = self.max_nodes {
            self.current_nodes < max
        } else {
            // Check CIDR capacity (subtract network and broadcast)
            let capacity = self.cidr.hosts().count() as i32;
            self.current_nodes < capacity
        }
    }

    /// Get the number of available IPs
    pub fn available_ips(&self) -> usize {
        let total = self.cidr.hosts().count();
        let used = self.current_nodes as usize;
        total.saturating_sub(used)
    }

    /// Check if an IP is within this subnet
    pub fn contains_ip(&self, ip: Ipv4Addr) -> bool {
        self.cidr.contains(&ip)
    }

    /// Get the gateway IP (first usable host)
    pub fn gateway_ip(&self) -> Option<Ipv4Addr> {
        self.cidr.hosts().next()
    }

    /// Get the broadcast address
    pub fn broadcast(&self) -> Ipv4Addr {
        self.cidr.broadcast()
    }
}

/// A node's assignment to a subnet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetAssignment {
    /// Assignment ID
    pub id: Uuid,
    /// Node ID
    pub node_id: Uuid,
    /// Subnet ID
    pub subnet_id: Uuid,
    /// Assigned IP within the subnet
    pub assigned_ip: Ipv4Addr,
    /// WireGuard public key of the node
    pub wg_public_key: String,
    /// Assigned at timestamp
    pub assigned_at: DateTime<Utc>,
    /// Assignment method (policy name or "manual")
    pub assignment_method: String,
    /// Policy ID that triggered this assignment
    pub policy_id: Option<Uuid>,
    /// Is this a temporary dual-stack assignment during migration
    pub is_migration_temp: bool,
}

/// WireGuard peer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireGuardPeer {
    /// Peer's public key
    pub public_key: String,
    /// Allowed IPs for this peer
    pub allowed_ips: Vec<Ipv4Net>,
    /// Peer endpoint (if known)
    pub endpoint: Option<String>,
    /// Persistent keepalive interval
    pub persistent_keepalive: Option<u16>,
    /// Preshared key (for additional security)
    #[serde(skip_serializing)]
    pub preshared_key: Option<String>,
}

/// Statistics for a subnet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetStats {
    /// Subnet ID
    pub subnet_id: Uuid,
    /// Total capacity (hosts in CIDR)
    pub total_capacity: usize,
    /// Currently allocated IPs
    pub allocated_ips: usize,
    /// Available IPs
    pub available_ips: usize,
    /// Utilization percentage
    pub utilization_percent: f64,
    /// Active connections
    pub active_connections: usize,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_subnet_creation() {
        let cidr = Ipv4Net::from_str("10.100.0.0/20").unwrap();
        let subnet = Subnet::new("Test Tenant", cidr, SubnetPurpose::Tenant, 51820);

        assert_eq!(subnet.name, "Test Tenant");
        assert_eq!(subnet.wg_interface, "wg-test-tenant");
        assert_eq!(subnet.status, SubnetStatus::Provisioning);
        assert!(subnet.can_accept_nodes() == false); // Not active yet
    }

    #[test]
    fn test_subnet_capacity() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let mut subnet = Subnet::new("Small", cidr, SubnetPurpose::Tenant, 51820);
        subnet.status = SubnetStatus::Active;

        // /24 = 254 usable hosts
        assert_eq!(subnet.available_ips(), 254);
        assert!(subnet.can_accept_nodes());

        // With max_nodes limit
        subnet.max_nodes = Some(10);
        subnet.current_nodes = 10;
        assert!(!subnet.can_accept_nodes());
    }

    #[test]
    fn test_contains_ip() {
        let cidr = Ipv4Net::from_str("10.100.0.0/24").unwrap();
        let subnet = Subnet::new("Test", cidr, SubnetPurpose::Tenant, 51820);

        assert!(subnet.contains_ip(Ipv4Addr::new(10, 100, 0, 50)));
        assert!(!subnet.contains_ip(Ipv4Addr::new(10, 101, 0, 50)));
    }

    #[test]
    fn test_node_type_spaces() {
        assert!(NodeType::DataCenter
            .address_space()
            .contains(&Ipv4Addr::new(10, 112, 0, 1)));
        assert!(NodeType::Workstation
            .address_space()
            .contains(&Ipv4Addr::new(10, 113, 0, 1)));
        assert!(NodeType::Laptop
            .address_space()
            .contains(&Ipv4Addr::new(10, 114, 0, 1)));
        assert!(NodeType::Edge
            .address_space()
            .contains(&Ipv4Addr::new(10, 115, 0, 1)));
    }
}
