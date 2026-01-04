//! Node class to subnet attribute mapping
//!
//! Maps cluster-mesh NodeClass and hardware attributes to
//! subnet-manager NodeAttributes for policy evaluation.

use crate::models::NodeType;
use crate::policy_engine::NodeAttributes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents the class of a node (mirrors cluster-mesh::NodeClass)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeClass {
    /// High-performance server in data center
    DataCenter,
    /// Desktop workstation
    Workstation,
    /// Laptop/mobile device
    Laptop,
    /// Edge device (IoT, embedded)
    Edge,
}

impl NodeClass {
    /// Convert to subnet-manager NodeType
    pub fn to_node_type(&self) -> NodeType {
        match self {
            NodeClass::DataCenter => NodeType::DataCenter,
            NodeClass::Workstation => NodeType::Workstation,
            NodeClass::Laptop => NodeType::Laptop,
            NodeClass::Edge => NodeType::Edge,
        }
    }
}

impl From<NodeClass> for NodeType {
    fn from(class: NodeClass) -> Self {
        class.to_node_type()
    }
}

/// Hardware information from a cluster node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_cores: Option<u32>,
    pub ram_gb: Option<u32>,
    pub gpu_count: Option<u32>,
    pub gpu_memory_gb: Option<u32>,
    pub gpu_model: Option<String>,
    pub storage_gb: Option<u64>,
    pub network_bandwidth_mbps: Option<u32>,
}

/// Network information from a cluster node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub hostname: Option<String>,
    pub public_ip: Option<std::net::Ipv4Addr>,
    pub region: Option<String>,
    pub zone: Option<String>,
    pub datacenter: Option<String>,
}

/// Represents a node from cluster-mesh for mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNodeInfo {
    pub id: Uuid,
    pub hostname: String,
    pub class: NodeClass,
    pub hardware: HardwareInfo,
    pub network: NetworkInfo,
    pub labels: HashMap<String, String>,
    pub tenant_id: Option<Uuid>,
    pub wg_public_key: Option<String>,
}

impl ClusterNodeInfo {
    /// Create a new cluster node info
    pub fn new(id: Uuid, hostname: &str, class: NodeClass) -> Self {
        Self {
            id,
            hostname: hostname.to_string(),
            class,
            hardware: HardwareInfo::default(),
            network: NetworkInfo::default(),
            labels: HashMap::new(),
            tenant_id: None,
            wg_public_key: None,
        }
    }

    /// Set hardware info
    pub fn with_hardware(mut self, hardware: HardwareInfo) -> Self {
        self.hardware = hardware;
        self
    }

    /// Set network info
    pub fn with_network(mut self, network: NetworkInfo) -> Self {
        self.network = network;
        self
    }

    /// Set labels
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Set tenant ID
    pub fn with_tenant(mut self, tenant_id: Uuid) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Set WireGuard public key
    pub fn with_wg_key(mut self, key: String) -> Self {
        self.wg_public_key = Some(key);
        self
    }
}

/// Maps cluster node information to subnet policy attributes
#[derive(Debug, Clone, Default)]
pub struct NodeClassMapper {
    /// Custom label mappings (cluster label key → subnet label format)
    label_mappings: HashMap<String, String>,
    /// Region normalization mappings
    region_mappings: HashMap<String, String>,
}

impl NodeClassMapper {
    /// Create a new mapper with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom label mapping
    pub fn with_label_mapping(mut self, from: &str, to: &str) -> Self {
        self.label_mappings.insert(from.to_string(), to.to_string());
        self
    }

    /// Add region normalization
    pub fn with_region_mapping(mut self, from: &str, to: &str) -> Self {
        self.region_mappings
            .insert(from.to_string(), to.to_string());
        self
    }

    /// Map a cluster node to subnet NodeAttributes
    pub fn map_to_attributes(&self, node: &ClusterNodeInfo) -> NodeAttributes {
        let mut attrs = NodeAttributes::new().with_node_type(node.class.to_node_type());

        // Map tenant
        if let Some(tenant_id) = node.tenant_id {
            attrs = attrs.with_tenant(tenant_id);
        }

        // Map region with normalization
        if let Some(region) = &node.network.region {
            let normalized = self
                .region_mappings
                .get(region)
                .cloned()
                .unwrap_or_else(|| region.clone());
            attrs = attrs.with_region(&normalized);
        }

        // Map hostname
        if !node.hostname.is_empty() {
            attrs.hostname = Some(node.hostname.clone());
        }

        // Map hardware: GPU
        if let Some(gpu_count) = node.hardware.gpu_count {
            let gpu_memory = node.hardware.gpu_memory_gb.unwrap_or(0);
            let gpu_model = node.hardware.gpu_model.clone().unwrap_or_default();
            attrs = attrs.with_gpu(gpu_count as i32, gpu_memory as i32, &gpu_model);
        }

        // Map hardware: CPU/RAM
        attrs.cpu_cores = node.hardware.cpu_cores.map(|c| c as i32);
        attrs.ram_gb = node.hardware.ram_gb.map(|r| r as i32);

        // Map labels with custom mappings
        let mut labels = Vec::new();
        for (key, value) in &node.labels {
            let label = if let Some(mapped_key) = self.label_mappings.get(key) {
                format!("{}={}", mapped_key, value)
            } else {
                format!("{}={}", key, value)
            };
            labels.push(label);
        }

        // Add zone/datacenter as labels if present
        if let Some(zone) = &node.network.zone {
            labels.push(format!("zone={}", zone));
        }
        if let Some(dc) = &node.network.datacenter {
            labels.push(format!("datacenter={}", dc));
        }

        if !labels.is_empty() {
            attrs = attrs.with_labels(labels);
        }

        attrs
    }

    /// Quick mapping for just node type
    pub fn map_class(class: NodeClass) -> NodeType {
        class.to_node_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_class_to_type() {
        assert_eq!(NodeClass::DataCenter.to_node_type(), NodeType::DataCenter);
        assert_eq!(NodeClass::Workstation.to_node_type(), NodeType::Workstation);
        assert_eq!(NodeClass::Laptop.to_node_type(), NodeType::Laptop);
        assert_eq!(NodeClass::Edge.to_node_type(), NodeType::Edge);
    }

    #[test]
    fn test_basic_mapping() {
        let mapper = NodeClassMapper::new();

        let node = ClusterNodeInfo::new(Uuid::new_v4(), "gpu-server-01", NodeClass::DataCenter)
            .with_tenant(Uuid::new_v4())
            .with_hardware(HardwareInfo {
                cpu_cores: Some(128),
                ram_gb: Some(512),
                gpu_count: Some(8),
                gpu_memory_gb: Some(80),
                gpu_model: Some("A100".to_string()),
                ..Default::default()
            })
            .with_network(NetworkInfo {
                region: Some("us-east-1".to_string()),
                zone: Some("us-east-1a".to_string()),
                ..Default::default()
            });

        let attrs = mapper.map_to_attributes(&node);

        assert_eq!(attrs.node_type, Some(NodeType::DataCenter));
        assert_eq!(attrs.region, Some("us-east-1".to_string()));
        assert_eq!(attrs.gpu_count, Some(8));
        assert_eq!(attrs.gpu_memory_gb, Some(80));
        assert_eq!(attrs.cpu_cores, Some(128));
        assert_eq!(attrs.ram_gb, Some(512));
        assert!(attrs.tenant_id.is_some());
    }

    #[test]
    fn test_region_normalization() {
        let mapper = NodeClassMapper::new()
            .with_region_mapping("east-us", "us-east-1")
            .with_region_mapping("west-us", "us-west-2");

        let node = ClusterNodeInfo::new(Uuid::new_v4(), "server", NodeClass::DataCenter)
            .with_network(NetworkInfo {
                region: Some("east-us".to_string()),
                ..Default::default()
            });

        let attrs = mapper.map_to_attributes(&node);
        assert_eq!(attrs.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_label_mapping() {
        let mapper = NodeClassMapper::new()
            .with_label_mapping("env", "environment")
            .with_label_mapping("team", "team");

        let mut labels = HashMap::new();
        labels.insert("env".to_string(), "production".to_string());
        labels.insert("team".to_string(), "platform".to_string());

        let node = ClusterNodeInfo::new(Uuid::new_v4(), "server", NodeClass::DataCenter)
            .with_labels(labels);

        let attrs = mapper.map_to_attributes(&node);
        assert!(attrs.labels.contains(&"environment=production".to_string()));
        assert!(attrs.labels.contains(&"team=platform".to_string()));
    }
}
