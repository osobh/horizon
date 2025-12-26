//! Subnet template models for standardized subnet creation

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{NodeType, PolicyRule, SubnetPurpose};

/// Template for creating subnets with predefined settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetTemplate {
    /// Unique template identifier
    pub id: Uuid,
    /// Template name (e.g., "tenant-isolation-standard")
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Purpose for subnets created from this template
    pub purpose: SubnetPurpose,
    /// Whether this is a system-provided template
    pub is_system: bool,
    /// CIDR prefix length for new subnets
    pub prefix_length: u8,
    /// Default settings for new subnets
    pub defaults: TemplateDefaults,
    /// Default policy rules to create with new subnets
    pub default_policies: Vec<PolicyRule>,
    /// Default cross-subnet routes to create
    pub default_routes: Vec<TemplateRoute>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Created by user/system
    pub created_by: Option<Uuid>,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

/// Default values for new subnets created from a template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateDefaults {
    /// Default WireGuard listen port (or port range start)
    pub wg_listen_port: u16,
    /// Auto-assign sequential ports
    pub wg_auto_port: bool,
    /// Maximum nodes (None = unlimited)
    pub max_nodes: Option<i32>,
    /// Default node type for NodeType purpose
    pub node_type: Option<NodeType>,
    /// Default persistent keepalive
    pub persistent_keepalive: Option<u16>,
    /// Enable automatic cleanup of expired assignments
    pub auto_cleanup: bool,
    /// Default DNS servers to push to clients
    pub dns_servers: Option<Vec<String>>,
    /// MTU override
    pub mtu: Option<u16>,
}

impl Default for TemplateDefaults {
    fn default() -> Self {
        Self {
            wg_listen_port: 51820,
            wg_auto_port: true,
            max_nodes: None,
            node_type: None,
            persistent_keepalive: Some(25),
            auto_cleanup: true,
            dns_servers: None,
            mtu: None,
        }
    }
}

/// Route template for auto-creating routes with new subnets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateRoute {
    /// Target template name or subnet ID
    pub target: RouteTarget,
    /// Create bidirectional route
    pub bidirectional: bool,
    /// Allowed ports
    pub allowed_ports: Option<Vec<super::PortRange>>,
    /// Allowed protocols
    pub allowed_protocols: Option<Vec<String>>,
}

/// Target for template routes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RouteTarget {
    /// Route to a specific subnet ID
    SubnetId(Uuid),
    /// Route to all subnets created from a template
    TemplateName(String),
    /// Route to all subnets of a purpose
    Purpose(SubnetPurpose),
}

impl SubnetTemplate {
    /// Create a new template
    pub fn new(
        name: impl Into<String>,
        purpose: SubnetPurpose,
        prefix_length: u8,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            purpose,
            is_system: false,
            prefix_length,
            defaults: TemplateDefaults::default(),
            default_policies: Vec::new(),
            default_routes: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
            metadata: None,
        }
    }

    /// Create system templates for each purpose
    pub fn system_templates() -> Vec<Self> {
        vec![
            // Tenant isolation templates
            Self::tenant_isolation_standard(),
            // Node type templates
            Self::nodetype_datacenter(),
            Self::nodetype_workstation(),
            Self::nodetype_laptop(),
            Self::nodetype_edge(),
            // Geographic templates
            Self::geographic_region(),
            // Resource pool templates
            Self::resource_pool_hackathon(),
            Self::resource_pool_research(),
            Self::resource_pool_training(),
        ]
    }

    /// Standard tenant isolation template
    pub fn tenant_isolation_standard() -> Self {
        let mut template = Self::new(
            "tenant-isolation-standard",
            SubnetPurpose::Tenant,
            20, // /20 = 4094 hosts
        );
        template.is_system = true;
        template.description = Some("Standard tenant isolation subnet".to_string());
        template
    }

    /// DataCenter nodes template
    pub fn nodetype_datacenter() -> Self {
        let mut template = Self::new(
            "nodetype-datacenter",
            SubnetPurpose::NodeType,
            16, // /16 = 65K hosts
        );
        template.is_system = true;
        template.description = Some("Subnet for datacenter nodes (high reliability)".to_string());
        template.defaults.node_type = Some(NodeType::DataCenter);
        template.default_policies.push(PolicyRule::node_type_equals(NodeType::DataCenter));
        template
    }

    /// Workstation nodes template
    pub fn nodetype_workstation() -> Self {
        let mut template = Self::new(
            "nodetype-workstation",
            SubnetPurpose::NodeType,
            16,
        );
        template.is_system = true;
        template.description = Some("Subnet for workstation nodes".to_string());
        template.defaults.node_type = Some(NodeType::Workstation);
        template.default_policies.push(PolicyRule::node_type_equals(NodeType::Workstation));
        template
    }

    /// Laptop nodes template
    pub fn nodetype_laptop() -> Self {
        let mut template = Self::new(
            "nodetype-laptop",
            SubnetPurpose::NodeType,
            16,
        );
        template.is_system = true;
        template.description = Some("Subnet for laptop nodes (variable connectivity)".to_string());
        template.defaults.node_type = Some(NodeType::Laptop);
        template.defaults.persistent_keepalive = Some(15); // More aggressive keepalive
        template.default_policies.push(PolicyRule::node_type_equals(NodeType::Laptop));
        template
    }

    /// Edge devices template
    pub fn nodetype_edge() -> Self {
        let mut template = Self::new(
            "nodetype-edge",
            SubnetPurpose::NodeType,
            16,
        );
        template.is_system = true;
        template.description = Some("Subnet for edge devices (IoT, embedded)".to_string());
        template.defaults.node_type = Some(NodeType::Edge);
        template.defaults.persistent_keepalive = Some(10); // Very aggressive keepalive
        template.defaults.mtu = Some(1280); // Lower MTU for constrained networks
        template.default_policies.push(PolicyRule::node_type_equals(NodeType::Edge));
        template
    }

    /// Geographic region template
    pub fn geographic_region() -> Self {
        let mut template = Self::new(
            "geographic-region",
            SubnetPurpose::Geographic,
            18, // /18 = 16K hosts
        );
        template.is_system = true;
        template.description = Some("Regional subnet for geographic isolation".to_string());
        template
    }

    /// Hackathon resource pool template
    pub fn resource_pool_hackathon() -> Self {
        let mut template = Self::new(
            "resource-pool-hackathon",
            SubnetPurpose::ResourcePool,
            22, // /22 = 1022 hosts (smaller, time-limited)
        );
        template.is_system = true;
        template.description = Some("Subnet for hackathon resource pools".to_string());
        template.defaults.max_nodes = Some(100);
        template
    }

    /// Research resource pool template
    pub fn resource_pool_research() -> Self {
        let mut template = Self::new(
            "resource-pool-research",
            SubnetPurpose::ResourcePool,
            20, // /20 = 4094 hosts
        );
        template.is_system = true;
        template.description = Some("Subnet for research resource pools".to_string());
        template
    }

    /// Training resource pool template
    pub fn resource_pool_training() -> Self {
        let mut template = Self::new(
            "resource-pool-training",
            SubnetPurpose::ResourcePool,
            20,
        );
        template.is_system = true;
        template.description = Some("Subnet for ML training resource pools".to_string());
        template
    }
}

/// Request to create a subnet from a template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateFromTemplateRequest {
    /// Template ID or name
    pub template: TemplateRef,
    /// Override name
    pub name: Option<String>,
    /// Override description
    pub description: Option<String>,
    /// Tenant ID (for tenant subnets)
    pub tenant_id: Option<Uuid>,
    /// Resource pool ID (for pool subnets)
    pub resource_pool_id: Option<Uuid>,
    /// Region (for geographic subnets)
    pub region: Option<super::Region>,
    /// Override max nodes
    pub max_nodes: Option<i32>,
    /// Custom metadata
    pub metadata: Option<serde_json::Value>,
}

/// Reference to a template by ID or name
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateRef {
    Id(Uuid),
    Name(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_templates() {
        let templates = SubnetTemplate::system_templates();
        assert!(templates.len() >= 9);

        // Verify all are system templates
        for template in &templates {
            assert!(template.is_system);
        }
    }

    #[test]
    fn test_datacenter_template() {
        let template = SubnetTemplate::nodetype_datacenter();

        assert_eq!(template.name, "nodetype-datacenter");
        assert_eq!(template.purpose, SubnetPurpose::NodeType);
        assert_eq!(template.prefix_length, 16);
        assert_eq!(template.defaults.node_type, Some(NodeType::DataCenter));
        assert!(!template.default_policies.is_empty());
    }

    #[test]
    fn test_edge_template_has_lower_mtu() {
        let template = SubnetTemplate::nodetype_edge();

        assert_eq!(template.defaults.mtu, Some(1280));
        assert_eq!(template.defaults.persistent_keepalive, Some(10));
    }
}
