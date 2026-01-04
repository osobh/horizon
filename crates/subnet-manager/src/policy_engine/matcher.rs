//! Attribute matching logic for policy rules

use crate::models::{MatchOperator, NodeAttribute, NodeType, PolicyRule, PolicyValue};
use regex::Regex;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use uuid::Uuid;

/// Node attributes for policy matching
#[derive(Debug, Clone)]
pub struct NodeAttributes {
    /// Node type
    pub node_type: Option<NodeType>,
    /// Tenant ID
    pub tenant_id: Option<Uuid>,
    /// Region code
    pub region: Option<String>,
    /// Availability zone
    pub availability_zone: Option<String>,
    /// Resource pool ID
    pub resource_pool_id: Option<Uuid>,
    /// Resource pool type
    pub resource_pool_type: Option<String>,
    /// CPU cores
    pub cpu_cores: Option<i32>,
    /// RAM in GB
    pub ram_gb: Option<i32>,
    /// GPU count
    pub gpu_count: Option<i32>,
    /// GPU memory in GB
    pub gpu_memory_gb: Option<i32>,
    /// GPU model
    pub gpu_model: Option<String>,
    /// Storage in TB
    pub storage_tb: Option<f64>,
    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: Option<f64>,
    /// Device reliability tier (0-3)
    pub reliability_tier: Option<i32>,
    /// Labels/tags
    pub labels: Vec<String>,
    /// Hostname
    pub hostname: Option<String>,
    /// Operating system
    pub operating_system: Option<String>,
    /// Kubernetes namespace
    pub namespace: Option<String>,
    /// Custom attributes
    pub custom: HashMap<u32, JsonValue>,
}

impl Default for NodeAttributes {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeAttributes {
    /// Create empty node attributes
    pub fn new() -> Self {
        Self {
            node_type: None,
            tenant_id: None,
            region: None,
            availability_zone: None,
            resource_pool_id: None,
            resource_pool_type: None,
            cpu_cores: None,
            ram_gb: None,
            gpu_count: None,
            gpu_memory_gb: None,
            gpu_model: None,
            storage_tb: None,
            network_bandwidth_gbps: None,
            reliability_tier: None,
            labels: Vec::new(),
            hostname: None,
            operating_system: None,
            namespace: None,
            custom: HashMap::new(),
        }
    }

    /// Builder: set node type
    pub fn with_node_type(mut self, node_type: NodeType) -> Self {
        self.node_type = Some(node_type);
        self
    }

    /// Builder: set tenant ID
    pub fn with_tenant(mut self, tenant_id: Uuid) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Builder: set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Builder: set resource pool
    pub fn with_resource_pool(mut self, pool_id: Uuid, pool_type: impl Into<String>) -> Self {
        self.resource_pool_id = Some(pool_id);
        self.resource_pool_type = Some(pool_type.into());
        self
    }

    /// Builder: set GPU specs
    pub fn with_gpu(mut self, count: i32, memory_gb: i32, model: impl Into<String>) -> Self {
        self.gpu_count = Some(count);
        self.gpu_memory_gb = Some(memory_gb);
        self.gpu_model = Some(model.into());
        self
    }

    /// Builder: set labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Builder: add a label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Builder: set hostname
    pub fn with_hostname(mut self, hostname: impl Into<String>) -> Self {
        self.hostname = Some(hostname.into());
        self
    }

    /// Get attribute value for matching
    fn get_value(&self, attribute: &NodeAttribute) -> Option<AttributeValue> {
        match attribute {
            NodeAttribute::NodeType => self.node_type.map(AttributeValue::NodeType),
            NodeAttribute::TenantId => self.tenant_id.map(AttributeValue::Uuid),
            NodeAttribute::Region => self.region.clone().map(AttributeValue::String),
            NodeAttribute::AvailabilityZone => {
                self.availability_zone.clone().map(AttributeValue::String)
            }
            NodeAttribute::ResourcePoolId => self.resource_pool_id.map(AttributeValue::Uuid),
            NodeAttribute::ResourcePoolType => {
                self.resource_pool_type.clone().map(AttributeValue::String)
            }
            NodeAttribute::CpuCores => self.cpu_cores.map(|v| AttributeValue::Integer(v as i64)),
            NodeAttribute::RamGb => self.ram_gb.map(|v| AttributeValue::Integer(v as i64)),
            NodeAttribute::GpuCount => self.gpu_count.map(|v| AttributeValue::Integer(v as i64)),
            NodeAttribute::GpuMemoryGb => self
                .gpu_memory_gb
                .map(|v| AttributeValue::Integer(v as i64)),
            NodeAttribute::GpuModel => self.gpu_model.clone().map(AttributeValue::String),
            NodeAttribute::StorageTb => self.storage_tb.map(AttributeValue::Float),
            NodeAttribute::NetworkBandwidthGbps => {
                self.network_bandwidth_gbps.map(AttributeValue::Float)
            }
            NodeAttribute::ReliabilityTier => self
                .reliability_tier
                .map(|v| AttributeValue::Integer(v as i64)),
            NodeAttribute::Labels => Some(AttributeValue::StringList(self.labels.clone())),
            NodeAttribute::Hostname => self.hostname.clone().map(AttributeValue::String),
            NodeAttribute::OperatingSystem => {
                self.operating_system.clone().map(AttributeValue::String)
            }
            NodeAttribute::Namespace => self.namespace.clone().map(AttributeValue::String),
            NodeAttribute::Custom(id) => self.custom.get(id).cloned().map(AttributeValue::Json),
        }
    }
}

/// Internal representation of attribute values for matching
#[derive(Debug, Clone)]
enum AttributeValue {
    String(String),
    Integer(i64),
    Float(f64),
    NodeType(NodeType),
    Uuid(Uuid),
    StringList(Vec<String>),
    Json(JsonValue),
}

/// Attribute matcher for policy evaluation
pub struct AttributeMatcher;

impl AttributeMatcher {
    /// Check if a node's attributes match a policy rule
    pub fn matches(node: &NodeAttributes, rule: &PolicyRule) -> bool {
        let node_value = match node.get_value(&rule.attribute) {
            Some(v) => v,
            None => {
                // Attribute doesn't exist on node
                return matches!(rule.operator, MatchOperator::NotExists);
            }
        };

        match rule.operator {
            MatchOperator::Exists => true,
            MatchOperator::NotExists => false,
            _ => Self::evaluate_operator(&node_value, &rule.operator, &rule.value),
        }
    }

    /// Evaluate an operator against attribute and policy values
    fn evaluate_operator(
        node_value: &AttributeValue,
        operator: &MatchOperator,
        policy_value: &PolicyValue,
    ) -> bool {
        match operator {
            MatchOperator::Equals => Self::equals(node_value, policy_value),
            MatchOperator::NotEquals => !Self::equals(node_value, policy_value),
            MatchOperator::In => Self::in_list(node_value, policy_value),
            MatchOperator::NotIn => !Self::in_list(node_value, policy_value),
            MatchOperator::GreaterThan => Self::compare(node_value, policy_value, |a, b| a > b),
            MatchOperator::GreaterThanOrEqual => {
                Self::compare(node_value, policy_value, |a, b| a >= b)
            }
            MatchOperator::LessThan => Self::compare(node_value, policy_value, |a, b| a < b),
            MatchOperator::LessThanOrEqual => {
                Self::compare(node_value, policy_value, |a, b| a <= b)
            }
            MatchOperator::Contains => Self::contains(node_value, policy_value),
            MatchOperator::StartsWith => Self::starts_with(node_value, policy_value),
            MatchOperator::EndsWith => Self::ends_with(node_value, policy_value),
            MatchOperator::Regex => Self::regex_match(node_value, policy_value),
            MatchOperator::Exists | MatchOperator::NotExists => unreachable!(),
        }
    }

    /// Check equality between node value and policy value
    fn equals(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::String(b)) => a == b,
            (AttributeValue::Integer(a), PolicyValue::Integer(b)) => a == b,
            (AttributeValue::Float(a), PolicyValue::Float(b)) => (a - b).abs() < f64::EPSILON,
            (AttributeValue::NodeType(a), PolicyValue::NodeType(b)) => a == b,
            (AttributeValue::Uuid(a), PolicyValue::Uuid(b)) => a == b,
            _ => false,
        }
    }

    /// Check if node value is in policy value list
    fn in_list(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::StringList(list)) => list.contains(a),
            (AttributeValue::Uuid(a), PolicyValue::UuidList(list)) => list.contains(a),
            _ => false,
        }
    }

    /// Numeric comparison
    fn compare<F>(node_value: &AttributeValue, policy_value: &PolicyValue, cmp: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        let node_num = match node_value {
            AttributeValue::Integer(v) => *v as f64,
            AttributeValue::Float(v) => *v,
            _ => return false,
        };

        let policy_num = match policy_value {
            PolicyValue::Integer(v) => *v as f64,
            PolicyValue::Float(v) => *v,
            _ => return false,
        };

        cmp(node_num, policy_num)
    }

    /// String contains check
    fn contains(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::String(b)) => a.contains(b),
            (AttributeValue::StringList(list), PolicyValue::String(b)) => list.contains(b),
            _ => false,
        }
    }

    /// String starts with check
    fn starts_with(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::String(b)) => a.starts_with(b),
            _ => false,
        }
    }

    /// String ends with check
    fn ends_with(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::String(b)) => a.ends_with(b),
            _ => false,
        }
    }

    /// Regex match
    fn regex_match(node_value: &AttributeValue, policy_value: &PolicyValue) -> bool {
        match (node_value, policy_value) {
            (AttributeValue::String(a), PolicyValue::String(pattern)) => Regex::new(pattern)
                .map(|re| re.is_match(a))
                .unwrap_or(false),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_type_equals() {
        let attrs = NodeAttributes::new().with_node_type(NodeType::DataCenter);
        let rule = PolicyRule::node_type_equals(NodeType::DataCenter);

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let rule_mismatch = PolicyRule::node_type_equals(NodeType::Laptop);
        assert!(!AttributeMatcher::matches(&attrs, &rule_mismatch));
    }

    #[test]
    fn test_tenant_equals() {
        let tenant_id = Uuid::new_v4();
        let attrs = NodeAttributes::new().with_tenant(tenant_id);
        let rule = PolicyRule::tenant_equals(tenant_id);

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let other_tenant = Uuid::new_v4();
        let rule_mismatch = PolicyRule::tenant_equals(other_tenant);
        assert!(!AttributeMatcher::matches(&attrs, &rule_mismatch));
    }

    #[test]
    fn test_gpu_memory_gte() {
        let attrs = NodeAttributes::new().with_gpu(2, 48, "A100");
        let rule = PolicyRule::gpu_memory_gte(24);

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let rule_too_high = PolicyRule::gpu_memory_gte(80);
        assert!(!AttributeMatcher::matches(&attrs, &rule_too_high));
    }

    #[test]
    fn test_has_label() {
        let attrs = NodeAttributes::new()
            .with_label("gpu")
            .with_label("high-memory");

        let rule = PolicyRule::has_label("gpu");
        assert!(AttributeMatcher::matches(&attrs, &rule));

        let rule_missing = PolicyRule::has_label("cpu-only");
        assert!(!AttributeMatcher::matches(&attrs, &rule_missing));
    }

    #[test]
    fn test_region_equals() {
        let attrs = NodeAttributes::new().with_region("us-east-1");
        let rule = PolicyRule::region_equals("us-east-1");

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let rule_mismatch = PolicyRule::region_equals("eu-west-1");
        assert!(!AttributeMatcher::matches(&attrs, &rule_mismatch));
    }

    #[test]
    fn test_hostname_regex() {
        let attrs = NodeAttributes::new().with_hostname("gpu-node-42.cluster.local");
        let rule = PolicyRule::new(
            NodeAttribute::Hostname,
            MatchOperator::Regex,
            PolicyValue::String(r"gpu-node-\d+\.cluster\.local".to_string()),
        );

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let attrs_no_match = NodeAttributes::new().with_hostname("cpu-node-1.cluster.local");
        assert!(!AttributeMatcher::matches(&attrs_no_match, &rule));
    }

    #[test]
    fn test_exists_not_exists() {
        let attrs = NodeAttributes::new().with_region("us-east-1");

        let exists_rule = PolicyRule::new(
            NodeAttribute::Region,
            MatchOperator::Exists,
            PolicyValue::Boolean(true),
        );
        assert!(AttributeMatcher::matches(&attrs, &exists_rule));

        let not_exists_rule = PolicyRule::new(
            NodeAttribute::AvailabilityZone,
            MatchOperator::NotExists,
            PolicyValue::Boolean(true),
        );
        assert!(AttributeMatcher::matches(&attrs, &not_exists_rule));
    }

    #[test]
    fn test_in_list() {
        let attrs = NodeAttributes::new().with_region("us-east-1");
        let rule = PolicyRule::new(
            NodeAttribute::Region,
            MatchOperator::In,
            PolicyValue::StringList(vec!["us-east-1".to_string(), "us-west-2".to_string()]),
        );

        assert!(AttributeMatcher::matches(&attrs, &rule));

        let attrs_eu = NodeAttributes::new().with_region("eu-west-1");
        assert!(!AttributeMatcher::matches(&attrs_eu, &rule));
    }
}
