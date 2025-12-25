//! Assignment policy models for automatic subnet assignment

use chrono::{DateTime, Datelike, NaiveTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Policy for automatic node-to-subnet assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentPolicy {
    /// Unique policy identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Priority (higher = evaluated first)
    pub priority: i32,
    /// Whether policy is enabled
    pub enabled: bool,
    /// Target subnet to assign matching nodes to
    pub target_subnet_id: Uuid,
    /// Rules that must ALL match (AND logic)
    pub rules: Vec<PolicyRule>,
    /// Optional time constraints
    pub time_constraints: Option<TimeConstraint>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Created by user/system
    pub created_by: Option<Uuid>,
}

impl AssignmentPolicy {
    /// Create a new policy
    pub fn new(
        name: impl Into<String>,
        target_subnet_id: Uuid,
        priority: i32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            priority,
            enabled: true,
            target_subnet_id,
            rules: Vec::new(),
            time_constraints: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
        }
    }

    /// Add a rule to this policy
    pub fn with_rule(mut self, rule: PolicyRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add time constraints
    pub fn with_time_constraint(mut self, constraint: TimeConstraint) -> Self {
        self.time_constraints = Some(constraint);
        self
    }

    /// Check if policy is currently active (considering time constraints)
    pub fn is_active(&self) -> bool {
        if !self.enabled {
            return false;
        }

        if let Some(ref tc) = self.time_constraints {
            let now = Utc::now();

            // Check date range
            if let Some(start) = tc.start_date {
                if now < start {
                    return false;
                }
            }
            if let Some(end) = tc.end_date {
                if now > end {
                    return false;
                }
            }

            // Check time of day
            if let (Some(start_time), Some(end_time)) = (&tc.active_hours_start, &tc.active_hours_end) {
                let current_time = now.time();
                if current_time < *start_time || current_time > *end_time {
                    return false;
                }
            }

            // Check day of week
            if let Some(ref days) = tc.active_days {
                let today = now.weekday().num_days_from_monday() as u8;
                if !days.contains(&today) {
                    return false;
                }
            }
        }

        true
    }
}

/// A single rule within an assignment policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Attribute to match against
    pub attribute: NodeAttribute,
    /// Match operator
    pub operator: MatchOperator,
    /// Value(s) to match
    pub value: PolicyValue,
}

impl PolicyRule {
    /// Create a new rule
    pub fn new(attribute: NodeAttribute, operator: MatchOperator, value: PolicyValue) -> Self {
        Self {
            attribute,
            operator,
            value,
        }
    }

    /// Convenience: node_type equals
    pub fn node_type_equals(node_type: super::NodeType) -> Self {
        Self::new(
            NodeAttribute::NodeType,
            MatchOperator::Equals,
            PolicyValue::NodeType(node_type),
        )
    }

    /// Convenience: tenant_id equals
    pub fn tenant_equals(tenant_id: Uuid) -> Self {
        Self::new(
            NodeAttribute::TenantId,
            MatchOperator::Equals,
            PolicyValue::Uuid(tenant_id),
        )
    }

    /// Convenience: region equals
    pub fn region_equals(region: impl Into<String>) -> Self {
        Self::new(
            NodeAttribute::Region,
            MatchOperator::Equals,
            PolicyValue::String(region.into()),
        )
    }

    /// Convenience: resource pool equals
    pub fn resource_pool_equals(pool_id: Uuid) -> Self {
        Self::new(
            NodeAttribute::ResourcePoolId,
            MatchOperator::Equals,
            PolicyValue::Uuid(pool_id),
        )
    }

    /// Convenience: GPU memory >= threshold
    pub fn gpu_memory_gte(gb: u32) -> Self {
        Self::new(
            NodeAttribute::GpuMemoryGb,
            MatchOperator::GreaterThanOrEqual,
            PolicyValue::Integer(gb as i64),
        )
    }

    /// Convenience: has label
    pub fn has_label(label: impl Into<String>) -> Self {
        Self::new(
            NodeAttribute::Labels,
            MatchOperator::Contains,
            PolicyValue::String(label.into()),
        )
    }
}

/// Node attributes that can be matched in policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeAttribute {
    /// Node type (DataCenter, Workstation, Laptop, Edge)
    NodeType,
    /// Tenant/organization ID
    TenantId,
    /// Geographic region code
    Region,
    /// Availability zone
    AvailabilityZone,
    /// Resource pool ID
    ResourcePoolId,
    /// Resource pool type (Hackathon, Research, etc.)
    ResourcePoolType,
    /// CPU core count
    CpuCores,
    /// RAM in GB
    RamGb,
    /// GPU count
    GpuCount,
    /// GPU memory in GB
    GpuMemoryGb,
    /// GPU model/type
    GpuModel,
    /// Storage in TB
    StorageTb,
    /// Network bandwidth in Gbps
    NetworkBandwidthGbps,
    /// Device reliability tier (0-3)
    ReliabilityTier,
    /// Node labels/tags
    Labels,
    /// Hostname pattern
    Hostname,
    /// Operating system
    OperatingSystem,
    /// Kubernetes namespace (if applicable)
    Namespace,
    /// Custom attribute
    Custom(u32),
}

/// Match operators for policy rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchOperator {
    /// Exact equality
    Equals,
    /// Not equal
    NotEquals,
    /// Value is in list
    In,
    /// Value is not in list
    NotIn,
    /// Greater than (numeric)
    GreaterThan,
    /// Greater than or equal (numeric)
    GreaterThanOrEqual,
    /// Less than (numeric)
    LessThan,
    /// Less than or equal (numeric)
    LessThanOrEqual,
    /// String contains
    Contains,
    /// String starts with
    StartsWith,
    /// String ends with
    EndsWith,
    /// Regex match
    Regex,
    /// Attribute exists (non-null)
    Exists,
    /// Attribute does not exist (null)
    NotExists,
}

/// Values that can be matched in policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PolicyValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// UUID value
    Uuid(Uuid),
    /// Node type value
    NodeType(super::NodeType),
    /// List of strings
    StringList(Vec<String>),
    /// List of UUIDs
    UuidList(Vec<Uuid>),
    /// Range (min, max) for numeric comparisons
    Range { min: Option<i64>, max: Option<i64> },
}

/// Time constraints for policy activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraint {
    /// Start date (policy active after this date)
    pub start_date: Option<DateTime<Utc>>,
    /// End date (policy active until this date)
    pub end_date: Option<DateTime<Utc>>,
    /// Start of active hours (daily)
    pub active_hours_start: Option<NaiveTime>,
    /// End of active hours (daily)
    pub active_hours_end: Option<NaiveTime>,
    /// Active days of week (0=Monday, 6=Sunday)
    pub active_days: Option<Vec<u8>>,
    /// Timezone for time calculations
    pub timezone: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::NodeType;

    #[test]
    fn test_policy_creation() {
        let subnet_id = Uuid::new_v4();
        let policy = AssignmentPolicy::new("DataCenter Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        assert_eq!(policy.name, "DataCenter Policy");
        assert_eq!(policy.priority, 100);
        assert!(policy.enabled);
        assert_eq!(policy.rules.len(), 1);
        assert!(policy.is_active());
    }

    #[test]
    fn test_policy_disabled() {
        let subnet_id = Uuid::new_v4();
        let mut policy = AssignmentPolicy::new("Test", subnet_id, 50);
        policy.enabled = false;

        assert!(!policy.is_active());
    }

    #[test]
    fn test_rule_convenience_methods() {
        let tenant_id = Uuid::new_v4();

        let rule = PolicyRule::tenant_equals(tenant_id);
        assert_eq!(rule.attribute, NodeAttribute::TenantId);
        assert_eq!(rule.operator, MatchOperator::Equals);

        let rule = PolicyRule::gpu_memory_gte(24);
        assert_eq!(rule.attribute, NodeAttribute::GpuMemoryGb);
        assert_eq!(rule.operator, MatchOperator::GreaterThanOrEqual);
    }
}
