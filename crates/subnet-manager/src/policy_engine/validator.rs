//! Policy validation and conflict detection

use crate::models::{AssignmentPolicy, MatchOperator, NodeAttribute, PolicyRule, PolicyValue};
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Policy validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the policy is valid
    pub is_valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<ValidationError>,
    /// Validation warnings (non-fatal issues)
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create an invalid result with errors
    pub fn invalid(errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning
    pub fn with_warning(mut self, warning: ValidationWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Add an error
    pub fn with_error(mut self, error: ValidationError) -> Self {
        self.errors.push(error);
        self.is_valid = false;
        self
    }
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Policy has no rules
    NoRules,
    /// Invalid rule configuration
    InvalidRule {
        rule_index: usize,
        reason: String,
    },
    /// Incompatible operator for attribute type
    IncompatibleOperator {
        attribute: NodeAttribute,
        operator: MatchOperator,
    },
    /// Invalid value type for attribute
    InvalidValueType {
        attribute: NodeAttribute,
        expected: String,
        got: String,
    },
    /// Invalid time constraint
    InvalidTimeConstraint {
        reason: String,
    },
    /// Target subnet doesn't exist
    TargetSubnetMissing {
        subnet_id: Uuid,
    },
    /// Policy name is empty
    EmptyName,
    /// Duplicate policy name
    DuplicateName {
        name: String,
    },
}

/// Validation warnings (non-fatal issues)
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// Policy has very low priority
    LowPriority {
        priority: i32,
    },
    /// Policy has very high priority
    HighPriority {
        priority: i32,
    },
    /// Policy uses regex which may be slow
    RegexPerformance {
        rule_index: usize,
    },
    /// Policy time constraint may never be active
    NeverActive {
        reason: String,
    },
    /// Overlapping rules with another policy
    OverlappingRules {
        other_policy_id: Uuid,
        overlap_description: String,
    },
    /// Policy targets a draining subnet
    TargetSubnetDraining {
        subnet_id: Uuid,
    },
}

/// Policy validator
pub struct PolicyValidator {
    /// Existing policy names for duplicate checking
    existing_names: HashSet<String>,
    /// Existing subnet IDs for target validation
    existing_subnets: HashSet<Uuid>,
}

impl PolicyValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            existing_names: HashSet::new(),
            existing_subnets: HashSet::new(),
        }
    }

    /// Add existing policy names for duplicate checking
    pub fn with_existing_names(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.existing_names.extend(names);
        self
    }

    /// Add existing subnet IDs for target validation
    pub fn with_existing_subnets(mut self, subnets: impl IntoIterator<Item = Uuid>) -> Self {
        self.existing_subnets.extend(subnets);
        self
    }

    /// Validate a policy
    pub fn validate(&self, policy: &AssignmentPolicy) -> ValidationResult {
        let mut result = ValidationResult::valid();

        // Check for empty name
        if policy.name.trim().is_empty() {
            result = result.with_error(ValidationError::EmptyName);
        }

        // Check for duplicate name
        if self.existing_names.contains(&policy.name) {
            result = result.with_error(ValidationError::DuplicateName {
                name: policy.name.clone(),
            });
        }

        // Check for no rules
        if policy.rules.is_empty() {
            result = result.with_error(ValidationError::NoRules);
        }

        // Validate each rule
        for (idx, rule) in policy.rules.iter().enumerate() {
            if let Some(error) = self.validate_rule(rule, idx) {
                result = result.with_error(error);
            }

            // Check for regex performance warning
            if rule.operator == MatchOperator::Regex {
                result = result.with_warning(ValidationWarning::RegexPerformance { rule_index: idx });
            }
        }

        // Validate target subnet exists
        if !self.existing_subnets.is_empty()
            && !self.existing_subnets.contains(&policy.target_subnet_id)
        {
            result = result.with_error(ValidationError::TargetSubnetMissing {
                subnet_id: policy.target_subnet_id,
            });
        }

        // Validate time constraints
        if let Some(ref tc) = policy.time_constraints {
            if let Err(reason) = self.validate_time_constraint(tc) {
                result = result.with_error(ValidationError::InvalidTimeConstraint { reason });
            }
        }

        // Priority warnings
        if policy.priority < -1000 {
            result = result.with_warning(ValidationWarning::LowPriority {
                priority: policy.priority,
            });
        } else if policy.priority > 10000 {
            result = result.with_warning(ValidationWarning::HighPriority {
                priority: policy.priority,
            });
        }

        result
    }

    /// Validate a single rule
    fn validate_rule(&self, rule: &PolicyRule, idx: usize) -> Option<ValidationError> {
        // Check operator compatibility with attribute type
        match rule.attribute {
            // Numeric attributes
            NodeAttribute::CpuCores
            | NodeAttribute::RamGb
            | NodeAttribute::GpuCount
            | NodeAttribute::GpuMemoryGb
            | NodeAttribute::ReliabilityTier => {
                if !self.is_numeric_operator(rule.operator) && !self.is_existence_operator(rule.operator) {
                    return Some(ValidationError::IncompatibleOperator {
                        attribute: rule.attribute,
                        operator: rule.operator,
                    });
                }
            }
            // Float attributes
            NodeAttribute::StorageTb | NodeAttribute::NetworkBandwidthGbps => {
                if !self.is_numeric_operator(rule.operator) && !self.is_existence_operator(rule.operator) {
                    return Some(ValidationError::IncompatibleOperator {
                        attribute: rule.attribute,
                        operator: rule.operator,
                    });
                }
            }
            // String attributes - most operators valid
            NodeAttribute::Region
            | NodeAttribute::AvailabilityZone
            | NodeAttribute::GpuModel
            | NodeAttribute::Hostname
            | NodeAttribute::OperatingSystem
            | NodeAttribute::Namespace
            | NodeAttribute::ResourcePoolType => {}
            // UUID attributes
            NodeAttribute::TenantId | NodeAttribute::ResourcePoolId => {
                if !matches!(
                    rule.operator,
                    MatchOperator::Equals
                        | MatchOperator::NotEquals
                        | MatchOperator::In
                        | MatchOperator::NotIn
                        | MatchOperator::Exists
                        | MatchOperator::NotExists
                ) {
                    return Some(ValidationError::IncompatibleOperator {
                        attribute: rule.attribute,
                        operator: rule.operator,
                    });
                }
            }
            // NodeType - equality only
            NodeAttribute::NodeType => {
                if !matches!(
                    rule.operator,
                    MatchOperator::Equals | MatchOperator::NotEquals | MatchOperator::Exists | MatchOperator::NotExists
                ) {
                    return Some(ValidationError::IncompatibleOperator {
                        attribute: rule.attribute,
                        operator: rule.operator,
                    });
                }
            }
            // Labels - contains and list operations
            NodeAttribute::Labels => {
                if !matches!(
                    rule.operator,
                    MatchOperator::Contains
                        | MatchOperator::In
                        | MatchOperator::NotIn
                        | MatchOperator::Exists
                        | MatchOperator::NotExists
                ) {
                    return Some(ValidationError::IncompatibleOperator {
                        attribute: rule.attribute,
                        operator: rule.operator,
                    });
                }
            }
            NodeAttribute::Custom(_) => {} // Custom attributes - all operators allowed
        }

        // Validate value type matches operator expectations
        match rule.operator {
            MatchOperator::In | MatchOperator::NotIn => {
                if !matches!(rule.value, PolicyValue::StringList(_) | PolicyValue::UuidList(_)) {
                    return Some(ValidationError::InvalidRule {
                        rule_index: idx,
                        reason: "In/NotIn operators require a list value".to_string(),
                    });
                }
            }
            MatchOperator::GreaterThan
            | MatchOperator::GreaterThanOrEqual
            | MatchOperator::LessThan
            | MatchOperator::LessThanOrEqual => {
                if !matches!(rule.value, PolicyValue::Integer(_) | PolicyValue::Float(_)) {
                    return Some(ValidationError::InvalidRule {
                        rule_index: idx,
                        reason: "Comparison operators require numeric value".to_string(),
                    });
                }
            }
            _ => {}
        }

        None
    }

    /// Check if operator is numeric
    fn is_numeric_operator(&self, op: MatchOperator) -> bool {
        matches!(
            op,
            MatchOperator::Equals
                | MatchOperator::NotEquals
                | MatchOperator::GreaterThan
                | MatchOperator::GreaterThanOrEqual
                | MatchOperator::LessThan
                | MatchOperator::LessThanOrEqual
        )
    }

    /// Check if operator is existence check
    fn is_existence_operator(&self, op: MatchOperator) -> bool {
        matches!(op, MatchOperator::Exists | MatchOperator::NotExists)
    }

    /// Validate time constraint
    fn validate_time_constraint(
        &self,
        tc: &crate::models::TimeConstraint,
    ) -> std::result::Result<(), String> {
        // Check start/end date ordering
        if let (Some(start), Some(end)) = (tc.start_date, tc.end_date) {
            if start >= end {
                return Err("Start date must be before end date".to_string());
            }
        }

        // Check active hours ordering
        if let (Some(start), Some(end)) = (&tc.active_hours_start, &tc.active_hours_end) {
            if start >= end {
                return Err("Active hours start must be before end".to_string());
            }
        }

        // Validate day of week values
        if let Some(ref days) = tc.active_days {
            for &day in days {
                if day > 6 {
                    return Err(format!("Invalid day of week: {} (must be 0-6)", day));
                }
            }
        }

        Ok(())
    }
}

impl Default for PolicyValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Conflict detection between policies
pub struct ConflictDetector;

impl ConflictDetector {
    /// Detect potential conflicts between policies
    pub fn detect_conflicts(policies: &[AssignmentPolicy]) -> Vec<PolicyConflict> {
        let mut conflicts = Vec::new();

        for (i, policy_a) in policies.iter().enumerate() {
            for policy_b in policies.iter().skip(i + 1) {
                if let Some(conflict) = Self::check_conflict(policy_a, policy_b) {
                    conflicts.push(conflict);
                }
            }
        }

        conflicts
    }

    /// Check if two policies conflict
    fn check_conflict(a: &AssignmentPolicy, b: &AssignmentPolicy) -> Option<PolicyConflict> {
        // Same priority targeting different subnets with overlapping rules
        if a.priority == b.priority && a.target_subnet_id != b.target_subnet_id {
            if Self::rules_may_overlap(&a.rules, &b.rules) {
                return Some(PolicyConflict {
                    policy_a_id: a.id,
                    policy_b_id: b.id,
                    conflict_type: ConflictType::SamePriorityOverlap,
                    description: format!(
                        "Policies '{}' and '{}' have same priority {} with potentially overlapping rules",
                        a.name, b.name, a.priority
                    ),
                    severity: ConflictSeverity::Warning,
                });
            }
        }

        // Identical rules targeting different subnets
        if a.target_subnet_id != b.target_subnet_id && Self::rules_identical(&a.rules, &b.rules) {
            return Some(PolicyConflict {
                policy_a_id: a.id,
                policy_b_id: b.id,
                conflict_type: ConflictType::IdenticalRules,
                description: format!(
                    "Policies '{}' and '{}' have identical rules but target different subnets",
                    a.name, b.name
                ),
                severity: ConflictSeverity::Error,
            });
        }

        // Contradictory rules (one requires X, other requires NOT X)
        if Self::rules_contradict(&a.rules, &b.rules) {
            return Some(PolicyConflict {
                policy_a_id: a.id,
                policy_b_id: b.id,
                conflict_type: ConflictType::ContradictoryRules,
                description: format!(
                    "Policies '{}' and '{}' have contradictory rules",
                    a.name, b.name
                ),
                severity: ConflictSeverity::Info,
            });
        }

        None
    }

    /// Check if two rule sets may overlap
    fn rules_may_overlap(rules_a: &[PolicyRule], rules_b: &[PolicyRule]) -> bool {
        // Check if they share any common attributes
        let attrs_a: HashSet<_> = rules_a.iter().map(|r| &r.attribute).collect();
        let attrs_b: HashSet<_> = rules_b.iter().map(|r| &r.attribute).collect();

        attrs_a.intersection(&attrs_b).count() > 0
    }

    /// Check if two rule sets are identical
    fn rules_identical(rules_a: &[PolicyRule], rules_b: &[PolicyRule]) -> bool {
        if rules_a.len() != rules_b.len() {
            return false;
        }

        // Simple check - same attributes and operators
        let set_a: HashSet<_> = rules_a
            .iter()
            .map(|r| (r.attribute, r.operator))
            .collect();
        let set_b: HashSet<_> = rules_b
            .iter()
            .map(|r| (r.attribute, r.operator))
            .collect();

        set_a == set_b
    }

    /// Check if rules contradict each other
    fn rules_contradict(rules_a: &[PolicyRule], rules_b: &[PolicyRule]) -> bool {
        for rule_a in rules_a {
            for rule_b in rules_b {
                if rule_a.attribute == rule_b.attribute {
                    // Check for Equals vs NotEquals on same value
                    if (rule_a.operator == MatchOperator::Equals
                        && rule_b.operator == MatchOperator::NotEquals)
                        || (rule_a.operator == MatchOperator::NotEquals
                            && rule_b.operator == MatchOperator::Equals)
                    {
                        if rule_a.value == rule_b.value {
                            return true;
                        }
                    }

                    // Check for Exists vs NotExists
                    if (rule_a.operator == MatchOperator::Exists
                        && rule_b.operator == MatchOperator::NotExists)
                        || (rule_a.operator == MatchOperator::NotExists
                            && rule_b.operator == MatchOperator::Exists)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Policy conflict information
#[derive(Debug, Clone)]
pub struct PolicyConflict {
    /// First policy ID
    pub policy_a_id: Uuid,
    /// Second policy ID
    pub policy_b_id: Uuid,
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Human-readable description
    pub description: String,
    /// Severity of the conflict
    pub severity: ConflictSeverity,
}

/// Types of policy conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Same priority with overlapping rules
    SamePriorityOverlap,
    /// Identical rules targeting different subnets
    IdenticalRules,
    /// Rules that contradict each other
    ContradictoryRules,
    /// Time constraints overlap
    TimeOverlap,
}

/// Conflict severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    /// Informational - not necessarily a problem
    Info,
    /// Warning - may cause unexpected behavior
    Warning,
    /// Error - will definitely cause issues
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::NodeType;

    #[test]
    fn test_validate_empty_name() {
        let validator = PolicyValidator::new();
        let mut policy = AssignmentPolicy::new("", Uuid::new_v4(), 100);
        policy.rules.push(PolicyRule::node_type_equals(NodeType::DataCenter));

        let result = validator.validate(&policy);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::EmptyName)));
    }

    #[test]
    fn test_validate_no_rules() {
        let validator = PolicyValidator::new();
        let policy = AssignmentPolicy::new("Test", Uuid::new_v4(), 100);

        let result = validator.validate(&policy);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::NoRules)));
    }

    #[test]
    fn test_validate_incompatible_operator() {
        let validator = PolicyValidator::new();
        let mut policy = AssignmentPolicy::new("Test", Uuid::new_v4(), 100);
        policy.rules.push(PolicyRule::new(
            NodeAttribute::CpuCores,
            MatchOperator::Contains, // Invalid for numeric attribute
            PolicyValue::Integer(4),
        ));

        let result = validator.validate(&policy);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::IncompatibleOperator { .. })));
    }

    #[test]
    fn test_validate_valid_policy() {
        let subnet_id = Uuid::new_v4();
        let validator = PolicyValidator::new()
            .with_existing_subnets(vec![subnet_id]);

        let policy = AssignmentPolicy::new("Test", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        let result = validator.validate(&policy);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_detect_same_priority_conflict() {
        let subnet_a = Uuid::new_v4();
        let subnet_b = Uuid::new_v4();

        let policy_a = AssignmentPolicy::new("Policy A", subnet_a, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        let policy_b = AssignmentPolicy::new("Policy B", subnet_b, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        let conflicts = ConflictDetector::detect_conflicts(&[policy_a, policy_b]);
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, ConflictType::SamePriorityOverlap);
    }

    #[test]
    fn test_detect_contradictory_rules() {
        let subnet_a = Uuid::new_v4();
        let subnet_b = Uuid::new_v4();

        let policy_a = AssignmentPolicy::new("Policy A", subnet_a, 100)
            .with_rule(PolicyRule::new(
                NodeAttribute::Region,
                MatchOperator::Equals,
                PolicyValue::String("us-east-1".to_string()),
            ));
        let policy_b = AssignmentPolicy::new("Policy B", subnet_b, 50)
            .with_rule(PolicyRule::new(
                NodeAttribute::Region,
                MatchOperator::NotEquals,
                PolicyValue::String("us-east-1".to_string()),
            ));

        let conflicts = ConflictDetector::detect_conflicts(&[policy_a, policy_b]);
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, ConflictType::ContradictoryRules);
    }

    #[test]
    fn test_no_conflict_different_attributes() {
        let subnet_a = Uuid::new_v4();
        let subnet_b = Uuid::new_v4();

        let policy_a = AssignmentPolicy::new("Policy A", subnet_a, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        let policy_b = AssignmentPolicy::new("Policy B", subnet_b, 100)
            .with_rule(PolicyRule::region_equals("us-east-1"));

        let conflicts = ConflictDetector::detect_conflicts(&[policy_a, policy_b]);
        // No conflict because they use different attributes
        assert!(conflicts.is_empty());
    }
}
