//! Policy evaluator for automatic subnet assignment

use crate::models::{AssignmentPolicy, Subnet};
use crate::{Error, Result};
use std::collections::HashMap;
use uuid::Uuid;

use super::matcher::{AttributeMatcher, NodeAttributes};

/// Trait for policy evaluation
pub trait PolicyEvaluator: Send + Sync {
    /// Evaluate policies for a node and return the matching subnet ID
    fn evaluate(&self, node: &NodeAttributes) -> Result<Uuid>;

    /// Dry-run evaluation to preview assignment without making changes
    fn evaluate_dry_run(&self, node: &NodeAttributes) -> EvaluationResult;
}

/// Result of policy evaluation
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The matched subnet ID (if any)
    pub subnet_id: Option<Uuid>,
    /// The matched policy (if any)
    pub matched_policy: Option<MatchedPolicy>,
    /// All evaluated policies with their results
    pub evaluated: Vec<PolicyEvaluation>,
}

/// Information about a matched policy
#[derive(Debug, Clone)]
pub struct MatchedPolicy {
    pub policy_id: Uuid,
    pub policy_name: String,
    pub priority: i32,
}

/// Result of evaluating a single policy
#[derive(Debug, Clone)]
pub struct PolicyEvaluation {
    pub policy_id: Uuid,
    pub policy_name: String,
    pub priority: i32,
    pub matched: bool,
    pub active: bool,
    pub rule_results: Vec<RuleEvaluation>,
}

/// Result of evaluating a single rule
#[derive(Debug, Clone)]
pub struct RuleEvaluation {
    pub attribute: String,
    pub operator: String,
    pub matched: bool,
}

/// Policy engine for automatic subnet assignment
///
/// Evaluates node attributes against assignment policies to determine
/// which subnet a node should be assigned to.
#[derive(Debug, Default)]
pub struct PolicyEngine {
    /// Policies indexed by ID
    policies: HashMap<Uuid, AssignmentPolicy>,
    /// Subnets indexed by ID
    subnets: HashMap<Uuid, Subnet>,
    /// Sorted policy IDs by priority (highest first)
    sorted_policies: Vec<Uuid>,
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            subnets: HashMap::new(),
            sorted_policies: Vec::new(),
        }
    }

    /// Add a policy to the engine
    pub fn add_policy(&mut self, policy: AssignmentPolicy) {
        let id = policy.id;
        self.policies.insert(id, policy);
        self.rebuild_sorted_policies();
    }

    /// Remove a policy from the engine
    pub fn remove_policy(&mut self, policy_id: Uuid) -> Option<AssignmentPolicy> {
        let policy = self.policies.remove(&policy_id);
        if policy.is_some() {
            self.rebuild_sorted_policies();
        }
        policy
    }

    /// Update a policy in the engine
    pub fn update_policy(&mut self, policy: AssignmentPolicy) {
        let id = policy.id;
        self.policies.insert(id, policy);
        self.rebuild_sorted_policies();
    }

    /// Get a policy by ID
    pub fn get_policy(&self, policy_id: Uuid) -> Option<&AssignmentPolicy> {
        self.policies.get(&policy_id)
    }

    /// Get all policies
    pub fn policies(&self) -> Vec<&AssignmentPolicy> {
        self.policies.values().collect()
    }

    /// Add a subnet to the engine (for validation)
    pub fn add_subnet(&mut self, subnet: Subnet) {
        self.subnets.insert(subnet.id, subnet);
    }

    /// Remove a subnet from the engine
    pub fn remove_subnet(&mut self, subnet_id: Uuid) -> Option<Subnet> {
        self.subnets.remove(&subnet_id)
    }

    /// Rebuild the sorted policy list
    fn rebuild_sorted_policies(&mut self) {
        let mut policies: Vec<_> = self.policies.values().collect();
        policies.sort_by(|a, b| b.priority.cmp(&a.priority)); // Highest priority first
        self.sorted_policies = policies.iter().map(|p| p.id).collect();
    }

    /// Check if a policy matches a node
    fn policy_matches(&self, policy: &AssignmentPolicy, node: &NodeAttributes) -> bool {
        // Check if policy is active (enabled and within time constraints)
        if !policy.is_active() {
            return false;
        }

        // All rules must match (AND logic)
        for rule in &policy.rules {
            if !AttributeMatcher::matches(node, rule) {
                return false;
            }
        }

        true
    }

    /// Evaluate a single policy and return detailed results
    fn evaluate_policy(
        &self,
        policy: &AssignmentPolicy,
        node: &NodeAttributes,
    ) -> PolicyEvaluation {
        let active = policy.is_active();
        let mut rule_results = Vec::new();
        let mut all_matched = active;

        for rule in &policy.rules {
            let matched = AttributeMatcher::matches(node, rule);
            if !matched {
                all_matched = false;
            }

            rule_results.push(RuleEvaluation {
                attribute: format!("{:?}", rule.attribute),
                operator: format!("{:?}", rule.operator),
                matched,
            });
        }

        PolicyEvaluation {
            policy_id: policy.id,
            policy_name: policy.name.clone(),
            priority: policy.priority,
            matched: all_matched,
            active,
            rule_results,
        }
    }
}

impl PolicyEvaluator for PolicyEngine {
    fn evaluate(&self, node: &NodeAttributes) -> Result<Uuid> {
        // Iterate through policies in priority order
        for &policy_id in &self.sorted_policies {
            let policy = self.policies.get(&policy_id).unwrap();

            if self.policy_matches(policy, node) {
                // Verify the target subnet exists
                if !self.subnets.contains_key(&policy.target_subnet_id) {
                    continue; // Skip policies targeting non-existent subnets
                }

                return Ok(policy.target_subnet_id);
            }
        }

        Err(Error::NoMatchingPolicy(Uuid::nil()))
    }

    fn evaluate_dry_run(&self, node: &NodeAttributes) -> EvaluationResult {
        let mut evaluated = Vec::new();
        let mut matched_policy: Option<MatchedPolicy> = None;

        for &policy_id in &self.sorted_policies {
            let policy = self.policies.get(&policy_id).unwrap();
            let evaluation = self.evaluate_policy(policy, node);

            // Record the first match
            if evaluation.matched && matched_policy.is_none() {
                // Verify the target subnet exists
                if self.subnets.contains_key(&policy.target_subnet_id) {
                    matched_policy = Some(MatchedPolicy {
                        policy_id: policy.id,
                        policy_name: policy.name.clone(),
                        priority: policy.priority,
                    });
                }
            }

            evaluated.push(evaluation);
        }

        EvaluationResult {
            subnet_id: matched_policy.as_ref().map(|p| {
                self.policies.get(&p.policy_id).unwrap().target_subnet_id
            }),
            matched_policy,
            evaluated,
        }
    }
}

/// Result of batch evaluation for a single node
#[derive(Debug, Clone)]
pub struct BatchEvaluationItem {
    /// Node identifier (index or ID)
    pub node_index: usize,
    /// Node ID if provided
    pub node_id: Option<Uuid>,
    /// Evaluation result
    pub result: Result<Uuid>,
    /// Matched policy (if any)
    pub matched_policy: Option<MatchedPolicy>,
}

/// Result of batch evaluation
#[derive(Debug, Clone)]
pub struct BatchEvaluationResult {
    /// Total nodes evaluated
    pub total: usize,
    /// Number of successful matches
    pub matched: usize,
    /// Number of failed matches
    pub unmatched: usize,
    /// Individual results
    pub results: Vec<BatchEvaluationItem>,
    /// Subnet distribution (subnet_id -> count)
    pub subnet_distribution: HashMap<Uuid, usize>,
}

impl PolicyEngine {
    /// Evaluate multiple nodes in batch
    ///
    /// More efficient than calling evaluate() repeatedly as it avoids
    /// repeated policy sorting and can be parallelized in the future.
    pub fn evaluate_batch(
        &self,
        nodes: &[(Option<Uuid>, NodeAttributes)],
    ) -> BatchEvaluationResult {
        let mut results = Vec::with_capacity(nodes.len());
        let mut matched = 0;
        let mut subnet_distribution: HashMap<Uuid, usize> = HashMap::new();

        for (idx, (node_id, attrs)) in nodes.iter().enumerate() {
            let result = self.evaluate(attrs);
            let matched_policy = if result.is_ok() {
                matched += 1;
                let subnet_id = result.as_ref().unwrap();
                *subnet_distribution.entry(*subnet_id).or_insert(0) += 1;

                // Find the matched policy
                self.sorted_policies
                    .iter()
                    .filter_map(|pid| self.policies.get(pid))
                    .find(|p| &p.target_subnet_id == subnet_id && self.policy_matches(p, attrs))
                    .map(|p| MatchedPolicy {
                        policy_id: p.id,
                        policy_name: p.name.clone(),
                        priority: p.priority,
                    })
            } else {
                None
            };

            results.push(BatchEvaluationItem {
                node_index: idx,
                node_id: *node_id,
                result,
                matched_policy,
            });
        }

        BatchEvaluationResult {
            total: nodes.len(),
            matched,
            unmatched: nodes.len() - matched,
            results,
            subnet_distribution,
        }
    }

    /// Get policies that would match a given node, sorted by priority
    pub fn matching_policies(&self, node: &NodeAttributes) -> Vec<&AssignmentPolicy> {
        self.sorted_policies
            .iter()
            .filter_map(|pid| self.policies.get(pid))
            .filter(|p| self.policy_matches(p, node))
            .collect()
    }

    /// Get the number of active policies
    pub fn active_policy_count(&self) -> usize {
        self.policies.values().filter(|p| p.is_active()).count()
    }

    /// Get policies targeting a specific subnet
    pub fn policies_for_subnet(&self, subnet_id: Uuid) -> Vec<&AssignmentPolicy> {
        self.policies
            .values()
            .filter(|p| p.target_subnet_id == subnet_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{NodeType, PolicyRule, SubnetPurpose, SubnetStatus};
    use ipnet::Ipv4Net;
    use std::str::FromStr;

    fn create_test_subnet(id: Uuid, name: &str) -> Subnet {
        let cidr = Ipv4Net::from_str("10.100.0.0/20").unwrap();
        let mut subnet = Subnet::new(name, cidr, SubnetPurpose::NodeType, 51820);
        subnet.id = id;
        subnet.status = SubnetStatus::Active;
        subnet
    }

    #[test]
    fn test_single_policy_match() {
        let mut engine = PolicyEngine::new();

        // Create a subnet
        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "DataCenter Subnet"));

        // Create a policy targeting datacenter nodes
        let policy = AssignmentPolicy::new("DC Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        engine.add_policy(policy);

        // Test matching node
        let node = NodeAttributes::new().with_node_type(NodeType::DataCenter);
        let result = engine.evaluate(&node);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), subnet_id);

        // Test non-matching node
        let laptop_node = NodeAttributes::new().with_node_type(NodeType::Laptop);
        let result = engine.evaluate(&laptop_node);
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_priority() {
        let mut engine = PolicyEngine::new();

        // Create two subnets
        let subnet1_id = Uuid::new_v4();
        let subnet2_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet1_id, "Low Priority Subnet"));
        engine.add_subnet(create_test_subnet(subnet2_id, "High Priority Subnet"));

        // Create two policies - same match, different priority
        let low_priority = AssignmentPolicy::new("Low", subnet1_id, 50)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        let high_priority = AssignmentPolicy::new("High", subnet2_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));

        engine.add_policy(low_priority);
        engine.add_policy(high_priority);

        // Higher priority policy should match first
        let node = NodeAttributes::new().with_node_type(NodeType::DataCenter);
        let result = engine.evaluate(&node).unwrap();
        assert_eq!(result, subnet2_id);
    }

    #[test]
    fn test_multiple_rules_and_logic() {
        let mut engine = PolicyEngine::new();

        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "GPU Datacenter"));

        // Policy requires BOTH datacenter AND GPU
        let policy = AssignmentPolicy::new("GPU DC Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter))
            .with_rule(PolicyRule::gpu_memory_gte(24));
        engine.add_policy(policy);

        // Node with only datacenter - should not match
        let dc_only = NodeAttributes::new().with_node_type(NodeType::DataCenter);
        assert!(engine.evaluate(&dc_only).is_err());

        // Node with only GPU - should not match
        let gpu_only = NodeAttributes::new().with_gpu(2, 48, "A100");
        assert!(engine.evaluate(&gpu_only).is_err());

        // Node with both - should match
        let dc_gpu = NodeAttributes::new()
            .with_node_type(NodeType::DataCenter)
            .with_gpu(2, 48, "A100");
        assert_eq!(engine.evaluate(&dc_gpu).unwrap(), subnet_id);
    }

    #[test]
    fn test_dry_run() {
        let mut engine = PolicyEngine::new();

        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "Test Subnet"));

        let policy = AssignmentPolicy::new("Test Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter))
            .with_rule(PolicyRule::region_equals("us-east-1"));
        engine.add_policy(policy);

        // Partially matching node
        let node = NodeAttributes::new()
            .with_node_type(NodeType::DataCenter)
            .with_region("eu-west-1");

        let result = engine.evaluate_dry_run(&node);

        // Should not match (wrong region)
        assert!(result.subnet_id.is_none());
        assert!(result.matched_policy.is_none());

        // But should have evaluation details
        assert_eq!(result.evaluated.len(), 1);
        assert!(!result.evaluated[0].matched);
        assert_eq!(result.evaluated[0].rule_results.len(), 2);
        assert!(result.evaluated[0].rule_results[0].matched); // NodeType matched
        assert!(!result.evaluated[0].rule_results[1].matched); // Region didn't match
    }

    #[test]
    fn test_disabled_policy_skipped() {
        let mut engine = PolicyEngine::new();

        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "Test Subnet"));

        let mut policy = AssignmentPolicy::new("Disabled Policy", subnet_id, 100)
            .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter));
        policy.enabled = false;
        engine.add_policy(policy);

        let node = NodeAttributes::new().with_node_type(NodeType::DataCenter);
        assert!(engine.evaluate(&node).is_err());
    }

    #[test]
    fn test_policy_crud() {
        let mut engine = PolicyEngine::new();

        let subnet_id = Uuid::new_v4();
        engine.add_subnet(create_test_subnet(subnet_id, "Test"));

        // Add policy
        let policy = AssignmentPolicy::new("Test", subnet_id, 50);
        let policy_id = policy.id;
        engine.add_policy(policy);

        assert!(engine.get_policy(policy_id).is_some());
        assert_eq!(engine.policies().len(), 1);

        // Update policy
        let mut updated = engine.get_policy(policy_id).unwrap().clone();
        updated.priority = 100;
        engine.update_policy(updated);
        assert_eq!(engine.get_policy(policy_id).unwrap().priority, 100);

        // Remove policy
        let removed = engine.remove_policy(policy_id);
        assert!(removed.is_some());
        assert!(engine.get_policy(policy_id).is_none());
        assert!(engine.policies().is_empty());
    }
}
