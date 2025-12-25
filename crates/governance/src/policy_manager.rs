//! Policy definition and enforcement module
//!
//! The PolicyManager handles the creation, validation, and enforcement of governance
//! policies across the ExoRust system. It supports various policy types including
//! resource, security, evolution, and compliance policies.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{GovernanceError, Result};
use stratoswarm_agent_core::agent::AgentId;

/// Policy manager for defining and enforcing governance rules
pub struct PolicyManager {
    policies: DashMap<Uuid, Policy>,
    policy_index: DashMap<PolicyType, Vec<Uuid>>,
    agent_policies: DashMap<AgentId, Vec<Uuid>>,
    global_policies: Arc<RwLock<Vec<Uuid>>>,
    conflict_resolver: Arc<ConflictResolver>,
}

/// Policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub policy_type: PolicyType,
    pub scope: PolicyScope,
    pub rules: Vec<PolicyRule>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Types of policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolicyType {
    Resource,
    Security,
    Evolution,
    Compliance,
    Communication,
    DataAccess,
}

/// Policy scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    Global,
    Agent(AgentId),
    AgentGroup(Vec<AgentId>),
    ResourceType(String),
}

/// Individual policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub condition: PolicyCondition,
    pub action: PolicyAction,
    pub parameters: serde_json::Value,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    Always,
    ResourceLimit {
        resource: String,
        operator: ComparisonOperator,
        value: f64,
    },
    TimeWindow {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    AgentProperty {
        property: String,
        operator: ComparisonOperator,
        value: serde_json::Value,
    },
    Combined {
        operator: LogicalOperator,
        conditions: Vec<PolicyCondition>,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Logical operators for combining conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Policy actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    Allow,
    Deny,
    Limit(serde_json::Value),
    Notify(String),
    Escalate(String),
}

/// Conflict resolver for handling policy conflicts
struct ConflictResolver {
    resolution_strategy: ConflictResolutionStrategy,
}

/// Strategies for resolving policy conflicts
#[derive(Debug, Clone)]
enum ConflictResolutionStrategy {
    PriorityBased,
    MostRestrictive,
    MostPermissive,
    Custom(Box<dyn Fn(&Policy, &Policy) -> &Policy + Send + Sync>),
}

impl PolicyManager {
    /// Create a new policy manager
    pub fn new() -> Self {
        Self {
            policies: DashMap::new(),
            policy_index: DashMap::new(),
            agent_policies: DashMap::new(),
            global_policies: Arc::new(RwLock::new(Vec::new())),
            conflict_resolver: Arc::new(ConflictResolver {
                resolution_strategy: ConflictResolutionStrategy::PriorityBased,
            }),
        }
    }

    /// Create a new policy
    pub async fn create_policy(&self, mut policy: Policy) -> Result<Uuid> {
        info!("Creating new policy: {}", policy.name);

        // Generate ID if not provided
        if policy.id == Uuid::nil() {
            policy.id = Uuid::new_v4();
        }

        // Set timestamps
        let now = Utc::now();
        policy.created_at = now;
        policy.updated_at = now;

        // Validate policy
        self.validate_policy(&policy)?;

        // Index by type
        self.policy_index
            .entry(policy.policy_type)
            .or_insert_with(Vec::new)
            .push(policy.id);

        // Handle scope-based indexing
        match &policy.scope {
            PolicyScope::Global => {
                self.global_policies.write().push(policy.id);
            }
            PolicyScope::Agent(agent_id) => {
                self.agent_policies
                    .entry(agent_id.clone())
                    .or_insert_with(Vec::new)
                    .push(policy.id);
            }
            PolicyScope::AgentGroup(agents) => {
                for agent_id in agents {
                    self.agent_policies
                        .entry(agent_id.clone())
                        .or_insert_with(Vec::new)
                        .push(policy.id);
                }
            }
            _ => {}
        }

        let id = policy.id;
        self.policies.insert(id, policy);

        Ok(id)
    }

    /// Update an existing policy
    pub async fn update_policy(&self, id: Uuid, updates: PolicyUpdate) -> Result<()> {
        let mut policy = self
            .policies
            .get_mut(&id)
            .ok_or_else(|| GovernanceError::InternalError("Policy not found".to_string()))?;

        // Apply updates
        if let Some(name) = updates.name {
            policy.name = name;
        }
        if let Some(description) = updates.description {
            policy.description = description;
        }
        if let Some(rules) = updates.rules {
            policy.rules = rules;
        }
        if let Some(priority) = updates.priority {
            policy.priority = priority;
        }
        if let Some(enabled) = updates.enabled {
            policy.enabled = enabled;
        }
        if let Some(expires_at) = updates.expires_at {
            policy.expires_at = Some(expires_at);
        }

        policy.updated_at = Utc::now();

        // Re-validate
        self.validate_policy(&policy)?;

        Ok(())
    }

    /// Delete a policy
    pub async fn delete_policy(&self, id: Uuid) -> Result<()> {
        let policy = self
            .policies
            .remove(&id)
            .ok_or_else(|| GovernanceError::InternalError("Policy not found".to_string()))?
            .1;

        // Remove from indexes
        if let Some(mut type_policies) = self.policy_index.get_mut(&policy.policy_type) {
            type_policies.retain(|&pid| pid != id);
        }

        // Remove from scope-based indexes
        match policy.scope {
            PolicyScope::Global => {
                self.global_policies.write().retain(|&pid| pid != id);
            }
            PolicyScope::Agent(agent_id) => {
                if let Some(mut agent_policies) = self.agent_policies.get_mut(&agent_id) {
                    agent_policies.retain(|&pid| pid != id);
                }
            }
            PolicyScope::AgentGroup(agents) => {
                for agent_id in agents {
                    if let Some(mut agent_policies) = self.agent_policies.get_mut(&agent_id) {
                        agent_policies.retain(|&pid| pid != id);
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Get applicable policies for an agent and policy type
    pub async fn get_applicable_policies(
        &self,
        agent_id: &AgentId,
        policy_type: PolicyType,
    ) -> Result<Vec<Policy>> {
        let mut applicable_policies = Vec::new();

        // Get global policies of this type
        for &policy_id in self.global_policies.read().iter() {
            if let Some(policy) = self.policies.get(&policy_id) {
                if policy.policy_type == policy_type && policy.enabled && !self.is_expired(&policy)
                {
                    applicable_policies.push(policy.clone());
                }
            }
        }

        // Get agent-specific policies
        if let Some(agent_policy_ids) = self.agent_policies.get(agent_id) {
            for &policy_id in agent_policy_ids.iter() {
                if let Some(policy) = self.policies.get(&policy_id) {
                    if policy.policy_type == policy_type
                        && policy.enabled
                        && !self.is_expired(&policy)
                    {
                        applicable_policies.push(policy.clone());
                    }
                }
            }
        }

        // Sort by priority
        applicable_policies.sort_by_key(|p| std::cmp::Reverse(p.priority));

        Ok(applicable_policies)
    }

    /// Evaluate a policy against a context
    pub async fn evaluate_policy<T: Serialize>(
        &self,
        policy: &Policy,
        context: &T,
    ) -> Result<bool> {
        debug!("Evaluating policy: {}", policy.name);

        let context_value = serde_json::to_value(context).map_err(|e| {
            GovernanceError::InternalError(format!("Failed to serialize context: {}", e))
        })?;

        for rule in &policy.rules {
            let condition_met = self.evaluate_condition(&rule.condition, &context_value)?;

            if condition_met {
                match &rule.action {
                    PolicyAction::Allow => return Ok(true),
                    PolicyAction::Deny => return Ok(false),
                    PolicyAction::Limit(limit) => {
                        // Check if context meets the limit
                        return self.check_limit(&context_value, limit);
                    }
                    PolicyAction::Notify(target) => {
                        warn!(
                            "Policy {} triggered notification to {}",
                            policy.name, target
                        );
                        // Continue evaluation
                    }
                    PolicyAction::Escalate(level) => {
                        warn!(
                            "Policy {} triggered escalation level {}",
                            policy.name, level
                        );
                        return Ok(false); // Deny by default on escalation
                    }
                }
            }
        }

        // Default allow if no rules matched
        Ok(true)
    }

    /// Validate and resolve conflicts between policies
    pub async fn resolve_conflicts(&self, policies: Vec<&Policy>) -> Result<Vec<&Policy>> {
        if policies.len() <= 1 {
            return Ok(policies);
        }

        let mut resolved = policies;

        // Group by action type
        let mut allows = Vec::new();
        let mut denies = Vec::new();
        let mut others = Vec::new();

        for policy in resolved.iter() {
            let has_deny = policy
                .rules
                .iter()
                .any(|r| matches!(r.action, PolicyAction::Deny));
            let has_allow = policy
                .rules
                .iter()
                .any(|r| matches!(r.action, PolicyAction::Allow));

            if has_deny {
                denies.push(*policy);
            } else if has_allow {
                allows.push(*policy);
            } else {
                others.push(*policy);
            }
        }

        // Apply resolution strategy
        match &self.conflict_resolver.resolution_strategy {
            ConflictResolutionStrategy::PriorityBased => {
                resolved.sort_by_key(|p| std::cmp::Reverse(p.priority));
            }
            ConflictResolutionStrategy::MostRestrictive => {
                // Denies take precedence
                resolved = denies;
                resolved.extend(others);
                resolved.extend(allows);
            }
            ConflictResolutionStrategy::MostPermissive => {
                // Allows take precedence
                resolved = allows;
                resolved.extend(others);
                resolved.extend(denies);
            }
            ConflictResolutionStrategy::Custom(_) => {
                // Custom resolution logic would go here
            }
        }

        Ok(resolved)
    }

    /// Validate a policy
    fn validate_policy(&self, policy: &Policy) -> Result<()> {
        // Check name
        if policy.name.is_empty() {
            return Err(GovernanceError::ConfigurationError(
                "Policy name cannot be empty".to_string(),
            ));
        }

        // Check rules
        if policy.rules.is_empty() {
            return Err(GovernanceError::ConfigurationError(
                "Policy must have at least one rule".to_string(),
            ));
        }

        // Validate each rule
        for rule in &policy.rules {
            self.validate_condition(&rule.condition)?;
        }

        // Check expiration
        if let Some(expires_at) = policy.expires_at {
            if expires_at <= Utc::now() {
                return Err(GovernanceError::ConfigurationError(
                    "Policy expiration date must be in the future".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate a policy condition
    fn validate_condition(&self, condition: &PolicyCondition) -> Result<()> {
        match condition {
            PolicyCondition::Always => Ok(()),
            PolicyCondition::ResourceLimit { resource, .. } => {
                if resource.is_empty() {
                    return Err(GovernanceError::ConfigurationError(
                        "Resource name cannot be empty".to_string(),
                    ));
                }
                Ok(())
            }
            PolicyCondition::TimeWindow { start, end } => {
                if start >= end {
                    return Err(GovernanceError::ConfigurationError(
                        "Time window start must be before end".to_string(),
                    ));
                }
                Ok(())
            }
            PolicyCondition::AgentProperty { property, .. } => {
                if property.is_empty() {
                    return Err(GovernanceError::ConfigurationError(
                        "Agent property name cannot be empty".to_string(),
                    ));
                }
                Ok(())
            }
            PolicyCondition::Combined { conditions, .. } => {
                if conditions.is_empty() {
                    return Err(GovernanceError::ConfigurationError(
                        "Combined condition must have at least one sub-condition".to_string(),
                    ));
                }
                for condition in conditions {
                    self.validate_condition(condition)?;
                }
                Ok(())
            }
        }
    }

    /// Evaluate a policy condition
    fn evaluate_condition(
        &self,
        condition: &PolicyCondition,
        context: &serde_json::Value,
    ) -> Result<bool> {
        match condition {
            PolicyCondition::Always => Ok(true),

            PolicyCondition::ResourceLimit {
                resource,
                operator,
                value,
            } => {
                let resource_value = context
                    .get(resource)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                Ok(self.compare_values(resource_value, *operator, *value))
            }

            PolicyCondition::TimeWindow { start, end } => {
                let now = Utc::now();
                Ok(now >= *start && now <= *end)
            }

            PolicyCondition::AgentProperty {
                property,
                operator,
                value,
            } => {
                let prop_value = context.get(property);
                if let Some(prop_value) = prop_value {
                    Ok(self.compare_json_values(prop_value, *operator, value))
                } else {
                    Ok(false)
                }
            }

            PolicyCondition::Combined {
                operator,
                conditions,
            } => match operator {
                LogicalOperator::And => {
                    for condition in conditions {
                        if !self.evaluate_condition(condition, context)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                LogicalOperator::Or => {
                    for condition in conditions {
                        if self.evaluate_condition(condition, context)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                LogicalOperator::Not => {
                    if conditions.len() != 1 {
                        return Err(GovernanceError::InternalError(
                            "NOT operator requires exactly one condition".to_string(),
                        ));
                    }
                    Ok(!self.evaluate_condition(&conditions[0], context)?)
                }
            },
        }
    }

    /// Compare numeric values
    fn compare_values(&self, left: f64, operator: ComparisonOperator, right: f64) -> bool {
        match operator {
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::LessThanOrEqual => left <= right,
        }
    }

    /// Compare JSON values
    fn compare_json_values(
        &self,
        left: &serde_json::Value,
        operator: ComparisonOperator,
        right: &serde_json::Value,
    ) -> bool {
        match (left, right) {
            (serde_json::Value::Number(l), serde_json::Value::Number(r)) => {
                if let (Some(l), Some(r)) = (l.as_f64(), r.as_f64()) {
                    self.compare_values(l, operator, r)
                } else {
                    false
                }
            }
            _ => match operator {
                ComparisonOperator::Equal => left == right,
                ComparisonOperator::NotEqual => left != right,
                _ => false, // Other operators not applicable to non-numeric values
            },
        }
    }

    /// Check if a value meets a limit
    fn check_limit(&self, context: &serde_json::Value, limit: &serde_json::Value) -> Result<bool> {
        // Simple implementation - can be extended
        if let (Some(context_val), Some(limit_val)) = (context.as_f64(), limit.as_f64()) {
            Ok(context_val <= limit_val)
        } else {
            Ok(true) // Allow if can't compare
        }
    }

    /// Check if a policy has expired
    fn is_expired(&self, policy: &Policy) -> bool {
        if let Some(expires_at) = policy.expires_at {
            expires_at <= Utc::now()
        } else {
            false
        }
    }

    /// Get all policies
    pub async fn get_all_policies(&self) -> Vec<Policy> {
        self.policies
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get policy by ID
    pub async fn get_policy(&self, id: Uuid) -> Result<Policy> {
        self.policies
            .get(&id)
            .map(|entry| entry.clone())
            .ok_or_else(|| GovernanceError::InternalError("Policy not found".to_string()))
    }
}

/// Policy update structure
#[derive(Debug, Clone)]
pub struct PolicyUpdate {
    pub name: Option<String>,
    pub description: Option<String>,
    pub rules: Option<Vec<PolicyRule>>,
    pub priority: Option<u32>,
    pub enabled: Option<bool>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    fn create_test_policy(name: &str, policy_type: PolicyType) -> Policy {
        Policy {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: "Test policy".to_string(),
            policy_type,
            scope: PolicyScope::Global,
            rules: vec![PolicyRule {
                condition: PolicyCondition::Always,
                action: PolicyAction::Allow,
                parameters: serde_json::json!({}),
            }],
            priority: 100,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: None,
        }
    }

    #[test]
    async fn test_policy_creation() {
        let manager = PolicyManager::new();
        let policy = create_test_policy("test_policy", PolicyType::Resource);

        let id = manager.create_policy(policy.clone()).await?;
        assert_ne!(id, Uuid::nil());

        let retrieved = manager.get_policy(id).await?;
        assert_eq!(retrieved.name, "test_policy");
    }

    #[test]
    async fn test_policy_update() {
        let manager = PolicyManager::new();
        let policy = create_test_policy("test_policy", PolicyType::Resource);
        let id = manager.create_policy(policy).await?;

        let update = PolicyUpdate {
            name: Some("updated_policy".to_string()),
            description: Some("Updated description".to_string()),
            rules: None,
            priority: Some(200),
            enabled: Some(false),
            expires_at: None,
        };

        manager.update_policy(id, update).await?;

        let updated = manager.get_policy(id).await?;
        assert_eq!(updated.name, "updated_policy");
        assert_eq!(updated.priority, 200);
        assert!(!updated.enabled);
    }

    #[test]
    async fn test_policy_deletion() {
        let manager = PolicyManager::new();
        let policy = create_test_policy("test_policy", PolicyType::Resource);
        let id = manager.create_policy(policy).await?;

        manager.delete_policy(id).await?;

        let result = manager.get_policy(id).await;
        assert!(result.is_err());
    }

    #[test]
    async fn test_global_policy_indexing() {
        let manager = PolicyManager::new();
        let policy = create_test_policy("global_policy", PolicyType::Security);

        manager.create_policy(policy).await?;

        assert_eq!(manager.global_policies.read().len(), 1);
    }

    #[test]
    async fn test_agent_specific_policy() {
        let manager = PolicyManager::new();
        let agent_id = AgentId::new();

        let mut policy = create_test_policy("agent_policy", PolicyType::Resource);
        policy.scope = PolicyScope::Agent(agent_id.clone());

        manager.create_policy(policy).await?;

        let policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "agent_policy");
    }

    #[test]
    async fn test_agent_group_policy() {
        let manager = PolicyManager::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let mut policy = create_test_policy("group_policy", PolicyType::Communication);
        policy.scope = PolicyScope::AgentGroup(vec![agent1.clone(), agent2.clone()]);

        manager.create_policy(policy).await?;

        let policies1 = manager
            .get_applicable_policies(&agent1, PolicyType::Communication)
            .await
            .unwrap();
        let policies2 = manager
            .get_applicable_policies(&agent2, PolicyType::Communication)
            .await
            .unwrap();

        assert_eq!(policies1.len(), 1);
        assert_eq!(policies2.len(), 1);
    }

    #[test]
    async fn test_policy_type_filtering() {
        let manager = PolicyManager::new();
        let agent_id = AgentId::new();

        // Create policies of different types
        let policy1 = create_test_policy("resource_policy", PolicyType::Resource);
        let policy2 = create_test_policy("security_policy", PolicyType::Security);

        manager.create_policy(policy1).await?;
        manager.create_policy(policy2).await?;

        let resource_policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        let security_policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Security)
            .await
            .unwrap();

        assert_eq!(resource_policies.len(), 1);
        assert_eq!(security_policies.len(), 1);
        assert_eq!(resource_policies[0].name, "resource_policy");
        assert_eq!(security_policies[0].name, "security_policy");
    }

    #[test]
    async fn test_policy_priority_ordering() {
        let manager = PolicyManager::new();
        let agent_id = AgentId::new();

        let mut policy1 = create_test_policy("low_priority", PolicyType::Resource);
        policy1.priority = 50;

        let mut policy2 = create_test_policy("high_priority", PolicyType::Resource);
        policy2.priority = 150;

        manager.create_policy(policy1).await?;
        manager.create_policy(policy2).await?;

        let policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        assert_eq!(policies[0].name, "high_priority");
        assert_eq!(policies[1].name, "low_priority");
    }

    #[test]
    async fn test_expired_policy_filtering() {
        let manager = PolicyManager::new();
        let agent_id = AgentId::new();

        let mut policy = create_test_policy("expired_policy", PolicyType::Resource);
        policy.expires_at = Some(Utc::now() - chrono::Duration::days(1));
        policy.enabled = true; // Even if enabled, should be filtered out

        // Need to bypass validation for testing
        let id = policy.id;
        manager.policies.insert(id, policy);
        manager.global_policies.write().push(id);

        let policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        assert_eq!(policies.len(), 0); // Expired policy should be filtered out
    }

    #[test]
    async fn test_disabled_policy_filtering() {
        let manager = PolicyManager::new();
        let agent_id = AgentId::new();

        let mut policy = create_test_policy("disabled_policy", PolicyType::Resource);
        policy.enabled = false;

        manager.create_policy(policy).await?;

        let policies = manager
            .get_applicable_policies(&agent_id, PolicyType::Resource)
            .await
            .unwrap();
        assert_eq!(policies.len(), 0); // Disabled policy should be filtered out
    }

    #[test]
    async fn test_resource_limit_condition() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("resource_limit", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::ResourceLimit {
                resource: "memory_mb".to_string(),
                operator: ComparisonOperator::LessThan,
                value: 1024.0,
            },
            action: PolicyAction::Allow,
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!({
            "memory_mb": 512.0
        });

        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(result);

        let context_exceed = serde_json::json!({
            "memory_mb": 2048.0
        });

        let result = manager
            .evaluate_policy(&policy, &context_exceed)
            .await
            .unwrap();
        assert!(result); // Default allow when condition not met
    }

    #[test]
    async fn test_time_window_condition() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("time_window", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::TimeWindow {
                start: Utc::now() - chrono::Duration::hours(1),
                end: Utc::now() + chrono::Duration::hours(1),
            },
            action: PolicyAction::Allow,
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!({});
        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(result);
    }

    #[test]
    async fn test_combined_condition_and() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("combined_and", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::Combined {
                operator: LogicalOperator::And,
                conditions: vec![
                    PolicyCondition::ResourceLimit {
                        resource: "memory_mb".to_string(),
                        operator: ComparisonOperator::LessThan,
                        value: 1024.0,
                    },
                    PolicyCondition::ResourceLimit {
                        resource: "cpu_cores".to_string(),
                        operator: ComparisonOperator::LessThan,
                        value: 2.0,
                    },
                ],
            },
            action: PolicyAction::Allow,
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!({
            "memory_mb": 512.0,
            "cpu_cores": 1.0
        });

        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(result);

        let context_fail = serde_json::json!({
            "memory_mb": 512.0,
            "cpu_cores": 4.0
        });

        let result = manager
            .evaluate_policy(&policy, &context_fail)
            .await
            .unwrap();
        assert!(result); // Default allow when condition not met
    }

    #[test]
    async fn test_combined_condition_or() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("combined_or", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::Combined {
                operator: LogicalOperator::Or,
                conditions: vec![
                    PolicyCondition::ResourceLimit {
                        resource: "memory_mb".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        value: 2048.0,
                    },
                    PolicyCondition::ResourceLimit {
                        resource: "cpu_cores".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        value: 4.0,
                    },
                ],
            },
            action: PolicyAction::Deny,
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!({
            "memory_mb": 3072.0,
            "cpu_cores": 2.0
        });

        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(!result); // Should deny
    }

    #[test]
    async fn test_deny_action() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("deny_policy", PolicyType::Security);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::Always,
            action: PolicyAction::Deny,
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!({});
        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(!result);
    }

    #[test]
    async fn test_limit_action() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("limit_policy", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::Always,
            action: PolicyAction::Limit(serde_json::json!(1024.0)),
            parameters: serde_json::json!({}),
        }];

        let context = serde_json::json!(512.0);
        let result = manager.evaluate_policy(&policy, &context).await?;
        assert!(result);

        let context_exceed = serde_json::json!(2048.0);
        let result = manager
            .evaluate_policy(&policy, &context_exceed)
            .await
            .unwrap();
        assert!(!result);
    }

    #[test]
    async fn test_conflict_resolution_priority_based() {
        let manager = PolicyManager::new();

        let mut policy1 = create_test_policy("low_priority", PolicyType::Resource);
        policy1.priority = 50;

        let mut policy2 = create_test_policy("high_priority", PolicyType::Resource);
        policy2.priority = 150;

        let policies = vec![&policy1, &policy2];
        let resolved = manager.resolve_conflicts(policies).await?;

        assert_eq!(resolved[0].name, "high_priority");
        assert_eq!(resolved[1].name, "low_priority");
    }

    #[test]
    async fn test_empty_policy_name_validation() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("", PolicyType::Resource);
        policy.name = "".to_string();

        let result = manager.create_policy(policy).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ConfigurationError(_))
        ));
    }

    #[test]
    async fn test_empty_rules_validation() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("no_rules", PolicyType::Resource);
        policy.rules = vec![];

        let result = manager.create_policy(policy).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ConfigurationError(_))
        ));
    }

    #[test]
    async fn test_time_window_validation() {
        let manager = PolicyManager::new();

        let mut policy = create_test_policy("invalid_time", PolicyType::Resource);
        policy.rules = vec![PolicyRule {
            condition: PolicyCondition::TimeWindow {
                start: Utc::now(),
                end: Utc::now() - chrono::Duration::hours(1), // End before start
            },
            action: PolicyAction::Allow,
            parameters: serde_json::json!({}),
        }];

        let result = manager.create_policy(policy).await;
        assert!(matches!(
            result,
            Err(GovernanceError::ConfigurationError(_))
        ));
    }
}
