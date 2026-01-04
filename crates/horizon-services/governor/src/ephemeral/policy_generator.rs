//! Ephemeral policy generator for dynamic policy creation.
//!
//! Generates Governor policies for ephemeral access scenarios:
//! - Per-user ephemeral identity policies
//! - Time-bounded quota access policies
//! - Resource pool participation policies

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for generating an ephemeral policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EphemeralPolicyConfig {
    /// Unique identifier for this ephemeral access
    pub ephemeral_id: Uuid,
    /// The principal (user) this policy applies to
    pub principal_id: String,
    /// Type of ephemeral access
    pub policy_type: EphemeralPolicyType,
    /// Scope of resources the policy applies to
    pub scope: PolicyScope,
    /// Time window configuration
    pub time_window: Option<TimeWindowConfig>,
    /// Risk-based access configuration
    pub risk_access: RiskBasedAccess,
    /// When this policy should expire
    pub expires_at: DateTime<Utc>,
    /// Sponsor who created this ephemeral access
    pub sponsor_id: String,
    /// Organization/tenant context
    pub tenant_id: Uuid,
}

/// Type of ephemeral policy to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EphemeralPolicyType {
    /// Identity-based access (for ephemeral identities)
    Identity,
    /// Quota-based access (for ephemeral quotas)
    Quota,
    /// Pool-based access (for resource pools)
    Pool,
    /// Federated training access
    Federated,
}

/// Scope of resources the policy applies to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyScope {
    /// Resource types this policy covers
    pub resource_types: Vec<String>,
    /// Resource patterns (glob patterns)
    pub resource_patterns: Vec<String>,
    /// Allowed actions
    pub allowed_actions: Vec<String>,
    /// Explicitly denied actions
    pub denied_actions: Vec<String>,
}

impl Default for PolicyScope {
    fn default() -> Self {
        Self {
            resource_types: vec!["job".to_string(), "notebook".to_string()],
            resource_patterns: vec!["ephemeral/*".to_string()],
            allowed_actions: vec!["view".to_string(), "submit".to_string()],
            denied_actions: vec!["delete".to_string()],
        }
    }
}

impl PolicyScope {
    /// Create a scope for collaboration access.
    pub fn collaboration() -> Self {
        Self {
            resource_types: vec!["notebook".to_string(), "session".to_string()],
            resource_patterns: vec!["*".to_string()],
            allowed_actions: vec![
                "view".to_string(),
                "edit".to_string(),
                "collaborate".to_string(),
            ],
            denied_actions: vec!["delete".to_string(), "admin".to_string()],
        }
    }

    /// Create a scope for training access.
    pub fn training() -> Self {
        Self {
            resource_types: vec![
                "training_run".to_string(),
                "model".to_string(),
                "checkpoint".to_string(),
            ],
            resource_patterns: vec!["*".to_string()],
            allowed_actions: vec![
                "view".to_string(),
                "submit".to_string(),
                "download".to_string(),
            ],
            denied_actions: vec!["delete".to_string()],
        }
    }

    /// Create a scope for read-only access.
    pub fn read_only() -> Self {
        Self {
            resource_types: vec!["*".to_string()],
            resource_patterns: vec!["*".to_string()],
            allowed_actions: vec!["view".to_string(), "read".to_string()],
            denied_actions: vec![
                "write".to_string(),
                "delete".to_string(),
                "execute".to_string(),
            ],
        }
    }
}

/// Time window configuration for when access is allowed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindowConfig {
    /// Start time in HH:MM format
    pub start_time: String,
    /// End time in HH:MM format
    pub end_time: String,
    /// Timezone (e.g., "UTC", "America/New_York")
    pub timezone: String,
    /// Allowed days of the week (0 = Monday, 6 = Sunday)
    pub allowed_days: Vec<u8>,
}

impl Default for TimeWindowConfig {
    fn default() -> Self {
        Self {
            start_time: "00:00".to_string(),
            end_time: "23:59".to_string(),
            timezone: "UTC".to_string(),
            allowed_days: vec![0, 1, 2, 3, 4, 5, 6], // All days
        }
    }
}

impl TimeWindowConfig {
    /// Create business hours configuration (9 AM - 5 PM, Mon-Fri).
    pub fn business_hours() -> Self {
        Self {
            start_time: "09:00".to_string(),
            end_time: "17:00".to_string(),
            timezone: "UTC".to_string(),
            allowed_days: vec![0, 1, 2, 3, 4], // Mon-Fri
        }
    }
}

/// Risk-based access configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskBasedAccess {
    /// Allow full access for low risk
    pub low_risk_actions: Vec<String>,
    /// Allow limited access for medium risk
    pub medium_risk_actions: Vec<String>,
    /// Allow minimal access for high risk
    pub high_risk_actions: Vec<String>,
    /// Risk threshold for suspension (0.0-1.0)
    pub suspension_threshold: f64,
    /// Risk threshold for revocation (0.0-1.0)
    pub revocation_threshold: f64,
}

impl Default for RiskBasedAccess {
    fn default() -> Self {
        Self {
            low_risk_actions: vec![
                "read".to_string(),
                "write".to_string(),
                "execute".to_string(),
                "collaborate".to_string(),
            ],
            medium_risk_actions: vec!["read".to_string(), "collaborate".to_string()],
            high_risk_actions: vec!["read".to_string()],
            suspension_threshold: 0.6,
            revocation_threshold: 0.85,
        }
    }
}

/// A generated policy ready for creation in Governor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPolicy {
    /// Policy name (unique identifier)
    pub name: String,
    /// Policy content (YAML format)
    pub content: String,
    /// Description
    pub description: String,
    /// When this policy should expire
    pub expires_at: DateTime<Utc>,
    /// The ephemeral ID this policy is for
    pub ephemeral_id: Uuid,
}

/// Generator for ephemeral access policies.
pub struct EphemeralPolicyGenerator;

impl EphemeralPolicyGenerator {
    /// Generate a policy from configuration.
    pub fn generate(config: &EphemeralPolicyConfig) -> GeneratedPolicy {
        let name = format!(
            "ephemeral-{}-{}",
            config.policy_type.as_str(),
            config
                .ephemeral_id
                .to_string()
                .split('-')
                .next()
                .unwrap_or("unknown")
        );

        let description = format!(
            "Ephemeral {} access for {} sponsored by {} (expires: {})",
            config.policy_type.as_str(),
            config.principal_id,
            config.sponsor_id,
            config.expires_at.format("%Y-%m-%d %H:%M UTC")
        );

        let content = Self::generate_yaml(config);

        GeneratedPolicy {
            name,
            content,
            description,
            expires_at: config.expires_at,
            ephemeral_id: config.ephemeral_id,
        }
    }

    /// Generate the YAML policy content.
    fn generate_yaml(config: &EphemeralPolicyConfig) -> String {
        let resources_yaml = Self::format_resources(&config.scope);
        let rules_yaml = Self::format_rules(config);

        format!(
            r#"apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: ephemeral-{policy_type}-{id}
  version: "1"
  description: "Auto-generated ephemeral {policy_type} access for {principal}"
spec:
  principals:
    - type: user
      value: "{principal}"
{resources}
{rules}
"#,
            policy_type = config.policy_type.as_str(),
            id = config
                .ephemeral_id
                .to_string()
                .split('-')
                .next()
                .unwrap_or("unknown"),
            principal = config.principal_id,
            resources = resources_yaml,
            rules = rules_yaml,
        )
    }

    /// Format resources section of the policy.
    fn format_resources(scope: &PolicyScope) -> String {
        let mut lines = Vec::new();
        lines.push("  resources:".to_string());

        for (rt, pattern) in scope
            .resource_types
            .iter()
            .zip(scope.resource_patterns.iter())
        {
            lines.push(format!("    - type: {}", rt));
            lines.push(format!("      pattern: \"{}\"", pattern));
        }

        lines.join("\n")
    }

    /// Format rules section of the policy.
    fn format_rules(config: &EphemeralPolicyConfig) -> String {
        let mut lines = Vec::new();
        lines.push("  rules:".to_string());

        // Allow rule with conditions based on policy type
        lines.push("    # Allow access when conditions are met".to_string());
        lines.push("    - effect: allow".to_string());
        lines.push(format!(
            "      actions: [{}]",
            config.scope.allowed_actions.join(", ")
        ));
        lines.push("      conditions:".to_string());

        match config.policy_type {
            EphemeralPolicyType::Identity => {
                lines.push("        - field: principal.attributes.ephemeral_identity".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: true".to_string());
                lines.push("        - field: principal.attributes.identity_state".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: \"active\"".to_string());
            }
            EphemeralPolicyType::Quota => {
                lines.push("        - field: principal.attributes.has_ephemeral_quota".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: true".to_string());
                lines.push("        - field: principal.attributes.ephemeral_status".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: \"active\"".to_string());
            }
            EphemeralPolicyType::Pool => {
                lines.push("        - field: principal.attributes.pool_member".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: true".to_string());
                lines.push("        - field: principal.attributes.pool_status".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: \"active\"".to_string());
            }
            EphemeralPolicyType::Federated => {
                lines.push(
                    "        - field: principal.attributes.federated_participant".to_string(),
                );
                lines.push("          operator: eq".to_string());
                lines.push("          value: true".to_string());
                lines.push("        - field: principal.attributes.session_active".to_string());
                lines.push("          operator: eq".to_string());
                lines.push("          value: true".to_string());
            }
        }

        // Add time window condition if configured
        if config.time_window.is_some() {
            lines.push("        - field: principal.attributes.in_time_window".to_string());
            lines.push("          operator: eq".to_string());
            lines.push("          value: true".to_string());
        }

        // Add deny rule for expired/revoked access
        lines.push("".to_string());
        lines.push("    # Deny access when status is not active".to_string());
        lines.push("    - effect: deny".to_string());
        lines.push(format!(
            "      actions: [{}]",
            config.scope.allowed_actions.join(", ")
        ));
        lines.push("      conditions:".to_string());

        let status_field = match config.policy_type {
            EphemeralPolicyType::Identity => "principal.attributes.identity_state",
            EphemeralPolicyType::Quota => "principal.attributes.ephemeral_status",
            EphemeralPolicyType::Pool => "principal.attributes.pool_status",
            EphemeralPolicyType::Federated => "principal.attributes.session_active",
        };

        lines.push(format!("        - field: {}", status_field));

        if config.policy_type == EphemeralPolicyType::Federated {
            lines.push("          operator: eq".to_string());
            lines.push("          value: false".to_string());
        } else {
            lines.push("          operator: in".to_string());
            lines.push("          value: [\"expired\", \"revoked\", \"suspended\"]".to_string());
        }

        lines.join("\n")
    }

    /// Generate a minimal read-only policy for auditors.
    pub fn generate_auditor_policy(
        ephemeral_id: Uuid,
        principal_id: &str,
        expires_at: DateTime<Utc>,
    ) -> GeneratedPolicy {
        let config = EphemeralPolicyConfig {
            ephemeral_id,
            principal_id: principal_id.to_string(),
            policy_type: EphemeralPolicyType::Identity,
            scope: PolicyScope::read_only(),
            time_window: None,
            risk_access: RiskBasedAccess::default(),
            expires_at,
            sponsor_id: "system".to_string(),
            tenant_id: Uuid::nil(),
        };

        Self::generate(&config)
    }

    /// Generate a collaboration policy for notebook sharing.
    pub fn generate_collaboration_policy(
        ephemeral_id: Uuid,
        principal_id: &str,
        notebook_id: &str,
        expires_at: DateTime<Utc>,
        sponsor_id: &str,
    ) -> GeneratedPolicy {
        let mut scope = PolicyScope::collaboration();
        scope.resource_patterns = vec![format!("notebooks/{}", notebook_id)];

        let config = EphemeralPolicyConfig {
            ephemeral_id,
            principal_id: principal_id.to_string(),
            policy_type: EphemeralPolicyType::Identity,
            scope,
            time_window: None,
            risk_access: RiskBasedAccess::default(),
            expires_at,
            sponsor_id: sponsor_id.to_string(),
            tenant_id: Uuid::nil(),
        };

        Self::generate(&config)
    }
}

impl EphemeralPolicyType {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            EphemeralPolicyType::Identity => "identity",
            EphemeralPolicyType::Quota => "quota",
            EphemeralPolicyType::Pool => "pool",
            EphemeralPolicyType::Federated => "federated",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_identity_policy() {
        let config = EphemeralPolicyConfig {
            ephemeral_id: Uuid::new_v4(),
            principal_id: "user@example.com".to_string(),
            policy_type: EphemeralPolicyType::Identity,
            scope: PolicyScope::default(),
            time_window: None,
            risk_access: RiskBasedAccess::default(),
            expires_at: Utc::now() + chrono::Duration::hours(24),
            sponsor_id: "sponsor@example.com".to_string(),
            tenant_id: Uuid::new_v4(),
        };

        let policy = EphemeralPolicyGenerator::generate(&config);

        assert!(policy.name.starts_with("ephemeral-identity-"));
        assert!(policy.content.contains("apiVersion: policy.horizon.dev/v1"));
        assert!(policy.content.contains("user@example.com"));
        assert!(policy.content.contains("ephemeral_identity"));
    }

    #[test]
    fn test_generate_quota_policy() {
        let config = EphemeralPolicyConfig {
            ephemeral_id: Uuid::new_v4(),
            principal_id: "user@example.com".to_string(),
            policy_type: EphemeralPolicyType::Quota,
            scope: PolicyScope::training(),
            time_window: Some(TimeWindowConfig::business_hours()),
            risk_access: RiskBasedAccess::default(),
            expires_at: Utc::now() + chrono::Duration::days(7),
            sponsor_id: "sponsor@example.com".to_string(),
            tenant_id: Uuid::new_v4(),
        };

        let policy = EphemeralPolicyGenerator::generate(&config);

        assert!(policy.name.starts_with("ephemeral-quota-"));
        assert!(policy.content.contains("has_ephemeral_quota"));
        assert!(policy.content.contains("in_time_window"));
    }

    #[test]
    fn test_generate_pool_policy() {
        let config = EphemeralPolicyConfig {
            ephemeral_id: Uuid::new_v4(),
            principal_id: "contributor@example.com".to_string(),
            policy_type: EphemeralPolicyType::Pool,
            scope: PolicyScope::default(),
            time_window: None,
            risk_access: RiskBasedAccess::default(),
            expires_at: Utc::now() + chrono::Duration::hours(4),
            sponsor_id: "hackathon@example.com".to_string(),
            tenant_id: Uuid::new_v4(),
        };

        let policy = EphemeralPolicyGenerator::generate(&config);

        assert!(policy.name.starts_with("ephemeral-pool-"));
        assert!(policy.content.contains("pool_member"));
    }

    #[test]
    fn test_generate_federated_policy() {
        let config = EphemeralPolicyConfig {
            ephemeral_id: Uuid::new_v4(),
            principal_id: "node123".to_string(),
            policy_type: EphemeralPolicyType::Federated,
            scope: PolicyScope::training(),
            time_window: None,
            risk_access: RiskBasedAccess::default(),
            expires_at: Utc::now() + chrono::Duration::hours(2),
            sponsor_id: "training-coordinator".to_string(),
            tenant_id: Uuid::new_v4(),
        };

        let policy = EphemeralPolicyGenerator::generate(&config);

        assert!(policy.name.starts_with("ephemeral-federated-"));
        assert!(policy.content.contains("federated_participant"));
        assert!(policy.content.contains("session_active"));
    }

    #[test]
    fn test_generate_auditor_policy() {
        let policy = EphemeralPolicyGenerator::generate_auditor_policy(
            Uuid::new_v4(),
            "auditor@compliance.com",
            Utc::now() + chrono::Duration::hours(8),
        );

        assert!(policy.content.contains("view"));
        assert!(policy.content.contains("read"));
    }

    #[test]
    fn test_generate_collaboration_policy() {
        let policy = EphemeralPolicyGenerator::generate_collaboration_policy(
            Uuid::new_v4(),
            "guest@example.com",
            "notebook-123",
            Utc::now() + chrono::Duration::hours(2),
            "owner@example.com",
        );

        assert!(policy.content.contains("notebooks/notebook-123"));
        assert!(policy.content.contains("collaborate"));
    }

    #[test]
    fn test_policy_scope_collaboration() {
        let scope = PolicyScope::collaboration();
        assert!(scope.resource_types.contains(&"notebook".to_string()));
        assert!(scope.allowed_actions.contains(&"collaborate".to_string()));
        assert!(scope.denied_actions.contains(&"delete".to_string()));
    }

    #[test]
    fn test_policy_scope_training() {
        let scope = PolicyScope::training();
        assert!(scope.resource_types.contains(&"training_run".to_string()));
        assert!(scope.allowed_actions.contains(&"submit".to_string()));
    }

    #[test]
    fn test_policy_scope_read_only() {
        let scope = PolicyScope::read_only();
        assert!(scope.allowed_actions.contains(&"view".to_string()));
        assert!(scope.denied_actions.contains(&"write".to_string()));
    }

    #[test]
    fn test_time_window_business_hours() {
        let tw = TimeWindowConfig::business_hours();
        assert_eq!(tw.start_time, "09:00");
        assert_eq!(tw.end_time, "17:00");
        assert_eq!(tw.allowed_days.len(), 5);
    }

    #[test]
    fn test_risk_based_access_default() {
        let risk = RiskBasedAccess::default();
        assert!(risk.low_risk_actions.contains(&"execute".to_string()));
        assert!(!risk.high_risk_actions.contains(&"execute".to_string()));
        assert!(risk.high_risk_actions.contains(&"read".to_string()));
    }
}
