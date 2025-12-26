use serde::{Deserialize, Serialize};

/// Policy document structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    #[serde(rename = "apiVersion")]
    pub api_version: String,
    pub kind: String,
    pub metadata: PolicyMetadata,
    pub spec: PolicySpec,
}

/// Policy metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Policy specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicySpec {
    pub principals: Vec<Principal>,
    pub resources: Vec<Resource>,
    pub rules: Vec<Rule>,
}

/// Principal (user, role, team)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Principal {
    #[serde(rename = "type")]
    pub principal_type: PrincipalType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

/// Principal type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrincipalType {
    User,
    Role,
    Team,
}

/// Resource definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resource {
    #[serde(rename = "type")]
    pub resource_type: String,
    pub pattern: String,
}

/// Policy rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub effect: Effect,
    pub actions: Vec<String>,
    #[serde(default)]
    pub conditions: Vec<Condition>,
}

/// Rule effect
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effect {
    Allow,
    Deny,
}

/// Condition for rule evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub operator: Operator,
    pub value: serde_json::Value,
}

/// Condition operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operator {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    In,
    Contains,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_policy_creation() {
        let policy = Policy {
            api_version: "policy.horizon.dev/v1".to_string(),
            kind: "Policy".to_string(),
            metadata: PolicyMetadata {
                name: "test-policy".to_string(),
                version: Some("1.0".to_string()),
                description: Some("Test policy".to_string()),
            },
            spec: PolicySpec {
                principals: vec![Principal {
                    principal_type: PrincipalType::Role,
                    value: Some("admin".to_string()),
                    pattern: None,
                }],
                resources: vec![Resource {
                    resource_type: "job".to_string(),
                    pattern: "jobs/*".to_string(),
                }],
                rules: vec![Rule {
                    effect: Effect::Allow,
                    actions: vec!["submit".to_string()],
                    conditions: vec![],
                }],
            },
        };

        assert_eq!(policy.metadata.name, "test-policy");
        assert_eq!(policy.spec.principals.len(), 1);
        assert_eq!(policy.spec.resources.len(), 1);
        assert_eq!(policy.spec.rules.len(), 1);
    }

    #[test]
    fn test_principal_type_role() {
        let principal = Principal {
            principal_type: PrincipalType::Role,
            value: Some("admin".to_string()),
            pattern: None,
        };
        assert_eq!(principal.principal_type, PrincipalType::Role);
    }

    #[test]
    fn test_principal_type_user() {
        let principal = Principal {
            principal_type: PrincipalType::User,
            value: Some("user@example.com".to_string()),
            pattern: None,
        };
        assert_eq!(principal.principal_type, PrincipalType::User);
    }

    #[test]
    fn test_principal_type_team() {
        let principal = Principal {
            principal_type: PrincipalType::Team,
            value: Some("ml-team".to_string()),
            pattern: None,
        };
        assert_eq!(principal.principal_type, PrincipalType::Team);
    }

    #[test]
    fn test_effect_allow() {
        let rule = Rule {
            effect: Effect::Allow,
            actions: vec!["submit".to_string()],
            conditions: vec![],
        };
        assert_eq!(rule.effect, Effect::Allow);
    }

    #[test]
    fn test_effect_deny() {
        let rule = Rule {
            effect: Effect::Deny,
            actions: vec!["delete".to_string()],
            conditions: vec![],
        };
        assert_eq!(rule.effect, Effect::Deny);
    }

    #[test]
    fn test_rule_with_conditions() {
        let rule = Rule {
            effect: Effect::Allow,
            actions: vec!["submit".to_string()],
            conditions: vec![Condition {
                field: "resource.gpu_count".to_string(),
                operator: Operator::Lte,
                value: json!(8),
            }],
        };
        assert_eq!(rule.conditions.len(), 1);
        assert_eq!(rule.conditions[0].operator, Operator::Lte);
    }

    #[test]
    fn test_operator_eq() {
        let condition = Condition {
            field: "principal.role".to_string(),
            operator: Operator::Eq,
            value: json!("admin"),
        };
        assert_eq!(condition.operator, Operator::Eq);
    }

    #[test]
    fn test_operator_in() {
        let condition = Condition {
            field: "principal.team".to_string(),
            operator: Operator::In,
            value: json!(["ml-team", "research-team"]),
        };
        assert_eq!(condition.operator, Operator::In);
    }

    #[test]
    fn test_resource_pattern() {
        let resource = Resource {
            resource_type: "job".to_string(),
            pattern: "jobs/*".to_string(),
        };
        assert_eq!(resource.pattern, "jobs/*");
    }

    #[test]
    fn test_policy_serialization() {
        let policy = Policy {
            api_version: "policy.horizon.dev/v1".to_string(),
            kind: "Policy".to_string(),
            metadata: PolicyMetadata {
                name: "test".to_string(),
                version: None,
                description: None,
            },
            spec: PolicySpec {
                principals: vec![],
                resources: vec![],
                rules: vec![],
            },
        };

        let json = serde_json::to_string(&policy).unwrap();
        assert!(json.contains("test"));
    }

    #[test]
    fn test_policy_deserialization() {
        let json = r#"{
            "apiVersion": "policy.horizon.dev/v1",
            "kind": "Policy",
            "metadata": {"name": "test"},
            "spec": {
                "principals": [],
                "resources": [],
                "rules": []
            }
        }"#;

        let policy: Policy = serde_json::from_str(json).unwrap();
        assert_eq!(policy.metadata.name, "test");
    }
}
