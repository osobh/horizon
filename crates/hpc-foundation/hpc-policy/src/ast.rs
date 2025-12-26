use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation context containing principal, resource, and action information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationContext {
    pub principal: PrincipalContext,
    pub resource: ResourceContext,
    pub action: String,
}

/// Principal context for evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrincipalContext {
    pub user: Option<String>,
    pub roles: Vec<String>,
    pub teams: Vec<String>,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Resource context for evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceContext {
    pub resource_type: String,
    pub resource_id: String,
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Policy decision result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
}

impl EvaluationContext {
    /// Create a new evaluation context
    pub fn new(
        principal: PrincipalContext,
        resource: ResourceContext,
        action: String,
    ) -> Self {
        Self {
            principal,
            resource,
            action,
        }
    }

    /// Get a field value from the context
    pub fn get_field(&self, field: &str) -> Option<serde_json::Value> {
        let parts: Vec<&str> = field.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        match parts[0] {
            "principal" => self.get_principal_field(&parts[1..]),
            "resource" => self.get_resource_field(&parts[1..]),
            "action" => {
                if parts.len() == 1 {
                    Some(serde_json::Value::String(self.action.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn get_principal_field(&self, parts: &[&str]) -> Option<serde_json::Value> {
        if parts.is_empty() {
            return None;
        }

        match parts[0] {
            "user" => self
                .principal
                .user
                .as_ref()
                .map(|u| serde_json::Value::String(u.clone())),
            "roles" => Some(serde_json::json!(self.principal.roles)),
            "teams" => Some(serde_json::json!(self.principal.teams)),
            attr => self.principal.attributes.get(attr).cloned(),
        }
    }

    fn get_resource_field(&self, parts: &[&str]) -> Option<serde_json::Value> {
        if parts.is_empty() {
            return None;
        }

        match parts[0] {
            "type" => Some(serde_json::Value::String(
                self.resource.resource_type.clone(),
            )),
            "id" => Some(serde_json::Value::String(self.resource.resource_id.clone())),
            attr => self.resource.attributes.get(attr).cloned(),
        }
    }
}

impl PrincipalContext {
    /// Create a new principal context
    pub fn new(user: Option<String>, roles: Vec<String>, teams: Vec<String>) -> Self {
        Self {
            user,
            roles,
            teams,
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to the principal context
    pub fn with_attribute(mut self, key: String, value: serde_json::Value) -> Self {
        self.attributes.insert(key, value);
        self
    }
}

impl ResourceContext {
    /// Create a new resource context
    pub fn new(resource_type: String, resource_id: String) -> Self {
        Self {
            resource_type,
            resource_id,
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to the resource context
    pub fn with_attribute(mut self, key: String, value: serde_json::Value) -> Self {
        self.attributes.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_evaluation_context_creation() {
        let principal = PrincipalContext::new(
            Some("user@example.com".to_string()),
            vec!["admin".to_string()],
            vec!["ml-team".to_string()],
        );
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string());
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        assert_eq!(ctx.action, "submit");
    }

    #[test]
    fn test_principal_context_roles() {
        let principal = PrincipalContext::new(
            Some("user@example.com".to_string()),
            vec!["admin".to_string(), "gpu-user".to_string()],
            vec![],
        );
        assert_eq!(principal.roles.len(), 2);
        assert!(principal.roles.contains(&"admin".to_string()));
    }

    #[test]
    fn test_principal_context_teams() {
        let principal = PrincipalContext::new(
            None,
            vec![],
            vec!["ml-team".to_string(), "research-team".to_string()],
        );
        assert_eq!(principal.teams.len(), 2);
    }

    #[test]
    fn test_principal_context_with_attribute() {
        let principal = PrincipalContext::new(Some("user@example.com".to_string()), vec![], vec![])
            .with_attribute("department".to_string(), json!("engineering"));

        assert_eq!(
            principal.attributes.get("department"),
            Some(&json!("engineering"))
        );
    }

    #[test]
    fn test_resource_context_with_attribute() {
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string())
            .with_attribute("gpu_count".to_string(), json!(8));

        assert_eq!(resource.attributes.get("gpu_count"), Some(&json!(8)));
    }

    #[test]
    fn test_get_field_principal_user() {
        let principal = PrincipalContext::new(Some("user@example.com".to_string()), vec![], vec![]);
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string());
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        let value = ctx.get_field("principal.user");
        assert_eq!(value, Some(json!("user@example.com")));
    }

    #[test]
    fn test_get_field_principal_roles() {
        let principal = PrincipalContext::new(
            None,
            vec!["admin".to_string(), "gpu-user".to_string()],
            vec![],
        );
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string());
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        let value = ctx.get_field("principal.roles");
        assert_eq!(value, Some(json!(["admin", "gpu-user"])));
    }

    #[test]
    fn test_get_field_resource_type() {
        let principal = PrincipalContext::new(None, vec![], vec![]);
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string());
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        let value = ctx.get_field("resource.type");
        assert_eq!(value, Some(json!("job")));
    }

    #[test]
    fn test_get_field_resource_attribute() {
        let principal = PrincipalContext::new(None, vec![], vec![]);
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string())
            .with_attribute("gpu_count".to_string(), json!(8));
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        let value = ctx.get_field("resource.gpu_count");
        assert_eq!(value, Some(json!(8)));
    }

    #[test]
    fn test_get_field_invalid() {
        let principal = PrincipalContext::new(None, vec![], vec![]);
        let resource = ResourceContext::new("job".to_string(), "job-123".to_string());
        let ctx = EvaluationContext::new(principal, resource, "submit".to_string());

        let value = ctx.get_field("invalid.field");
        assert_eq!(value, None);
    }

    #[test]
    fn test_decision_allow() {
        let decision = Decision::Allow;
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_decision_deny() {
        let decision = Decision::Deny;
        assert_eq!(decision, Decision::Deny);
    }
}
