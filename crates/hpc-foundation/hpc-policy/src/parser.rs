use crate::error::{Error, Result};
use crate::policy::Policy;

/// Parse a policy from YAML string
pub fn parse_policy(yaml: &str) -> Result<Policy> {
    let policy: Policy = serde_yaml::from_str(yaml)?;
    validate_policy(&policy)?;
    Ok(policy)
}

/// Validate a policy document
fn validate_policy(policy: &Policy) -> Result<()> {
    // Validate API version
    if policy.api_version != "policy.horizon.dev/v1" {
        return Err(Error::UnsupportedVersion(policy.api_version.clone()));
    }

    // Validate kind
    if policy.kind != "Policy" {
        return Err(Error::ValidationError(format!(
            "Invalid kind: expected 'Policy', got '{}'",
            policy.kind
        )));
    }

    // Validate metadata
    if policy.metadata.name.is_empty() {
        return Err(Error::MissingField("metadata.name".to_string()));
    }

    // Validate spec
    if policy.spec.principals.is_empty() {
        return Err(Error::ValidationError(
            "spec.principals must not be empty".to_string(),
        ));
    }

    if policy.spec.resources.is_empty() {
        return Err(Error::ValidationError(
            "spec.resources must not be empty".to_string(),
        ));
    }

    if policy.spec.rules.is_empty() {
        return Err(Error::ValidationError(
            "spec.rules must not be empty".to_string(),
        ));
    }

    // Validate principals
    for (i, principal) in policy.spec.principals.iter().enumerate() {
        if principal.value.is_none() && principal.pattern.is_none() {
            return Err(Error::ValidationError(format!(
                "spec.principals[{}]: must have either 'value' or 'pattern'",
                i
            )));
        }
    }

    // Validate resources
    for (i, resource) in policy.spec.resources.iter().enumerate() {
        if resource.resource_type.is_empty() {
            return Err(Error::ValidationError(format!(
                "spec.resources[{}].type must not be empty",
                i
            )));
        }
        if resource.pattern.is_empty() {
            return Err(Error::ValidationError(format!(
                "spec.resources[{}].pattern must not be empty",
                i
            )));
        }
    }

    // Validate rules
    for (i, rule) in policy.spec.rules.iter().enumerate() {
        if rule.actions.is_empty() {
            return Err(Error::ValidationError(format!(
                "spec.rules[{}].actions must not be empty",
                i
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_policy() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let policy = parse_policy(yaml).unwrap();
        assert_eq!(policy.metadata.name, "test-policy");
        assert_eq!(policy.spec.principals.len(), 1);
        assert_eq!(policy.spec.resources.len(), 1);
        assert_eq!(policy.spec.rules.len(), 1);
    }

    #[test]
    fn test_parse_policy_with_conditions() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: gpu-policy
spec:
  principals:
    - type: role
      value: gpu-user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
      conditions:
        - field: resource.gpu_count
          operator: lte
          value: 8
"#;

        let policy = parse_policy(yaml).unwrap();
        assert_eq!(policy.spec.rules[0].conditions.len(), 1);
    }

    #[test]
    fn test_parse_policy_with_multiple_rules() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-rule-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
    - effect: deny
      actions: [delete]
"#;

        let policy = parse_policy(yaml).unwrap();
        assert_eq!(policy.spec.rules.len(), 2);
    }

    #[test]
    fn test_parse_invalid_yaml() {
        let yaml = "invalid: yaml: syntax:";
        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unsupported_api_version() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v2
kind: Policy
metadata:
  name: test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
        match result {
            Err(Error::UnsupportedVersion(v)) => {
                assert_eq!(v, "policy.horizon.dev/v2");
            }
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }

    #[test]
    fn test_parse_invalid_kind() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: InvalidKind
metadata:
  name: test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_metadata_name() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: ""
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_principals() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
spec:
  principals: []
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_resources() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
spec:
  principals:
    - type: role
      value: admin
  resources: []
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_rules() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules: []
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_principal_without_value_or_pattern() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
spec:
  principals:
    - type: role
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_rule_without_actions() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: []
"#;

        let result = parse_policy(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_policy_with_version() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
  version: "1.0"
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let policy = parse_policy(yaml).unwrap();
        assert_eq!(policy.metadata.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_parse_policy_with_description() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test
  description: "Test policy"
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let policy = parse_policy(yaml).unwrap();
        assert_eq!(policy.metadata.description, Some("Test policy".to_string()));
    }
}
