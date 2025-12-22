use crate::ast::{Decision, EvaluationContext};
use crate::error::{Error, Result};
use crate::matcher::{match_principal, match_resource};
use crate::policy::{Effect, Operator, Policy, Rule};

/// Evaluate a policy against an evaluation context
pub fn evaluate(policy: &Policy, context: &EvaluationContext) -> Result<Decision> {
    // Check if principal matches
    if !matches_any_principal(policy, context)? {
        return Ok(Decision::Deny);
    }

    // Check if resource matches
    if !matches_any_resource(policy, context)? {
        return Ok(Decision::Deny);
    }

    // Evaluate rules (first match wins)
    for rule in &policy.spec.rules {
        if matches_rule(rule, context)? {
            return match rule.effect {
                Effect::Allow => Ok(Decision::Allow),
                Effect::Deny => Ok(Decision::Deny),
            };
        }
    }

    // Default deny
    Ok(Decision::Deny)
}

/// Check if the context matches any principal in the policy
fn matches_any_principal(policy: &Policy, context: &EvaluationContext) -> Result<bool> {
    for principal in &policy.spec.principals {
        let principal_type = match principal.principal_type {
            crate::policy::PrincipalType::Role => "role",
            crate::policy::PrincipalType::User => "user",
            crate::policy::PrincipalType::Team => "team",
        };

        let matched = match_principal(
            principal_type,
            principal.value.as_deref(),
            principal.pattern.as_deref(),
            context.principal.user.as_deref(),
            &context.principal.roles,
            &context.principal.teams,
        );

        if matched {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if the context matches any resource in the policy
fn matches_any_resource(policy: &Policy, context: &EvaluationContext) -> Result<bool> {
    for resource in &policy.spec.resources {
        if resource.resource_type != context.resource.resource_type {
            continue;
        }

        if match_resource(&resource.pattern, &context.resource.resource_id)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if a rule matches the context
fn matches_rule(rule: &Rule, context: &EvaluationContext) -> Result<bool> {
    // Check if action matches
    if !rule.actions.iter().any(|a| a == &context.action) {
        return Ok(false);
    }

    // Evaluate all conditions
    for condition in &rule.conditions {
        if !evaluate_condition(condition, context)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Evaluate a single condition
fn evaluate_condition(
    condition: &crate::policy::Condition,
    context: &EvaluationContext,
) -> Result<bool> {
    let field_value = context.get_field(&condition.field);

    let field_value = match field_value {
        Some(v) => v,
        None => return Ok(false),
    };

    match condition.operator {
        Operator::Eq => Ok(field_value == condition.value),
        Operator::Ne => Ok(field_value != condition.value),
        Operator::Gt => compare_values(&field_value, &condition.value, |a, b| a > b),
        Operator::Lt => compare_values(&field_value, &condition.value, |a, b| a < b),
        Operator::Gte => compare_values(&field_value, &condition.value, |a, b| a >= b),
        Operator::Lte => compare_values(&field_value, &condition.value, |a, b| a <= b),
        Operator::In => {
            if let serde_json::Value::Array(arr) = &condition.value {
                // If field_value is an array, check if any element is in the condition array
                if let serde_json::Value::Array(field_arr) = &field_value {
                    Ok(field_arr.iter().any(|v| arr.contains(v)))
                } else {
                    Ok(arr.contains(&field_value))
                }
            } else {
                Ok(false)
            }
        }
        Operator::Contains => {
            if let (serde_json::Value::String(haystack), serde_json::Value::String(needle)) =
                (&field_value, &condition.value)
            {
                Ok(haystack.contains(needle))
            } else if let (serde_json::Value::Array(arr), _) = (&field_value, &condition.value) {
                Ok(arr.contains(&condition.value))
            } else {
                Ok(false)
            }
        }
    }
}

/// Compare two JSON values using a comparison function
fn compare_values<F>(a: &serde_json::Value, b: &serde_json::Value, cmp: F) -> Result<bool>
where
    F: Fn(f64, f64) -> bool,
{
    match (a, b) {
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            let v1 = n1.as_f64().ok_or_else(|| {
                Error::EvaluationError("Failed to convert number to f64".to_string())
            })?;
            let v2 = n2.as_f64().ok_or_else(|| {
                Error::EvaluationError("Failed to convert number to f64".to_string())
            })?;
            Ok(cmp(v1, v2))
        }
        _ => Err(Error::EvaluationError(
            "Cannot compare non-numeric values".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{PrincipalContext, ResourceContext};
    use crate::parser::parse_policy;
    use serde_json::json;

    #[test]
    fn test_evaluate_simple_allow() {
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
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_simple_deny() {
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
    - effect: deny
      actions: [delete]
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "delete".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_wrong_principal() {
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
        let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_wrong_resource() {
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
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("user".to_string(), "users/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_wrong_action() {
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
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "delete".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_condition_lte() {
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
        let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("gpu_count".to_string(), json!(4));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_condition_lte_fail() {
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
        let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("gpu_count".to_string(), json!(16));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_condition_eq() {
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
      conditions:
        - field: resource.priority
          operator: eq
          value: "high"
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("priority".to_string(), json!("high"));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_condition_in() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
      conditions:
        - field: principal.teams
          operator: in
          value: ["ml-team", "research-team"]
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(
            None,
            vec!["user".to_string()],
            vec!["ml-team".to_string()],
        );
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_multiple_conditions() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: user
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
        - field: resource.priority
          operator: eq
          value: "normal"
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("gpu_count".to_string(), json!(4))
            .with_attribute("priority".to_string(), json!("normal"));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_multiple_rules_first_match() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: role
      value: user
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
    - effect: deny
      actions: [submit]
      conditions:
        - field: resource.gpu_count
          operator: gt
          value: 8
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("gpu_count".to_string(), json!(4));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_default_deny() {
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
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "unknown-action".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Deny);
    }

    #[test]
    fn test_evaluate_user_principal() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: user
      value: user@example.com
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(Some("user@example.com".to_string()), vec![], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_team_principal() {
        let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: test-policy
spec:
  principals:
    - type: team
      value: ml-team
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec![], vec!["ml-team".to_string()]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_operator_gt() {
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
      conditions:
        - field: resource.priority
          operator: gt
          value: 5
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("priority".to_string(), json!(10));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }

    #[test]
    fn test_evaluate_operator_contains() {
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
      conditions:
        - field: resource.name
          operator: contains
          value: "test"
"#;

        let policy = parse_policy(yaml).unwrap();
        let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
        let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
            .with_attribute("name".to_string(), json!("my-test-job"));
        let context = EvaluationContext::new(principal, resource, "submit".to_string());

        let decision = evaluate(&policy, &context).unwrap();
        assert_eq!(decision, Decision::Allow);
    }
}
