use hpc_policy::{
    evaluate, parse_policy, Decision, EvaluationContext, PrincipalContext, ResourceContext,
};
use serde_json::json;

#[test]
fn test_end_to_end_rbac_policy() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: rbac-policy
  version: "1.0"
  description: "Simple RBAC policy for job submission"
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel, view]
    - effect: deny
      actions: [delete]
"#;

    let policy = parse_policy(yaml).unwrap();

    // Test allow case
    let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
    let context = EvaluationContext::new(principal.clone(), resource.clone(), "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test deny case
    let context = EvaluationContext::new(principal, resource, "delete".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}

#[test]
fn test_end_to_end_abac_policy() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: abac-gpu-policy
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
        - field: resource.priority
          operator: eq
          value: "normal"
    - effect: deny
      actions: [submit]
      conditions:
        - field: resource.gpu_count
          operator: gt
          value: 8
"#;

    let policy = parse_policy(yaml).unwrap();

    // Test allow case with conditions
    let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
        .with_attribute("gpu_count".to_string(), json!(4))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal.clone(), resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test deny case with exceeded GPU count
    let resource = ResourceContext::new("job".to_string(), "jobs/456".to_string())
        .with_attribute("gpu_count".to_string(), json!(16))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}

#[test]
fn test_end_to_end_multi_principal_policy() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-principal-policy
spec:
  principals:
    - type: role
      value: admin
    - type: role
      value: gpu-user
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

    // Test with admin role
    let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
    let context = EvaluationContext::new(principal, resource.clone(), "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with gpu-user role
    let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
    let context = EvaluationContext::new(principal, resource.clone(), "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with ml-team
    let principal = PrincipalContext::new(None, vec![], vec!["ml-team".to_string()]);
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);
}

#[test]
fn test_end_to_end_pattern_matching() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: pattern-matching-policy
spec:
  principals:
    - type: role
      pattern: "*-admin"
  resources:
    - type: job
      pattern: "jobs/team-*/*"
  rules:
    - effect: allow
      actions: [submit, view]
"#;

    let policy = parse_policy(yaml).unwrap();

    // Test with matching principal pattern
    let principal = PrincipalContext::new(None, vec!["system-admin".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/team-ml/123".to_string());
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with non-matching principal pattern
    let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/team-ml/123".to_string());
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);

    // Test with non-matching resource pattern
    let principal = PrincipalContext::new(None, vec!["system-admin".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}

#[test]
fn test_end_to_end_complex_conditions() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: complex-conditions-policy
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
          operator: gte
          value: 1
        - field: resource.gpu_count
          operator: lte
          value: 8
        - field: resource.memory_gb
          operator: lte
          value: 128
        - field: resource.priority
          operator: in
          value: ["normal", "low"]
"#;

    let policy = parse_policy(yaml).unwrap();

    // Test all conditions pass
    let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
        .with_attribute("gpu_count".to_string(), json!(4))
        .with_attribute("memory_gb".to_string(), json!(64))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal.clone(), resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test one condition fails (memory exceeds limit)
    let resource = ResourceContext::new("job".to_string(), "jobs/456".to_string())
        .with_attribute("gpu_count".to_string(), json!(4))
        .with_attribute("memory_gb".to_string(), json!(256))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}

#[test]
fn test_end_to_end_user_principal() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: user-policy
spec:
  principals:
    - type: user
      value: alice@example.com
    - type: user
      value: bob@example.com
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
"#;

    let policy = parse_policy(yaml).unwrap();

    // Test with alice
    let principal = PrincipalContext::new(Some("alice@example.com".to_string()), vec![], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
    let context = EvaluationContext::new(principal, resource.clone(), "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with bob
    let principal = PrincipalContext::new(Some("bob@example.com".to_string()), vec![], vec![]);
    let context = EvaluationContext::new(principal, resource.clone(), "cancel".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with unauthorized user
    let principal = PrincipalContext::new(Some("charlie@example.com".to_string()), vec![], vec![]);
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}

#[test]
fn test_end_to_end_multiple_resources() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-resource-policy
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
    - type: job
      pattern: "experiments/*"
    - type: cluster
      pattern: "clusters/*"
  rules:
    - effect: allow
      actions: [view, submit]
"#;

    let policy = parse_policy(yaml).unwrap();
    let principal = PrincipalContext::new(None, vec!["admin".to_string()], vec![]);

    // Test job resource
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string());
    let context = EvaluationContext::new(principal.clone(), resource, "view".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test experiment resource
    let resource = ResourceContext::new("job".to_string(), "experiments/exp-1".to_string());
    let context = EvaluationContext::new(principal.clone(), resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test cluster resource
    let resource = ResourceContext::new("cluster".to_string(), "clusters/prod".to_string());
    let context = EvaluationContext::new(principal, resource, "view".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);
}

#[test]
fn test_end_to_end_operator_ne() {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: ne-operator-policy
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
        - field: resource.status
          operator: ne
          value: "failed"
"#;

    let policy = parse_policy(yaml).unwrap();
    let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);

    // Test with non-failed status
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
        .with_attribute("status".to_string(), json!("pending"));
    let context = EvaluationContext::new(principal.clone(), resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Allow);

    // Test with failed status
    let resource = ResourceContext::new("job".to_string(), "jobs/456".to_string())
        .with_attribute("status".to_string(), json!("failed"));
    let context = EvaluationContext::new(principal, resource, "submit".to_string());
    assert_eq!(evaluate(&policy, &context).unwrap(), Decision::Deny);
}
