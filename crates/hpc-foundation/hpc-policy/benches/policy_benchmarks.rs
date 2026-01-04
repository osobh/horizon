use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hpc_policy::{evaluate, parse_policy, EvaluationContext, PrincipalContext, ResourceContext};
use serde_json::json;

fn bench_parse_simple_policy(c: &mut Criterion) {
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

    c.bench_function("parse_simple_policy", |b| {
        b.iter(|| parse_policy(black_box(yaml)))
    });
}

fn bench_parse_complex_policy(c: &mut Criterion) {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: complex-policy
  version: "1.0"
  description: "Complex ABAC policy with multiple conditions"
spec:
  principals:
    - type: role
      value: gpu-user
    - type: team
      value: ml-team
  resources:
    - type: job
      pattern: "jobs/*"
    - type: job
      pattern: "experiments/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
      conditions:
        - field: resource.gpu_count
          operator: lte
          value: 8
        - field: resource.priority
          operator: in
          value: ["normal", "low"]
    - effect: deny
      actions: [delete]
"#;

    c.bench_function("parse_complex_policy", |b| {
        b.iter(|| parse_policy(black_box(yaml)))
    });
}

fn bench_evaluate_simple_allow(c: &mut Criterion) {
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

    c.bench_function("evaluate_simple_allow", |b| {
        b.iter(|| evaluate(black_box(&policy), black_box(&context)))
    });
}

fn bench_evaluate_with_conditions(c: &mut Criterion) {
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
        - field: resource.priority
          operator: eq
          value: "normal"
"#;

    let policy = parse_policy(yaml).unwrap();
    let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
        .with_attribute("gpu_count".to_string(), json!(4))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal, resource, "submit".to_string());

    c.bench_function("evaluate_with_conditions", |b| {
        b.iter(|| evaluate(black_box(&policy), black_box(&context)))
    });
}

fn bench_evaluate_multiple_rules(c: &mut Criterion) {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: multi-rule-policy
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
          value: 4
    - effect: allow
      actions: [submit]
      conditions:
        - field: resource.gpu_count
          operator: lte
          value: 8
        - field: resource.priority
          operator: eq
          value: "high"
    - effect: deny
      actions: [submit]
"#;

    let policy = parse_policy(yaml).unwrap();
    let principal = PrincipalContext::new(None, vec!["user".to_string()], vec![]);
    let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
        .with_attribute("gpu_count".to_string(), json!(2))
        .with_attribute("priority".to_string(), json!("normal"));
    let context = EvaluationContext::new(principal, resource, "submit".to_string());

    c.bench_function("evaluate_multiple_rules", |b| {
        b.iter(|| evaluate(black_box(&policy), black_box(&context)))
    });
}

fn bench_pattern_matching(c: &mut Criterion) {
    let yaml = r#"
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: pattern-policy
spec:
  principals:
    - type: role
      pattern: "*-admin"
  resources:
    - type: job
      pattern: "jobs/team-*/project-*/*"
  rules:
    - effect: allow
      actions: [submit]
"#;

    let policy = parse_policy(yaml).unwrap();
    let principal = PrincipalContext::new(None, vec!["system-admin".to_string()], vec![]);
    let resource = ResourceContext::new(
        "job".to_string(),
        "jobs/team-ml/project-gpu/123".to_string(),
    );
    let context = EvaluationContext::new(principal, resource, "submit".to_string());

    c.bench_function("pattern_matching", |b| {
        b.iter(|| evaluate(black_box(&policy), black_box(&context)))
    });
}

criterion_group!(
    benches,
    bench_parse_simple_policy,
    bench_parse_complex_policy,
    bench_evaluate_simple_allow,
    bench_evaluate_with_conditions,
    bench_evaluate_multiple_rules,
    bench_pattern_matching
);
criterion_main!(benches);
