# Horizon PolicyX

A high-performance declarative policy engine for authorization and access control in the Horizon platform.

## Features

- **YAML-based Policy DSL**: Define policies in human-readable YAML format
- **RBAC & ABAC Support**: Role-based and attribute-based access control
- **Pattern Matching**: Glob patterns for flexible resource and principal matching
- **Rich Condition Operators**: eq, ne, gt, lt, gte, lte, in, contains
- **Ultra-fast Evaluation**: Sub-microsecond policy evaluation
- **Type-safe**: Leverages Rust's type system for compile-time safety

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
horizon-policyx = { path = "../crates/policyx" }
```

## Quick Start

### Define a Policy

```yaml
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: gpu-access-policy
  version: "1.0"
spec:
  principals:
    - type: role
      value: gpu-user
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel]
      conditions:
        - field: resource.gpu_count
          operator: lte
          value: 8
```

### Evaluate the Policy

```rust
use horizon_policyx::{
    parse_policy, evaluate, Decision,
    EvaluationContext, PrincipalContext, ResourceContext
};
use serde_json::json;

// Parse policy from YAML
let yaml = std::fs::read_to_string("policy.yaml")?;
let policy = parse_policy(&yaml)?;

// Create evaluation context
let principal = PrincipalContext::new(
    None,
    vec!["gpu-user".to_string()],
    vec![]
);

let resource = ResourceContext::new(
    "job".to_string(),
    "jobs/123".to_string()
)
.with_attribute("gpu_count".to_string(), json!(4));

let context = EvaluationContext::new(
    principal,
    resource,
    "submit".to_string()
);

// Evaluate policy
let decision = evaluate(&policy, &context)?;

match decision {
    Decision::Allow => println!("Access granted!"),
    Decision::Deny => println!("Access denied!"),
}
```

## Policy Structure

### API Version

All policies must specify the API version:

```yaml
apiVersion: policy.horizon.dev/v1
```

### Metadata

Policy metadata includes name, optional version, and description:

```yaml
metadata:
  name: my-policy
  version: "1.0"
  description: "Policy description"
```

### Principals

Define who the policy applies to:

```yaml
principals:
  # Role-based
  - type: role
    value: admin

  # User-based
  - type: user
    value: user@example.com

  # Team-based
  - type: team
    value: ml-team

  # Pattern matching
  - type: role
    pattern: "*-admin"
```

### Resources

Define what resources the policy controls:

```yaml
resources:
  - type: job
    pattern: "jobs/*"

  - type: job
    pattern: "jobs/team-*/project-*/*"
```

### Rules

Define access rules with effects and conditions:

```yaml
rules:
  # Allow rule with conditions
  - effect: allow
    actions: [submit, cancel]
    conditions:
      - field: resource.gpu_count
        operator: lte
        value: 8
      - field: principal.team
        operator: in
        value: [ml-team, research-team]

  # Deny rule
  - effect: deny
    actions: [delete]
```

## Condition Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equals | `field: value` equals `"admin"` |
| `ne` | Not equals | `field: status` not equals `"failed"` |
| `gt` | Greater than | `field: priority` > `5` |
| `lt` | Less than | `field: priority` < `10` |
| `gte` | Greater than or equal | `field: gpu_count` >= `1` |
| `lte` | Less than or equal | `field: gpu_count` <= `8` |
| `in` | Value in array | `field: team` in `[ml-team, research]` |
| `contains` | Contains substring/element | `field: name` contains `"test"` |

## Context Fields

Access context information in conditions:

### Principal Fields

- `principal.user` - User ID
- `principal.roles` - Array of roles
- `principal.teams` - Array of teams
- `principal.*` - Custom attributes

### Resource Fields

- `resource.type` - Resource type
- `resource.id` - Resource ID
- `resource.*` - Custom attributes

### Action Field

- `action` - The action being performed

## Performance

PolicyX is designed for high performance:

- **Parse Simple Policy**: ~5 µs
- **Parse Complex Policy**: ~11 µs
- **Evaluate Simple Allow**: ~92 ns
- **Evaluate with Conditions**: ~173 ns
- **Pattern Matching**: ~341 ns

All operations complete in microseconds or nanoseconds, making it suitable for high-throughput systems.

## Error Handling

PolicyX provides comprehensive error types:

```rust
use horizon_policyx::Error;

match parse_policy(yaml) {
    Ok(policy) => { /* use policy */ }
    Err(Error::ParseError(msg)) => {
        eprintln!("YAML parse error: {}", msg);
    }
    Err(Error::ValidationError(msg)) => {
        eprintln!("Policy validation error: {}", msg);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Examples

### RBAC Policy

```yaml
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: admin-rbac
spec:
  principals:
    - type: role
      value: admin
  resources:
    - type: job
      pattern: "jobs/*"
  rules:
    - effect: allow
      actions: [submit, cancel, view, delete]
```

### ABAC Policy with Multiple Conditions

```yaml
apiVersion: policy.horizon.dev/v1
kind: Policy
metadata:
  name: gpu-quota-policy
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
        - field: resource.priority
          operator: in
          value: [normal, low]
        - field: principal.teams
          operator: in
          value: [ml-team, research-team]
```

### Pattern Matching Policy

```yaml
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
      pattern: "jobs/team-*/project-*/*"
  rules:
    - effect: allow
      actions: [submit, view]
```

## Testing

Run tests:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## Contributing

For questions or issues, contact the Horizon team.
