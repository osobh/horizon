//! # Horizon Policy Engine (policyx)
//!
//! A declarative policy engine for authorization and access control in the Horizon platform.
//!
//! ## Features
//!
//! - **YAML-based Policy DSL**: Define policies in a human-readable YAML format
//! - **RBAC & ABAC Support**: Role-based and attribute-based access control
//! - **Pattern Matching**: Glob patterns for resource and principal matching
//! - **Condition Evaluation**: Rich condition operators (eq, ne, gt, lt, gte, lte, in, contains)
//! - **Fast Evaluation**: Sub-5ms p99 policy evaluation
//!
//! ## Example
//!
//! ```rust
//! use hpc_policy::{parse_policy, evaluate, EvaluationContext, PrincipalContext, ResourceContext};
//! use serde_json::json;
//!
//! let yaml = r#"
//! apiVersion: policy.horizon.dev/v1
//! kind: Policy
//! metadata:
//!   name: gpu-access-policy
//! spec:
//!   principals:
//!     - type: role
//!       value: gpu-user
//!   resources:
//!     - type: job
//!       pattern: "jobs/*"
//!   rules:
//!     - effect: allow
//!       actions: [submit]
//!       conditions:
//!         - field: resource.gpu_count
//!           operator: lte
//!           value: 8
//! "#;
//!
//! // Parse the policy
//! let policy = parse_policy(yaml).unwrap();
//!
//! // Create evaluation context
//! let principal = PrincipalContext::new(None, vec!["gpu-user".to_string()], vec![]);
//! let resource = ResourceContext::new("job".to_string(), "jobs/123".to_string())
//!     .with_attribute("gpu_count".to_string(), json!(4));
//! let context = EvaluationContext::new(principal, resource, "submit".to_string());
//!
//! // Evaluate the policy
//! let decision = evaluate(&policy, &context).unwrap();
//! ```

pub mod ast;
pub mod error;
pub mod evaluator;
pub mod matcher;
pub mod parser;
pub mod policy;

// Re-export commonly used types
pub use ast::{Decision, EvaluationContext, PrincipalContext, ResourceContext};
pub use error::{Error, Result};
pub use evaluator::evaluate;
pub use parser::parse_policy;
pub use policy::{Effect, Operator, Policy, PolicyMetadata, PolicySpec, Principal, Resource, Rule};
