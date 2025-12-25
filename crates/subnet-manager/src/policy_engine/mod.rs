//! Policy engine for automatic subnet assignment
//!
//! Evaluates node attributes against assignment policies to automatically
//! determine which subnet a node should be assigned to.
//!
//! # Components
//!
//! - **Evaluator**: Core policy evaluation logic with priority-based matching
//! - **Matcher**: Attribute matching for policy rules
//! - **Validator**: Policy validation and conflict detection
//! - **Analytics**: Statistics, metrics, and health checks

mod analytics;
mod evaluator;
mod matcher;
mod validator;

pub use analytics::{
    health_check, EvaluationRecord, HealthIssue, HealthStatus, PolicyAnalyzer, PolicyHealth,
    PolicyHealthCheck, PolicyStats,
};
pub use evaluator::{
    BatchEvaluationItem, BatchEvaluationResult, EvaluationResult, MatchedPolicy, PolicyEngine,
    PolicyEvaluator,
};
pub use matcher::{AttributeMatcher, NodeAttributes};
pub use validator::{
    ConflictDetector, ConflictSeverity, ConflictType, PolicyConflict, PolicyValidator,
    ValidationError, ValidationResult, ValidationWarning,
};
