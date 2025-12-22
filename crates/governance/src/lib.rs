//! # ExoRust Governance System
//!
//! A comprehensive centralized governance system that unifies all agent governance
//! across the ExoRust system. This crate provides:
//!
//! - Agent lifecycle governance (creation, evolution, termination)
//! - Resource allocation and permission systems
//! - Policy definition and enforcement
//! - Cross-agent coordination
//! - Compliance integration
//! - Emergency override capabilities
//! - Comprehensive audit logging

pub mod compliance_integration;
pub mod coordination_manager;
pub mod governance_engine;
pub mod lifecycle_governance;
pub mod monitoring_governance;
pub mod permission_system;
pub mod policy_manager;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod e2e_tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod additional_tests;

// Re-export main types for convenience
pub use compliance_integration::{ComplianceIntegration, ComplianceStatus};
pub use coordination_manager::{CoordinationManager, CoordinationRequest};
pub use governance_engine::{GovernanceConfig, GovernanceEngine};
pub use lifecycle_governance::{LifecycleDecision, LifecycleGovernor};
pub use monitoring_governance::{GovernanceMetrics, GovernanceMonitor};
pub use permission_system::{Permission, PermissionSystem, Role};
pub use policy_manager::{Policy, PolicyManager, PolicyType};

/// Error types for the governance system
#[derive(Debug, thiserror::Error)]
pub enum GovernanceError {
    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    #[error("Lifecycle error: {0}")]
    LifecycleError(String),

    #[error("Coordination error: {0}")]
    CoordinationError(String),

    #[error("Compliance error: {0}")]
    ComplianceError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, GovernanceError>;
