//! Compliance framework for GDPR, HIPAA, SOC2, and AI safety
//!
//! This crate provides comprehensive compliance capabilities for:
//! - Data classification and handling
//! - Regulatory compliance (GDPR, HIPAA, SOC2)
//! - AI safety and ethics validation
//! - Immutable audit logging
//! - Region-specific data retention
//!
//! # Shared Security Types
//! Core security types are now provided by the `security-common` crate.
//! This crate re-exports them for backward compatibility.

#![warn(missing_docs)]

pub mod ai_safety;
pub mod audit;
pub mod audit_framework;
pub mod gdpr;
pub mod hipaa;
pub mod retention;
pub mod soc2;

// Re-export shared security types from security-common for backward compatibility
pub mod data_classification {
    //! Data classification module - re-exported from security-common
    pub use security_common::data_classification::*;
}

pub mod encryption {
    //! Encryption module - re-exported from security-common
    pub use security_common::encryption::*;
}

pub mod error {
    //! Error types - re-exported from security-common
    pub use security_common::error::*;
}

pub use audit_framework::{ComplianceConfig, ComplianceEngine};
pub use security_common::{DataCategory, DataClassification, ComplianceError, ComplianceResult};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod basic_tests {
    use super::*;

    #[test]
    fn test_compliance_framework_creation() {
        // This will fail initially (RED phase)
        let config = ComplianceConfig::default();
        let engine = ComplianceEngine::new(config);
        assert!(engine.is_ok());
    }
}
