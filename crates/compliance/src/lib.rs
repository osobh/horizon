//! Compliance framework for GDPR, HIPAA, SOC2, and AI safety
//!
//! This crate provides comprehensive compliance capabilities for:
//! - Data classification and handling
//! - Regulatory compliance (GDPR, HIPAA, SOC2)
//! - AI safety and ethics validation
//! - Immutable audit logging
//! - Region-specific data retention

#![warn(missing_docs)]

pub mod ai_safety;
pub mod audit;
pub mod audit_framework;
pub mod data_classification;
pub mod encryption;
pub mod error;
pub mod gdpr;
pub mod hipaa;
pub mod retention;
pub mod soc2;

pub use audit_framework::{ComplianceConfig, ComplianceEngine};
pub use data_classification::{DataCategory, DataClassification};
pub use error::{ComplianceError, ComplianceResult};

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
