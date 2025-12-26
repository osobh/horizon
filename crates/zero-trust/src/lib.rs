//! Zero-trust security framework with identity verification and continuous authentication
//!
//! This crate provides comprehensive zero-trust security capabilities for:
//! - Identity verification and continuous authentication
//! - Device trust validation and attestation
//! - Network microsegmentation and policy enforcement
//! - Behavioral analysis and anomaly detection
//! - Risk scoring and adaptive access control
//! - Session management and token validation

#![warn(missing_docs)]

pub mod attestation;
pub mod behavior_analysis;
pub mod device_trust;
pub mod error;
pub mod identity;
pub mod network_policy;
pub mod risk_engine;
pub mod session_manager;

pub use device_trust::{DeviceAttestation, DeviceTrustManager};
pub use error::{ZeroTrustError, ZeroTrustResult};
pub use identity::{IdentityConfig, IdentityProvider};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_trust_framework_creation() {
        // This will fail initially (RED phase)
        let config = IdentityConfig::default();
        let provider = IdentityProvider::new(config);
        assert!(provider.is_ok());
    }
}
