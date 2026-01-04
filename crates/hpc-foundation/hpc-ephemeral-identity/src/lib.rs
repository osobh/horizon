//! # HPC Ephemeral Identity
//!
//! Time-limited identity management for external access in the HPC Platform.
//!
//! This crate provides:
//! - Ephemeral identities with automatic expiration
//! - Capability-based access control with scoped permissions
//! - Secure invitation links with two-factor redemption
//! - Cryptographically signed tokens with encryption
//!
//! ## Architecture
//!
//! The ephemeral identity system integrates with zero-trust security:
//! - All identities have bounded lifetimes
//! - Capabilities can only be reduced, never expanded
//! - Risk scoring increases as expiry approaches
//! - Full audit trail for compliance

mod capabilities;
mod error;
mod identity;
pub mod integration;
mod invitation;
mod service;
mod token;
pub mod workers;

pub use capabilities::{Capability, CapabilitySet, RateLimits, TimeRestrictions};
pub use error::{EphemeralError, Result};
pub use identity::{DeviceBinding, EphemeralIdentity, EphemeralIdentityState, IdentityMetadata};
pub use integration::{
    EphemeralSessionActivity, EphemeralSessionAdapter, EphemeralSessionBinding,
    EphemeralSessionConfig, EphemeralVerificationResult,
};
pub use invitation::{InvitationLink, InvitationPayload, InvitationStatus, RedemptionResult};
pub use service::{EphemeralIdentityService, ServiceConfig};
pub use token::{EphemeralToken, TokenClaims, TokenValidation};
pub use workers::{CleanupStats, CleanupWorker, CleanupWorkerConfig, CleanupWorkerHandle};
