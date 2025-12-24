//! Integration module for connecting ephemeral identities with zero-trust infrastructure.

pub mod session_adapter;

pub use session_adapter::{
    EphemeralSessionAdapter, EphemeralSessionConfig, EphemeralSessionBinding,
    EphemeralVerificationResult, EphemeralSessionActivity,
};
