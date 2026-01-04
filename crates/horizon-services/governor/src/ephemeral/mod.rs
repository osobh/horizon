//! Ephemeral policy generation and management.
//!
//! This module provides utilities for generating and managing policies
//! for ephemeral access scenarios including:
//! - Ephemeral identity access
//! - Time-limited quota access
//! - Resource pool participation
//! - Federated training access

pub mod policy_generator;

pub use policy_generator::{
    EphemeralPolicyConfig, EphemeralPolicyGenerator, GeneratedPolicy, PolicyScope, RiskBasedAccess,
    TimeWindowConfig,
};
