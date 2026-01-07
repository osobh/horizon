//! Core types and state management for HPC-AI CLI
//!
//! This module provides shared types used by both CLI and TUI modes:
//! - Application configuration and profiles
//! - Project definitions and dependency management
//! - Runtime state for deployment tracking
//! - Node inventory management

pub mod config;
pub mod inventory;
pub mod profile;
pub mod project;
pub mod state;

pub use config::{AppConfig, TuiSettings};
pub use inventory::{
    CredentialRef, CredentialStore, InventoryStore, InventorySummary, NodeInfo, NodeMode,
    NodeStatus,
};
pub use profile::{Environment, Profile};
pub use project::{Project, ProjectCategory};
pub use state::{AppState, DeploymentStatus, SharedState};
