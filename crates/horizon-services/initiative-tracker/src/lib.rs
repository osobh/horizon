//! Initiative tracking service for managing cost optimization projects and their ROI.

pub mod api;
pub mod config;
pub mod db;
pub mod error;
pub mod measurement;
pub mod models;
pub mod portfolio;
pub mod registry;

pub use config::Config;
pub use error::{HpcError, InitiativeErrorExt, Result};
