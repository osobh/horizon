//! Efficiency intelligence service for detecting waste and optimizing resource usage.

pub mod analyzer;
pub mod api;
pub mod config;
pub mod db;
pub mod detector;
pub mod error;
pub mod models;
pub mod recommender;

pub use config::Config;
pub use error::{EfficiencyErrorExt, HpcError, Result};
