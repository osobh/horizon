//! Capacity modeling service for GPU demand forecasting and resource planning.

pub mod api;
pub mod config;
pub mod error;
pub mod forecaster;
pub mod models;
pub mod service;

pub use config::Config;
pub use error::{CapacityErrorExt, HpcError, Result};
pub use service::ForecastService;
