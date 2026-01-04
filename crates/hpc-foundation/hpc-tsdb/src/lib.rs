//! Time-series database abstractions for Horizon GPU capacity management.
//!
//! This crate provides trait-based abstractions for time-series databases, with a
//! primary implementation for InfluxDB. It enables querying historical metrics for
//! capacity forecasting and analysis.

pub mod client;
pub mod error;
pub mod query;
pub mod types;

// Re-exports
pub use client::influxdb::InfluxDbClient;
pub use error::TsdbError;
pub use query::QueryBuilder;
pub use types::{Aggregation, DataPoint, TimeRange, TimeSeries};

/// Result type for tsdbx operations
pub type Result<T> = std::result::Result<T, TsdbError>;
