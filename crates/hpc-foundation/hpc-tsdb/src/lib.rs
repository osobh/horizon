//! Time-series database abstractions for Horizon GPU capacity management.
//!
//! This crate provides trait-based abstractions for time-series databases, with a
//! primary implementation for InfluxDB. It enables querying historical metrics for
//! capacity forecasting and analysis.

pub mod error;
pub mod types;
pub mod client;
pub mod query;

// Re-exports
pub use error::TsdbError;
pub use types::{TimeRange, DataPoint, TimeSeries, Aggregation};
pub use client::influxdb::InfluxDbClient;
pub use query::QueryBuilder;

/// Result type for tsdbx operations
pub type Result<T> = std::result::Result<T, TsdbError>;
