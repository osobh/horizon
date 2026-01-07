//! Telemetry collection service for metrics, alerts, and observability data.

pub mod backpressure;
pub mod cardinality;
pub mod channels;
pub mod collector;
pub mod config;
pub mod handler;
pub mod listener;
pub mod writers;

pub use channels::TelemetryChannels;
