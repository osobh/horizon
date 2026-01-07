//! Cost ingestion service for collecting and normalizing billing data from cloud providers.

pub mod api;
pub mod config;
pub mod db;
pub mod error;
pub mod ingest;
pub mod models;
pub mod normalize;

pub use config::Config;
pub use error::{HpcError, IngestorErrorExt, Result};
