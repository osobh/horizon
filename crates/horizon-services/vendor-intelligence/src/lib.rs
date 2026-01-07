//! Vendor intelligence service for contract analysis and vendor relationship management.

pub mod analysis;
pub mod api;
pub mod config;
pub mod contracts;
pub mod db;
pub mod error;
pub mod models;
pub mod rfp;

pub use config::Config;
pub use error::{HpcError, Result, VendorErrorExt};
