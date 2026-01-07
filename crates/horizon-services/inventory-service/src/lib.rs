//! Inventory service for tracking cloud assets and resource discovery.

pub mod api;
pub mod config;
pub mod db;
pub mod error;
pub mod models;
pub mod repository;

pub use error::{HpcError, InventoryErrorExt, Result};
