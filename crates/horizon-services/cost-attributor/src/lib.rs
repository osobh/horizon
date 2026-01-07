//! Cost attribution service for tracking and allocating cloud spend to teams and projects.

pub mod api;
pub mod attribution;
pub mod config;
pub mod db;
pub mod error;
pub mod models;

pub use config::Config;
pub use error::{AttributorErrorExt, HpcError, Result};
