//! Checksum verification, corruption detection, repair strategies, audit trails
//!
//! This module provides comprehensive data integrity management including:
//! - Multi-algorithm checksum calculation and verification
//! - Real-time corruption detection and alerting
//! - Automated data repair strategies with multiple recovery options
//! - Comprehensive audit trails for all integrity operations
//! - Silent corruption detection through continuous monitoring
//! - Integrity reporting and compliance tracking
//! - Cross-system integrity verification

// Public exports
pub use algorithms::*;
pub use audit::*;
pub use config::*;
pub use corruption::*;
pub use manager::*;
pub use metrics::*;
pub use repair::*;
pub use types::*;
pub use verification::*;

// Module declarations
mod algorithms;
mod audit;
mod config;
mod corruption;
mod manager;
mod metrics;
mod repair;
mod types;
mod verification;

#[cfg(test)]
mod tests;
