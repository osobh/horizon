//! Workload Orchestration Benchmarks
//!
//! Comprehensive benchmarking for swarmlet workload orchestration including
//! container startup time, concurrent workload handling, and resource utilization
//! efficiency. Implements TDD methodology for systematic optimization.

pub mod types;
pub mod metrics;
pub mod workloads;
pub mod constraints;
pub mod monitor;
pub mod benchmarks;

// Re-export all public types for convenience
pub use types::*;
pub use metrics::*;
pub use workloads::*;
pub use constraints::*;
pub use monitor::*;
pub use benchmarks::*;

// Tests are in their own module
#[cfg(test)]
mod tests;