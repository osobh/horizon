//! Trend analysis for performance regression

mod analyzer;
mod config;
mod metrics;
mod types;

pub use analyzer::TrendAnalyzer;
pub use config::TrendConfig;
pub use metrics::TrendMetrics;
pub use types::{Trend, TrendDirection};
