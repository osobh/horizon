//! Baseline management for performance regression detection

mod config;
mod manager;
mod metrics;
mod storage;
mod types;

pub use config::BaselineConfig;
pub use manager::BaselineManager;
pub use metrics::BaselineMetrics;
pub use storage::BaselineStorage;
pub use types::{Baseline, BaselineType, MetricBaseline};
