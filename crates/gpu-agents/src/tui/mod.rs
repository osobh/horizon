//! Terminal User Interface module for GPU Agents monitoring dashboard
//!
//! This module provides a real-time TUI monitoring interface for the GPU agents
//! benchmark suite, replacing the flickering bash-based dashboard with a
//! professional Rust TUI using ratatui and crossterm.

pub mod app;
pub mod dashboard;
pub mod events;
pub mod log_parser;
pub mod resource_monitor;
pub mod ui;

pub use app::App;
pub use dashboard::Dashboard;
pub use events::{Event, EventHandler};
pub use log_parser::ProgressLogParser;
pub use resource_monitor::ResourceMonitor;
pub use ui::{ColorTheme, UiUtils};

/// Main TUI result type
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// TUI configuration
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Progress log file path
    pub log_file_path: String,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Maximum log entries to keep in memory
    pub max_log_entries: usize,
    /// Tick rate for UI updates
    pub tick_rate_ms: u64,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            log_file_path: "benchmark_progress.log".to_string(),
            enable_resource_monitoring: true,
            max_log_entries: 1000,
            tick_rate_ms: 250,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_config_default() {
        let config = TuiConfig::default();
        assert_eq!(config.update_interval_ms, 1000);
        assert_eq!(config.log_file_path, "benchmark_progress.log");
        assert!(config.enable_resource_monitoring);
        assert_eq!(config.max_log_entries, 1000);
        assert_eq!(config.tick_rate_ms, 250);
    }
}
