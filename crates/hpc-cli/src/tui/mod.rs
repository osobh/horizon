//! Terminal User Interface module for HPC-AI CLI
//!
//! Provides a real-time TUI dashboard for monitoring and managing
//! HPC-AI platform components with a three-panel layout:
//! - Dashboard (left): GPU usage, cluster health, jobs
//! - Command Menu (right): Tabbed navigation for commands
//! - Log Viewer (bottom): Real-time log streaming

pub mod app;
pub mod events;
pub mod modals;
pub mod panels;

pub use app::TuiApp;
pub use events::{Event, EventHandler};
pub use modals::{AddNodeModal, Modal, ModalAction};
pub use panels::{CommandMenuPanel, DashboardPanel, LogViewerPanel, Panel, PanelFocus};

use crate::core::AppConfig;

/// TUI configuration
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Tick rate for UI updates in milliseconds
    pub tick_rate_ms: u64,
    /// Maximum log entries to keep in memory
    pub max_log_entries: usize,
    /// Color theme
    pub theme: Theme,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            tick_rate_ms: 250,
            max_log_entries: 1000,
            theme: Theme::default(),
        }
    }
}

impl From<&AppConfig> for TuiConfig {
    fn from(config: &AppConfig) -> Self {
        Self {
            tick_rate_ms: config.tui.tick_rate_ms,
            max_log_entries: config.tui.max_log_entries,
            theme: Theme::from_name(&config.tui.theme),
        }
    }
}

/// Color theme for the TUI
#[derive(Debug, Clone)]
pub struct Theme {
    /// Primary accent color
    pub primary: ratatui::style::Color,
    /// Secondary accent color
    pub secondary: ratatui::style::Color,
    /// Success color
    pub success: ratatui::style::Color,
    /// Warning color
    pub warning: ratatui::style::Color,
    /// Error color
    pub error: ratatui::style::Color,
    /// Background color
    pub background: ratatui::style::Color,
    /// Foreground/text color
    pub foreground: ratatui::style::Color,
    /// Border color
    pub border: ratatui::style::Color,
    /// Highlight/selection color
    pub highlight: ratatui::style::Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme (default)
    pub fn dark() -> Self {
        use ratatui::style::Color;
        Self {
            primary: Color::Cyan,
            secondary: Color::Blue,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            background: Color::Reset,
            foreground: Color::White,
            border: Color::DarkGray,
            highlight: Color::LightCyan,
        }
    }

    /// Light theme
    pub fn light() -> Self {
        use ratatui::style::Color;
        Self {
            primary: Color::Blue,
            secondary: Color::Cyan,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            background: Color::White,
            foreground: Color::Black,
            border: Color::Gray,
            highlight: Color::LightBlue,
        }
    }

    /// Create theme from name
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "light" => Self::light(),
            _ => Self::dark(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_config_default() {
        let config = TuiConfig::default();
        assert_eq!(config.tick_rate_ms, 250);
        assert_eq!(config.max_log_entries, 1000);
    }

    #[test]
    fn test_theme_from_name() {
        let dark = Theme::from_name("dark");
        assert!(matches!(dark.primary, ratatui::style::Color::Cyan));

        let light = Theme::from_name("light");
        assert!(matches!(light.primary, ratatui::style::Color::Blue));

        // Unknown defaults to dark
        let unknown = Theme::from_name("unknown");
        assert!(matches!(unknown.primary, ratatui::style::Color::Cyan));
    }
}
