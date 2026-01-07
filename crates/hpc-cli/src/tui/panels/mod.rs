//! TUI Panel components
//!
//! Implements the panel layout:
//! - Dashboard (left): GPU usage, cluster health, jobs
//! - Command Menu (right): Tabbed navigation for commands
//! - Log Viewer (bottom): Real-time log streaming
//! - Inventory: Node inventory management

mod command_menu;
mod dashboard;
mod inventory;
mod log_viewer;

pub use command_menu::CommandMenuPanel;
pub use dashboard::DashboardPanel;
pub use inventory::InventoryPanel;
pub use log_viewer::LogViewerPanel;

use ratatui::{layout::Rect, Frame};

use crate::core::AppState;
use crate::tui::{events::KeyAction, Theme};

/// Panel focus indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanelFocus {
    /// Dashboard panel (left)
    #[default]
    Dashboard,
    /// Command menu panel (right)
    CommandMenu,
    /// Log viewer panel (bottom)
    LogViewer,
    /// Inventory panel
    Inventory,
}

impl PanelFocus {
    /// Get next panel in cycle
    pub fn next(&self) -> Self {
        match self {
            Self::Dashboard => Self::CommandMenu,
            Self::CommandMenu => Self::LogViewer,
            Self::LogViewer => Self::Inventory,
            Self::Inventory => Self::Dashboard,
        }
    }

    /// Get previous panel in cycle
    pub fn prev(&self) -> Self {
        match self {
            Self::Dashboard => Self::Inventory,
            Self::CommandMenu => Self::Dashboard,
            Self::LogViewer => Self::CommandMenu,
            Self::Inventory => Self::LogViewer,
        }
    }
}

/// Trait for renderable TUI panels
pub trait Panel {
    /// Render the panel
    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme);

    /// Handle key action, returns true if handled
    fn handle_action(&mut self, action: KeyAction) -> bool;

    /// Update panel state from app state
    fn update(&mut self, state: &AppState);

    /// Get panel title
    fn title(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_focus_cycle() {
        let focus = PanelFocus::Dashboard;
        assert_eq!(focus.next(), PanelFocus::CommandMenu);
        assert_eq!(focus.next().next(), PanelFocus::LogViewer);
        assert_eq!(focus.next().next().next(), PanelFocus::Inventory);
        assert_eq!(focus.next().next().next().next(), PanelFocus::Dashboard);
    }

    #[test]
    fn test_panel_focus_prev_cycle() {
        let focus = PanelFocus::Dashboard;
        assert_eq!(focus.prev(), PanelFocus::Inventory);
        assert_eq!(focus.prev().prev(), PanelFocus::LogViewer);
        assert_eq!(focus.prev().prev().prev(), PanelFocus::CommandMenu);
        assert_eq!(focus.prev().prev().prev().prev(), PanelFocus::Dashboard);
    }
}
