//! Inventory panel for the HPC-AI TUI
//!
//! Displays:
//! - Node inventory list with status
//! - Node details on selection
//! - Quick actions (bootstrap, remove, refresh)

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame,
};

use crate::core::inventory::{InventoryStore, NodeInfo, NodeStatus};
use crate::core::AppState;
use crate::tui::{events::KeyAction, Theme};

use super::Panel;

/// Inventory panel state
#[derive(Debug)]
pub struct InventoryPanel {
    /// Loaded nodes
    nodes: Vec<NodeInfo>,
    /// Table state for selection
    table_state: TableState,
    /// Selected node index
    selected: usize,
    /// Error message if any
    error: Option<String>,
    /// Last refresh time
    last_refresh: Option<std::time::Instant>,
}

impl Default for InventoryPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl InventoryPanel {
    /// Create a new inventory panel
    pub fn new() -> Self {
        let mut panel = Self {
            nodes: Vec::new(),
            table_state: TableState::default(),
            selected: 0,
            error: None,
            last_refresh: None,
        };
        panel.refresh();
        panel
    }

    /// Refresh node list from store
    pub fn refresh(&mut self) {
        match InventoryStore::new() {
            Ok(store) => {
                self.nodes = store.list_nodes().into_iter().cloned().collect();
                self.error = None;
                self.last_refresh = Some(std::time::Instant::now());

                // Adjust selection if needed
                if !self.nodes.is_empty() && self.selected >= self.nodes.len() {
                    self.selected = self.nodes.len() - 1;
                }
                self.table_state.select(Some(self.selected));
            }
            Err(e) => {
                self.error = Some(e.to_string());
            }
        }
    }

    /// Get currently selected node
    pub fn selected_node(&self) -> Option<&NodeInfo> {
        self.nodes.get(self.selected)
    }

    /// Move selection up
    fn select_prev(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        self.selected = if self.selected == 0 {
            self.nodes.len() - 1
        } else {
            self.selected - 1
        };
        self.table_state.select(Some(self.selected));
    }

    /// Move selection down
    fn select_next(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        self.selected = (self.selected + 1) % self.nodes.len();
        self.table_state.select(Some(self.selected));
    }

    /// Get status symbol and color for a node
    fn status_display(status: &NodeStatus, theme: &Theme) -> (Span<'static>, Style) {
        match status {
            NodeStatus::Connected => (
                Span::raw(" ● "),
                Style::default().fg(theme.success),
            ),
            NodeStatus::Pending => (
                Span::raw(" ○ "),
                Style::default().fg(theme.border),
            ),
            NodeStatus::Connecting => (
                Span::raw(" ◐ "),
                Style::default().fg(theme.primary),
            ),
            NodeStatus::Bootstrapping => (
                Span::raw(" ◑ "),
                Style::default().fg(theme.primary),
            ),
            NodeStatus::Unreachable => (
                Span::raw(" ◌ "),
                Style::default().fg(theme.warning),
            ),
            NodeStatus::Offline => (
                Span::raw(" ○ "),
                Style::default().fg(theme.border),
            ),
            NodeStatus::Failed => (
                Span::raw(" ✗ "),
                Style::default().fg(theme.error),
            ),
        }
    }

    /// Render the node table
    fn render_table(&mut self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let header_style = Style::default()
            .fg(theme.primary)
            .add_modifier(Modifier::BOLD);

        let header = Row::new(vec![
            Cell::from("Status"),
            Cell::from("Name"),
            Cell::from("Address"),
            Cell::from("Platform"),
            Cell::from("Mode"),
        ])
        .style(header_style)
        .height(1);

        let rows: Vec<Row> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let (status_span, status_style) = Self::status_display(&node.status, theme);

                let platform = match (&node.os, &node.arch) {
                    (Some(os), Some(arch)) => format!("{}/{}", os, arch),
                    _ => "-".to_string(),
                };

                let selected = i == self.selected;
                let row_style = if selected && focused {
                    Style::default().bg(theme.highlight)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(status_span).style(status_style),
                    Cell::from(node.name.clone()),
                    Cell::from(format!("{}:{}", node.address, node.port)),
                    Cell::from(platform),
                    Cell::from(format!("{}", node.mode)),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(6),
            Constraint::Min(15),
            Constraint::Min(20),
            Constraint::Length(12),
            Constraint::Length(8),
        ];

        let border_style = if focused {
            Style::default().fg(theme.primary)
        } else {
            Style::default().fg(theme.border)
        };

        let table = Table::new(rows, widths)
            .header(header)
            .block(
                Block::default()
                    .title(" Inventory ")
                    .borders(Borders::ALL)
                    .border_style(border_style),
            )
            .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));

        frame.render_stateful_widget(table, area, &mut self.table_state);
    }

    /// Render node details section
    fn render_details(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Block::default()
            .title(" Node Details ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border));

        let content = if let Some(node) = self.selected_node() {
            let mut lines = vec![
                Line::from(vec![
                    Span::styled("ID: ", Style::default().fg(theme.border)),
                    Span::styled(&node.id, Style::default().fg(theme.foreground)),
                ]),
                Line::from(vec![
                    Span::styled("Name: ", Style::default().fg(theme.border)),
                    Span::styled(&node.name, Style::default().fg(theme.foreground)),
                ]),
                Line::from(vec![
                    Span::styled("Address: ", Style::default().fg(theme.border)),
                    Span::styled(
                        format!("{}:{}", node.address, node.port),
                        Style::default().fg(theme.foreground),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("User: ", Style::default().fg(theme.border)),
                    Span::styled(&node.username, Style::default().fg(theme.foreground)),
                ]),
                Line::from(vec![
                    Span::styled("Status: ", Style::default().fg(theme.border)),
                    Span::styled(
                        format!("{}", node.status),
                        Style::default().fg(theme.primary),
                    ),
                ]),
            ];

            // Add hardware info if available
            if let Some(ref hw) = node.hardware {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![Span::styled(
                    "Hardware:",
                    Style::default().fg(theme.border).add_modifier(Modifier::BOLD),
                )]));
                lines.push(Line::from(vec![
                    Span::styled("  CPU: ", Style::default().fg(theme.border)),
                    Span::styled(&hw.cpu_model, Style::default().fg(theme.foreground)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("  Cores: ", Style::default().fg(theme.border)),
                    Span::styled(
                        format!("{}", hw.cpu_cores),
                        Style::default().fg(theme.foreground),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("  Memory: ", Style::default().fg(theme.border)),
                    Span::styled(
                        format!("{:.1} GB", hw.memory_gb),
                        Style::default().fg(theme.foreground),
                    ),
                ]));

                if !hw.gpus.is_empty() {
                    lines.push(Line::from(vec![
                        Span::styled("  GPUs: ", Style::default().fg(theme.border)),
                        Span::styled(
                            format!("{}", hw.gpus.len()),
                            Style::default().fg(theme.success),
                        ),
                    ]));
                }
            }

            // Add last heartbeat if available
            if let Some(ref hb) = node.last_heartbeat {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("Last Heartbeat: ", Style::default().fg(theme.border)),
                    Span::styled(
                        hb.format("%Y-%m-%d %H:%M:%S").to_string(),
                        Style::default().fg(theme.foreground),
                    ),
                ]));
            }

            lines
        } else {
            vec![Line::from(Span::styled(
                "No node selected",
                Style::default().fg(theme.border),
            ))]
        };

        let paragraph = Paragraph::new(content).block(block);
        frame.render_widget(paragraph, area);
    }

    /// Render help bar
    fn render_help(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let help_spans = vec![
            Span::styled("[a] ", Style::default().fg(theme.border)),
            Span::styled("Add  ", Style::default().fg(theme.foreground)),
            Span::styled("[j/k] ", Style::default().fg(theme.border)),
            Span::styled("Navigate  ", Style::default().fg(theme.foreground)),
            Span::styled("[r] ", Style::default().fg(theme.border)),
            Span::styled("Refresh  ", Style::default().fg(theme.foreground)),
            Span::styled("[b] ", Style::default().fg(theme.border)),
            Span::styled("Bootstrap  ", Style::default().fg(theme.foreground)),
            Span::styled("[d] ", Style::default().fg(theme.border)),
            Span::styled("Delete  ", Style::default().fg(theme.foreground)),
        ];

        let help = Paragraph::new(Line::from(help_spans))
            .style(Style::default().fg(theme.border));

        frame.render_widget(help, area);
    }
}

impl Panel for InventoryPanel {
    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        // Clone self for mutable table state access
        let mut panel = Self {
            nodes: self.nodes.clone(),
            table_state: self.table_state.clone(),
            selected: self.selected,
            error: self.error.clone(),
            last_refresh: self.last_refresh,
        };

        // Check for error
        if let Some(ref err) = self.error {
            let error_block = Block::default()
                .title(" Inventory Error ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme.error));

            let error_msg = Paragraph::new(err.as_str())
                .style(Style::default().fg(theme.error))
                .block(error_block);

            frame.render_widget(error_msg, area);
            return;
        }

        // Layout: table on top, details on bottom, help at very bottom
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),      // Table
                Constraint::Length(12),   // Details
                Constraint::Length(1),    // Help
            ])
            .split(area);

        panel.render_table(frame, layout[0], focused, theme);
        panel.render_details(frame, layout[1], theme);
        panel.render_help(frame, layout[2], theme);
    }

    fn handle_action(&mut self, action: KeyAction) -> bool {
        match action {
            KeyAction::Up => {
                self.select_prev();
                true
            }
            KeyAction::Down => {
                self.select_next();
                true
            }
            KeyAction::Refresh => {
                self.refresh();
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, _state: &AppState) {
        // Periodically refresh (every 30 seconds)
        if let Some(last) = self.last_refresh {
            if last.elapsed().as_secs() > 30 {
                self.refresh();
            }
        }
    }

    fn title(&self) -> &str {
        "Inventory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_panel_new() {
        let panel = InventoryPanel::new();
        assert_eq!(panel.selected, 0);
        assert!(panel.last_refresh.is_some());
    }

    #[test]
    fn test_inventory_panel_navigation() {
        use crate::core::inventory::{CredentialRef, NodeMode};

        let mut panel = InventoryPanel::new();

        // Add some test nodes manually
        panel.nodes = vec![
            NodeInfo::new(
                "node-1".to_string(),
                "192.168.1.1".to_string(),
                22,
                "admin".to_string(),
                CredentialRef::SshAgent,
                NodeMode::Docker,
            ),
            NodeInfo::new(
                "node-2".to_string(),
                "192.168.1.2".to_string(),
                22,
                "admin".to_string(),
                CredentialRef::SshAgent,
                NodeMode::Docker,
            ),
            NodeInfo::new(
                "node-3".to_string(),
                "192.168.1.3".to_string(),
                22,
                "admin".to_string(),
                CredentialRef::SshAgent,
                NodeMode::Docker,
            ),
        ];
        panel.table_state.select(Some(0));

        // Test navigation
        panel.select_next();
        assert_eq!(panel.selected, 1);

        panel.select_next();
        assert_eq!(panel.selected, 2);

        panel.select_next();
        assert_eq!(panel.selected, 0); // Wrap around

        panel.select_prev();
        assert_eq!(panel.selected, 2); // Wrap around back
    }

    #[test]
    fn test_status_display() {
        let theme = Theme::dark();

        let (span, _style) = InventoryPanel::status_display(&NodeStatus::Connected, &theme);
        let content = span.content.to_string();
        assert!(content.contains("●"));

        let (span, _style) = InventoryPanel::status_display(&NodeStatus::Pending, &theme);
        let content = span.content.to_string();
        assert!(content.contains("○"));

        let (span, _style) = InventoryPanel::status_display(&NodeStatus::Failed, &theme);
        let content = span.content.to_string();
        assert!(content.contains("✗"));
    }

    #[test]
    fn test_handle_action() {
        use crate::core::inventory::{CredentialRef, NodeMode};

        let mut panel = InventoryPanel::new();
        panel.nodes = vec![
            NodeInfo::new(
                "node-1".to_string(),
                "192.168.1.1".to_string(),
                22,
                "admin".to_string(),
                CredentialRef::SshAgent,
                NodeMode::Docker,
            ),
            NodeInfo::new(
                "node-2".to_string(),
                "192.168.1.2".to_string(),
                22,
                "admin".to_string(),
                CredentialRef::SshAgent,
                NodeMode::Docker,
            ),
        ];
        panel.table_state.select(Some(0));

        // Test up/down navigation
        assert!(panel.handle_action(KeyAction::Down));
        assert_eq!(panel.selected, 1);

        assert!(panel.handle_action(KeyAction::Up));
        assert_eq!(panel.selected, 0);

        // Test refresh
        assert!(panel.handle_action(KeyAction::Refresh));

        // Test unhandled action
        assert!(!panel.handle_action(KeyAction::None));
    }
}
