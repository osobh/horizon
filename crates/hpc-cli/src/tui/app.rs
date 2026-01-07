//! Main TUI application for HPC-AI CLI
//!
//! Implements the three-panel layout:
//! ```text
//! ┌─────────────────────────┬─────────────────────────┐
//! │      DASHBOARD          │      COMMAND MENU       │
//! │                         │                         │
//! │  GPU Usage              │  [Commands] Deploy Proj │
//! │  ████████████░░ 78%     │                         │
//! │                         │  > slai detect          │
//! │  Cluster Health         │    swarm status         │
//! │  Nodes: 4 | Running: 12 │    argus status         │
//! │                         │                         │
//! └─────────────────────────┴─────────────────────────┘
//! │                        LOGS                        │
//! │ [12:34:56] INFO  Deploying rnccl to local...      │
//! │ [12:34:58] INFO  GPU 0: NVIDIA A100 detected      │
//! └────────────────────────────────────────────────────┘
//! ```

use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};

use std::path::PathBuf;

use tokio::sync::mpsc;

use crate::core::{config::AppConfig, state::AppState};
use crate::core::inventory::{
    bootstrap_node_async, BootstrapParams, CredentialRef, InventoryStore, NodeInfo, NodeMode,
};
use crate::tui::{
    events::{Event, EventHandler, KeyAction},
    modals::{AddNodeModal, AddNodeResult, Modal, ModalAction},
    panels::{CommandMenuPanel, DashboardPanel, InventoryPanel, LogViewerPanel, Panel, PanelFocus},
    TuiConfig,
};

/// Main TUI application
pub struct TuiApp {
    /// Application state
    state: AppState,
    /// TUI configuration
    config: TuiConfig,
    /// Event handler
    event_handler: EventHandler,
    /// Current panel focus
    focus: PanelFocus,
    /// Dashboard panel
    dashboard: DashboardPanel,
    /// Command menu panel
    command_menu: CommandMenuPanel,
    /// Log viewer panel
    log_viewer: LogViewerPanel,
    /// Inventory panel
    inventory: InventoryPanel,
    /// Whether to quit
    should_quit: bool,
    /// Show help overlay
    show_help: bool,
    /// Active modal dialog (using concrete type for type-safe access)
    add_node_modal: Option<AddNodeModal>,
}

impl TuiApp {
    /// Create a new TUI application
    pub fn new(app_config: AppConfig) -> Result<Self> {
        let config = TuiConfig::from(&app_config);
        let state = AppState::new(app_config.clone());
        let event_handler = EventHandler::new(Duration::from_millis(config.tick_rate_ms))?;

        Ok(Self {
            state,
            config: config.clone(),
            event_handler,
            focus: PanelFocus::Dashboard,
            dashboard: DashboardPanel::new(),
            command_menu: CommandMenuPanel::new(),
            log_viewer: LogViewerPanel::new(config.max_log_entries),
            inventory: InventoryPanel::new(),
            should_quit: false,
            show_help: false,
            add_node_modal: None,
        })
    }

    /// Run the TUI application
    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Clear screen
        terminal.clear()?;

        // Main event loop
        loop {
            // Draw UI
            terminal.draw(|frame| self.render(frame))?;

            // Handle events
            if let Some(event) = self.event_handler.next().await {
                self.handle_event(event)?;

                if self.should_quit {
                    break;
                }
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    /// Render the TUI
    fn render(&self, frame: &mut Frame) {
        let size = frame.area();

        // Main layout: top panels and bottom log viewer
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),    // Top panels
                Constraint::Length(8),  // Log viewer
            ])
            .split(size);

        // Top panels: Dashboard (left) and Command Menu (right)
        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(main_chunks[0]);

        // Render panels
        // Left panel: Dashboard or Inventory (toggle with Tab)
        if self.focus == PanelFocus::Inventory {
            self.inventory.render(
                frame,
                top_chunks[0],
                true,
                &self.config.theme,
            );
        } else {
            self.dashboard.render(
                frame,
                top_chunks[0],
                self.focus == PanelFocus::Dashboard,
                &self.config.theme,
            );
        }

        self.command_menu.render(
            frame,
            top_chunks[1],
            self.focus == PanelFocus::CommandMenu,
            &self.config.theme,
        );

        self.log_viewer.render(
            frame,
            main_chunks[1],
            self.focus == PanelFocus::LogViewer,
            &self.config.theme,
        );

        // Render status bar at the bottom of main area
        self.render_status_bar(frame, main_chunks[0]);

        // Render help overlay if active
        if self.show_help {
            self.render_help_overlay(frame, size);
        }

        // Render add node modal on top of everything
        if let Some(ref modal) = self.add_node_modal {
            modal.render(frame, size, &self.config.theme);
        }
    }

    /// Render status bar
    fn render_status_bar(&self, frame: &mut Frame, _area: Rect) {
        let size = frame.area();

        // Status bar at very bottom
        let status_area = Rect {
            x: 0,
            y: size.height.saturating_sub(1),
            width: size.width,
            height: 1,
        };

        let focus_name = match self.focus {
            PanelFocus::Dashboard => "Dashboard",
            PanelFocus::CommandMenu => "Commands",
            PanelFocus::LogViewer => "Logs",
            PanelFocus::Inventory => "Inventory",
        };

        let status_text = Line::from(vec![
            Span::styled(
                " HPC-AI TUI ",
                Style::default()
                    .fg(self.config.theme.background)
                    .bg(self.config.theme.primary)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                format!("Focus: {}", focus_name),
                Style::default().fg(self.config.theme.foreground),
            ),
            Span::raw(" | "),
            Span::styled(
                "Tab: switch panel",
                Style::default().fg(self.config.theme.border),
            ),
            Span::raw(" | "),
            Span::styled(
                "?: help",
                Style::default().fg(self.config.theme.border),
            ),
            Span::raw(" | "),
            Span::styled(
                "q: quit",
                Style::default().fg(self.config.theme.border),
            ),
        ]);

        let status_widget = Paragraph::new(status_text)
            .style(Style::default().bg(self.config.theme.secondary));

        frame.render_widget(status_widget, status_area);
    }

    /// Render help overlay
    fn render_help_overlay(&self, frame: &mut Frame, area: Rect) {
        // Calculate centered overlay
        let overlay_width = 50.min(area.width.saturating_sub(4));
        let overlay_height = 18.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(overlay_width)) / 2;
        let y = (area.height.saturating_sub(overlay_height)) / 2;

        let overlay_area = Rect {
            x,
            y,
            width: overlay_width,
            height: overlay_height,
        };

        // Clear background
        let block = Block::default()
            .title(" Keyboard Shortcuts ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.config.theme.primary))
            .style(Style::default().bg(self.config.theme.background));

        let help_text = vec![
            "",
            "  Navigation:",
            "    Tab / Shift+Tab  Switch panel focus",
            "    Arrow keys / hjkl  Move within panel",
            "    1 / 2 / 3        Switch menu tabs",
            "",
            "  Actions:",
            "    Enter / Space    Select / activate",
            "    r                Refresh data",
            "",
            "  General:",
            "    ?                Toggle this help",
            "    q / Esc          Quit",
            "",
            "  Press any key to close",
        ];

        let paragraph = Paragraph::new(help_text.join("\n"))
            .block(block)
            .style(Style::default().fg(self.config.theme.foreground));

        frame.render_widget(paragraph, overlay_area);
    }

    /// Handle events
    fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Key(key_event) => {
                use crossterm::event::KeyCode;

                // Handle add node modal input first if active
                if let Some(ref mut modal) = self.add_node_modal {
                    match key_event.code {
                        KeyCode::Char(c) => {
                            modal.handle_char(c);
                            return Ok(());
                        }
                        KeyCode::Backspace => {
                            modal.handle_backspace();
                            return Ok(());
                        }
                        _ => {
                            let action = KeyAction::from_key_event(&key_event);
                            match modal.handle_key(action) {
                                ModalAction::Cancel => {
                                    self.add_node_modal = None;
                                }
                                ModalAction::Submit => {
                                    // Handle add node submission
                                    self.handle_add_node_submit();
                                }
                                ModalAction::Continue => {}
                                ModalAction::Passthrough(action) => {
                                    // Handle passthrough action
                                    self.handle_passthrough_action(action);
                                }
                            }
                            return Ok(());
                        }
                    }
                }

                let action = KeyAction::from_key_event(&key_event);

                // Handle help overlay first
                if self.show_help {
                    self.show_help = false;
                    return Ok(());
                }

                // Global actions
                match action {
                    KeyAction::Quit => {
                        self.should_quit = true;
                    }
                    KeyAction::NextPanel => {
                        self.focus = self.focus.next();
                    }
                    KeyAction::PrevPanel => {
                        self.focus = self.focus.prev();
                    }
                    KeyAction::Help => {
                        self.show_help = true;
                    }
                    KeyAction::Refresh => {
                        self.update_state();
                    }
                    KeyAction::Add => {
                        // Open add node modal when in inventory panel
                        if self.focus == PanelFocus::Inventory {
                            self.add_node_modal = Some(AddNodeModal::new());
                        }
                    }
                    _ => {
                        // Delegate to focused panel
                        match self.focus {
                            PanelFocus::Dashboard => {
                                self.dashboard.handle_action(action);
                            }
                            PanelFocus::CommandMenu => {
                                self.command_menu.handle_action(action);
                            }
                            PanelFocus::LogViewer => {
                                self.log_viewer.handle_action(action);
                            }
                            PanelFocus::Inventory => {
                                self.inventory.handle_action(action);
                            }
                        }
                    }
                }
            }
            Event::Tick => {
                // Update panels from state
                self.update_panels();
                // Update modal animation
                if let Some(ref mut modal) = self.add_node_modal {
                    modal.tick();
                }
            }
            Event::Resize(_, _) => {
                // Terminal will handle resize automatically
            }
            Event::Mouse(_) => {
                // Mouse support can be added later
            }
        }

        Ok(())
    }

    /// Handle add node modal submit action
    fn handle_add_node_submit(&mut self) {
        if let Some(ref modal) = self.add_node_modal {
            if let Some(result) = modal.get_result() {
                // Create the node and save to inventory first
                match self.add_node_from_result(&result) {
                    Ok(node_id) => {
                        // Create progress channel
                        let (progress_tx, progress_rx) = mpsc::channel(16);

                        // Build bootstrap params
                        let credential = if let Some(key_path) = &result.key_path {
                            let expanded_path = if key_path.starts_with("~/") {
                                if let Some(home) = dirs::home_dir() {
                                    home.join(&key_path[2..])
                                } else {
                                    PathBuf::from(key_path)
                                }
                            } else {
                                PathBuf::from(key_path)
                            };
                            CredentialRef::SshKey { path: expanded_path }
                        } else {
                            CredentialRef::SshAgent
                        };

                        let params = BootstrapParams {
                            node_id: node_id.clone(),
                            address: result.address.clone(),
                            port: result.port,
                            username: result.username.clone(),
                            credential,
                            mode: result.mode.clone(),
                            name: result.name.clone().unwrap_or_else(|| node_id.clone()),
                        };

                        // Spawn bootstrap task
                        tokio::spawn(async move {
                            if let Err(e) = bootstrap_node_async(params, progress_tx).await {
                                // Error is already sent via progress channel
                                eprintln!("Bootstrap error: {}", e);
                            }
                        });

                        // Start bootstrap in modal (pass receiver)
                        if let Some(ref mut modal) = self.add_node_modal {
                            modal.start_bootstrap(progress_rx);
                        }

                        // Refresh inventory to show new node
                        self.inventory.refresh();
                    }
                    Err(e) => {
                        // Set error on modal
                        if let Some(ref mut modal) = self.add_node_modal {
                            modal.set_error(format!("Failed to save node: {}", e));
                        }
                    }
                }
            }
        }
    }

    /// Handle passthrough action from modal
    fn handle_passthrough_action(&mut self, action: KeyAction) {
        if action == KeyAction::Quit {
            self.should_quit = true;
        }
    }

    /// Open the add node modal
    pub fn open_add_node_modal(&mut self) {
        self.add_node_modal = Some(AddNodeModal::new());
    }

    /// Close the active modal
    pub fn close_modal(&mut self) {
        self.add_node_modal = None;
    }

    /// Add a node from the modal result, returns node ID
    fn add_node_from_result(&mut self, result: &AddNodeResult) -> Result<String> {
        let credential = if let Some(ref key_path) = result.key_path {
            // Expand ~ to home directory
            let expanded_path = if key_path.starts_with("~/") {
                if let Some(home) = dirs::home_dir() {
                    home.join(&key_path[2..])
                } else {
                    PathBuf::from(key_path)
                }
            } else {
                PathBuf::from(key_path)
            };
            CredentialRef::SshKey { path: expanded_path }
        } else {
            CredentialRef::SshAgent
        };

        let name = result.name.clone().unwrap_or_else(|| {
            format!("node-{}", &uuid::Uuid::new_v4().to_string()[..8])
        });

        let node = NodeInfo::new(
            name,
            result.address.clone(),
            result.port,
            result.username.clone(),
            credential,
            result.mode.clone(),
        );

        let node_id = node.id.clone();

        // Add to inventory store
        let mut store = InventoryStore::new()?;
        store.add_node(node)?;
        store.save()?;

        Ok(node_id)
    }

    /// Update application state
    fn update_state(&mut self) {
        // In a real implementation, this would fetch data from services
        // For now, we just update panels with current state
        self.update_panels();
    }

    /// Update all panels from app state
    fn update_panels(&mut self) {
        self.dashboard.update(&self.state);
        self.command_menu.update(&self.state);
        self.log_viewer.update(&self.state);
        self.inventory.update(&self.state);
    }
}

/// Launch the TUI application
pub async fn launch(config: AppConfig) -> Result<()> {
    let mut app = TuiApp::new(config)?;
    app.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_app_creation() {
        // Note: Can't fully test TUI without a terminal
        // This just tests the struct can be instantiated conceptually
        let config = AppConfig::default();
        // TuiApp::new requires tokio runtime for event handler
        // so we just verify config conversion works
        let tui_config = TuiConfig::from(&config);
        assert_eq!(tui_config.tick_rate_ms, config.tui.tick_rate_ms);
    }
}
