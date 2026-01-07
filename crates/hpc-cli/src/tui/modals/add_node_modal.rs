//! Add Node modal dialog
//!
//! Interactive form for adding new nodes to the inventory.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};
use tokio::sync::mpsc;

use crate::core::inventory::{BootstrapProgress, BootstrapStage, NodeMode};
use crate::tui::{events::KeyAction, Theme};

use super::{InputField, Modal, ModalAction, SelectField};

/// Current state of the add node modal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddNodeState {
    /// Editing form fields
    Editing,
    /// Bootstrap in progress
    Bootstrapping,
    /// Successfully added
    Complete,
    /// Error occurred
    Error(String),
}

/// Progress step for display
#[derive(Debug, Clone)]
struct ProgressStep {
    stage: BootstrapStage,
    message: String,
    details: Option<String>,
    completed: bool,
}

/// Which field is currently focused
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddNodeField {
    Address,
    Port,
    Username,
    KeyPath,
    Name,
    Mode,
    Cancel,
    Submit,
}

impl AddNodeField {
    fn next(&self) -> Self {
        match self {
            Self::Address => Self::Port,
            Self::Port => Self::Username,
            Self::Username => Self::KeyPath,
            Self::KeyPath => Self::Name,
            Self::Name => Self::Mode,
            Self::Mode => Self::Cancel,
            Self::Cancel => Self::Submit,
            Self::Submit => Self::Address,
        }
    }

    fn prev(&self) -> Self {
        match self {
            Self::Address => Self::Submit,
            Self::Port => Self::Address,
            Self::Username => Self::Port,
            Self::KeyPath => Self::Username,
            Self::Name => Self::KeyPath,
            Self::Mode => Self::Name,
            Self::Cancel => Self::Mode,
            Self::Submit => Self::Cancel,
        }
    }
}

/// Result of a successful add node operation
#[derive(Debug, Clone)]
pub struct AddNodeResult {
    pub address: String,
    pub port: u16,
    pub username: String,
    pub key_path: Option<String>,
    pub name: Option<String>,
    pub mode: NodeMode,
}

/// Add Node modal dialog
pub struct AddNodeModal {
    /// Current state
    state: AddNodeState,
    /// Currently focused field
    focused_field: AddNodeField,
    /// IP/hostname field
    address_field: InputField,
    /// SSH port field
    port_field: InputField,
    /// Username field
    username_field: InputField,
    /// SSH key path field (optional)
    key_path_field: InputField,
    /// Node name field (optional)
    name_field: InputField,
    /// Deployment mode selector
    mode_field: SelectField<NodeMode>,
    /// Status message during connection
    status_message: Option<String>,
    /// Animation tick counter
    tick_count: u32,
    /// Progress receiver for bootstrap updates
    progress_rx: Option<mpsc::Receiver<BootstrapProgress>>,
    /// Completed progress steps
    progress_steps: Vec<ProgressStep>,
    /// Current bootstrap stage
    current_stage: Option<BootstrapStage>,
}

impl AddNodeModal {
    /// Create a new add node modal
    pub fn new() -> Self {
        Self {
            state: AddNodeState::Editing,
            focused_field: AddNodeField::Address,
            address_field: InputField::new("IP / Hostname")
                .with_placeholder("192.168.1.100 or node.example.com")
                .required(),
            port_field: InputField::new("SSH Port")
                .with_value("22")
                .with_max_length(5),
            username_field: InputField::new("Username")
                .with_placeholder("admin")
                .required(),
            key_path_field: InputField::new("SSH Key Path")
                .with_placeholder("~/.ssh/id_ed25519 (empty for SSH agent)"),
            name_field: InputField::new("Node Name")
                .with_placeholder("Auto-generated if empty"),
            mode_field: SelectField::new("Deploy Mode", vec![NodeMode::Docker, NodeMode::Binary]),
            status_message: None,
            tick_count: 0,
            progress_rx: None,
            progress_steps: Vec::new(),
            current_stage: None,
        }
    }

    /// Get the result if form is valid
    pub fn get_result(&self) -> Option<AddNodeResult> {
        if !self.validate_all() {
            return None;
        }

        let port = self.port_field.value.parse::<u16>().unwrap_or(22);
        let key_path = if self.key_path_field.value.trim().is_empty() {
            None
        } else {
            Some(self.key_path_field.value.clone())
        };
        let name = if self.name_field.value.trim().is_empty() {
            None
        } else {
            Some(self.name_field.value.clone())
        };

        Some(AddNodeResult {
            address: self.address_field.value.clone(),
            port,
            username: self.username_field.value.clone(),
            key_path,
            name,
            mode: self.mode_field.value_cloned().unwrap_or(NodeMode::Docker),
        })
    }

    /// Validate all required fields
    fn validate_all(&self) -> bool {
        !self.address_field.value.trim().is_empty()
            && !self.username_field.value.trim().is_empty()
            && self.port_field.value.parse::<u16>().is_ok()
    }

    /// Validate and mark errors on fields
    fn validate_with_errors(&mut self) -> bool {
        let mut valid = true;

        if self.address_field.value.trim().is_empty() {
            self.address_field.set_error("IP or hostname is required");
            valid = false;
        } else {
            self.address_field.clear_error();
        }

        if self.username_field.value.trim().is_empty() {
            self.username_field.set_error("Username is required");
            valid = false;
        } else {
            self.username_field.clear_error();
        }

        if self.port_field.value.parse::<u16>().is_err() {
            self.port_field.set_error("Invalid port number");
            valid = false;
        } else {
            self.port_field.clear_error();
        }

        valid
    }

    /// Start bootstrap with progress channel
    pub fn start_bootstrap(&mut self, progress_rx: mpsc::Receiver<BootstrapProgress>) {
        self.state = AddNodeState::Bootstrapping;
        self.progress_rx = Some(progress_rx);
        self.progress_steps.clear();
        self.current_stage = Some(BootstrapStage::Saving);
        self.status_message = Some("Starting bootstrap...".to_string());
    }

    /// Update progress from bootstrap task
    fn update_progress(&mut self, progress: BootstrapProgress) {
        // Add to progress steps
        let completed = progress.stage.is_terminal() ||
            matches!(progress.stage, BootstrapStage::Complete);

        // Mark previous step as completed
        if let Some(last) = self.progress_steps.last_mut() {
            if !last.completed {
                last.completed = true;
            }
        }

        // Add new step
        self.progress_steps.push(ProgressStep {
            stage: progress.stage.clone(),
            message: progress.message.clone(),
            details: progress.details.clone(),
            completed,
        });

        self.current_stage = Some(progress.stage.clone());
        self.status_message = Some(progress.message);

        // Handle terminal states
        match &progress.stage {
            BootstrapStage::Complete => {
                self.state = AddNodeState::Complete;
            }
            BootstrapStage::Failed(err) => {
                self.state = AddNodeState::Error(err.clone());
            }
            _ => {}
        }
    }

    /// Set error state
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.state = AddNodeState::Error(error.into());
    }

    /// Set complete state
    pub fn set_complete(&mut self) {
        self.state = AddNodeState::Complete;
        self.status_message = Some("Node added successfully!".to_string());
    }

    /// Reset to editing state
    pub fn reset_to_editing(&mut self) {
        self.state = AddNodeState::Editing;
        self.status_message = None;
        self.progress_rx = None;
        self.progress_steps.clear();
        self.current_stage = None;
    }

    /// Update focus indicators on fields
    fn update_focus(&mut self) {
        self.address_field.focused = self.focused_field == AddNodeField::Address;
        self.port_field.focused = self.focused_field == AddNodeField::Port;
        self.username_field.focused = self.focused_field == AddNodeField::Username;
        self.key_path_field.focused = self.focused_field == AddNodeField::KeyPath;
        self.name_field.focused = self.focused_field == AddNodeField::Name;
        self.mode_field.focused = self.focused_field == AddNodeField::Mode;
    }

    /// Get loading animation character
    fn loading_char(&self) -> &'static str {
        const CHARS: &[&str] = &["◐", "◓", "◑", "◒"];
        CHARS[(self.tick_count as usize / 2) % CHARS.len()]
    }

    /// Render the progress steps during bootstrap
    fn render_progress(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let total_stages = BootstrapStage::total();

        // Build lines for each stage
        let mut lines: Vec<Line> = Vec::new();

        // Current status line with spinner
        if let Some(msg) = &self.status_message {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{} ", self.loading_char()),
                    Style::default().fg(theme.primary),
                ),
                Span::styled(msg, Style::default().fg(theme.foreground)),
            ]));
        }

        lines.push(Line::from("")); // Separator

        // Stage checklist
        let stages = [
            (BootstrapStage::Saving, "Save to inventory"),
            (BootstrapStage::Connecting, "SSH connect"),
            (BootstrapStage::DetectingPlatform, "Detect platform"),
            (BootstrapStage::DetectingHardware, "Detect hardware"),
            (BootstrapStage::InstallingAgent, "Install agent"),
            (BootstrapStage::VerifyingAgent, "Verify agent"),
        ];

        for (i, (stage, label)) in stages.iter().enumerate() {
            let stage_num = i + 1;

            // Find matching progress step
            let step = self.progress_steps.iter().find(|s| &s.stage == stage);

            let (icon, style) = if let Some(s) = step {
                if s.completed || self.is_stage_completed(stage) {
                    ("✓", Style::default().fg(theme.success))
                } else {
                    (self.loading_char(), Style::default().fg(theme.primary))
                }
            } else if self.is_stage_pending(stage) {
                ("○", Style::default().fg(theme.border))
            } else {
                ("○", Style::default().fg(theme.border))
            };

            let mut spans = vec![
                Span::styled(
                    format!("[{}/{}] ", stage_num, total_stages),
                    Style::default().fg(theme.border),
                ),
                Span::styled(format!("{} ", icon), style),
                Span::styled(*label, Style::default().fg(theme.foreground)),
            ];

            // Add details if available
            if let Some(s) = step {
                if let Some(details) = &s.details {
                    spans.push(Span::styled(
                        format!(": {}", details),
                        Style::default().fg(theme.border),
                    ));
                }
            }

            lines.push(Line::from(spans));
        }

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, area);
    }

    /// Check if a stage is completed based on current progress
    fn is_stage_completed(&self, target: &BootstrapStage) -> bool {
        if let Some(current) = &self.current_stage {
            target.number() < current.number()
        } else {
            false
        }
    }

    /// Check if a stage is pending (not yet reached)
    fn is_stage_pending(&self, target: &BootstrapStage) -> bool {
        if let Some(current) = &self.current_stage {
            target.number() > current.number()
        } else {
            true
        }
    }
}

impl Default for AddNodeModal {
    fn default() -> Self {
        Self::new()
    }
}

impl Modal for AddNodeModal {
    fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        // Calculate modal dimensions (centered) - larger when bootstrapping to show progress
        let is_bootstrapping = matches!(self.state, AddNodeState::Bootstrapping);
        let modal_height = if is_bootstrapping {
            28.min(area.height.saturating_sub(4))
        } else {
            22.min(area.height.saturating_sub(4))
        };
        let modal_width = 60.min(area.width.saturating_sub(4));
        let x = (area.width.saturating_sub(modal_width)) / 2;
        let y = (area.height.saturating_sub(modal_height)) / 2;

        let modal_area = Rect {
            x,
            y,
            width: modal_width,
            height: modal_height,
        };

        // Clear the area behind the modal
        frame.render_widget(Clear, modal_area);

        // Outer border
        let block = Block::default()
            .title(" Add Node ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.primary))
            .style(Style::default().bg(theme.background));
        frame.render_widget(block, modal_area);

        // Inner content area
        let inner = Rect {
            x: modal_area.x + 2,
            y: modal_area.y + 1,
            width: modal_area.width.saturating_sub(4),
            height: modal_area.height.saturating_sub(2),
        };

        // Layout for form fields
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Address
                Constraint::Length(3), // Port + Username row
                Constraint::Length(3), // Key path
                Constraint::Length(3), // Name
                Constraint::Length(3), // Mode
                Constraint::Length(1), // Separator
                Constraint::Min(2),    // Status/Progress area
                Constraint::Length(1), // Buttons
                Constraint::Length(1), // Help
            ])
            .split(inner);

        // Render fields
        // Clone self to get mutable access for rendering
        let mut address_field = self.address_field.clone();
        let mut port_field = self.port_field.clone();
        let mut username_field = self.username_field.clone();
        let mut key_path_field = self.key_path_field.clone();
        let mut name_field = self.name_field.clone();
        let mut mode_field = self.mode_field.clone();

        address_field.focused = self.focused_field == AddNodeField::Address;
        port_field.focused = self.focused_field == AddNodeField::Port;
        username_field.focused = self.focused_field == AddNodeField::Username;
        key_path_field.focused = self.focused_field == AddNodeField::KeyPath;
        name_field.focused = self.focused_field == AddNodeField::Name;
        mode_field.focused = self.focused_field == AddNodeField::Mode;

        address_field.render(frame, layout[0], theme);

        // Port and username side by side
        let port_user_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(15), Constraint::Min(10)])
            .split(layout[1]);

        port_field.render(frame, port_user_layout[0], theme);
        username_field.render(frame, port_user_layout[1], theme);

        key_path_field.render(frame, layout[2], theme);
        name_field.render(frame, layout[3], theme);
        mode_field.render(frame, layout[4], theme);

        // Status/Progress area
        match &self.state {
            AddNodeState::Bootstrapping => {
                self.render_progress(frame, layout[6], theme);
            }
            AddNodeState::Error(err) => {
                let error_text = Line::from(Span::styled(
                    format!("✗ Error: {}", err),
                    Style::default().fg(theme.error),
                ));
                let error_para = Paragraph::new(error_text);
                frame.render_widget(error_para, layout[6]);
            }
            AddNodeState::Complete => {
                let success_text = Line::from(Span::styled(
                    "✓ Node added and bootstrapped successfully!",
                    Style::default().fg(theme.success),
                ));
                let success_para = Paragraph::new(success_text);
                frame.render_widget(success_para, layout[6]);
            }
            AddNodeState::Editing => {
                // Empty status area when editing
            }
        }

        // Buttons
        let cancel_style = if self.focused_field == AddNodeField::Cancel {
            Style::default()
                .fg(theme.background)
                .bg(theme.warning)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.foreground)
        };

        let submit_style = if self.focused_field == AddNodeField::Submit {
            Style::default()
                .fg(theme.background)
                .bg(theme.primary)
                .add_modifier(Modifier::BOLD)
        } else if self.validate_all() {
            Style::default().fg(theme.primary)
        } else {
            Style::default().fg(theme.border)
        };

        // Show different buttons during bootstrap
        let buttons = if is_bootstrapping {
            Line::from(vec![
                Span::raw("                 "),
                Span::styled(" Cancel ", cancel_style),
            ])
        } else {
            Line::from(vec![
                Span::raw("       "),
                Span::styled(" Cancel ", cancel_style),
                Span::raw("          "),
                Span::styled(" Add Node ", submit_style),
            ])
        };
        let buttons_para = Paragraph::new(buttons);
        frame.render_widget(buttons_para, layout[7]);

        // Help text
        let help = if is_bootstrapping {
            Line::from(vec![
                Span::styled("Esc", Style::default().fg(theme.border)),
                Span::raw(": cancel bootstrap"),
            ])
        } else {
            Line::from(vec![
                Span::styled("Tab", Style::default().fg(theme.border)),
                Span::raw(": next  "),
                Span::styled("Shift+Tab", Style::default().fg(theme.border)),
                Span::raw(": prev  "),
                Span::styled("Esc", Style::default().fg(theme.border)),
                Span::raw(": cancel"),
            ])
        };
        let help_para =
            Paragraph::new(help).style(Style::default().fg(theme.border).add_modifier(Modifier::DIM));
        frame.render_widget(help_para, layout[8]);
    }

    fn handle_key(&mut self, action: KeyAction) -> ModalAction {
        // Handle error/complete states
        if matches!(self.state, AddNodeState::Error(_)) {
            // Any key returns to editing
            self.reset_to_editing();
            return ModalAction::Continue;
        }

        if self.state == AddNodeState::Complete {
            return ModalAction::Cancel; // Close modal on success
        }

        // Block input during bootstrap
        if matches!(self.state, AddNodeState::Bootstrapping) {
            if action == KeyAction::Quit || action == KeyAction::Back {
                self.reset_to_editing();
                return ModalAction::Continue;
            }
            return ModalAction::Continue;
        }

        match action {
            KeyAction::Quit | KeyAction::Back => ModalAction::Cancel,

            KeyAction::NextPanel => {
                self.focused_field = self.focused_field.next();
                self.update_focus();
                ModalAction::Continue
            }

            KeyAction::PrevPanel => {
                self.focused_field = self.focused_field.prev();
                self.update_focus();
                ModalAction::Continue
            }

            KeyAction::Up => {
                self.focused_field = self.focused_field.prev();
                self.update_focus();
                ModalAction::Continue
            }

            KeyAction::Down => {
                self.focused_field = self.focused_field.next();
                self.update_focus();
                ModalAction::Continue
            }

            KeyAction::Left => {
                if self.focused_field == AddNodeField::Mode {
                    self.mode_field.prev();
                }
                ModalAction::Continue
            }

            KeyAction::Right => {
                if self.focused_field == AddNodeField::Mode {
                    self.mode_field.next();
                }
                ModalAction::Continue
            }

            KeyAction::Select => {
                if self.focused_field == AddNodeField::Cancel {
                    return ModalAction::Cancel;
                }
                if self.focused_field == AddNodeField::Submit {
                    if self.validate_with_errors() {
                        return ModalAction::Submit;
                    }
                }
                if self.focused_field == AddNodeField::Mode {
                    self.mode_field.next();
                }
                ModalAction::Continue
            }

            _ => ModalAction::Continue,
        }
    }

    fn handle_char(&mut self, c: char) {
        if !matches!(self.state, AddNodeState::Editing) {
            return;
        }

        match self.focused_field {
            AddNodeField::Address => self.address_field.insert_char(c),
            AddNodeField::Port => {
                if c.is_ascii_digit() {
                    self.port_field.insert_char(c);
                }
            }
            AddNodeField::Username => self.username_field.insert_char(c),
            AddNodeField::KeyPath => self.key_path_field.insert_char(c),
            AddNodeField::Name => self.name_field.insert_char(c),
            AddNodeField::Mode => {
                // Space toggles mode
                if c == ' ' {
                    self.mode_field.next();
                }
            }
            AddNodeField::Cancel | AddNodeField::Submit => {}
        }
    }

    fn handle_backspace(&mut self) {
        if !matches!(self.state, AddNodeState::Editing) {
            return;
        }

        match self.focused_field {
            AddNodeField::Address => self.address_field.delete_char(),
            AddNodeField::Port => self.port_field.delete_char(),
            AddNodeField::Username => self.username_field.delete_char(),
            AddNodeField::KeyPath => self.key_path_field.delete_char(),
            AddNodeField::Name => self.name_field.delete_char(),
            _ => {}
        }
    }

    fn title(&self) -> &str {
        "Add Node"
    }

    fn can_submit(&self) -> bool {
        self.validate_all() && matches!(self.state, AddNodeState::Editing)
    }

    fn is_loading(&self) -> bool {
        matches!(self.state, AddNodeState::Bootstrapping)
    }

    fn tick(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);

        // Poll progress channel during bootstrap
        // Collect updates first to avoid borrow conflicts
        let updates: Vec<_> = if let Some(ref mut rx) = self.progress_rx {
            let mut updates = Vec::new();
            while let Ok(progress) = rx.try_recv() {
                updates.push(progress);
            }
            updates
        } else {
            Vec::new()
        };

        // Apply updates
        for progress in updates {
            self.update_progress(progress);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_modal_new() {
        let modal = AddNodeModal::new();
        assert_eq!(modal.state, AddNodeState::Editing);
        assert_eq!(modal.focused_field, AddNodeField::Address);
        assert_eq!(modal.port_field.value, "22");
    }

    #[test]
    fn test_field_navigation() {
        let mut modal = AddNodeModal::new();

        modal.handle_key(KeyAction::NextPanel);
        assert_eq!(modal.focused_field, AddNodeField::Port);

        modal.handle_key(KeyAction::NextPanel);
        assert_eq!(modal.focused_field, AddNodeField::Username);

        modal.handle_key(KeyAction::PrevPanel);
        assert_eq!(modal.focused_field, AddNodeField::Port);
    }

    #[test]
    fn test_char_input() {
        let mut modal = AddNodeModal::new();

        modal.handle_char('1');
        modal.handle_char('9');
        modal.handle_char('2');
        modal.handle_char('.');
        modal.handle_char('1');
        modal.handle_char('6');
        modal.handle_char('8');

        assert_eq!(modal.address_field.value, "192.168");
    }

    #[test]
    fn test_validation() {
        let mut modal = AddNodeModal::new();
        assert!(!modal.validate_all());

        modal.address_field.value = "192.168.1.1".to_string();
        assert!(!modal.validate_all());

        modal.username_field.value = "admin".to_string();
        assert!(modal.validate_all());
    }

    #[test]
    fn test_get_result() {
        let mut modal = AddNodeModal::new();
        assert!(modal.get_result().is_none());

        modal.address_field.value = "192.168.1.100".to_string();
        modal.username_field.value = "admin".to_string();
        modal.port_field.value = "22".to_string();

        let result = modal.get_result().unwrap();
        assert_eq!(result.address, "192.168.1.100");
        assert_eq!(result.username, "admin");
        assert_eq!(result.port, 22);
        assert!(result.key_path.is_none());
        assert!(result.name.is_none());
    }

    #[test]
    fn test_state_transitions() {
        let mut modal = AddNodeModal::new();

        // Test bootstrap progress
        let (tx, rx) = mpsc::channel(16);
        modal.start_bootstrap(rx);
        assert_eq!(modal.state, AddNodeState::Bootstrapping);
        assert!(modal.is_loading());

        modal.set_error("Connection failed");
        assert!(matches!(modal.state, AddNodeState::Error(_)));

        modal.reset_to_editing();
        assert_eq!(modal.state, AddNodeState::Editing);

        // Clean up sender
        drop(tx);
    }

    #[test]
    fn test_cancel_action() {
        let mut modal = AddNodeModal::new();
        let action = modal.handle_key(KeyAction::Quit);
        assert!(matches!(action, ModalAction::Cancel));
    }

    #[test]
    fn test_mode_selection() {
        let mut modal = AddNodeModal::new();

        // Navigate to mode field
        modal.focused_field = AddNodeField::Mode;
        modal.update_focus();

        assert_eq!(modal.mode_field.value(), Some(&NodeMode::Docker));

        modal.handle_key(KeyAction::Right);
        assert_eq!(modal.mode_field.value(), Some(&NodeMode::Binary));

        modal.handle_key(KeyAction::Left);
        assert_eq!(modal.mode_field.value(), Some(&NodeMode::Docker));
    }
}
