//! Command menu panel for the HPC-AI TUI
//!
//! Provides tabbed navigation:
//! - Commands: Quick access to common commands
//! - Deploy: Project deployment interface
//! - Projects: Project status overview

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs},
    Frame,
};

use crate::core::{project::get_all_projects, AppState};
use crate::tui::{events::KeyAction, Theme};

use super::Panel;

/// Command menu tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MenuTab {
    #[default]
    Commands,
    Deploy,
    Projects,
}

impl MenuTab {
    fn titles() -> Vec<&'static str> {
        vec!["Commands", "Deploy", "Projects"]
    }

    fn index(&self) -> usize {
        match self {
            Self::Commands => 0,
            Self::Deploy => 1,
            Self::Projects => 2,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Commands,
            1 => Self::Deploy,
            2 => Self::Projects,
            _ => Self::Commands,
        }
    }
}

/// Command menu item
#[derive(Debug, Clone)]
pub struct MenuItem {
    /// Display label
    pub label: String,
    /// Command to execute
    pub command: String,
    /// Description
    pub description: String,
    /// Whether item is enabled
    pub enabled: bool,
}

/// Project item for deploy/projects tabs
#[derive(Debug, Clone)]
pub struct ProjectItem {
    /// Project ID (e.g., "rnccl", "slai")
    pub id: String,
    /// Display name
    pub name: String,
    /// Whether selected for deployment
    pub selected: bool,
    /// Whether project is available
    pub available: bool,
}

/// Command menu panel state
#[derive(Debug)]
pub struct CommandMenuPanel {
    /// Current tab
    current_tab: MenuTab,
    /// Command items
    commands: Vec<MenuItem>,
    /// Project items
    projects: Vec<ProjectItem>,
    /// Selected index in current list
    selected_index: usize,
    /// Scroll offset
    scroll_offset: usize,
}

impl Default for CommandMenuPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandMenuPanel {
    /// Create a new command menu panel
    pub fn new() -> Self {
        let commands = vec![
            MenuItem {
                label: "slai detect".to_string(),
                command: "hpc slai detect".to_string(),
                description: "Detect available GPUs".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "swarm status".to_string(),
                command: "hpc swarm status".to_string(),
                description: "Show cluster status".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "argus status".to_string(),
                command: "hpc argus status".to_string(),
                description: "Observability status".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "torch models".to_string(),
                command: "hpc torch models".to_string(),
                description: "List ML models".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "rnccl info".to_string(),
                command: "hpc rnccl info".to_string(),
                description: "GPU collective info".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "stack list".to_string(),
                command: "hpc stack list".to_string(),
                description: "List stacks".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "deploy status".to_string(),
                command: "hpc deploy status".to_string(),
                description: "Deployment status".to_string(),
                enabled: true,
            },
            MenuItem {
                label: "info".to_string(),
                command: "hpc info".to_string(),
                description: "System information".to_string(),
                enabled: true,
            },
        ];

        // Initialize projects from registry
        let projects: Vec<ProjectItem> = get_all_projects()
            .into_iter()
            .filter(|p| p.deployable)
            .map(|p| ProjectItem {
                id: p.id.clone(),
                name: p.name.clone(),
                selected: false,
                available: true,
            })
            .collect();

        Self {
            current_tab: MenuTab::Commands,
            commands,
            projects,
            selected_index: 0,
            scroll_offset: 0,
        }
    }

    /// Get selected projects for deployment
    pub fn get_selected_projects(&self) -> Vec<String> {
        self.projects
            .iter()
            .filter(|p| p.selected)
            .map(|p| p.id.clone())
            .collect()
    }

    /// Render tabs
    fn render_tabs(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let titles: Vec<Line> = MenuTab::titles()
            .iter()
            .map(|t| Line::from(*t))
            .collect();

        let tabs = Tabs::new(titles)
            .select(self.current_tab.index())
            .style(Style::default().fg(theme.foreground))
            .highlight_style(
                Style::default()
                    .fg(theme.primary)
                    .add_modifier(Modifier::BOLD),
            )
            .divider("|");

        frame.render_widget(tabs, area);
    }

    /// Render commands list
    fn render_commands(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let items: Vec<ListItem> = self
            .commands
            .iter()
            .enumerate()
            .skip(self.scroll_offset)
            .map(|(i, cmd)| {
                let style = if i == self.selected_index && focused {
                    Style::default()
                        .fg(theme.foreground)
                        .bg(theme.secondary)
                        .add_modifier(Modifier::BOLD)
                } else if cmd.enabled {
                    Style::default().fg(theme.foreground)
                } else {
                    Style::default().fg(theme.border)
                };

                let prefix = if i == self.selected_index && focused {
                    "> "
                } else {
                    "  "
                };

                ListItem::new(Line::from(vec![
                    Span::raw(prefix),
                    Span::styled(&cmd.label, style),
                ]))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, area);
    }

    /// Render deploy tab (project selector)
    fn render_deploy(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(area);

        // Project list
        let items: Vec<ListItem> = self
            .projects
            .iter()
            .enumerate()
            .skip(self.scroll_offset)
            .map(|(i, proj)| {
                let checkbox = if proj.selected { "[x]" } else { "[ ]" };

                let style = if i == self.selected_index && focused {
                    Style::default()
                        .fg(theme.foreground)
                        .bg(theme.secondary)
                        .add_modifier(Modifier::BOLD)
                } else if proj.available {
                    Style::default().fg(theme.foreground)
                } else {
                    Style::default().fg(theme.border)
                };

                ListItem::new(Line::from(vec![
                    Span::styled(
                        checkbox,
                        Style::default().fg(if proj.selected {
                            theme.success
                        } else {
                            theme.foreground
                        }),
                    ),
                    Span::raw(" "),
                    Span::styled(&proj.name, style),
                ]))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, chunks[0]);

        // Action hint
        let selected_count = self.projects.iter().filter(|p| p.selected).count();
        let hint = if selected_count > 0 {
            format!(
                "{} selected - Press Enter to deploy",
                selected_count
            )
        } else {
            "Space to select, Enter to deploy".to_string()
        };

        let hint_widget = Paragraph::new(hint)
            .style(Style::default().fg(theme.primary))
            .block(Block::default().borders(Borders::TOP));
        frame.render_widget(hint_widget, chunks[1]);
    }

    /// Render projects status tab
    fn render_projects(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let items: Vec<ListItem> = self
            .projects
            .iter()
            .enumerate()
            .skip(self.scroll_offset)
            .map(|(i, proj)| {
                let status = if proj.available { "OK" } else { "--" };
                let status_color = if proj.available {
                    theme.success
                } else {
                    theme.border
                };

                let style = if i == self.selected_index && focused {
                    Style::default()
                        .fg(theme.foreground)
                        .bg(theme.secondary)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.foreground)
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:2} ", status), Style::default().fg(status_color)),
                    Span::styled(&proj.name, style),
                ]))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, area);
    }

    /// Get current list length
    fn current_list_len(&self) -> usize {
        match self.current_tab {
            MenuTab::Commands => self.commands.len(),
            MenuTab::Deploy | MenuTab::Projects => self.projects.len(),
        }
    }
}

impl Panel for CommandMenuPanel {
    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let border_color = if focused {
            theme.highlight
        } else {
            theme.border
        };

        let block = Block::default()
            .title(" Command Menu ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Split for tabs and content
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(inner);

        self.render_tabs(frame, chunks[0], theme);

        match self.current_tab {
            MenuTab::Commands => self.render_commands(frame, chunks[1], focused, theme),
            MenuTab::Deploy => self.render_deploy(frame, chunks[1], focused, theme),
            MenuTab::Projects => self.render_projects(frame, chunks[1], focused, theme),
        }
    }

    fn handle_action(&mut self, action: KeyAction) -> bool {
        match action {
            KeyAction::Tab1 => {
                self.current_tab = MenuTab::Commands;
                self.selected_index = 0;
                self.scroll_offset = 0;
                true
            }
            KeyAction::Tab2 => {
                self.current_tab = MenuTab::Deploy;
                self.selected_index = 0;
                self.scroll_offset = 0;
                true
            }
            KeyAction::Tab3 => {
                self.current_tab = MenuTab::Projects;
                self.selected_index = 0;
                self.scroll_offset = 0;
                true
            }
            KeyAction::Left => {
                let current = self.current_tab.index();
                if current > 0 {
                    self.current_tab = MenuTab::from_index(current - 1);
                    self.selected_index = 0;
                    self.scroll_offset = 0;
                }
                true
            }
            KeyAction::Right => {
                let current = self.current_tab.index();
                if current < 2 {
                    self.current_tab = MenuTab::from_index(current + 1);
                    self.selected_index = 0;
                    self.scroll_offset = 0;
                }
                true
            }
            KeyAction::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                    if self.selected_index < self.scroll_offset {
                        self.scroll_offset = self.selected_index;
                    }
                }
                true
            }
            KeyAction::Down => {
                let max = self.current_list_len().saturating_sub(1);
                if self.selected_index < max {
                    self.selected_index += 1;
                }
                true
            }
            KeyAction::Select => {
                // Toggle selection in Deploy tab
                if self.current_tab == MenuTab::Deploy {
                    if let Some(proj) = self.projects.get_mut(self.selected_index) {
                        proj.selected = !proj.selected;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, _state: &AppState) {
        // Could update project availability from state
    }

    fn title(&self) -> &str {
        "Commands"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_menu_tab_cycle() {
        assert_eq!(MenuTab::from_index(0), MenuTab::Commands);
        assert_eq!(MenuTab::from_index(1), MenuTab::Deploy);
        assert_eq!(MenuTab::from_index(2), MenuTab::Projects);
        assert_eq!(MenuTab::from_index(99), MenuTab::Commands); // Default
    }

    #[test]
    fn test_command_menu_new() {
        let panel = CommandMenuPanel::new();
        assert!(!panel.commands.is_empty());
        assert!(!panel.projects.is_empty());
        assert_eq!(panel.current_tab, MenuTab::Commands);
    }

    #[test]
    fn test_command_menu_navigation() {
        let mut panel = CommandMenuPanel::new();
        assert_eq!(panel.selected_index, 0);

        panel.handle_action(KeyAction::Down);
        assert_eq!(panel.selected_index, 1);

        panel.handle_action(KeyAction::Up);
        assert_eq!(panel.selected_index, 0);
    }

    #[test]
    fn test_command_menu_tab_switch() {
        let mut panel = CommandMenuPanel::new();
        assert_eq!(panel.current_tab, MenuTab::Commands);

        panel.handle_action(KeyAction::Tab2);
        assert_eq!(panel.current_tab, MenuTab::Deploy);

        panel.handle_action(KeyAction::Tab3);
        assert_eq!(panel.current_tab, MenuTab::Projects);

        panel.handle_action(KeyAction::Tab1);
        assert_eq!(panel.current_tab, MenuTab::Commands);
    }

    #[test]
    fn test_project_selection() {
        let mut panel = CommandMenuPanel::new();
        panel.handle_action(KeyAction::Tab2); // Switch to Deploy tab

        // Select first project
        assert!(!panel.projects[0].selected);
        panel.handle_action(KeyAction::Select);
        assert!(panel.projects[0].selected);

        // Toggle off
        panel.handle_action(KeyAction::Select);
        assert!(!panel.projects[0].selected);
    }
}
