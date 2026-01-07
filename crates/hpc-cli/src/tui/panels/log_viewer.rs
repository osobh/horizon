//! Log viewer panel for the HPC-AI TUI
//!
//! Displays real-time log streaming with:
//! - Color-coded log levels
//! - Timestamp display
//! - Scroll and filter support

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::core::{
    state::{LogEntry, LogLevel},
    AppState,
};
use crate::tui::{events::KeyAction, Theme};

use super::Panel;

/// Log viewer panel state
#[derive(Debug, Default)]
pub struct LogViewerPanel {
    /// Log entries
    entries: Vec<LogEntry>,
    /// Maximum entries to keep
    max_entries: usize,
    /// Scroll offset (from bottom)
    scroll_offset: usize,
    /// Auto-scroll mode
    auto_scroll: bool,
    /// Filter level (None = show all)
    filter_level: Option<LogLevel>,
}

impl LogViewerPanel {
    /// Create a new log viewer panel
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            scroll_offset: 0,
            auto_scroll: true,
            filter_level: None,
        }
    }

    /// Add a log entry
    pub fn add_entry(&mut self, entry: LogEntry) {
        self.entries.push(entry);

        // Trim old entries
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        // Reset scroll if auto-scroll is enabled
        if self.auto_scroll {
            self.scroll_offset = 0;
        }
    }

    /// Get filtered entries
    fn filtered_entries(&self) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| {
                self.filter_level
                    .as_ref()
                    .map(|level| &e.level == level)
                    .unwrap_or(true)
            })
            .collect()
    }

    /// Format a log entry as a line
    fn format_entry<'a>(entry: &'a LogEntry, theme: &Theme) -> Line<'a> {
        let level_style = match entry.level {
            LogLevel::Error => Style::default().fg(theme.error).add_modifier(Modifier::BOLD),
            LogLevel::Warn => Style::default().fg(theme.warning),
            LogLevel::Info => Style::default().fg(theme.primary),
            LogLevel::Debug => Style::default().fg(theme.border),
        };

        let timestamp = entry.timestamp.format("%H:%M:%S");

        Line::from(vec![
            Span::styled(
                format!("[{}]", timestamp),
                Style::default().fg(theme.border),
            ),
            Span::raw(" "),
            Span::styled(format!("{:5}", entry.level.as_str()), level_style),
            Span::raw(" "),
            Span::raw(&entry.message),
        ])
    }

    /// Toggle auto-scroll
    pub fn toggle_auto_scroll(&mut self) {
        self.auto_scroll = !self.auto_scroll;
    }

    /// Set filter level
    pub fn set_filter(&mut self, level: Option<LogLevel>) {
        self.filter_level = level;
    }

    /// Clear all logs
    pub fn clear(&mut self) {
        self.entries.clear();
        self.scroll_offset = 0;
    }
}

impl Panel for LogViewerPanel {
    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let border_color = if focused {
            theme.highlight
        } else {
            theme.border
        };

        // Build title with status indicators
        let auto_scroll_indicator = if self.auto_scroll { "A" } else { "-" };
        let filter_indicator = self.filter_level
            .as_ref()
            .map(|l| l.as_str())
            .unwrap_or("ALL");

        let title = format!(
            " Logs [{}|{}] ({} entries) ",
            auto_scroll_indicator,
            filter_indicator,
            self.entries.len()
        );

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Get visible entries
        let filtered = self.filtered_entries();
        let visible_height = inner.height as usize;

        let lines: Vec<Line> = filtered
            .iter()
            .rev()
            .skip(self.scroll_offset)
            .take(visible_height)
            .rev()
            .map(|entry| Self::format_entry(entry, theme))
            .collect();

        if lines.is_empty() {
            let empty_msg = if self.filter_level.is_some() {
                "No matching log entries"
            } else {
                "No log entries"
            };
            let paragraph = Paragraph::new(empty_msg)
                .style(Style::default().fg(theme.border));
            frame.render_widget(paragraph, inner);
        } else {
            let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
            frame.render_widget(paragraph, inner);
        }
    }

    fn handle_action(&mut self, action: KeyAction) -> bool {
        match action {
            KeyAction::Up => {
                let max_scroll = self.filtered_entries().len().saturating_sub(1);
                if self.scroll_offset < max_scroll {
                    self.scroll_offset += 1;
                    self.auto_scroll = false;
                }
                true
            }
            KeyAction::Down => {
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                }
                if self.scroll_offset == 0 {
                    self.auto_scroll = true;
                }
                true
            }
            KeyAction::Select => {
                // Toggle auto-scroll
                self.toggle_auto_scroll();
                if self.auto_scroll {
                    self.scroll_offset = 0;
                }
                true
            }
            KeyAction::Tab1 => {
                // Cycle filter: ALL -> ERROR -> WARN -> INFO -> ALL
                self.filter_level = match &self.filter_level {
                    None => Some(LogLevel::Error),
                    Some(LogLevel::Error) => Some(LogLevel::Warn),
                    Some(LogLevel::Warn) => Some(LogLevel::Info),
                    _ => None,
                };
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, state: &AppState) {
        // Sync log entries from app state
        // In a real implementation, we'd want incremental updates
        self.entries = state.logs.clone();

        if self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(0..excess);
        }
    }

    fn title(&self) -> &str {
        "Logs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_entry(level: LogLevel, message: &str) -> LogEntry {
        LogEntry {
            timestamp: Utc::now(),
            level,
            message: message.to_string(),
        }
    }

    #[test]
    fn test_log_viewer_new() {
        let panel = LogViewerPanel::new(100);
        assert!(panel.entries.is_empty());
        assert!(panel.auto_scroll);
        assert_eq!(panel.max_entries, 100);
    }

    #[test]
    fn test_log_viewer_add_entry() {
        let mut panel = LogViewerPanel::new(10);
        panel.add_entry(create_test_entry(LogLevel::Info, "Test message"));
        assert_eq!(panel.entries.len(), 1);
    }

    #[test]
    fn test_log_viewer_max_entries() {
        let mut panel = LogViewerPanel::new(5);
        for i in 0..10 {
            panel.add_entry(create_test_entry(LogLevel::Info, &format!("Message {}", i)));
        }
        assert_eq!(panel.entries.len(), 5);
        // Should keep the last 5 messages
        assert!(panel.entries[0].message.contains("5"));
    }

    #[test]
    fn test_log_viewer_filter() {
        let mut panel = LogViewerPanel::new(100);
        panel.add_entry(create_test_entry(LogLevel::Error, "Error message"));
        panel.add_entry(create_test_entry(LogLevel::Info, "Info message"));
        panel.add_entry(create_test_entry(LogLevel::Warn, "Warn message"));

        // No filter - all entries
        assert_eq!(panel.filtered_entries().len(), 3);

        // Filter to errors only
        panel.set_filter(Some(LogLevel::Error));
        assert_eq!(panel.filtered_entries().len(), 1);

        // Clear filter
        panel.set_filter(None);
        assert_eq!(panel.filtered_entries().len(), 3);
    }

    #[test]
    fn test_log_viewer_scroll() {
        let mut panel = LogViewerPanel::new(100);
        for i in 0..20 {
            panel.add_entry(create_test_entry(LogLevel::Info, &format!("Message {}", i)));
        }

        assert_eq!(panel.scroll_offset, 0);
        assert!(panel.auto_scroll);

        // Scroll up disables auto-scroll
        panel.handle_action(KeyAction::Up);
        assert_eq!(panel.scroll_offset, 1);
        assert!(!panel.auto_scroll);

        // Scroll back down to 0 re-enables auto-scroll
        panel.handle_action(KeyAction::Down);
        assert_eq!(panel.scroll_offset, 0);
        assert!(panel.auto_scroll);
    }

    #[test]
    fn test_log_viewer_clear() {
        let mut panel = LogViewerPanel::new(100);
        panel.add_entry(create_test_entry(LogLevel::Info, "Test"));
        assert!(!panel.entries.is_empty());

        panel.clear();
        assert!(panel.entries.is_empty());
    }
}
