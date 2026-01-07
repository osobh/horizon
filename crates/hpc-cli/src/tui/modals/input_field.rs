//! Text input field widget for modals
//!
//! Provides a reusable text input component with cursor management.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::tui::Theme;

/// Text input field for modal forms
#[derive(Debug, Clone)]
pub struct InputField {
    /// Field label
    pub label: String,
    /// Current value
    pub value: String,
    /// Placeholder text when empty
    pub placeholder: String,
    /// Cursor position (byte index)
    cursor: usize,
    /// Whether this field is focused
    pub focused: bool,
    /// Whether this field is required
    pub required: bool,
    /// Validation error message
    pub error: Option<String>,
    /// Maximum length (0 = unlimited)
    pub max_length: usize,
}

impl InputField {
    /// Create a new input field
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: String::new(),
            placeholder: String::new(),
            cursor: 0,
            focused: false,
            required: false,
            error: None,
            max_length: 0,
        }
    }

    /// Set placeholder text
    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Set initial value
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self.cursor = self.value.len();
        self
    }

    /// Set as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set max length
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }

    /// Get current cursor position
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Insert a character at cursor position
    pub fn insert_char(&mut self, c: char) {
        if self.max_length > 0 && self.value.len() >= self.max_length {
            return;
        }
        self.value.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.clear_error();
    }

    /// Delete character before cursor (backspace)
    pub fn delete_char(&mut self) {
        if self.cursor > 0 {
            // Find the previous character boundary
            let prev_boundary = self
                .value[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.value.remove(prev_boundary);
            self.cursor = prev_boundary;
            self.clear_error();
        }
    }

    /// Delete character at cursor (delete key)
    pub fn delete_char_forward(&mut self) {
        if self.cursor < self.value.len() {
            self.value.remove(self.cursor);
            self.clear_error();
        }
    }

    /// Move cursor left
    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            // Find the previous character boundary
            self.cursor = self.value[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    /// Move cursor right
    pub fn cursor_right(&mut self) {
        if self.cursor < self.value.len() {
            // Find the next character boundary
            self.cursor = self.value[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.value.len());
        }
    }

    /// Move cursor to start
    pub fn cursor_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end
    pub fn cursor_end(&mut self) {
        self.cursor = self.value.len();
    }

    /// Clear the field
    pub fn clear(&mut self) {
        self.value.clear();
        self.cursor = 0;
        self.clear_error();
    }

    /// Set error message
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }

    /// Clear error message
    pub fn clear_error(&mut self) {
        self.error = None;
    }

    /// Validate the field
    pub fn validate(&mut self) -> bool {
        if self.required && self.value.trim().is_empty() {
            self.set_error("This field is required");
            return false;
        }
        self.clear_error();
        true
    }

    /// Check if field is valid
    pub fn is_valid(&self) -> bool {
        if self.required && self.value.trim().is_empty() {
            return false;
        }
        self.error.is_none()
    }

    /// Render the input field
    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let border_color = if self.error.is_some() {
            theme.error
        } else if self.focused {
            theme.primary
        } else {
            theme.border
        };

        let label = if self.required {
            format!("{} *", self.label)
        } else {
            self.label.clone()
        };

        let block = Block::default()
            .title(format!(" {} ", label))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        // Display value or placeholder
        let display_text = if self.value.is_empty() && !self.focused {
            Span::styled(&self.placeholder, Style::default().fg(theme.border))
        } else {
            Span::styled(&self.value, Style::default().fg(theme.foreground))
        };

        let paragraph = Paragraph::new(Line::from(display_text)).block(block);
        frame.render_widget(paragraph, area);

        // Show cursor when focused
        if self.focused {
            // Calculate cursor position in terminal coordinates
            let cursor_x = area.x + 1 + self.cursor as u16;
            let cursor_y = area.y + 1;
            if cursor_x < area.x + area.width - 1 {
                frame.set_cursor_position((cursor_x, cursor_y));
            }
        }

        // Show error below if present
        if let Some(ref err) = self.error {
            if area.height > 2 {
                let error_area = Rect {
                    x: area.x,
                    y: area.y + area.height - 1,
                    width: area.width,
                    height: 1,
                };
                let error_text = Paragraph::new(Span::styled(
                    err.as_str(),
                    Style::default().fg(theme.error).add_modifier(Modifier::ITALIC),
                ));
                frame.render_widget(error_text, error_area);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_field_new() {
        let field = InputField::new("Test");
        assert_eq!(field.label, "Test");
        assert!(field.value.is_empty());
        assert!(!field.required);
    }

    #[test]
    fn test_input_field_with_value() {
        let field = InputField::new("Test").with_value("hello");
        assert_eq!(field.value, "hello");
        assert_eq!(field.cursor, 5);
    }

    #[test]
    fn test_insert_char() {
        let mut field = InputField::new("Test");
        field.insert_char('a');
        field.insert_char('b');
        field.insert_char('c');
        assert_eq!(field.value, "abc");
        assert_eq!(field.cursor, 3);
    }

    #[test]
    fn test_delete_char() {
        let mut field = InputField::new("Test").with_value("abc");
        field.delete_char();
        assert_eq!(field.value, "ab");
        assert_eq!(field.cursor, 2);
    }

    #[test]
    fn test_cursor_movement() {
        let mut field = InputField::new("Test").with_value("abc");
        assert_eq!(field.cursor, 3);

        field.cursor_left();
        assert_eq!(field.cursor, 2);

        field.cursor_home();
        assert_eq!(field.cursor, 0);

        field.cursor_right();
        assert_eq!(field.cursor, 1);

        field.cursor_end();
        assert_eq!(field.cursor, 3);
    }

    #[test]
    fn test_validation() {
        let mut field = InputField::new("Test").required();
        assert!(!field.validate());
        assert!(field.error.is_some());

        field.insert_char('x');
        assert!(field.validate());
        assert!(field.error.is_none());
    }

    #[test]
    fn test_max_length() {
        let mut field = InputField::new("Test").with_max_length(3);
        field.insert_char('a');
        field.insert_char('b');
        field.insert_char('c');
        field.insert_char('d'); // Should be ignored
        assert_eq!(field.value, "abc");
    }
}
