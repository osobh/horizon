//! Selection field widget for modals
//!
//! Provides a reusable selection component for choosing from options.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use std::fmt::Display;

use crate::tui::Theme;

/// Selection field for modal forms
#[derive(Debug, Clone)]
pub struct SelectField<T>
where
    T: Clone + Display + PartialEq,
{
    /// Field label
    pub label: String,
    /// Available options
    options: Vec<T>,
    /// Currently selected index
    selected: usize,
    /// Whether this field is focused
    pub focused: bool,
}

impl<T> SelectField<T>
where
    T: Clone + Display + PartialEq,
{
    /// Create a new select field with options
    pub fn new(label: impl Into<String>, options: Vec<T>) -> Self {
        Self {
            label: label.into(),
            options,
            selected: 0,
            focused: false,
        }
    }

    /// Set initial selected value
    pub fn with_selected(mut self, value: &T) -> Self {
        if let Some(idx) = self.options.iter().position(|o| o == value) {
            self.selected = idx;
        }
        self
    }

    /// Get currently selected value
    pub fn value(&self) -> Option<&T> {
        self.options.get(self.selected)
    }

    /// Get currently selected value (cloned)
    pub fn value_cloned(&self) -> Option<T> {
        self.options.get(self.selected).cloned()
    }

    /// Select next option
    pub fn next(&mut self) {
        if !self.options.is_empty() {
            self.selected = (self.selected + 1) % self.options.len();
        }
    }

    /// Select previous option
    pub fn prev(&mut self) {
        if !self.options.is_empty() {
            self.selected = if self.selected == 0 {
                self.options.len() - 1
            } else {
                self.selected - 1
            };
        }
    }

    /// Render the select field
    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let border_color = if self.focused {
            theme.primary
        } else {
            theme.border
        };

        let block = Block::default()
            .title(format!(" {} ", self.label))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        // Build option display: < Option >
        let value_str = self
            .value()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "None".to_string());

        let left_arrow = if self.focused { "◀ " } else { "  " };
        let right_arrow = if self.focused { " ▶" } else { "  " };

        let content = Line::from(vec![
            Span::styled(left_arrow, Style::default().fg(theme.primary)),
            Span::styled(
                value_str,
                Style::default()
                    .fg(theme.foreground)
                    .add_modifier(if self.focused {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            Span::styled(right_arrow, Style::default().fg(theme.primary)),
        ]);

        let paragraph = Paragraph::new(content).block(block);
        frame.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    enum TestOption {
        A,
        B,
        C,
    }

    impl Display for TestOption {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestOption::A => write!(f, "Option A"),
                TestOption::B => write!(f, "Option B"),
                TestOption::C => write!(f, "Option C"),
            }
        }
    }

    #[test]
    fn test_select_field_new() {
        let field = SelectField::new(
            "Test",
            vec![TestOption::A, TestOption::B, TestOption::C],
        );
        assert_eq!(field.label, "Test");
        assert_eq!(field.value(), Some(&TestOption::A));
    }

    #[test]
    fn test_select_field_navigation() {
        let mut field = SelectField::new(
            "Test",
            vec![TestOption::A, TestOption::B, TestOption::C],
        );

        field.next();
        assert_eq!(field.value(), Some(&TestOption::B));

        field.next();
        assert_eq!(field.value(), Some(&TestOption::C));

        field.next(); // Wrap around
        assert_eq!(field.value(), Some(&TestOption::A));

        field.prev(); // Wrap backwards
        assert_eq!(field.value(), Some(&TestOption::C));
    }

    #[test]
    fn test_select_field_with_selected() {
        let field = SelectField::new(
            "Test",
            vec![TestOption::A, TestOption::B, TestOption::C],
        )
        .with_selected(&TestOption::B);

        assert_eq!(field.value(), Some(&TestOption::B));
    }

    #[test]
    fn test_value_cloned() {
        let field = SelectField::new(
            "Test",
            vec![TestOption::A, TestOption::B],
        );
        assert_eq!(field.value_cloned(), Some(TestOption::A));
    }
}
