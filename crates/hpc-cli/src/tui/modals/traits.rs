//! Modal trait and common types
//!
//! Defines the interface for modal dialogs in the TUI.

use ratatui::{layout::Rect, Frame};

use crate::tui::{events::KeyAction, Theme};

/// Result of handling a key action in a modal
#[derive(Debug)]
pub enum ModalAction {
    /// Continue with modal open (action was handled)
    Continue,
    /// User cancelled the modal (Escape pressed)
    Cancel,
    /// User submitted the form (Enter on submit button)
    Submit,
    /// Pass action through to parent (not handled by modal)
    Passthrough(KeyAction),
}

/// Trait for modal dialogs
pub trait Modal: Send {
    /// Render the modal overlay
    fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme);

    /// Handle a key action, returns the result
    fn handle_key(&mut self, action: KeyAction) -> ModalAction;

    /// Handle character input for text fields
    fn handle_char(&mut self, c: char);

    /// Handle backspace for text fields
    fn handle_backspace(&mut self);

    /// Get the modal title
    fn title(&self) -> &str;

    /// Check if the modal is ready for submission
    fn can_submit(&self) -> bool;

    /// Check if the modal is in a loading/async state
    fn is_loading(&self) -> bool {
        false
    }

    /// Called on tick events to update async state
    fn tick(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modal_action_debug() {
        let action = ModalAction::Continue;
        assert!(format!("{:?}", action).contains("Continue"));

        let action = ModalAction::Passthrough(KeyAction::Up);
        assert!(format!("{:?}", action).contains("Passthrough"));
    }
}
