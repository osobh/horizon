//! Event handling for the HPC-AI TUI dashboard

use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};
use std::time::Duration;
use tokio::sync::mpsc;

/// TUI Events
#[derive(Debug, Clone)]
pub enum Event {
    /// Key press event
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
    /// Terminal resize event
    Resize(u16, u16),
    /// Tick event for regular updates
    Tick,
}

/// Event handler for processing terminal events
pub struct EventHandler {
    /// Event receiver
    receiver: mpsc::UnboundedReceiver<Event>,
    /// Event sender (kept for potential future use)
    _sender: mpsc::UnboundedSender<Event>,
}

impl std::fmt::Debug for EventHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventHandler").finish()
    }
}

impl EventHandler {
    /// Create a new event handler with specified tick interval
    pub fn new(tick_interval: Duration) -> Result<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let event_sender = sender.clone();

        // Spawn event polling task
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                loop {
                    // Wait for the tick interval or a terminal event
                    if crossterm::event::poll(tick_interval).unwrap_or(false) {
                        match event::read() {
                            Ok(CrosstermEvent::Key(key)) => {
                                if event_sender.send(Event::Key(key)).is_err() {
                                    break;
                                }
                            }
                            Ok(CrosstermEvent::Mouse(mouse)) => {
                                if event_sender.send(Event::Mouse(mouse)).is_err() {
                                    break;
                                }
                            }
                            Ok(CrosstermEvent::Resize(w, h)) => {
                                if event_sender.send(Event::Resize(w, h)).is_err() {
                                    break;
                                }
                            }
                            Ok(
                                CrosstermEvent::FocusGained
                                | CrosstermEvent::FocusLost
                                | CrosstermEvent::Paste(_),
                            ) => {
                                // Ignore these events
                            }
                            Err(_) => {
                                break;
                            }
                        }
                    } else {
                        // Send tick event for periodic updates
                        if event_sender.send(Event::Tick).is_err() {
                            break;
                        }
                    }
                }
            });
        }

        Ok(Self {
            receiver,
            _sender: sender,
        })
    }

    /// Get the next event
    pub async fn next(&mut self) -> Option<Event> {
        self.receiver.recv().await
    }
}

/// Key action mappings for the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Quit the application
    Quit,
    /// Move focus to next panel
    NextPanel,
    /// Move focus to previous panel
    PrevPanel,
    /// Move selection up
    Up,
    /// Move selection down
    Down,
    /// Move selection left
    Left,
    /// Move selection right
    Right,
    /// Select/activate current item
    Select,
    /// Go back / cancel
    Back,
    /// Switch to tab 1
    Tab1,
    /// Switch to tab 2
    Tab2,
    /// Switch to tab 3
    Tab3,
    /// Toggle help overlay
    Help,
    /// Refresh data
    Refresh,
    /// Add item (e.g., add node in inventory)
    Add,
    /// Delete item
    Delete,
    /// Bootstrap action
    Bootstrap,
    /// No action
    None,
}

impl KeyAction {
    /// Parse key event into action
    pub fn from_key_event(key: &KeyEvent) -> Self {
        use crossterm::event::{KeyCode, KeyModifiers};

        match (key.code, key.modifiers) {
            // Quit
            (KeyCode::Char('q'), KeyModifiers::NONE) => Self::Quit,
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => Self::Quit,
            (KeyCode::Esc, _) => Self::Quit,

            // Panel navigation
            (KeyCode::Tab, KeyModifiers::NONE) => Self::NextPanel,
            (KeyCode::BackTab, _) => Self::PrevPanel,

            // Movement
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => Self::Up,
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => Self::Down,
            (KeyCode::Left, _) | (KeyCode::Char('h'), KeyModifiers::NONE) => Self::Left,
            (KeyCode::Right, _) | (KeyCode::Char('l'), KeyModifiers::NONE) => Self::Right,

            // Selection
            (KeyCode::Enter, _) | (KeyCode::Char(' '), KeyModifiers::NONE) => Self::Select,
            (KeyCode::Backspace, _) => Self::Back,

            // Tabs
            (KeyCode::Char('1'), _) => Self::Tab1,
            (KeyCode::Char('2'), _) => Self::Tab2,
            (KeyCode::Char('3'), _) => Self::Tab3,

            // Help
            (KeyCode::Char('?'), _) | (KeyCode::F(1), _) => Self::Help,

            // Refresh
            (KeyCode::Char('r'), KeyModifiers::NONE) => Self::Refresh,

            // Add/Delete/Bootstrap
            (KeyCode::Char('a'), KeyModifiers::NONE) => Self::Add,
            (KeyCode::Char('d'), KeyModifiers::NONE) => Self::Delete,
            (KeyCode::Char('b'), KeyModifiers::NONE) => Self::Bootstrap,

            _ => Self::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn test_key_action_quit() {
        let q_key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&q_key), KeyAction::Quit);

        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(KeyAction::from_key_event(&ctrl_c), KeyAction::Quit);

        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&esc), KeyAction::Quit);
    }

    #[test]
    fn test_key_action_navigation() {
        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&tab), KeyAction::NextPanel);

        let up = KeyEvent::new(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&up), KeyAction::Up);

        let k = KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&k), KeyAction::Up);
    }

    #[test]
    fn test_key_action_tabs() {
        let one = KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&one), KeyAction::Tab1);

        let two = KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from_key_event(&two), KeyAction::Tab2);
    }
}
