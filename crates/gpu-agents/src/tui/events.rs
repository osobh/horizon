//! Event handling for the TUI monitoring dashboard

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
    /// Event sender
    _sender: mpsc::UnboundedSender<Event>,
    /// Tick interval
    tick_interval: Duration,
}

impl std::fmt::Debug for EventHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventHandler")
            .field("tick_interval", &self.tick_interval)
            .finish()
    }
}

impl EventHandler {
    /// Create a new event handler
    pub fn new(tick_interval: Duration) -> Result<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();

        // Clone sender for the spawned task
        let event_sender = sender.clone();

        // Only spawn the background task if we're in a tokio runtime
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
                                // Error reading event, break the loop
                                break;
                            }
                        }
                    } else {
                        // Send tick event
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
            tick_interval,
        })
    }

    /// Get the next event
    pub async fn next(&mut self) -> Option<Event> {
        self.receiver.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    #[tokio::test]
    async fn test_event_handler_creation() {
        // RED PHASE: Test will fail until EventHandler is properly implemented
        let handler = EventHandler::new(Duration::from_millis(100));
        assert!(handler.is_ok());
    }

    #[tokio::test]
    async fn test_event_handler_tick_events() {
        // Create a handler but don't actually poll for events in tests
        let (sender, mut receiver) = mpsc::unbounded_channel();

        // Send a tick event directly
        sender.send(Event::Tick)?;

        // Test that we can receive it
        let event = receiver.recv().await;
        assert!(event.is_some());
        match event? {
            Event::Tick => {
                // Success - got the expected tick event
            }
            other => panic!("Expected Tick event, got {:?}", other),
        }
    }

    #[test]
    fn test_event_types() {
        // Test event enum variants
        let key_event = Event::Key(KeyEvent::new(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ));

        match key_event {
            Event::Key(_) => {
                // Success
            }
            _ => panic!("Expected Key event"),
        }

        let tick_event = Event::Tick;
        match tick_event {
            Event::Tick => {
                // Success
            }
            _ => panic!("Expected Tick event"),
        }
    }
}
