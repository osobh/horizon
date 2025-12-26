//! Main TUI application state and logic

use anyhow::Result;
use std::time::{Duration, Instant};

use crate::benchmarks::progress_monitor::{BenchmarkPhase, ProgressState, ResourceSnapshot};
use crate::tui::{Event, EventHandler, ProgressLogParser, ResourceMonitor, TuiConfig};

/// Application state for the TUI monitoring dashboard
#[derive(Debug)]
pub struct App {
    /// Current benchmark phase
    pub current_phase: BenchmarkPhase,
    /// Overall progress (0.0 to 1.0)
    pub overall_progress: f64,
    /// Phase progress (0.0 to 1.0)
    pub phase_progress: f64,
    /// Current operation description
    pub current_operation: String,
    /// Tests completed
    pub tests_completed: usize,
    /// Total tests
    pub total_tests: usize,
    /// Elapsed time since start
    pub elapsed_time: Duration,
    /// Recent log entries
    pub log_entries: Vec<String>,
    /// Resource usage history
    pub resource_history: Vec<ResourceSnapshot>,
    /// Whether the application should quit
    pub should_quit: bool,
    /// Configuration
    pub config: TuiConfig,
    /// Log parser
    log_parser: ProgressLogParser,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
    /// Event handler
    pub event_handler: EventHandler,
    /// Start time for elapsed calculation
    start_time: Instant,
}

impl App {
    /// Create a new TUI application
    pub fn new(config: TuiConfig) -> Result<Self> {
        let log_parser = ProgressLogParser::new(&config.log_file_path)?;
        let resource_monitor = ResourceMonitor::new(config.enable_resource_monitoring)?;
        let event_handler = EventHandler::new(Duration::from_millis(config.tick_rate_ms))?;

        Ok(Self {
            current_phase: BenchmarkPhase::NotStarted,
            overall_progress: 0.0,
            phase_progress: 0.0,
            current_operation: "Waiting for benchmark to start...".to_string(),
            tests_completed: 0,
            total_tests: 0,
            elapsed_time: Duration::ZERO,
            log_entries: Vec::new(),
            resource_history: Vec::new(),
            should_quit: false,
            config,
            log_parser,
            resource_monitor,
            event_handler,
            start_time: Instant::now(),
        })
    }

    /// Run the main application loop
    pub async fn run(&mut self) -> Result<()> {
        // GREEN phase implementation - minimal working version
        use crossterm::{
            event::{DisableMouseCapture, EnableMouseCapture},
            execute,
            terminal::{
                disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
            },
        };
        use ratatui::{backend::CrosstermBackend, Terminal};
        use std::io;

        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Main event loop
        loop {
            // Draw the UI
            terminal.draw(|f| {
                use ratatui::{
                    layout::{Constraint, Direction, Layout},
                    style::{Color, Style},
                    text::Span,
                    widgets::{Block, Borders, Gauge, Paragraph},
                };

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .margin(1)
                    .constraints([
                        Constraint::Length(3), // Progress bar
                        Constraint::Length(3), // Phase info
                        Constraint::Min(0),    // Rest for logs
                    ])
                    .split(f.size());

                // Progress gauge
                let progress = Gauge::default()
                    .block(
                        Block::default()
                            .title("Overall Progress")
                            .borders(Borders::ALL),
                    )
                    .gauge_style(Style::default().fg(Color::Blue))
                    .percent((self.overall_progress * 100.0) as u16)
                    .label(format!("{:.1}%", self.overall_progress * 100.0));
                f.render_widget(progress, chunks[0]);

                // Phase information
                let phase_text = format!(
                    "Phase: {:?} | Operation: {} | Tests: {}/{}",
                    self.current_phase,
                    self.current_operation,
                    self.tests_completed,
                    self.total_tests
                );
                let phase_info = Paragraph::new(phase_text)
                    .block(Block::default().title("Status").borders(Borders::ALL));
                f.render_widget(phase_info, chunks[1]);

                // Log entries
                let logs_text = self.log_entries.join("\n");
                let logs_widget = Paragraph::new(logs_text).block(
                    Block::default()
                        .title("Recent Activity")
                        .borders(Borders::ALL),
                );
                f.render_widget(logs_widget, chunks[2]);
            })?;

            // Handle events
            if let Some(event) = self.event_handler.next().await {
                self.handle_event(event).await?;

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

    /// Update application state from progress log and resource monitoring
    pub async fn update(&mut self) -> Result<()> {
        // Update elapsed time
        self.elapsed_time = self.start_time.elapsed();

        // Parse latest log entries
        if let Some(latest_state) = self.log_parser.get_latest_state()? {
            self.current_phase = latest_state.current_phase;
            self.overall_progress = latest_state.overall_progress;
            self.phase_progress = latest_state.phase_progress;
            self.current_operation = latest_state.current_test;
            self.tests_completed = latest_state.tests_completed;
            self.total_tests = latest_state.total_tests;
        }

        // Get recent log entries
        self.log_entries = self.log_parser.get_recent_entries(10)?;

        // Update resource monitoring
        if let Some(snapshot) = self.resource_monitor.get_latest_snapshot().await? {
            self.resource_history.push(snapshot);

            // Keep only recent history to prevent memory growth
            if self.resource_history.len() > self.config.max_log_entries {
                self.resource_history.remove(0);
            }
        }

        Ok(())
    }

    /// Handle user input events
    pub async fn handle_event(&mut self, event: Event) -> Result<()> {
        match event {
            Event::Key(key_event) => match key_event.code {
                crossterm::event::KeyCode::Char('q') => {
                    self.should_quit = true;
                }
                crossterm::event::KeyCode::Esc => {
                    self.should_quit = true;
                }
                _ => {}
            },
            Event::Tick => {
                // Update application state on each tick
                self.update().await?;
            }
            Event::Mouse(_) => {
                // Handle mouse events - will be implemented later
            }
            Event::Resize(_, _) => {
                // Handle terminal resize - will be implemented later
            }
        }

        Ok(())
    }

    /// Get current progress state for display
    pub fn get_progress_state(&self) -> ProgressState {
        ProgressState {
            current_phase: self.current_phase.clone(),
            phase_progress: self.phase_progress,
            overall_progress: self.overall_progress,
            current_test: self.current_operation.clone(),
            tests_completed: self.tests_completed,
            total_tests: self.total_tests,
            elapsed_time: self.elapsed_time,
            estimated_remaining: self.calculate_estimated_remaining(),
            current_operation: self.current_operation.clone(),
        }
    }

    /// Calculate estimated time remaining
    fn calculate_estimated_remaining(&self) -> Duration {
        if self.overall_progress > 0.0 {
            let total_estimated = self.elapsed_time.as_secs_f64() / self.overall_progress;
            let remaining = total_estimated - self.elapsed_time.as_secs_f64();
            Duration::from_secs_f64(remaining.max(0.0))
        } else {
            Duration::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;

    fn create_test_config() -> TuiConfig {
        let temp_file = NamedTempFile::new()?;
        TuiConfig {
            log_file_path: temp_file.path().to_string_lossy().to_string(),
            update_interval_ms: 100,
            enable_resource_monitoring: false,
            max_log_entries: 10,
            tick_rate_ms: 50,
        }
    }

    #[test]
    fn test_app_new_creates_with_default_state() {
        // RED PHASE: This test will fail until we implement the constructor properly
        let config = create_test_config();
        let app = App::new(config)?;

        assert_eq!(app.current_phase, BenchmarkPhase::Initialization);
        assert_eq!(app.overall_progress, 0.0);
        assert_eq!(app.phase_progress, 0.0);
        assert!(!app.should_quit);
        assert_eq!(app.tests_completed, 0);
        assert_eq!(app.total_tests, 0);
        assert!(app.log_entries.is_empty());
        assert!(app.resource_history.is_empty());
    }

    #[test]
    fn test_app_calculate_estimated_remaining() {
        let config = create_test_config();
        let mut app = App::new(config)?;

        // Test with no progress
        assert_eq!(app.calculate_estimated_remaining(), Duration::ZERO);

        // Test with 50% progress after 10 seconds
        app.overall_progress = 0.5;
        app.elapsed_time = Duration::from_secs(10);
        let remaining = app.calculate_estimated_remaining();
        assert_eq!(remaining, Duration::from_secs(10)); // Should be 10 more seconds
    }

    #[tokio::test]
    async fn test_app_handle_quit_events() {
        let config = create_test_config();
        let mut app = App::new(config)?;

        // Test 'q' key
        let quit_event = Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ));
        app.handle_event(quit_event).await?;
        assert!(app.should_quit);

        // Reset and test Escape key
        app.should_quit = false;
        let esc_event = Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        ));
        app.handle_event(esc_event).await?;
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn test_app_update_elapsed_time() {
        let mut config = create_test_config();
        // Disable resource monitoring to prevent hanging
        config.enable_resource_monitoring = false;
        let mut app = App::new(config)?;

        // Wait a bit and update
        tokio::time::sleep(Duration::from_millis(10)).await;
        app.update().await?;

        assert!(app.elapsed_time > Duration::ZERO);
    }

    #[test]
    fn test_app_get_progress_state() {
        let config = create_test_config();
        let mut app = App::new(config)?;

        app.current_phase = BenchmarkPhase::ScalabilityTests;
        app.overall_progress = 0.75;
        app.phase_progress = 0.5;
        app.current_operation = "Testing 1M agents".to_string();
        app.tests_completed = 3;
        app.total_tests = 4;

        let state = app.get_progress_state();
        assert_eq!(state.current_phase, BenchmarkPhase::ScalabilityTests);
        assert_eq!(state.overall_progress, 0.75);
        assert_eq!(state.phase_progress, 0.5);
        assert_eq!(state.current_test, "Testing 1M agents");
        assert_eq!(state.tests_completed, 3);
        assert_eq!(state.total_tests, 4);
    }
}
