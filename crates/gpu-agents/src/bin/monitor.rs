//! GPU Agents Benchmark Monitor - Terminal User Interface
//!
//! This binary provides a real-time monitoring dashboard for the GPU agents
//! benchmark suite. It replaces the bash-based monitor_dashboard.sh with
//! a smooth, professional TUI that updates without flickering.

use anyhow::Result;
use clap::Parser;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use gpu_agents::tui::{App, Dashboard, TuiConfig};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use tokio::time::{interval, Duration};

/// GPU Agents Benchmark Monitor
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the benchmark progress log file
    #[arg(short, long, default_value = "benchmark_progress.log")]
    log_file: String,

    /// Update interval in milliseconds
    #[arg(short, long, default_value = "1000")]
    update_interval: u64,

    /// Enable resource monitoring (GPU/CPU/Memory)
    #[arg(short, long, default_value = "true")]
    enable_resources: bool,

    /// Maximum number of log entries to keep in memory
    #[arg(short, long, default_value = "1000")]
    max_log_entries: usize,

    /// UI tick rate in milliseconds
    #[arg(short, long, default_value = "250")]
    tick_rate: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create TUI configuration
    let config = TuiConfig {
        log_file_path: args.log_file,
        update_interval_ms: args.update_interval,
        enable_resource_monitoring: args.enable_resources,
        max_log_entries: args.max_log_entries,
        tick_rate_ms: args.tick_rate,
    };

    // Create app and dashboard
    let mut app = App::new(config)?;
    let mut dashboard = Dashboard::new()?;

    // Run the application
    let res = run_app(&mut terminal, &mut app, &mut dashboard).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {:?}", err);
    }

    Ok(())
}

/// Run the main application loop
async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    dashboard: &mut Dashboard,
) -> Result<()> {
    let mut update_interval = interval(Duration::from_millis(app.config.update_interval_ms));

    loop {
        // Draw the UI
        terminal.draw(|f| {
            if let Err(e) = dashboard.render::<CrosstermBackend<io::Stdout>>(f, app) {
                eprintln!("Render error: {:?}", e);
            }
        })?;

        // Handle events and updates
        tokio::select! {
            _ = update_interval.tick() => {
                app.update().await?;
            }
            Some(event) = app.event_handler.next() => {
                use gpu_agents::tui::Event;
                match event {
                    Event::Key(key) => {
                        match key.code {
                            crossterm::event::KeyCode::Char('q') | crossterm::event::KeyCode::Esc => {
                                return Ok(());
                            }
                            crossterm::event::KeyCode::Tab => {
                                dashboard.next_tab();
                            }
                            crossterm::event::KeyCode::BackTab => {
                                dashboard.previous_tab();
                            }
                            _ => {}
                        }
                    }
                    Event::Tick => {
                        // Regular tick update
                        app.update().await?;
                    }
                    _ => {}
                }

                app.handle_event(event).await?;

                if app.should_quit {
                    return Ok(());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = vec![
            "monitor",
            "--log-file",
            "test.log",
            "--update-interval",
            "500",
            "--enable-resources",
            "false",
        ];

        let parsed = Args::try_parse_from(args);
        assert!(parsed.is_ok());

        let args = parsed.unwrap();
        assert_eq!(args.log_file, "test.log");
        assert_eq!(args.update_interval, 500);
        assert!(!args.enable_resources);
    }
}
