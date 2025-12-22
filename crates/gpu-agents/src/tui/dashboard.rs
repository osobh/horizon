//! Dashboard rendering logic for the TUI monitoring interface

use anyhow::Result;
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap},
    Frame,
};
use std::time::Duration;

use crate::benchmarks::progress_monitor::BenchmarkPhase;
use crate::tui::App;

/// Dashboard renderer for the TUI monitoring interface
pub struct Dashboard {
    /// Current selected tab
    selected_tab: usize,
    /// Available tabs
    tabs: Vec<String>,
}

impl Dashboard {
    /// Format benchmark phase for display
    fn format_phase(phase: &BenchmarkPhase) -> String {
        match phase {
            BenchmarkPhase::NotStarted => "Not Started".to_string(),
            BenchmarkPhase::Initialization => "Initialization".to_string(),
            BenchmarkPhase::SystemCheck => "System Check".to_string(),
            BenchmarkPhase::ScalabilityTests => "Scalability Tests".to_string(),
            BenchmarkPhase::LlmTests => "LLM Tests".to_string(),
            BenchmarkPhase::KnowledgeGraphTests => "Knowledge Graph Tests".to_string(),
            BenchmarkPhase::EvolutionTests => "Evolution Tests".to_string(),
            BenchmarkPhase::ReportGeneration => "Report Generation".to_string(),
            BenchmarkPhase::Complete => "Complete".to_string(),
            BenchmarkPhase::Custom(name) => name.clone(),
        }
    }
}

impl Dashboard {
    /// Create a new dashboard renderer
    pub fn new() -> Result<Self> {
        Ok(Self {
            selected_tab: 0,
            tabs: vec![
                "Overview".to_string(),
                "Progress".to_string(),
                "Resources".to_string(),
                "Logs".to_string(),
            ],
        })
    }

    /// Render the complete dashboard
    pub fn render<B: Backend>(&mut self, frame: &mut Frame, app: &App) -> Result<()> {
        let size = frame.size();

        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(0),    // Main content
            ])
            .split(size);

        // Render header with tabs
        self.render_header(frame, chunks[0]);

        // Render main content based on selected tab
        match self.selected_tab {
            0 => self.render_overview_tab(frame, chunks[1], app)?,
            1 => self.render_progress_tab(frame, chunks[1], app)?,
            2 => self.render_resources_tab(frame, chunks[1], app)?,
            3 => self.render_logs_tab(frame, chunks[1], app)?,
            _ => {}
        }

        Ok(())
    }

    /// Render the header with tab navigation
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let titles = self
            .tabs
            .iter()
            .map(|t| Line::from(vec![Span::styled(t, Style::default())]))
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("GPU Agents Monitor"),
            )
            .style(Style::default().fg(Color::White))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .select(self.selected_tab);

        frame.render_widget(tabs, area);
    }

    /// Render the overview tab
    fn render_overview_tab(&self, frame: &mut Frame, area: Rect, app: &App) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Current phase
                Constraint::Length(3), // Overall progress
                Constraint::Length(3), // Phase progress
                Constraint::Min(0),    // Current operation
            ])
            .split(area);

        // Current phase
        let phase_text = format!("Current Phase: {}", Self::format_phase(&app.current_phase));
        let phase_paragraph = Paragraph::new(phase_text)
            .block(Block::default().borders(Borders::ALL).title("Phase"))
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(phase_paragraph, chunks[0]);

        // Overall progress
        let overall_gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Overall Progress"),
            )
            .gauge_style(Style::default().fg(Color::Green))
            .percent((app.overall_progress * 100.0) as u16);
        frame.render_widget(overall_gauge, chunks[1]);

        // Phase progress
        let phase_gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Phase Progress"),
            )
            .gauge_style(Style::default().fg(Color::Blue))
            .percent((app.phase_progress * 100.0) as u16);
        frame.render_widget(phase_gauge, chunks[2]);

        // Current operation
        let operation_text = format!("Current Operation: {}", app.current_operation);
        let operation_paragraph = Paragraph::new(operation_text)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White));
        frame.render_widget(operation_paragraph, chunks[3]);

        Ok(())
    }

    /// Render the progress tab
    fn render_progress_tab(&self, frame: &mut Frame, area: Rect, app: &App) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Test progress
        let test_info = format!(
            "Tests: {}/{}\nElapsed: {:?}\nPhase: {:?}",
            app.tests_completed, app.total_tests, app.elapsed_time, app.current_phase
        );
        let test_paragraph = Paragraph::new(test_info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Test Progress"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(test_paragraph, chunks[0]);

        // Time information
        let time_info = format!(
            "Start Time: {:?}\nElapsed: {:?}\nEstimated Remaining: {:?}",
            Duration::ZERO, // Will be calculated properly later
            app.elapsed_time,
            Duration::ZERO // Will be calculated properly later
        );
        let time_paragraph = Paragraph::new(time_info)
            .block(Block::default().borders(Borders::ALL).title("Timing"))
            .wrap(Wrap { trim: true });
        frame.render_widget(time_paragraph, chunks[1]);

        Ok(())
    }

    /// Render the resources tab
    fn render_resources_tab(&self, frame: &mut Frame, area: Rect, app: &App) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // GPU usage
                Constraint::Length(3), // CPU usage
                Constraint::Length(3), // Memory usage
                Constraint::Min(0),    // Resource history chart
            ])
            .split(area);

        if let Some(latest_snapshot) = app.resource_history.last() {
            // GPU usage
            let gpu_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("GPU Usage"))
                .gauge_style(Style::default().fg(Color::Red))
                .percent(latest_snapshot.gpu_usage_percent as u16);
            frame.render_widget(gpu_gauge, chunks[0]);

            // CPU usage
            let cpu_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("CPU Usage"))
                .gauge_style(Style::default().fg(Color::Yellow))
                .percent(latest_snapshot.cpu_usage_percent as u16);
            frame.render_widget(cpu_gauge, chunks[1]);

            // Memory usage
            let memory_percent = if latest_snapshot.memory_total_mb > 0.0 {
                (latest_snapshot.memory_usage_mb / latest_snapshot.memory_total_mb * 100.0) as u16
            } else {
                0
            };
            let memory_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Memory Usage"))
                .gauge_style(Style::default().fg(Color::Magenta))
                .percent(memory_percent);
            frame.render_widget(memory_gauge, chunks[2]);

            // Resource history (simplified for now)
            let history_text = format!(
                "Resource History:\nGPU: {:.1}%\nCPU: {:.1}%\nMemory: {:.1} MB / {:.1} MB",
                latest_snapshot.gpu_usage_percent,
                latest_snapshot.cpu_usage_percent,
                latest_snapshot.memory_usage_mb,
                latest_snapshot.memory_total_mb
            );
            let history_paragraph = Paragraph::new(history_text)
                .block(Block::default().borders(Borders::ALL).title("Details"))
                .wrap(Wrap { trim: true });
            frame.render_widget(history_paragraph, chunks[3]);
        } else {
            // No resource data available
            let no_data_paragraph = Paragraph::new("No resource data available")
                .block(Block::default().borders(Borders::ALL).title("Resources"))
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(no_data_paragraph, area);
        }

        Ok(())
    }

    /// Render the logs tab
    fn render_logs_tab(&self, frame: &mut Frame, area: Rect, app: &App) -> Result<()> {
        let log_items: Vec<ListItem> = app
            .log_entries
            .iter()
            .map(|entry| ListItem::new(entry.clone()))
            .collect();

        let logs_list = List::new(log_items)
            .block(Block::default().borders(Borders::ALL).title("Recent Logs"))
            .style(Style::default().fg(Color::White));

        frame.render_widget(logs_list, area);

        Ok(())
    }

    /// Switch to next tab
    pub fn next_tab(&mut self) {
        self.selected_tab = (self.selected_tab + 1) % self.tabs.len();
    }

    /// Switch to previous tab
    pub fn previous_tab(&mut self) {
        if self.selected_tab > 0 {
            self.selected_tab -= 1;
        } else {
            self.selected_tab = self.tabs.len() - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_dashboard_new() {
        // RED PHASE: Test will fail until Dashboard is properly implemented
        let dashboard = Dashboard::new();
        assert!(dashboard.is_ok());

        let dashboard = dashboard?;
        assert_eq!(dashboard.selected_tab, 0);
        assert_eq!(dashboard.tabs.len(), 4);
        assert_eq!(dashboard.tabs[0], "Overview");
        assert_eq!(dashboard.tabs[1], "Progress");
        assert_eq!(dashboard.tabs[2], "Resources");
        assert_eq!(dashboard.tabs[3], "Logs");
    }

    #[test]
    fn test_dashboard_tab_navigation() -> Result<(), Box<dyn std::error::Error>>  {
        let mut dashboard = Dashboard::new()?;

        // Test next tab
        dashboard.next_tab();
        assert_eq!(dashboard.selected_tab, 1);

        dashboard.next_tab();
        assert_eq!(dashboard.selected_tab, 2);

        dashboard.next_tab();
        assert_eq!(dashboard.selected_tab, 3);

        // Should wrap around
        dashboard.next_tab();
        assert_eq!(dashboard.selected_tab, 0);

        // Test previous tab
        dashboard.previous_tab();
        assert_eq!(dashboard.selected_tab, 3);

        dashboard.previous_tab();
        assert_eq!(dashboard.selected_tab, 2);
    }

    #[test]
    fn test_dashboard_tab_wrapping() -> Result<(), Box<dyn std::error::Error>>  {
        let mut dashboard = Dashboard::new()?;

        // Test wrapping from beginning to end
        dashboard.previous_tab();
        assert_eq!(dashboard.selected_tab, 3);

        // Test wrapping from end to beginning
        dashboard.next_tab();
        assert_eq!(dashboard.selected_tab, 0);
    }
}
