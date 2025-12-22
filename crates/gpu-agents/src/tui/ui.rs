//! UI rendering utilities and helpers for the TUI monitoring dashboard

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Borders, Clear, Gauge, Paragraph, Wrap},
    Frame,
};

use crate::benchmarks::progress_monitor::{BenchmarkPhase, ResourceSnapshot};

/// UI utilities for rendering TUI components
pub struct UiUtils;

impl UiUtils {
    /// Create a styled block with title
    pub fn create_block(title: &str) -> Block {
        Block::default()
            .borders(Borders::ALL)
            .title(title)
            .style(Style::default().fg(Color::White))
    }

    /// Create a styled gauge widget
    pub fn create_gauge(title: &str, percentage: f64, color: Color) -> Gauge {
        Gauge::default()
            .block(Self::create_block(title))
            .gauge_style(Style::default().fg(color))
            .percent((percentage * 100.0).min(100.0).max(0.0) as u16)
    }

    /// Format duration for display
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_seconds = duration.as_secs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    /// Format memory size for display
    pub fn format_memory_size(size_mb: f64) -> String {
        if size_mb >= 1024.0 {
            format!("{:.1} GB", size_mb / 1024.0)
        } else {
            format!("{:.1} MB", size_mb)
        }
    }

    /// Get color for benchmark phase
    pub fn get_phase_color(phase: &BenchmarkPhase) -> Color {
        match phase {
            BenchmarkPhase::NotStarted => Color::Gray,
            BenchmarkPhase::Initialization => Color::Yellow,
            BenchmarkPhase::SystemCheck => Color::Blue,
            BenchmarkPhase::ScalabilityTests => Color::Green,
            BenchmarkPhase::LlmTests => Color::Magenta,
            BenchmarkPhase::KnowledgeGraphTests => Color::Cyan,
            BenchmarkPhase::EvolutionTests => Color::Red,
            BenchmarkPhase::ReportGeneration => Color::White,
            BenchmarkPhase::Complete => Color::Green,
            BenchmarkPhase::Custom(_) => Color::Yellow,
        }
    }

    /// Get formatted phase name
    pub fn format_phase_name(phase: &BenchmarkPhase) -> String {
        match phase {
            BenchmarkPhase::NotStarted => "⏸️  Not Started".to_string(),
            BenchmarkPhase::Initialization => "🚀 Initialization".to_string(),
            BenchmarkPhase::SystemCheck => "🔍 System Check".to_string(),
            BenchmarkPhase::ScalabilityTests => "📈 Scalability Tests".to_string(),
            BenchmarkPhase::LlmTests => "🧠 LLM Integration Tests".to_string(),
            BenchmarkPhase::KnowledgeGraphTests => "🕸️ Knowledge Graph Tests".to_string(),
            BenchmarkPhase::EvolutionTests => "🧬 Evolution Tests".to_string(),
            BenchmarkPhase::ReportGeneration => "📊 Report Generation".to_string(),
            BenchmarkPhase::Complete => "✅ Complete".to_string(),
            BenchmarkPhase::Custom(name) => format!("🔧 {}", name),
        }
    }

    /// Create a centered popup area
    pub fn create_popup_area(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ])
            .split(area);

        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ])
            .split(popup_layout[1])[1]
    }

    /// Render a help popup
    pub fn render_help_popup(frame: &mut Frame, area: Rect) {
        let popup_area = Self::create_popup_area(60, 70, area);

        // Clear the area
        frame.render_widget(Clear, popup_area);

        let help_text = vec![
            Line::from("📋 GPU Agents Monitor - Help"),
            Line::from(""),
            Line::from("🔑 Key Bindings:"),
            Line::from("  q, Esc    - Quit application"),
            Line::from("  Tab       - Next tab"),
            Line::from("  Shift+Tab - Previous tab"),
            Line::from("  h, ?      - Show this help"),
            Line::from("  r         - Refresh data"),
            Line::from(""),
            Line::from("📊 Tabs:"),
            Line::from("  Overview  - Current status & progress"),
            Line::from("  Progress  - Detailed progress information"),
            Line::from("  Resources - GPU/CPU/Memory usage"),
            Line::from("  Logs      - Recent benchmark logs"),
            Line::from(""),
            Line::from("Press any key to close this help"),
        ];

        let help_paragraph = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Help")
                    .style(Style::default().fg(Color::Yellow)),
            )
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });

        frame.render_widget(help_paragraph, popup_area);
    }

    /// Create resource usage sparkline data
    pub fn create_sparkline_data(snapshots: &[ResourceSnapshot], max_points: usize) -> Vec<u64> {
        let start_idx = if snapshots.len() > max_points {
            snapshots.len() - max_points
        } else {
            0
        };

        snapshots[start_idx..]
            .iter()
            .map(|snapshot| snapshot.gpu_usage_percent as u64)
            .collect()
    }

    /// Format resource snapshot for display
    pub fn format_resource_snapshot(snapshot: &ResourceSnapshot) -> Vec<Line> {
        vec![
            Line::from(format!("🖥️  GPU Usage: {:.1}%", snapshot.gpu_usage_percent)),
            Line::from(format!(
                "🧠 GPU Memory: {} / {}",
                Self::format_memory_size(snapshot.gpu_memory_used_mb),
                Self::format_memory_size(snapshot.gpu_memory_total_mb)
            )),
            Line::from(format!("🌡️  GPU Temp: {:.1}°C", snapshot.gpu_temperature_c)),
            Line::from(format!("💻 CPU Usage: {:.1}%", snapshot.cpu_usage_percent)),
            Line::from(format!(
                "💾 Memory: {} / {}",
                Self::format_memory_size(snapshot.memory_usage_mb),
                Self::format_memory_size(snapshot.memory_total_mb)
            )),
        ]
    }

    /// Get progress bar style based on progress value
    pub fn get_progress_style(progress: f64) -> Style {
        if progress < 0.3 {
            Style::default().fg(Color::Red)
        } else if progress < 0.7 {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        }
    }
}

/// Color theme for the TUI
pub struct ColorTheme {
    pub primary: Color,
    pub secondary: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,
    pub background: Color,
    pub text: Color,
}

impl Default for ColorTheme {
    fn default() -> Self {
        Self {
            primary: Color::Blue,
            secondary: Color::Cyan,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            info: Color::White,
            background: Color::Black,
            text: Color::White,
        }
    }
}

impl ColorTheme {
    /// Create a dark theme
    pub fn dark() -> Self {
        Self::default()
    }

    /// Create a light theme (placeholder for future implementation)
    pub fn light() -> Self {
        Self {
            primary: Color::Blue,
            secondary: Color::Cyan,
            success: Color::Green,
            warning: Color::Rgb(255, 165, 0), // Orange
            error: Color::Red,
            info: Color::Black,
            background: Color::White,
            text: Color::Black,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    #[test]
    fn test_format_duration() {
        // RED PHASE: Test will fail until UiUtils is properly implemented
        assert_eq!(UiUtils::format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(UiUtils::format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(
            UiUtils::format_duration(Duration::from_secs(3661)),
            "1h 1m 1s"
        );
        assert_eq!(
            UiUtils::format_duration(Duration::from_secs(7200)),
            "2h 0m 0s"
        );
    }

    #[test]
    fn test_format_memory_size() {
        assert_eq!(UiUtils::format_memory_size(512.0), "512.0 MB");
        assert_eq!(UiUtils::format_memory_size(1024.0), "1.0 GB");
        assert_eq!(UiUtils::format_memory_size(1536.0), "1.5 GB");
        assert_eq!(UiUtils::format_memory_size(2048.0), "2.0 GB");
    }

    #[test]
    fn test_get_phase_color() {
        assert_eq!(
            UiUtils::get_phase_color(&BenchmarkPhase::Initialization),
            Color::Yellow
        );
        assert_eq!(
            UiUtils::get_phase_color(&BenchmarkPhase::SystemCheck),
            Color::Blue
        );
        assert_eq!(
            UiUtils::get_phase_color(&BenchmarkPhase::ScalabilityTests),
            Color::Green
        );
        assert_eq!(
            UiUtils::get_phase_color(&BenchmarkPhase::LlmTests),
            Color::Magenta
        );
        assert_eq!(
            UiUtils::get_phase_color(&BenchmarkPhase::Complete),
            Color::Green
        );
    }

    #[test]
    fn test_format_phase_name() {
        assert_eq!(
            UiUtils::format_phase_name(&BenchmarkPhase::Initialization),
            "🚀 Initialization"
        );
        assert_eq!(
            UiUtils::format_phase_name(&BenchmarkPhase::SystemCheck),
            "🔍 System Check"
        );
        assert_eq!(
            UiUtils::format_phase_name(&BenchmarkPhase::ScalabilityTests),
            "📈 Scalability Tests"
        );
        assert_eq!(
            UiUtils::format_phase_name(&BenchmarkPhase::Complete),
            "✅ Complete"
        );
    }

    #[test]
    fn test_create_popup_area() {
        let area = Rect::new(0, 0, 100, 50);
        let popup = UiUtils::create_popup_area(60, 70, area);

        // Should be centered
        assert!(popup.x > 0);
        assert!(popup.y > 0);
        assert!(popup.width < area.width);
        assert!(popup.height < area.height);
    }

    #[test]
    fn test_get_progress_style() {
        // Test different progress ranges
        assert_eq!(UiUtils::get_progress_style(0.1).fg, Some(Color::Red));
        assert_eq!(UiUtils::get_progress_style(0.5).fg, Some(Color::Yellow));
        assert_eq!(UiUtils::get_progress_style(0.9).fg, Some(Color::Green));
    }

    #[test]
    fn test_color_theme_default() {
        let theme = ColorTheme::default();
        assert_eq!(theme.primary, Color::Blue);
        assert_eq!(theme.success, Color::Green);
        assert_eq!(theme.error, Color::Red);
        assert_eq!(theme.text, Color::White);
    }

    #[test]
    fn test_color_theme_dark() {
        let theme = ColorTheme::dark();
        assert_eq!(theme.background, Color::Black);
        assert_eq!(theme.text, Color::White);
    }

    #[test]
    fn test_color_theme_light() {
        let theme = ColorTheme::light();
        assert_eq!(theme.background, Color::White);
        assert_eq!(theme.text, Color::Black);
    }

    #[test]
    fn test_create_sparkline_data() {
        use crate::benchmarks::progress_monitor::ResourceSnapshot;

        let snapshots = vec![
            ResourceSnapshot {
                timestamp: 1,
                cpu_usage_percent: 10.0,
                memory_usage_mb: 1000.0,
                memory_total_mb: 8000.0,
                gpu_usage_percent: 25.0,
                gpu_memory_used_mb: 2000.0,
                gpu_memory_total_mb: 8000.0,
                gpu_temperature_c: 65.0,
                disk_io_read_mb_s: 50.0,
                disk_io_write_mb_s: 25.0,
            },
            ResourceSnapshot {
                timestamp: 2,
                cpu_usage_percent: 20.0,
                memory_usage_mb: 1200.0,
                memory_total_mb: 8000.0,
                gpu_usage_percent: 50.0,
                gpu_memory_used_mb: 3000.0,
                gpu_memory_total_mb: 8000.0,
                gpu_temperature_c: 70.0,
                disk_io_read_mb_s: 75.0,
                disk_io_write_mb_s: 35.0,
            },
        ];

        let data = UiUtils::create_sparkline_data(&snapshots, 10);
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], 25);
        assert_eq!(data[1], 50);
    }
}
