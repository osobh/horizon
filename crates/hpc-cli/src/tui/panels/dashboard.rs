//! Dashboard panel for the HPC-AI TUI
//!
//! Displays:
//! - GPU usage metrics
//! - Cluster health status
//! - Running jobs

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
    Frame,
};

use crate::core::{state::GpuMetric, AppState};
use crate::tui::{events::KeyAction, Theme};

use super::Panel;

/// Dashboard panel state
#[derive(Debug, Default)]
pub struct DashboardPanel {
    /// GPU metrics
    gpu_metrics: Vec<GpuMetric>,
    /// Cluster node count
    cluster_nodes: usize,
    /// Running services count
    running_services: usize,
    /// Jobs list
    jobs: Vec<JobDisplay>,
    /// Selected job index
    selected_job: usize,
    /// Scroll offset for jobs list
    scroll_offset: usize,
}

/// Job display info
#[derive(Debug, Clone)]
struct JobDisplay {
    name: String,
    status: JobStatus,
    progress: f64,
}

/// Job status for display
#[derive(Debug, Clone, Copy)]
enum JobStatus {
    Running,
    Pending,
    Completed,
    Failed,
}

impl JobStatus {
    fn symbol(&self) -> &'static str {
        match self {
            Self::Running => ">>",
            Self::Pending => "..",
            Self::Completed => "OK",
            Self::Failed => "XX",
        }
    }
}

impl DashboardPanel {
    /// Create a new dashboard panel
    pub fn new() -> Self {
        Self::default()
    }

    /// Render GPU usage section
    fn render_gpu_section(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Block::default()
            .title(" GPU Usage ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border));

        if self.gpu_metrics.is_empty() {
            let no_gpu = Paragraph::new("No GPUs detected")
                .style(Style::default().fg(theme.warning))
                .block(block);
            frame.render_widget(no_gpu, area);
            return;
        }

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Create gauge for each GPU (max 4 displayed)
        let gpu_count = self.gpu_metrics.len().min(4);
        let constraints: Vec<Constraint> = (0..gpu_count)
            .map(|_| Constraint::Length(2))
            .collect();

        let gpu_areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);

        for (i, metric) in self.gpu_metrics.iter().take(4).enumerate() {
            let usage_percent = metric.usage_percent as u16;
            let temp_str = metric.temperature_c
                .map(|t| format!("{}C", t as i32))
                .unwrap_or_else(|| "N/A".to_string());
            let label = format!(
                "GPU {}: {}% | {}MB/{}MB | {}",
                i,
                usage_percent,
                metric.memory_used_mb as i32,
                metric.memory_total_mb as i32,
                temp_str
            );

            let color = if metric.usage_percent > 90.0 {
                theme.error
            } else if metric.usage_percent > 70.0 {
                theme.warning
            } else {
                theme.success
            };

            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(color))
                .percent(usage_percent)
                .label(label);

            frame.render_widget(gauge, gpu_areas[i]);
        }
    }

    /// Render cluster health section
    fn render_cluster_section(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Block::default()
            .title(" Cluster Health ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let status_text = if self.cluster_nodes > 0 {
            vec![
                Line::from(vec![
                    Span::raw("Nodes: "),
                    Span::styled(
                        format!("{}", self.cluster_nodes),
                        Style::default()
                            .fg(theme.success)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::raw("Services: "),
                    Span::styled(
                        format!("{} running", self.running_services),
                        Style::default().fg(theme.primary),
                    ),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(
                "No cluster connected",
                Style::default().fg(theme.warning),
            ))]
        };

        let paragraph = Paragraph::new(status_text);
        frame.render_widget(paragraph, inner);
    }

    /// Render jobs section
    fn render_jobs_section(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        let border_color = if focused {
            theme.highlight
        } else {
            theme.border
        };

        let block = Block::default()
            .title(" Jobs ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        if self.jobs.is_empty() {
            let no_jobs = Paragraph::new("No active jobs")
                .style(Style::default().fg(theme.foreground))
                .block(block);
            frame.render_widget(no_jobs, area);
            return;
        }

        let items: Vec<ListItem> = self
            .jobs
            .iter()
            .enumerate()
            .skip(self.scroll_offset)
            .map(|(i, job)| {
                let status_color = match job.status {
                    JobStatus::Running => theme.primary,
                    JobStatus::Pending => theme.warning,
                    JobStatus::Completed => theme.success,
                    JobStatus::Failed => theme.error,
                };

                let style = if i == self.selected_job && focused {
                    Style::default()
                        .fg(theme.foreground)
                        .bg(theme.secondary)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.foreground)
                };

                let content = Line::from(vec![
                    Span::styled(format!("{} ", job.status.symbol()), Style::default().fg(status_color)),
                    Span::styled(&job.name, style),
                    Span::raw(format!(" ({:.0}%)", job.progress * 100.0)),
                ]);

                ListItem::new(content)
            })
            .collect();

        let list = List::new(items).block(block);
        frame.render_widget(list, area);
    }
}

impl Panel for DashboardPanel {
    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, theme: &Theme) {
        // Split dashboard into three vertical sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6),  // GPU usage
                Constraint::Length(5),  // Cluster health
                Constraint::Min(5),     // Jobs
            ])
            .split(area);

        self.render_gpu_section(frame, chunks[0], theme);
        self.render_cluster_section(frame, chunks[1], theme);
        self.render_jobs_section(frame, chunks[2], focused, theme);
    }

    fn handle_action(&mut self, action: KeyAction) -> bool {
        match action {
            KeyAction::Up => {
                if self.selected_job > 0 {
                    self.selected_job -= 1;
                    if self.selected_job < self.scroll_offset {
                        self.scroll_offset = self.selected_job;
                    }
                }
                true
            }
            KeyAction::Down => {
                if self.selected_job < self.jobs.len().saturating_sub(1) {
                    self.selected_job += 1;
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, state: &AppState) {
        // Update GPU metrics from state
        self.gpu_metrics = state.metrics.gpu_usage.clone();
        self.cluster_nodes = state.metrics.cluster_health.total_nodes as usize;
        self.running_services = state.metrics.cluster_health.healthy_nodes as usize;

        // Update jobs from state
        self.jobs = state
            .metrics
            .job_status
            .iter()
            .map(|j| JobDisplay {
                name: j.name.clone(),
                status: match j.status.as_str() {
                    "running" => JobStatus::Running,
                    "pending" => JobStatus::Pending,
                    "completed" => JobStatus::Completed,
                    _ => JobStatus::Failed,
                },
                progress: j.progress as f64,
            })
            .collect();
    }

    fn title(&self) -> &str {
        "Dashboard"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_panel_new() {
        let panel = DashboardPanel::new();
        assert!(panel.gpu_metrics.is_empty());
        assert_eq!(panel.cluster_nodes, 0);
        assert!(panel.jobs.is_empty());
    }

    #[test]
    fn test_job_status_symbol() {
        assert_eq!(JobStatus::Running.symbol(), ">>");
        assert_eq!(JobStatus::Pending.symbol(), "..");
        assert_eq!(JobStatus::Completed.symbol(), "OK");
        assert_eq!(JobStatus::Failed.symbol(), "XX");
    }

    #[test]
    fn test_dashboard_navigation() {
        let mut panel = DashboardPanel::new();
        panel.jobs = vec![
            JobDisplay {
                name: "job1".to_string(),
                status: JobStatus::Running,
                progress: 0.5,
            },
            JobDisplay {
                name: "job2".to_string(),
                status: JobStatus::Pending,
                progress: 0.0,
            },
        ];

        assert_eq!(panel.selected_job, 0);
        panel.handle_action(KeyAction::Down);
        assert_eq!(panel.selected_job, 1);
        panel.handle_action(KeyAction::Down);
        assert_eq!(panel.selected_job, 1); // Can't go past end
        panel.handle_action(KeyAction::Up);
        assert_eq!(panel.selected_job, 0);
    }
}
