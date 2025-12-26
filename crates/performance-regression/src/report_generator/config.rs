//! Configuration types for report generation

use super::types::ReportOutputFormat;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Report scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    /// Schedule identifier
    pub id: String,
    /// Report name
    pub name: String,
    /// Cron expression for scheduling
    pub cron_expression: String,
    /// Report format
    pub format: ReportOutputFormat,
    /// Recipients for email delivery
    pub recipients: Vec<String>,
    /// Whether schedule is enabled
    pub enabled: bool,
}

/// Report generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGeneratorConfig {
    /// Output directory for reports
    pub output_directory: PathBuf,
    /// Template directory for HTML/PDF generation
    pub template_directory: PathBuf,
    /// Default chart theme
    pub chart_theme: String,
    /// Enable data aggregation
    pub enable_aggregation: bool,
    /// Data retention period in days
    pub retention_days: u32,
    /// Maximum data points per chart
    pub max_chart_points: usize,
}

impl Default for ReportGeneratorConfig {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("./reports"),
            template_directory: PathBuf::from("./templates"),
            chart_theme: "dark".to_string(),
            enable_aggregation: true,
            retention_days: 30,
            max_chart_points: 1000,
        }
    }
}
