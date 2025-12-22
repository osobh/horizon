//! Basic types for report generation

use crate::metrics_collector::MetricType;
use serde::{Deserialize, Serialize};

/// Report output format types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportOutputFormat {
    /// HTML format with interactive charts
    Html,
    /// PDF format for distribution
    Pdf,
    /// JSON format for API consumption
    Json,
    /// CSV format for data analysis
    Csv,
    /// Markdown format for documentation
    Markdown,
}

/// Chart type for visualizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series
    TimeSeries,
    /// Bar chart for comparisons
    Bar,
    /// Heatmap for correlations
    Heatmap,
    /// Scatter plot for distributions
    Scatter,
    /// Histogram for frequency distribution
    Histogram,
    /// Gauge for current values
    Gauge,
}

/// Dashboard widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    /// Widget identifier
    pub id: String,
    /// Widget title
    pub title: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Metric types to display
    pub metrics: Vec<MetricType>,
    /// Widget position (row, column)
    pub position: (u32, u32),
    /// Widget size (width, height)
    pub size: (u32, u32),
    /// Refresh interval in seconds
    pub refresh_interval: u64,
}
