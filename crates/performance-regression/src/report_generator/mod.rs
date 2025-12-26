//! Performance report generation and dashboard creation

pub mod config;
pub mod data_analysis;
pub mod generator;
pub mod html_generator;
pub mod report_types;
#[cfg(test)]
pub mod tests;
pub mod types;

pub use config::{ReportGeneratorConfig, ReportSchedule};
pub use generator::ReportGenerator;
pub use report_types::{
    MetricPrediction, MetricTrend, PerformanceReport, RegressionDetail, RegressionSeverity,
    RegressionSummary, ReportSection, SectionContentType, TrendAnomaly, TrendDirection,
    TrendReport,
};
pub use types::{ChartType, DashboardWidget, ReportOutputFormat};
