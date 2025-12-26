//! Main report generator implementation

use super::config::{ReportGeneratorConfig, ReportSchedule};
use super::data_analysis::{
    aggregate_chart_data, analyze_trend, calculate_health_score, create_prediction,
    detect_anomalies, generate_recommendations, get_unique_metric_types,
};
use super::html_generator::{
    generate_csv_content, generate_dashboard_html, generate_html_report, generate_markdown_report,
};
use super::report_types::{
    PerformanceReport, RegressionDetail, RegressionSummary, ReportSection, SectionContentType,
    TrendReport,
};
use super::types::{DashboardWidget, ReportOutputFormat};
use crate::error::{PerformanceRegressionError, PerformanceRegressionResult};
use crate::metrics_collector::{MetricDataPoint, MetricStatistics, MetricType};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Report generator
pub struct ReportGenerator {
    config: ReportGeneratorConfig,
    scheduled_reports: Arc<RwLock<HashMap<String, ReportSchedule>>>,
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new(config: ReportGeneratorConfig) -> PerformanceRegressionResult<Self> {
        // Create output directory if it doesn't exist
        if !config.output_directory.exists() {
            std::fs::create_dir_all(&config.output_directory)
                .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
        }

        Ok(Self {
            config,
            scheduled_reports: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate performance report
    pub async fn generate_report(
        &self,
        title: String,
        format: ReportOutputFormat,
        metrics_data: Vec<MetricDataPoint>,
        statistics: Vec<MetricStatistics>,
    ) -> PerformanceRegressionResult<PerformanceReport> {
        let now = Utc::now();
        let time_range = if metrics_data.is_empty() {
            (now - Duration::hours(1), now)
        } else {
            let start = metrics_data
                .iter()
                .map(|m| m.timestamp)
                .min()
                .unwrap_or(now - Duration::hours(1));
            let end = metrics_data
                .iter()
                .map(|m| m.timestamp)
                .max()
                .unwrap_or(now);
            (start, end)
        };

        let sections = self.create_report_sections(&metrics_data, &statistics)?;

        let report = PerformanceReport {
            id: uuid::Uuid::new_v4().to_string(),
            title,
            generated_at: now,
            time_range,
            format,
            sections,
            metadata: HashMap::new(),
        };

        // Generate report file based on format
        self.save_report(&report).await?;

        Ok(report)
    }

    /// Create dashboard
    pub async fn create_dashboard(
        &self,
        title: String,
        widgets: Vec<DashboardWidget>,
        metrics_data: Vec<MetricDataPoint>,
    ) -> PerformanceRegressionResult<String> {
        let dashboard_id = uuid::Uuid::new_v4().to_string();

        // Create dashboard HTML
        let html = generate_dashboard_html(&title, &widgets, &metrics_data)?;

        // Save dashboard
        let dashboard_path = self
            .config
            .output_directory
            .join(format!("dashboard_{}.html", dashboard_id));

        tokio::fs::write(&dashboard_path, html)
            .await
            .map_err(|e| PerformanceRegressionError::IoError { source: e })?;

        Ok(dashboard_path.to_string_lossy().to_string())
    }

    /// Export metrics data
    pub async fn export_metrics_data(
        &self,
        format: ReportOutputFormat,
        metrics_data: Vec<MetricDataPoint>,
        output_path: PathBuf,
    ) -> PerformanceRegressionResult<()> {
        match format {
            ReportOutputFormat::Csv => {
                let csv = generate_csv_content(&metrics_data)?;
                tokio::fs::write(&output_path, csv)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
            ReportOutputFormat::Json => {
                let json = serde_json::to_string_pretty(&metrics_data)?;
                tokio::fs::write(&output_path, json)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
            _ => {
                // For other formats, default to JSON
                let json = serde_json::to_string_pretty(&metrics_data)?;
                tokio::fs::write(&output_path, json)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
        }

        Ok(())
    }

    /// Generate trend report
    pub async fn generate_trend_report(
        &self,
        metrics_data: Vec<MetricDataPoint>,
        analysis_period: Duration,
    ) -> PerformanceRegressionResult<TrendReport> {
        let now = Utc::now();
        let cutoff_time = now - analysis_period;

        // Filter data to analysis period
        let period_data: Vec<_> = metrics_data
            .into_iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        // Group by metric type
        let mut metrics_by_type: HashMap<MetricType, Vec<MetricDataPoint>> = HashMap::new();
        for metric in period_data {
            metrics_by_type
                .entry(metric.metric_type.clone())
                .or_insert_with(Vec::new)
                .push(metric);
        }

        // Analyze trends for each metric type
        let mut trends = Vec::new();
        let mut predictions = Vec::new();
        let mut anomalies = Vec::new();

        for (metric_type, data) in &metrics_by_type {
            // Sort by timestamp
            let mut sorted_data = data.clone();
            sorted_data.sort_by_key(|m| m.timestamp);

            // Trend analysis
            if let Some(trend) = analyze_trend(metric_type, &sorted_data) {
                trends.push(trend);
            }

            // Create predictions
            if let Some(prediction) = create_prediction(metric_type, &sorted_data, 1.0) {
                predictions.push(prediction);
            }

            // Detect anomalies
            let metric_anomalies = detect_anomalies(metric_type, &sorted_data);
            anomalies.extend(metric_anomalies);
        }

        Ok(TrendReport {
            id: uuid::Uuid::new_v4().to_string(),
            period: analysis_period,
            trends,
            predictions,
            anomalies,
        })
    }

    /// Create regression summary
    pub async fn create_regression_summary(
        &self,
        regressions_by_metric: HashMap<MetricType, Vec<RegressionDetail>>,
    ) -> PerformanceRegressionResult<RegressionSummary> {
        let now = Utc::now();
        let total_regressions = regressions_by_metric
            .values()
            .map(|v| v.len())
            .sum::<usize>();

        let health_score = calculate_health_score(&regressions_by_metric);
        let recommendations = generate_recommendations(&regressions_by_metric);

        // Determine analysis period from regressions
        let mut earliest = now;
        let mut latest = now - Duration::days(1);

        for regressions in regressions_by_metric.values() {
            for reg in regressions {
                if reg.detected_at < earliest {
                    earliest = reg.detected_at;
                }
                if reg.detected_at > latest {
                    latest = reg.detected_at;
                }
            }
        }

        Ok(RegressionSummary {
            id: uuid::Uuid::new_v4().to_string(),
            period: (earliest, latest),
            total_regressions,
            regressions_by_metric,
            health_score,
            recommendations,
        })
    }

    /// Schedule report generation
    pub async fn schedule_report_generation(
        &self,
        schedule: ReportSchedule,
    ) -> PerformanceRegressionResult<()> {
        let mut scheduled = self.scheduled_reports.write().await;
        scheduled.insert(schedule.id.clone(), schedule);
        Ok(())
    }

    // Helper methods

    fn create_report_sections(
        &self,
        metrics_data: &[MetricDataPoint],
        statistics: &[MetricStatistics],
    ) -> PerformanceRegressionResult<Vec<ReportSection>> {
        let mut sections = Vec::new();

        // Summary section
        sections.push(ReportSection {
            title: "Performance Summary".to_string(),
            content_type: SectionContentType::Summary,
            data: serde_json::json!({
                "total_metrics": metrics_data.len(),
                "statistics_count": statistics.len(),
                "metric_types": get_unique_metric_types(metrics_data),
            }),
        });

        // Statistics section
        if !statistics.is_empty() {
            sections.push(ReportSection {
                title: "Metric Statistics".to_string(),
                content_type: SectionContentType::Table,
                data: serde_json::to_value(statistics)?,
            });
        }

        // Time series chart section
        if !metrics_data.is_empty() {
            sections.push(ReportSection {
                title: "Performance Trends".to_string(),
                content_type: SectionContentType::Chart,
                data: serde_json::json!({
                    "chart_type": "time_series",
                    "data": metrics_data,
                }),
            });
        }

        Ok(sections)
    }

    async fn save_report(&self, report: &PerformanceReport) -> PerformanceRegressionResult<()> {
        let filename = format!(
            "report_{}_{}.{}",
            report.id,
            report.generated_at.timestamp(),
            match report.format {
                ReportOutputFormat::Html => "html",
                ReportOutputFormat::Pdf => "pdf",
                ReportOutputFormat::Json => "json",
                ReportOutputFormat::Csv => "csv",
                ReportOutputFormat::Markdown => "md",
            }
        );

        let path = self.config.output_directory.join(filename);

        match report.format {
            ReportOutputFormat::Json => {
                let json = serde_json::to_string_pretty(report)?;
                tokio::fs::write(&path, json)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
            ReportOutputFormat::Html => {
                let html = generate_html_report(report)?;
                tokio::fs::write(&path, html)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
            ReportOutputFormat::Markdown => {
                let markdown = generate_markdown_report(report)?;
                tokio::fs::write(&path, markdown)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
            _ => {
                // For PDF and CSV, we'd need additional libraries
                // For now, fallback to JSON
                let json = serde_json::to_string_pretty(report)?;
                tokio::fs::write(&path, json)
                    .await
                    .map_err(|e| PerformanceRegressionError::IoError { source: e })?;
            }
        }

        Ok(())
    }

    /// Aggregate chart data to fit within max points limit
    pub fn aggregate_chart_data(
        &self,
        metrics_data: &[MetricDataPoint],
    ) -> PerformanceRegressionResult<Vec<MetricDataPoint>> {
        Ok(aggregate_chart_data(
            metrics_data,
            self.config.max_chart_points,
        ))
    }

    /// Clean up old reports based on retention policy
    pub async fn cleanup_old_reports(&self) -> PerformanceRegressionResult<()> {
        let cutoff_date = Utc::now() - Duration::days(self.config.retention_days as i64);

        let mut entries = tokio::fs::read_dir(&self.config.output_directory)
            .await
            .map_err(|e| PerformanceRegressionError::IoError { source: e })?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| PerformanceRegressionError::IoError { source: e })?
        {
            let path = entry.path();
            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    let modified_time = DateTime::<Utc>::from(modified);
                    if modified_time < cutoff_date {
                        let _ = tokio::fs::remove_file(path).await;
                    }
                }
            }
        }

        Ok(())
    }
}
