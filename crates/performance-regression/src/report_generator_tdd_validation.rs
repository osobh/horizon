//! TDD Validation Tests for Report Generator Module Splitting
//!
//! These tests establish the expected behavior that must be preserved
//! during the refactoring process (RED phase of TDD).

use super::report_generator::*;
use crate::metrics_collector::{MetricDataPoint, MetricStatistics, MetricType};
use chrono::{Duration, Utc};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use tempfile::TempDir;

/// TDD Test Suite to validate all functionality before refactoring
pub struct ReportGeneratorTddValidation;

impl ReportGeneratorTddValidation {
    /// Create test metric data for validation
    pub fn create_test_metric_data(count: usize, metric_type: MetricType) -> Vec<MetricDataPoint> {
        let now = Utc::now();
        (0..count)
            .map(|i| MetricDataPoint {
                metric_type: metric_type.clone(),
                value: OrderedFloat(100.0 + (i as f64 * 10.0)),
                timestamp: now - Duration::minutes(count as i64 - i as i64),
                tags: HashMap::new(),
                source: "test".to_string(),
            })
            .collect()
    }

    /// Create test statistics for validation
    pub fn create_test_statistics(metric_type: MetricType) -> MetricStatistics {
        MetricStatistics {
            metric_type,
            average: OrderedFloat(150.0),
            minimum: OrderedFloat(100.0),
            maximum: OrderedFloat(200.0),
            std_deviation: OrderedFloat(25.0),
            p95: OrderedFloat(190.0),
            p99: OrderedFloat(199.0),
            sample_count: 100,
            window_start: Utc::now() - Duration::hours(1),
            window_end: Utc::now(),
        }
    }

    /// Validate all core ReportGenerator functionality
    pub async fn validate_all_functionality() -> Result<(), Box<dyn std::error::Error>> {
        println!("🔴 TDD RED Phase: Validating existing functionality...");

        // Test 1: Core functionality tests
        Self::test_core_functionality().await?;

        // Test 2: Type serialization/deserialization
        Self::test_type_serialization().await?;

        // Test 3: Report generation in all formats
        Self::test_report_generation_all_formats().await?;

        // Test 4: Dashboard creation
        Self::test_dashboard_functionality().await?;

        // Test 5: Trend analysis and predictions
        Self::test_analysis_functionality().await?;

        // Test 6: Export functionality
        Self::test_export_functionality().await?;

        // Test 7: Scheduling functionality
        Self::test_scheduling_functionality().await?;

        // Test 8: Edge cases and error handling
        Self::test_edge_cases().await?;

        println!("✅ All TDD validation tests passed - functionality preserved");
        Ok(())
    }

    async fn test_core_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            template_directory: temp_dir.path().join("templates"),
            chart_theme: "dark".to_string(),
            enable_aggregation: true,
            retention_days: 30,
            max_chart_points: 1000,
        };

        let _generator = ReportGenerator::new(config)?;

        // Validate that the generator can be created successfully
        // (The actual validation is that no error occurred in creation)

        println!("✓ Core ReportGenerator creation works");
        Ok(())
    }

    async fn test_type_serialization() -> Result<(), Box<dyn std::error::Error>> {
        // Test all enum serialization
        let formats = vec![
            ReportOutputFormat::Html,
            ReportOutputFormat::Pdf,
            ReportOutputFormat::Json,
            ReportOutputFormat::Csv,
            ReportOutputFormat::Markdown,
        ];

        for format in formats {
            let serialized = serde_json::to_string(&format)?;
            let deserialized: ReportOutputFormat = serde_json::from_str(&serialized)?;
            assert_eq!(format, deserialized);
        }

        let chart_types = vec![
            ChartType::TimeSeries,
            ChartType::Bar,
            ChartType::Heatmap,
            ChartType::Scatter,
            ChartType::Histogram,
            ChartType::Gauge,
        ];

        for chart_type in chart_types {
            let serialized = serde_json::to_string(&chart_type)?;
            let deserialized: ChartType = serde_json::from_str(&serialized)?;
            assert_eq!(chart_type, deserialized);
        }

        println!("✓ Type serialization/deserialization works");
        Ok(())
    }

    async fn test_report_generation_all_formats() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        let metrics_data = Self::create_test_metric_data(10, MetricType::ResponseTime);
        let statistics = vec![Self::create_test_statistics(MetricType::ResponseTime)];

        // Test all formats
        let formats = vec![
            ReportOutputFormat::Html,
            ReportOutputFormat::Json,
            ReportOutputFormat::Markdown,
            ReportOutputFormat::Pdf,
            ReportOutputFormat::Csv,
        ];

        for format in formats {
            let report = generator
                .generate_report(
                    format!("Test Report {:?}", format),
                    format.clone(),
                    metrics_data.clone(),
                    statistics.clone(),
                )
                .await?;

            assert_eq!(report.format, format);
            assert!(!report.sections.is_empty());
            assert!(!report.id.is_empty());
        }

        println!("✓ Report generation in all formats works");
        Ok(())
    }

    async fn test_dashboard_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        let widgets = vec![
            DashboardWidget {
                id: "widget1".to_string(),
                title: "CPU Usage".to_string(),
                chart_type: ChartType::Gauge,
                metrics: vec![MetricType::CpuUsage],
                position: (0, 0),
                size: (3, 3),
                refresh_interval: 10,
            },
            DashboardWidget {
                id: "widget2".to_string(),
                title: "Response Time".to_string(),
                chart_type: ChartType::TimeSeries,
                metrics: vec![MetricType::ResponseTime],
                position: (3, 0),
                size: (6, 4),
                refresh_interval: 30,
            },
        ];

        let metrics_data = Self::create_test_metric_data(50, MetricType::ResponseTime);
        let dashboard_path = generator
            .create_dashboard("Test Dashboard".to_string(), widgets, metrics_data)
            .await?;

        assert!(dashboard_path.contains("dashboard_"));
        assert!(dashboard_path.ends_with(".html"));

        println!("✓ Dashboard functionality works");
        Ok(())
    }

    async fn test_analysis_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        // Test trend analysis
        let metrics_data = Self::create_test_metric_data(100, MetricType::ResponseTime);
        let trend_report = generator
            .generate_trend_report(metrics_data.clone(), Duration::hours(24))
            .await?;

        assert_eq!(trend_report.period, Duration::hours(24));
        assert!(!trend_report.id.is_empty());

        // Test regression summary
        let now = Utc::now();
        let mut regressions_map = HashMap::new();
        regressions_map.insert(
            MetricType::ResponseTime,
            vec![RegressionDetail {
                detected_at: now - Duration::hours(2),
                baseline_value: 100.0,
                current_value: 150.0,
                degradation_percent: 50.0,
                severity: RegressionSeverity::High,
            }],
        );

        let summary = generator.create_regression_summary(regressions_map).await?;

        assert!(summary.total_regressions > 0);
        assert!(!summary.recommendations.is_empty());

        println!("✓ Analysis functionality works");
        Ok(())
    }

    async fn test_export_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        let metrics_data = Self::create_test_metric_data(10, MetricType::Throughput);

        // Test JSON export
        let json_path = temp_dir.path().join("export.json");
        generator
            .export_metrics_data(
                ReportOutputFormat::Json,
                metrics_data.clone(),
                json_path.clone(),
            )
            .await?;
        assert!(json_path.exists());

        // Test CSV export
        let csv_path = temp_dir.path().join("export.csv");
        generator
            .export_metrics_data(ReportOutputFormat::Csv, metrics_data, csv_path.clone())
            .await?;
        assert!(csv_path.exists());

        println!("✓ Export functionality works");
        Ok(())
    }

    async fn test_scheduling_functionality() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        let schedule = ReportSchedule {
            id: "test-schedule".to_string(),
            name: "Test Schedule".to_string(),
            cron_expression: "0 0 * * *".to_string(),
            format: ReportOutputFormat::Html,
            recipients: vec!["test@example.com".to_string()],
            enabled: true,
        };

        generator.schedule_report_generation(schedule).await?;

        println!("✓ Scheduling functionality works");
        Ok(())
    }

    async fn test_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            ..ReportGeneratorConfig::default()
        };
        let generator = ReportGenerator::new(config)?;

        // Test with empty data
        let empty_metrics: Vec<MetricDataPoint> = Vec::new();
        let empty_statistics: Vec<MetricStatistics> = Vec::new();

        let report = generator
            .generate_report(
                "Empty Report".to_string(),
                ReportOutputFormat::Json,
                empty_metrics,
                empty_statistics,
            )
            .await?;

        assert_eq!(report.title, "Empty Report");
        assert!(!report.sections.is_empty()); // Should still have summary section

        // Test disabled schedule
        let disabled_schedule = ReportSchedule {
            id: "disabled".to_string(),
            name: "Disabled".to_string(),
            cron_expression: "0 0 * * *".to_string(),
            format: ReportOutputFormat::Html,
            recipients: vec![],
            enabled: false,
        };

        generator
            .schedule_report_generation(disabled_schedule)
            .await?;

        println!("✓ Edge cases handled correctly");
        Ok(())
    }
}

#[cfg(test)]
mod tdd_validation_tests {
    use super::*;

    #[tokio::test]
    async fn run_full_tdd_validation() {
        ReportGeneratorTddValidation::validate_all_functionality()
            .await
            .expect("TDD validation should pass before refactoring");
    }

    #[tokio::test]
    async fn test_metric_data_creation() {
        let data = ReportGeneratorTddValidation::create_test_metric_data(5, MetricType::CpuUsage);
        assert_eq!(data.len(), 5);
        assert_eq!(data[0].metric_type, MetricType::CpuUsage);
        assert_eq!(data[0].value.0, 100.0);
        assert_eq!(data[4].value.0, 140.0);
    }

    #[tokio::test]
    async fn test_statistics_creation() {
        let stats = ReportGeneratorTddValidation::create_test_statistics(MetricType::MemoryUsage);
        assert_eq!(stats.metric_type, MetricType::MemoryUsage);
        assert_eq!(stats.average.0, 150.0);
        assert_eq!(stats.sample_count, 100);
    }
}
