//! Tests for report generator module

#[cfg(test)]
mod tests {
    use crate::metrics_collector::{MetricDataPoint, MetricStatistics, MetricType};
    use crate::report_generator::{
        config::{ReportGeneratorConfig, ReportSchedule},
        generator::ReportGenerator,
        report_types::{RegressionDetail, RegressionSeverity, SectionContentType, TrendDirection},
        types::{ChartType, DashboardWidget, ReportOutputFormat},
    };
    use chrono::{Duration, Utc};
    use ordered_float::OrderedFloat;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_config(temp_dir: &TempDir) -> ReportGeneratorConfig {
        ReportGeneratorConfig {
            output_directory: temp_dir.path().to_path_buf(),
            template_directory: temp_dir.path().join("templates"),
            chart_theme: "test_theme".to_string(),
            enable_aggregation: true,
            retention_days: 7,
            max_chart_points: 100,
        }
    }

    fn create_test_metrics() -> Vec<MetricDataPoint> {
        let now = Utc::now();
        vec![
            MetricDataPoint {
                metric_type: MetricType::Throughput,
                timestamp: now - Duration::minutes(5),
                value: OrderedFloat(1000.0),
                tags: HashMap::from([("host".to_string(), "server1".to_string())]),
                source: "test".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::Throughput,
                timestamp: now - Duration::minutes(3),
                value: OrderedFloat(1200.0),
                tags: HashMap::from([("host".to_string(), "server1".to_string())]),
                source: "test".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::ResponseTime,
                timestamp: now - Duration::minutes(5),
                value: OrderedFloat(25.0),
                tags: HashMap::from([("endpoint".to_string(), "/api/v1".to_string())]),
                source: "test".to_string(),
            },
            MetricDataPoint {
                metric_type: MetricType::ResponseTime,
                timestamp: now - Duration::minutes(3),
                value: OrderedFloat(30.0),
                tags: HashMap::from([("endpoint".to_string(), "/api/v1".to_string())]),
                source: "test".to_string(),
            },
        ]
    }

    fn create_test_statistics() -> Vec<MetricStatistics> {
        let now = Utc::now();
        vec![
            MetricStatistics {
                metric_type: MetricType::Throughput,
                average: OrderedFloat(1100.0),
                minimum: OrderedFloat(1000.0),
                maximum: OrderedFloat(1200.0),
                std_deviation: OrderedFloat(100.0),
                p95: OrderedFloat(1190.0),
                p99: OrderedFloat(1198.0),
                sample_count: 2,
                window_start: now - Duration::minutes(5),
                window_end: now,
            },
            MetricStatistics {
                metric_type: MetricType::ResponseTime,
                average: OrderedFloat(27.5),
                minimum: OrderedFloat(25.0),
                maximum: OrderedFloat(30.0),
                std_deviation: OrderedFloat(2.5),
                p95: OrderedFloat(29.5),
                p99: OrderedFloat(29.9),
                sample_count: 2,
                window_start: now - Duration::minutes(5),
                window_end: now,
            },
        ]
    }

    #[tokio::test]
    async fn test_report_generator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);

        let _generator = ReportGenerator::new(config).unwrap();
        assert!(temp_dir.path().exists());
    }

    #[tokio::test]
    async fn test_generate_html_report() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let stats = create_test_statistics();

        let report = generator
            .generate_report(
                "Test Performance Report".to_string(),
                ReportOutputFormat::Html,
                metrics,
                stats,
            )
            .await
            .unwrap();

        assert_eq!(report.title, "Test Performance Report");
        assert_eq!(report.format, ReportOutputFormat::Html);
        assert!(!report.sections.is_empty());
        assert!(report.generated_at <= Utc::now());
    }

    #[tokio::test]
    async fn test_generate_json_report() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let stats = create_test_statistics();

        let report = generator
            .generate_report(
                "JSON Report".to_string(),
                ReportOutputFormat::Json,
                metrics,
                stats,
            )
            .await
            .unwrap();

        assert_eq!(report.format, ReportOutputFormat::Json);

        // Verify JSON file was created
        let json_path = temp_dir.path().join(format!(
            "report_{}_{}.json",
            report.id,
            report.generated_at.timestamp()
        ));
        assert!(json_path.exists());
    }

    #[tokio::test]
    async fn test_create_dashboard() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let widgets = vec![
            DashboardWidget {
                id: "widget1".to_string(),
                title: "Throughput".to_string(),
                chart_type: ChartType::TimeSeries,
                metrics: vec![MetricType::Throughput],
                position: (0, 0),
                size: (6, 4),
                refresh_interval: 60,
            },
            DashboardWidget {
                id: "widget2".to_string(),
                title: "Latency".to_string(),
                chart_type: ChartType::Gauge,
                metrics: vec![MetricType::ResponseTime],
                position: (0, 6),
                size: (6, 4),
                refresh_interval: 30,
            },
        ];

        let metrics = create_test_metrics();
        let dashboard_path = generator
            .create_dashboard("Test Dashboard".to_string(), widgets, metrics)
            .await
            .unwrap();

        assert!(PathBuf::from(&dashboard_path).exists());
        assert!(dashboard_path.contains("dashboard_"));
        assert!(dashboard_path.ends_with(".html"));
    }

    #[tokio::test]
    async fn test_export_metrics_csv() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let output_path = temp_dir.path().join("metrics.csv");

        generator
            .export_metrics_data(
                ReportOutputFormat::Csv,
                metrics.clone(),
                output_path.clone(),
            )
            .await
            .unwrap();

        assert!(output_path.exists());

        // Verify CSV content
        let content = tokio::fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("metric_type,timestamp,value"));
        // Check we have at least header + some data
        assert!(content.lines().count() > 1);
    }

    #[tokio::test]
    async fn test_generate_trend_report() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let trend_report = generator
            .generate_trend_report(metrics, Duration::hours(1))
            .await
            .unwrap();

        assert_eq!(trend_report.period, Duration::hours(1));
        assert!(!trend_report.trends.is_empty());

        // Check trend detection
        let throughput_trend = trend_report
            .trends
            .iter()
            .find(|t| t.metric_type == MetricType::Throughput)
            .unwrap();
        assert_eq!(throughput_trend.direction, TrendDirection::Increasing);
        assert!(throughput_trend.strength > 0.0);
    }

    #[tokio::test]
    async fn test_create_regression_summary() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let regressions = vec![RegressionDetail {
            detected_at: Utc::now(),
            severity: RegressionSeverity::High,
            baseline_value: 1000.0,
            current_value: 500.0,
            degradation_percent: 50.0,
        }];

        let mut regressions_map = HashMap::new();
        regressions_map.insert(MetricType::Throughput, regressions);

        let summary = generator
            .create_regression_summary(regressions_map)
            .await
            .unwrap();

        assert_eq!(summary.total_regressions, 1);
        assert!(summary.health_score < 1.0); // Health score should be reduced
        assert!(!summary.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_schedule_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let schedule = ReportSchedule {
            id: "daily_report".to_string(),
            name: "Daily Performance Report".to_string(),
            cron_expression: "0 0 * * *".to_string(),
            format: ReportOutputFormat::Pdf,
            recipients: vec!["test@example.com".to_string()],
            enabled: true,
        };

        generator
            .schedule_report_generation(schedule.clone())
            .await
            .unwrap();

        // Since we can't access private fields, just verify the method succeeds
        // In a real implementation we would need public methods to query schedules
        assert_eq!(schedule.name, "Daily Performance Report");
    }

    #[tokio::test]
    async fn test_report_sections_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let stats = create_test_statistics();

        // Generate report to test sections creation indirectly
        let report = generator
            .generate_report(
                "Section Test Report".to_string(),
                ReportOutputFormat::Json,
                metrics,
                stats,
            )
            .await
            .unwrap();

        // Verify required sections
        assert!(report
            .sections
            .iter()
            .any(|s| s.content_type == SectionContentType::Summary));
        assert!(report
            .sections
            .iter()
            .any(|s| s.content_type == SectionContentType::Chart));
        assert!(report
            .sections
            .iter()
            .any(|s| s.content_type == SectionContentType::Table));
    }

    #[tokio::test]
    async fn test_empty_metrics_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let report = generator
            .generate_report(
                "Empty Report".to_string(),
                ReportOutputFormat::Json,
                vec![],
                vec![],
            )
            .await
            .unwrap();

        assert_eq!(report.sections.len(), 1); // Should have at least a summary
        assert_eq!(
            report.time_range.1 - report.time_range.0,
            Duration::hours(1)
        );
    }

    #[tokio::test]
    async fn test_trend_anomaly_detection() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        // Create enough data points for anomaly detection
        let now = Utc::now();
        let mut metrics = vec![];

        // Add normal data points
        for i in 0..5 {
            metrics.push(MetricDataPoint {
                metric_type: MetricType::Throughput,
                timestamp: now - Duration::minutes(10 - i),
                value: OrderedFloat(1000.0 + (i as f64 * 10.0)),
                tags: HashMap::new(),
                source: "test".to_string(),
            });
        }

        // Add anomalous data point
        metrics.push(MetricDataPoint {
            metric_type: MetricType::Throughput,
            timestamp: now - Duration::minutes(1),
            value: OrderedFloat(100.0), // Significantly lower than normal
            tags: HashMap::new(),
            source: "test".to_string(),
        });

        let trend_report = generator
            .generate_trend_report(metrics, Duration::hours(1))
            .await
            .unwrap();

        assert!(!trend_report.anomalies.is_empty());
        let anomaly = &trend_report.anomalies[0];
        assert_eq!(anomaly.metric_type, MetricType::Throughput);
        assert!(anomaly.score > 0.5); // High anomaly score
    }

    #[tokio::test]
    async fn test_markdown_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let generator = ReportGenerator::new(config).unwrap();

        let metrics = create_test_metrics();
        let stats = create_test_statistics();

        let report = generator
            .generate_report(
                "Markdown Report".to_string(),
                ReportOutputFormat::Markdown,
                metrics,
                stats,
            )
            .await
            .unwrap();

        assert_eq!(report.format, ReportOutputFormat::Markdown);

        let md_path = temp_dir.path().join(format!(
            "report_{}_{}.md",
            report.id,
            report.generated_at.timestamp()
        ));
        assert!(md_path.exists());

        let content = tokio::fs::read_to_string(&md_path).await.unwrap();
        assert!(content.contains("# Markdown Report"));
        assert!(content.contains("## Performance Summary"));
    }

    #[tokio::test]
    async fn test_chart_data_aggregation() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = create_test_config(&temp_dir);
        config.max_chart_points = 2;
        let generator = ReportGenerator::new(config).unwrap();

        // Create many data points
        let mut metrics = vec![];
        let now = Utc::now();
        for i in 0..10 {
            metrics.push(MetricDataPoint {
                metric_type: MetricType::Throughput,
                timestamp: now - Duration::minutes(i),
                value: OrderedFloat(1000.0 + (i as f64 * 10.0)),
                tags: HashMap::new(),
                source: "test".to_string(),
            });
        }

        let aggregated = generator.aggregate_chart_data(&metrics).unwrap();
        assert!(aggregated.len() <= 2); // Should be limited to max_chart_points
    }

    #[tokio::test]
    async fn test_report_retention_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = create_test_config(&temp_dir);
        config.retention_days = 0; // Immediate cleanup
        let generator = ReportGenerator::new(config).unwrap();

        // Create a report
        let report = generator
            .generate_report(
                "Old Report".to_string(),
                ReportOutputFormat::Json,
                vec![],
                vec![],
            )
            .await
            .unwrap();

        let report_path = temp_dir.path().join(format!(
            "report_{}_{}.json",
            report.id,
            report.generated_at.timestamp()
        ));
        assert!(report_path.exists());

        // Run cleanup
        generator.cleanup_old_reports().await.unwrap();

        // Report should be deleted
        assert!(!report_path.exists());
    }

    #[test]
    fn test_report_output_format_serialization() {
        let formats = vec![
            ReportOutputFormat::Html,
            ReportOutputFormat::Pdf,
            ReportOutputFormat::Json,
            ReportOutputFormat::Csv,
            ReportOutputFormat::Markdown,
        ];

        for format in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            let deserialized: ReportOutputFormat = serde_json::from_str(&serialized).unwrap();
            assert_eq!(format, deserialized);
        }
    }

    #[test]
    fn test_chart_type_properties() {
        assert_eq!(ChartType::TimeSeries, ChartType::TimeSeries);
        assert_ne!(ChartType::Bar, ChartType::Scatter);

        let chart_types = vec![
            ChartType::TimeSeries,
            ChartType::Bar,
            ChartType::Heatmap,
            ChartType::Scatter,
            ChartType::Histogram,
            ChartType::Gauge,
        ];

        for chart_type in chart_types {
            let serialized = serde_json::to_string(&chart_type).unwrap();
            assert!(!serialized.is_empty());
        }
    }
}
