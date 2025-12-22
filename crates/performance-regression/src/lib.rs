//! Performance regression detection with ML-based anomaly detection and automated analysis
//!
//! This crate provides comprehensive performance regression detection capabilities for:
//! - Real-time performance metric collection and analysis
//! - ML-based anomaly detection for performance degradation
//! - Automated baseline establishment and comparison
//! - Performance trend analysis and forecasting
//! - Resource bottleneck identification
//! - Automated performance testing and CI/CD integration

#![warn(missing_docs)]

pub mod alert_manager;
pub mod anomaly_detector;
pub mod baseline_manager;
pub mod bottleneck_detector;
pub mod error;
pub mod metrics_collector;
pub mod report_generator;
pub mod test_orchestrator;
pub mod trend_analyzer;

#[cfg(test)]
pub mod report_generator_tdd_validation;

pub use alert_manager::{
    Alert, AlertCondition, AlertManager, AlertManagerConfig, AlertSeverity, ComparisonOperator,
    NotificationChannel,
};
pub use anomaly_detector::{
    AnomalyConfig, AnomalyDetector, AnomalyInsights, AnomalyResult, MLAlgorithm,
};
pub use baseline_manager::{
    Baseline, BaselineConfig, BaselineManager, BaselineStrategy, RegressionResult,
    RegressionSeverity,
};
pub use bottleneck_detector::{
    BottleneckConfig, BottleneckDetector, BottleneckInsights, BottleneckSeverity, BottleneckType,
    DetectedBottleneck, TrendIndicator,
};
pub use error::{PerformanceRegressionError, PerformanceRegressionResult};
pub use metrics_collector::{
    MetricDataPoint, MetricStatistics, MetricType, MetricsCollector, MetricsConfig,
};
pub use report_generator::{
    ChartType, DashboardWidget, MetricPrediction, MetricTrend, PerformanceReport, RegressionDetail,
    RegressionSummary, ReportGenerator, ReportGeneratorConfig, ReportOutputFormat, ReportSchedule,
    ReportSection, SectionContentType, TrendAnomaly, TrendReport,
};
pub use test_orchestrator::{
    CiIntegrationConfig, CiMetadata, CiPlatform, EnduranceTestConfig, LoadTestConfig, ReportFormat,
    ResourceUsage, SchedulingConfig, SpikeTestConfig, StressTestConfig, TestExecutionResult,
    TestInsights, TestOrchestrator, TestOrchestratorConfig, TestReport, TestStatus, TestStrategy,
    TestSummary, VolumeTestConfig,
};
pub use trend_analyzer::{
    DegradationPattern, DetectedPattern, ForecastResult, SeasonalPattern, TrendAnalysis,
    TrendAnalyzer, TrendConfig, TrendDirection, TrendInsights,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_regression_framework_creation() {
        // Basic framework integration test
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config);
        assert!(collector.is_ok());

        // Test anomaly detector integration
        let anomaly_config = AnomalyConfig::default();
        let detector = AnomalyDetector::new(anomaly_config);

        // Verify basic functionality works together
        let metric = MetricDataPoint {
            metric_type: MetricType::ResponseTime,
            value: ordered_float::OrderedFloat(100.0),
            timestamp: chrono::Utc::now(),
            tags: std::collections::HashMap::new(),
            source: "test".to_string(),
        };

        detector.add_metric_to_history(metric).await.unwrap();
    }
}
