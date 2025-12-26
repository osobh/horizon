//! Performance regression detection error types

use thiserror::Error;

/// Performance regression detection error types
#[derive(Debug, Error)]
pub enum PerformanceRegressionError {
    /// Metrics collection failed
    #[error("Metrics collection failed: {source_name} - {details}")]
    MetricsCollectionFailed {
        source_name: String,
        details: String,
    },

    /// Anomaly detection failed
    #[error("Anomaly detection failed: {algorithm} - {reason}")]
    AnomalyDetectionFailed { algorithm: String, reason: String },

    /// Baseline establishment failed
    #[error("Baseline establishment failed: {metric} - {reason}")]
    BaselineEstablishmentFailed { metric: String, reason: String },

    /// Performance regression detected
    #[error("Performance regression detected: {metric} degraded by {degradation_percent}% (threshold: {threshold_percent}%)")]
    RegressionDetected {
        metric: String,
        degradation_percent: f64,
        threshold_percent: f64,
    },

    /// Insufficient data for analysis
    #[error("Insufficient data for {analysis_type}: need {required} samples, have {available}")]
    InsufficientData {
        analysis_type: String,
        required: usize,
        available: usize,
    },

    /// Bottleneck detection failed
    #[error("Bottleneck detection failed: {component} - {reason}")]
    BottleneckDetectionFailed { component: String, reason: String },

    /// Test orchestration failed
    #[error("Test orchestration failed: {test_suite} - {reason}")]
    TestOrchestrationFailed { test_suite: String, reason: String },

    /// Alert delivery failed
    #[error("Alert delivery failed: {channel} - {details}")]
    AlertDeliveryFailed { channel: String, details: String },

    /// Alert condition not found
    #[error("Alert condition not found: {condition_id}")]
    AlertConditionNotFound { condition_id: String },

    /// Alert not found
    #[error("Alert not found: {alert_id}")]
    AlertNotFound { alert_id: String },

    /// Report generation failed
    #[error("Report generation failed: {report_type} - {reason}")]
    ReportGenerationFailed { report_type: String, reason: String },

    /// Model training failed
    #[error("ML model training failed: {model_type} - {reason}")]
    ModelTrainingFailed { model_type: String, reason: String },

    /// Data validation failed
    #[error("Data validation failed: {validation_type} - {details}")]
    DataValidationFailed {
        validation_type: String,
        details: String,
    },

    /// Configuration error
    #[error("Configuration error: {parameter} - {message}")]
    ConfigurationError { parameter: String, message: String },

    /// Storage error
    #[error("Storage error: {operation} - {details}")]
    StorageError { operation: String, details: String },

    /// Network error
    #[error("Network error: {endpoint} - {details}")]
    NetworkError { endpoint: String, details: String },

    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    /// Statistical computation error
    #[error("Statistical computation error: {computation} - {reason}")]
    StatisticalError { computation: String, reason: String },

    /// Test not found
    #[error("Test not found: {test_id}")]
    TestNotFound { test_id: String },

    /// No test results available
    #[error("No test results available for report generation")]
    NoTestResults,

    /// Tokio join error
    #[error("Task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),

    /// Semaphore acquire error
    #[error("Semaphore acquire error: {0}")]
    AcquireError(#[from] tokio::sync::AcquireError),
}

/// Performance regression detection result type
pub type PerformanceRegressionResult<T> = Result<T, PerformanceRegressionError>;
