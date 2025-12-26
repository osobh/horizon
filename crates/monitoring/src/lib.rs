//! Monitoring and observability infrastructure for GPU container systems
//!
//! This crate provides comprehensive monitoring, metrics collection, and observability
//! tools for distributed GPU container environments.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod metrics;
pub mod profiler;
pub mod tracer;

/// Monitoring errors
#[derive(Error, Debug)]
pub enum MonitoringError {
    #[error("Metrics collection failed: {reason}")]
    MetricsFailed { reason: String },

    #[error("Profiler initialization failed: {reason}")]
    ProfilerFailed { reason: String },

    #[error("Tracing error: {reason}")]
    TracingFailed { reason: String },
}

/// System metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub gpu_usage_percent: f64,
    pub gpu_memory_usage_bytes: u64,
    pub network_io_bytes: u64,
    pub disk_io_bytes: u64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            gpu_usage_percent: 0.0,
            gpu_memory_usage_bytes: 0,
            network_io_bytes: 0,
            disk_io_bytes: 0,
        }
    }
}

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert!(metrics.timestamp > 0);
        assert_eq!(metrics.cpu_usage_percent, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
    }

    #[test]
    fn test_system_metrics_serialization() {
        let metrics = SystemMetrics {
            timestamp: 1234567890,
            cpu_usage_percent: 45.5,
            memory_usage_bytes: 8_589_934_592, // 8GB
            gpu_usage_percent: 82.3,
            gpu_memory_usage_bytes: 12_884_901_888, // 12GB
            network_io_bytes: 1_048_576,            // 1MB
            disk_io_bytes: 10_485_760,              // 10MB
        };

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: SystemMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(metrics.timestamp, deserialized.timestamp);
        assert_eq!(metrics.cpu_usage_percent, deserialized.cpu_usage_percent);
        assert_eq!(metrics.memory_usage_bytes, deserialized.memory_usage_bytes);
        assert_eq!(metrics.gpu_usage_percent, deserialized.gpu_usage_percent);
        assert_eq!(
            metrics.gpu_memory_usage_bytes,
            deserialized.gpu_memory_usage_bytes
        );
        assert_eq!(metrics.network_io_bytes, deserialized.network_io_bytes);
        assert_eq!(metrics.disk_io_bytes, deserialized.disk_io_bytes);
    }

    #[test]
    fn test_monitoring_error_display() {
        let error = MonitoringError::MetricsFailed {
            reason: "Connection timeout".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Metrics collection failed: Connection timeout"
        );

        let error = MonitoringError::ProfilerFailed {
            reason: "Nsight Compute not found".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Profiler initialization failed: Nsight Compute not found"
        );

        let error = MonitoringError::TracingFailed {
            reason: "Span not found".to_string(),
        };
        assert_eq!(error.to_string(), "Tracing error: Span not found");
    }

    #[test]
    fn test_system_metrics_with_extreme_values() {
        let metrics = SystemMetrics {
            timestamp: u64::MAX,
            cpu_usage_percent: 100.0,
            memory_usage_bytes: u64::MAX,
            gpu_usage_percent: 100.0,
            gpu_memory_usage_bytes: u64::MAX,
            network_io_bytes: u64::MAX,
            disk_io_bytes: u64::MAX,
        };

        // Should not panic
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: SystemMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(metrics.timestamp, deserialized.timestamp);
        assert_eq!(metrics.cpu_usage_percent, 100.0);
        assert_eq!(metrics.memory_usage_bytes, u64::MAX);
    }

    #[test]
    fn test_system_metrics_with_minimum_values() {
        let metrics = SystemMetrics {
            timestamp: 0,
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            gpu_usage_percent: 0.0,
            gpu_memory_usage_bytes: 0,
            network_io_bytes: 0,
            disk_io_bytes: 0,
        };

        assert_eq!(metrics.timestamp, 0);
        assert_eq!(metrics.cpu_usage_percent, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
    }

    #[test]
    fn test_system_metrics_typical_values() {
        let metrics = SystemMetrics {
            timestamp: 1700000000,
            cpu_usage_percent: 65.5,
            memory_usage_bytes: 16_777_216_000, // ~16GB
            gpu_usage_percent: 78.2,
            gpu_memory_usage_bytes: 24_696_061_952, // ~24GB
            network_io_bytes: 125_829_120,          // ~120MB
            disk_io_bytes: 536_870_912,             // 512MB
        };

        assert!(metrics.cpu_usage_percent > 0.0 && metrics.cpu_usage_percent <= 100.0);
        assert!(metrics.gpu_usage_percent > 0.0 && metrics.gpu_usage_percent <= 100.0);
        assert!(metrics.memory_usage_bytes > 0);
        assert!(metrics.gpu_memory_usage_bytes > 0);
    }

    #[test]
    fn test_system_metrics_timestamp_generation() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metrics = SystemMetrics::default();

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        assert!(metrics.timestamp >= before);
        assert!(metrics.timestamp <= after);
    }

    #[test]
    fn test_monitoring_error_variants() {
        use std::error::Error;

        let errors = vec![
            MonitoringError::MetricsFailed {
                reason: "test1".to_string(),
            },
            MonitoringError::ProfilerFailed {
                reason: "test2".to_string(),
            },
            MonitoringError::TracingFailed {
                reason: "test3".to_string(),
            },
        ];

        for error in errors {
            // Test that Error trait is implemented
            let _ = error.source();
            let _ = error.to_string();
        }
    }

    #[test]
    fn test_system_metrics_json_field_names() {
        let metrics = SystemMetrics::default();
        let json = serde_json::to_value(&metrics).unwrap();

        assert!(json.get("timestamp").is_some());
        assert!(json.get("cpu_usage_percent").is_some());
        assert!(json.get("memory_usage_bytes").is_some());
        assert!(json.get("gpu_usage_percent").is_some());
        assert!(json.get("gpu_memory_usage_bytes").is_some());
        assert!(json.get("network_io_bytes").is_some());
        assert!(json.get("disk_io_bytes").is_some());
    }

    #[test]
    fn test_system_metrics_memory_efficiency() {
        use std::mem::size_of;

        // Ensure the struct is reasonably sized
        assert!(size_of::<SystemMetrics>() <= 128); // Should be compact
    }

    #[test]
    fn test_system_metrics_clone() {
        let original = SystemMetrics {
            timestamp: 1234567890,
            cpu_usage_percent: 50.0,
            memory_usage_bytes: 1024,
            gpu_usage_percent: 75.0,
            gpu_memory_usage_bytes: 2048,
            network_io_bytes: 512,
            disk_io_bytes: 256,
        };

        let cloned = original.clone();

        assert_eq!(original.timestamp, cloned.timestamp);
        assert_eq!(original.cpu_usage_percent, cloned.cpu_usage_percent);
        assert_eq!(original.memory_usage_bytes, cloned.memory_usage_bytes);
        assert_eq!(original.gpu_usage_percent, cloned.gpu_usage_percent);
        assert_eq!(
            original.gpu_memory_usage_bytes,
            cloned.gpu_memory_usage_bytes
        );
        assert_eq!(original.network_io_bytes, cloned.network_io_bytes);
        assert_eq!(original.disk_io_bytes, cloned.disk_io_bytes);
    }

    #[test]
    fn test_system_metrics_debug_output() {
        let metrics = SystemMetrics {
            timestamp: 1234567890,
            cpu_usage_percent: 45.5,
            memory_usage_bytes: 1024,
            gpu_usage_percent: 75.5,
            gpu_memory_usage_bytes: 2048,
            network_io_bytes: 512,
            disk_io_bytes: 256,
        };

        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("SystemMetrics"));
        assert!(debug_str.contains("1234567890"));
        assert!(debug_str.contains("45.5"));
        assert!(debug_str.contains("75.5"));
    }

    #[test]
    fn test_monitoring_error_from_string() {
        let reasons = vec![
            "Network timeout",
            "Invalid configuration",
            "GPU not available",
            "Permission denied",
            "Resource exhausted",
        ];

        for reason in reasons {
            let metrics_error = MonitoringError::MetricsFailed {
                reason: reason.to_string(),
            };
            assert!(metrics_error.to_string().contains(reason));

            let profiler_error = MonitoringError::ProfilerFailed {
                reason: reason.to_string(),
            };
            assert!(profiler_error.to_string().contains(reason));

            let tracing_error = MonitoringError::TracingFailed {
                reason: reason.to_string(),
            };
            assert!(tracing_error.to_string().contains(reason));
        }
    }

    #[test]
    fn test_system_metrics_percentage_validation() {
        // Test that percentages can be properly represented
        let metrics = SystemMetrics {
            timestamp: 1000,
            cpu_usage_percent: 99.99,
            memory_usage_bytes: 1024,
            gpu_usage_percent: 0.01,
            gpu_memory_usage_bytes: 2048,
            network_io_bytes: 512,
            disk_io_bytes: 256,
        };

        assert!(metrics.cpu_usage_percent < 100.0);
        assert!(metrics.gpu_usage_percent > 0.0);
    }

    #[test]
    fn test_monitoring_error_match_patterns() {
        let error = MonitoringError::MetricsFailed {
            reason: "test".to_string(),
        };

        match error {
            MonitoringError::MetricsFailed { reason } => {
                assert_eq!(reason, "test");
            }
            _ => panic!("Expected MetricsFailed variant"),
        }
    }

    #[test]
    fn test_system_metrics_builder_pattern() {
        // Test creating metrics with specific values
        let mut metrics = SystemMetrics::default();

        metrics.cpu_usage_percent = 50.0;
        metrics.memory_usage_bytes = 1_073_741_824; // 1GB
        metrics.gpu_usage_percent = 80.0;
        metrics.gpu_memory_usage_bytes = 8_589_934_592; // 8GB
        metrics.network_io_bytes = 1_048_576; // 1MB
        metrics.disk_io_bytes = 10_485_760; // 10MB

        assert_eq!(metrics.cpu_usage_percent, 50.0);
        assert_eq!(metrics.memory_usage_bytes, 1_073_741_824);
        assert_eq!(metrics.gpu_usage_percent, 80.0);
        assert_eq!(metrics.gpu_memory_usage_bytes, 8_589_934_592);
        assert_eq!(metrics.network_io_bytes, 1_048_576);
        assert_eq!(metrics.disk_io_bytes, 10_485_760);
    }

    #[test]
    fn test_monitoring_module_exports() {
        // Verify modules are properly exported
        use crate::metrics::*;
        use crate::profiler::*;
        use crate::tracer::*;

        // This test ensures the modules are accessible
        let _: Option<PrometheusExporter> = None;
        let _: Option<NsightComputeProfiler> = None;
        let _: Option<DistributedTracer> = None;
    }
}
