//! Edge case tests for monitoring crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{MonitoringError, SystemMetrics};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    // MonitoringError edge cases

    #[test]
    fn test_error_empty_messages() {
        let errors = vec![
            MonitoringError::MetricsFailed {
                reason: "".to_string(),
            },
            MonitoringError::ProfilerFailed {
                reason: "".to_string(),
            },
            MonitoringError::TracingFailed {
                reason: "".to_string(),
            },
        ];

        for err in errors {
            let error_str = err.to_string();
            assert!(error_str.contains(": ")); // Should have the prefix and colon
        }
    }

    #[test]
    fn test_error_unicode_messages() {
        let unicode_reasons = vec![
            "错误：内存不足 🚨",
            "Erreur: Mémoire insuffisante 💾",
            "エラー：メモリ不足 ⚡",
            "خطأ: ذاكرة غير كافية 🔥",
            "🚨💥❌ Fatal Error ❌💥🚨",
            "Error\0with\0nulls",
            "Error\nwith\nnewlines\n",
            "Error\twith\ttabs\t",
            "Error\rwith\rcarriage\rreturns",
        ];

        for reason in unicode_reasons {
            let err = MonitoringError::MetricsFailed {
                reason: reason.to_string(),
            };
            let error_str = err.to_string();
            // Just verify it doesn't panic
            assert!(!error_str.is_empty());
        }
    }

    #[test]
    fn test_error_extremely_long_messages() {
        let long_reason = "x".repeat(10000);
        let err = MonitoringError::ProfilerFailed {
            reason: long_reason.clone(),
        };
        assert!(err.to_string().contains(&long_reason));

        // Test with repeated unicode
        let unicode_long = "🚨".repeat(1000);
        let err = MonitoringError::TracingFailed {
            reason: unicode_long.clone(),
        };
        let _ = err.to_string(); // Should not panic
    }

    // SystemMetrics edge cases with floating point

    #[test]
    fn test_system_metrics_float_extremes() {
        let metrics = SystemMetrics {
            timestamp: 0,
            cpu_usage_percent: f64::INFINITY,
            memory_usage_bytes: 0,
            gpu_usage_percent: f64::NEG_INFINITY,
            gpu_memory_usage_bytes: 0,
            network_io_bytes: 0,
            disk_io_bytes: 0,
        };

        // Test serialization with infinity values
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("null")); // Infinity serializes as null in JSON

        // Test NaN
        let mut metrics = SystemMetrics::default();
        metrics.cpu_usage_percent = f64::NAN;
        metrics.gpu_usage_percent = f64::NAN;

        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("null")); // NaN also serializes as null
    }

    #[test]
    fn test_system_metrics_float_precision() {
        let test_values = vec![
            0.0,
            0.1,
            0.01,
            0.001,
            0.0001,
            99.9999,
            50.5555555555,
            f64::EPSILON,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];

        for value in test_values {
            let metrics = SystemMetrics {
                timestamp: 1000,
                cpu_usage_percent: value,
                memory_usage_bytes: 1024,
                gpu_usage_percent: value,
                gpu_memory_usage_bytes: 2048,
                network_io_bytes: 512,
                disk_io_bytes: 256,
            };

            let json = serde_json::to_string(&metrics).unwrap();
            let deserialized: SystemMetrics = serde_json::from_str(&json).unwrap();

            // For normal values, should preserve
            if value.is_finite() {
                assert_eq!(metrics.cpu_usage_percent, deserialized.cpu_usage_percent);
                assert_eq!(metrics.gpu_usage_percent, deserialized.gpu_usage_percent);
            }
        }
    }

    // SystemMetrics timestamp edge cases

    #[test]
    fn test_system_metrics_timestamp_boundaries() {
        // Test UNIX epoch
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

        // Test year 2038 problem timestamp
        let y2038_timestamp = 2147483647u64; // Max 32-bit signed int
        let metrics = SystemMetrics {
            timestamp: y2038_timestamp,
            ..SystemMetrics::default()
        };
        assert_eq!(metrics.timestamp, y2038_timestamp);

        // Test far future
        let far_future = u64::MAX;
        let metrics = SystemMetrics {
            timestamp: far_future,
            ..SystemMetrics::default()
        };
        assert_eq!(metrics.timestamp, far_future);
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify types are Send + Sync
        assert_send::<MonitoringError>();
        assert_sync::<MonitoringError>();
        assert_send::<SystemMetrics>();
        assert_sync::<SystemMetrics>();
    }

    // Error conversion tests

    #[test]
    fn test_error_into_box() {
        let errors: Vec<Box<dyn std::error::Error + Send + Sync>> = vec![
            Box::new(MonitoringError::MetricsFailed {
                reason: "test1".to_string(),
            }),
            Box::new(MonitoringError::ProfilerFailed {
                reason: "test2".to_string(),
            }),
            Box::new(MonitoringError::TracingFailed {
                reason: "test3".to_string(),
            }),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
            // Verify source is None (no nested errors)
            assert!(err.source().is_none());
        }
    }

    // Percentage validation edge cases

    #[test]
    fn test_metrics_percentage_boundaries() {
        let percentages = vec![
            -100.0, -1.0, -0.1, -0.0, 0.0, 0.00001, 50.0, 99.99999, 100.0, 100.1, 200.0, 1000.0,
        ];

        for pct in percentages {
            let metrics = SystemMetrics {
                timestamp: 1000,
                cpu_usage_percent: pct,
                memory_usage_bytes: 1024,
                gpu_usage_percent: pct,
                gpu_memory_usage_bytes: 2048,
                network_io_bytes: 512,
                disk_io_bytes: 256,
            };

            // Should handle any percentage value without panic
            let json = serde_json::to_string(&metrics).unwrap();
            let _: SystemMetrics = serde_json::from_str(&json).unwrap();
        }
    }

    // Memory size edge cases

    #[test]
    fn test_metrics_memory_sizes() {
        let sizes = vec![
            0,                // Zero
            1,                // One byte
            1024,             // 1 KB
            1048576,          // 1 MB
            1073741824,       // 1 GB
            1099511627776,    // 1 TB
            1125899906842624, // 1 PB
            u64::MAX,         // Max possible
        ];

        for size in sizes {
            let metrics = SystemMetrics {
                timestamp: 1000,
                cpu_usage_percent: 50.0,
                memory_usage_bytes: size,
                gpu_usage_percent: 50.0,
                gpu_memory_usage_bytes: size,
                network_io_bytes: size,
                disk_io_bytes: size,
            };

            assert_eq!(metrics.memory_usage_bytes, size);
            assert_eq!(metrics.gpu_memory_usage_bytes, size);
            assert_eq!(metrics.network_io_bytes, size);
            assert_eq!(metrics.disk_io_bytes, size);
        }
    }

    // JSON field order independence

    #[test]
    fn test_metrics_json_field_order() {
        let metrics = SystemMetrics {
            timestamp: 1234567890,
            cpu_usage_percent: 45.5,
            memory_usage_bytes: 8192,
            gpu_usage_percent: 75.5,
            gpu_memory_usage_bytes: 16384,
            network_io_bytes: 4096,
            disk_io_bytes: 2048,
        };

        // Serialize to JSON object
        let json_value = serde_json::to_value(&metrics).unwrap();

        // Create JSON with different field order
        let reordered_json = serde_json::json!({
            "disk_io_bytes": 2048,
            "gpu_memory_usage_bytes": 16384,
            "timestamp": 1234567890,
            "network_io_bytes": 4096,
            "cpu_usage_percent": 45.5,
            "memory_usage_bytes": 8192,
            "gpu_usage_percent": 75.5,
        });

        // Both should deserialize to same metrics
        let metrics1: SystemMetrics = serde_json::from_value(json_value).unwrap();
        let metrics2: SystemMetrics = serde_json::from_value(reordered_json).unwrap();

        assert_eq!(metrics1.timestamp, metrics2.timestamp);
        assert_eq!(metrics1.cpu_usage_percent, metrics2.cpu_usage_percent);
        assert_eq!(metrics1.memory_usage_bytes, metrics2.memory_usage_bytes);
    }

    // Concurrent metrics creation

    #[test]
    fn test_concurrent_metrics_creation() {
        use std::sync::Arc;
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let mut metrics = SystemMetrics::default();
                    metrics.cpu_usage_percent = i as f64 * 10.0;
                    metrics.memory_usage_bytes = i as u64 * 1024;
                    metrics
                })
            })
            .collect();

        let mut all_metrics = Vec::new();
        for handle in handles {
            all_metrics.push(handle.join().unwrap());
        }

        // Verify all metrics were created with correct values
        for (i, metrics) in all_metrics.iter().enumerate() {
            assert_eq!(metrics.cpu_usage_percent, i as f64 * 10.0);
            assert_eq!(metrics.memory_usage_bytes, i as u64 * 1024);
        }
    }

    // Error pattern matching completeness

    #[test]
    fn test_error_exhaustive_matching() {
        fn classify_error(err: MonitoringError) -> &'static str {
            match err {
                MonitoringError::MetricsFailed { .. } => "metrics",
                MonitoringError::ProfilerFailed { .. } => "profiler",
                MonitoringError::TracingFailed { .. } => "tracing",
            }
        }

        assert_eq!(
            classify_error(MonitoringError::MetricsFailed {
                reason: "test".to_string()
            }),
            "metrics"
        );
        assert_eq!(
            classify_error(MonitoringError::ProfilerFailed {
                reason: "test".to_string()
            }),
            "profiler"
        );
        assert_eq!(
            classify_error(MonitoringError::TracingFailed {
                reason: "test".to_string()
            }),
            "tracing"
        );
    }

    // Metrics arithmetic edge cases

    #[test]
    fn test_metrics_arithmetic_operations() {
        let metrics1 = SystemMetrics {
            timestamp: 1000,
            cpu_usage_percent: 30.0,
            memory_usage_bytes: 1000,
            gpu_usage_percent: 40.0,
            gpu_memory_usage_bytes: 2000,
            network_io_bytes: 500,
            disk_io_bytes: 600,
        };

        let metrics2 = SystemMetrics {
            timestamp: 2000,
            cpu_usage_percent: 20.0,
            memory_usage_bytes: 1500,
            gpu_usage_percent: 35.0,
            gpu_memory_usage_bytes: 2500,
            network_io_bytes: 700,
            disk_io_bytes: 800,
        };

        // Calculate deltas (common monitoring operation)
        let cpu_delta = metrics2.cpu_usage_percent - metrics1.cpu_usage_percent;
        let memory_delta = metrics2
            .memory_usage_bytes
            .saturating_sub(metrics1.memory_usage_bytes);
        let time_delta = metrics2.timestamp - metrics1.timestamp;

        assert_eq!(cpu_delta, -10.0);
        assert_eq!(memory_delta, 500);
        assert_eq!(time_delta, 1000);
    }

    // Special timestamp values

    #[test]
    fn test_special_timestamps() {
        // Test with specific known timestamps
        let timestamps = vec![
            0,          // Unix epoch
            946684800,  // Y2K (2000-01-01)
            1234567890, // Notable timestamp
            2147483647, // 32-bit max (Y2038)
            4294967295, // 32-bit unsigned max
            9999999999, // Far future
        ];

        for ts in timestamps {
            let metrics = SystemMetrics {
                timestamp: ts,
                ..SystemMetrics::default()
            };

            let json = serde_json::to_string(&metrics).unwrap();
            let deserialized: SystemMetrics = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.timestamp, ts);
        }
    }

    // Zero-value edge cases

    #[test]
    fn test_all_zero_metrics() {
        let zero_metrics = SystemMetrics {
            timestamp: 0,
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            gpu_usage_percent: 0.0,
            gpu_memory_usage_bytes: 0,
            network_io_bytes: 0,
            disk_io_bytes: 0,
        };

        // Verify serialization of all-zero metrics
        let json = serde_json::to_string(&zero_metrics).unwrap();
        assert!(json.contains("\"timestamp\":0"));
        assert!(json.contains("\"cpu_usage_percent\":0.0"));
        assert!(json.contains("\"memory_usage_bytes\":0"));

        // Verify it's different from default (which has non-zero timestamp)
        let default_metrics = SystemMetrics::default();
        assert_ne!(zero_metrics.timestamp, default_metrics.timestamp);
    }
}
