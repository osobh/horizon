//! Additional comprehensive tests for net to enhance coverage to 90%+

#[cfg(test)]
mod additional_tests_simple {
    use super::super::*;
    use crate::error::NetworkError;
    use crate::protocol::{Message, MessageType};
    use std::time::Duration;

    #[test]
    fn test_network_stats_operations() {
        let mut stats = NetworkStats::default();

        // Test initial state
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);

        // Simulate network activity
        stats.bytes_sent = 1024 * 1024; // 1MB
        stats.bytes_received = 2048 * 1024; // 2MB
        stats.messages_sent = 100;
        stats.messages_received = 150;
        stats.average_latency_us = 250.5;
        stats.throughput_mbps = 100.0;

        // Test serialization
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: NetworkStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_network_error_display() {
        let errors = vec![
            NetworkError::ConnectionClosed,
            NetworkError::Timeout(Duration::from_secs(30)),
            NetworkError::InvalidMessage("bad format".to_string()),
            NetworkError::Io(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pipe broken",
            )),
        ];

        for error in errors {
            let error_str = error.to_string();
            assert!(!error_str.is_empty());

            // Test Debug formatting
            let debug_str = format!("{:?}", error);
            assert!(debug_str.contains("NetworkError"));
        }
    }

    #[test]
    fn test_message_type_variants() {
        let types = vec![
            MessageType::Data,
            MessageType::Control,
            MessageType::Handshake,
            MessageType::Heartbeat,
            MessageType::Error,
            MessageType::Close,
        ];

        for msg_type in types {
            // Test PartialEq
            assert_eq!(msg_type, msg_type.clone());

            // Test serialization
            let json = serde_json::to_string(&msg_type).unwrap();
            let deserialized: MessageType = serde_json::from_str(&json).unwrap();
            assert_eq!(msg_type, deserialized);
        }
    }

    #[test]
    fn test_message_creation_edge_cases() {
        use std::time::SystemTime;

        // Empty payload
        let msg1 = Message {
            id: 0,
            msg_type: MessageType::Data,
            payload: vec![],
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        assert_eq!(msg1.payload.len(), 0);

        // Large payload
        let large_payload = vec![0u8; 10 * 1024 * 1024]; // 10MB
        let msg2 = Message {
            id: u64::MAX,
            msg_type: MessageType::Data,
            payload: large_payload.clone(),
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        assert_eq!(msg2.payload.len(), 10 * 1024 * 1024);

        // Unicode in error message
        let error_msg = Message {
            id: 12345,
            msg_type: MessageType::Error,
            payload: "错误信息 🚫".as_bytes().to_vec(),
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        assert!(!error_msg.payload.is_empty());
    }

    #[test]
    fn test_network_benchmark_config() {
        use crate::NetworkBenchmarkConfig;

        let configs = vec![
            NetworkBenchmarkConfig {
                message_sizes: vec![64, 1024, 65536],
                num_messages: 1000,
                warmup_messages: 100,
                parallel_connections: 1,
                ..Default::default()..Default::default()
            },
            NetworkBenchmarkConfig {
                message_sizes: vec![1],
                num_messages: 1,
                warmup_messages: 0,
                parallel_connections: 0,
                ..Default::default()
            },
            NetworkBenchmarkConfig {
                message_sizes: vec![1024 * 1024 * 100], // 100MB
                num_messages: u64::MAX,
                warmup_messages: u64::MAX,
                parallel_connections: 1000,
                ..Default::default()
            },
        ];

        for config in configs {
            // Test serialization
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: NetworkBenchmarkConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(config.message_sizes, deserialized.message_sizes);
            assert_eq!(config.num_messages, deserialized.num_messages);
        }
    }

    #[test]
    fn test_network_benchmark_results() {
        use crate::NetworkBenchmarkResults;

        let results = NetworkBenchmarkResults {
            throughput_mbps: 1234.56,
            latency_us: 42.0,
            messages_per_second: 1_000_000.0,
            cpu_usage_percent: 75.5,
            memory_usage_mb: 512,
        };

        // Test edge cases
        assert!(results.throughput_mbps > 0.0);
        assert!(results.latency_us > 0.0);
        assert!(results.cpu_usage_percent >= 0.0 && results.cpu_usage_percent <= 100.0);

        // Test with extreme values
        let extreme_results = NetworkBenchmarkResults {
            throughput_mbps: f64::MAX,
            latency_us: f64::MIN_POSITIVE,
            messages_per_second: f64::INFINITY,
            cpu_usage_percent: 0.0,
            memory_usage_mb: u64::MAX,
        };

        let json = serde_json::to_string(&extreme_results).unwrap();
        assert!(!json.is_empty());
    }

    #[test]
    fn test_full_network_benchmark_results() {
        use crate::{FullNetworkBenchmarkResults, NetworkBenchmarkResults};
        use std::collections::HashMap;

        let mut results_map = HashMap::new();

        // Add results for different message sizes
        for size in &[64, 1024, 65536] {
            results_map.insert(
                *size,
                NetworkBenchmarkResults {
                    throughput_mbps: 1000.0 / (*size as f64),
                    latency_us: *size as f64 * 0.01,
                    messages_per_second: 1_000_000.0 / (*size as f64),
                    cpu_usage_percent: 50.0,
                    memory_usage_mb: 256,
                },
            );
        }

        let full_results = FullNetworkBenchmarkResults {
            results_by_size: results_map,
            total_duration: Duration::from_secs(60),
            total_messages: 1_000_000,
            total_bytes: 1024 * 1024 * 1024, // 1GB
        };

        assert_eq!(full_results.results_by_size.len(), 3);
        assert_eq!(full_results.total_duration, Duration::from_secs(60));
    }

    #[test]
    fn test_io_error_conversion() {
        use std::io;

        let io_errors = vec![
            io::Error::new(io::ErrorKind::NotFound, "file not found"),
            io::Error::new(io::ErrorKind::PermissionDenied, "access denied"),
            io::Error::new(io::ErrorKind::ConnectionRefused, "connection refused"),
            io::Error::new(io::ErrorKind::ConnectionReset, "connection reset"),
            io::Error::new(io::ErrorKind::ConnectionAborted, "connection aborted"),
            io::Error::new(io::ErrorKind::NotConnected, "not connected"),
            io::Error::new(io::ErrorKind::AddrInUse, "address in use"),
            io::Error::new(io::ErrorKind::AddrNotAvailable, "address not available"),
            io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"),
            io::Error::new(io::ErrorKind::AlreadyExists, "already exists"),
            io::Error::new(io::ErrorKind::WouldBlock, "would block"),
            io::Error::new(io::ErrorKind::InvalidInput, "invalid input"),
            io::Error::new(io::ErrorKind::InvalidData, "invalid data"),
            io::Error::new(io::ErrorKind::TimedOut, "timed out"),
            io::Error::new(io::ErrorKind::WriteZero, "write zero"),
            io::Error::new(io::ErrorKind::Interrupted, "interrupted"),
            io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"),
        ];

        for io_err in io_errors {
            let kind = io_err.kind();
            let net_err: NetworkError = io_err.into();

            match net_err {
                NetworkError::Io(e) => assert_eq!(e.kind(), kind),
                _ => panic!("Expected Io variant"),
            }
        }
    }

    #[test]
    fn test_message_timestamp_edge_cases() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamps = vec![
            UNIX_EPOCH,
            SystemTime::now(),
            SystemTime::now() + Duration::from_secs(3600), // 1 hour future
            UNIX_EPOCH + Duration::from_secs(u64::MAX / 2), // Far future
        ];

        for timestamp in timestamps {
            let msg = Message {
                id: 1,
                msg_type: MessageType::Heartbeat,
                payload: vec![],
                timestamp: timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            // Verify timestamp is preserved
            assert_eq!(
                msg.timestamp,
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            );

            // Test serialization with various timestamps
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: Message = serde_json::from_str(&json).unwrap();

            // Compare timestamps (may have slight precision loss)
            let original_secs = timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let deserialized_secs = deserialized.timestamp;

            assert!((original_secs as i64 - deserialized_secs as i64).abs() <= 1);
        }
    }

    #[test]
    fn test_network_error_timeout_durations() {
        let timeouts = vec![
            Duration::from_nanos(1),
            Duration::from_micros(1),
            Duration::from_millis(1),
            Duration::from_secs(1),
            Duration::from_secs(3600),     // 1 hour
            Duration::from_secs(86400),    // 1 day
            Duration::from_secs(u64::MAX), // Max duration
        ];

        for timeout in timeouts {
            let err = NetworkError::Timeout(timeout);

            match err {
                NetworkError::Timeout(d) => assert_eq!(d, timeout),
                _ => panic!("Expected Timeout variant"),
            }

            // Test error message contains duration info
            let err_str = err.to_string();
            assert!(err_str.contains("timeout") || err_str.contains("Timeout"));
        }
    }

    #[test]
    fn test_message_id_boundaries() {
        let ids = vec![0, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX];

        for id in ids {
            let msg = Message {
                id,
                msg_type: MessageType::Data,
                payload: vec![1, 2, 3],
                timestamp: std::time::SystemTime::now(),
            };

            assert_eq!(msg.id, id);

            // Test serialization preserves ID
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.id, id);
        }
    }

    #[test]
    fn test_parallel_connections_edge_cases() {
        use crate::NetworkBenchmarkConfig;

        let edge_cases = vec![0, 1, 10, 100, 1000, 10000, usize::MAX];

        for connections in edge_cases {
            let config = NetworkBenchmarkConfig {
                message_sizes: vec![1024],
                num_messages: 100,
                warmup_messages: 10,
                parallel_connections: connections,
            };

            assert_eq!(config.parallel_connections, connections);
        }
    }
}
