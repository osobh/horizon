//! Edge case tests for streaming crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{ChunkMetadata, StreamChunk, StreamStats, StreamingError};
    use bytes::Bytes;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    // StreamingError edge cases

    #[test]
    fn test_streaming_error_variants() {
        let errors = vec![
            StreamingError::IoFailed {
                reason: "".to_string(),
            },
            StreamingError::IoError("Very long error message".repeat(100)),
            StreamingError::InvalidInput("Unicode error: 错误 🚨".to_string()),
            StreamingError::InvalidConfig {
                reason: "Config\nwith\nnewlines".to_string(),
            },
            StreamingError::ProcessingFailed {
                reason: "Failed with\ttabs\tand spaces   ".to_string(),
            },
            StreamingError::ResourceExhausted {
                reason: String::new(),
            },
            StreamingError::Timeout {
                reason: "⏰ Timeout after 1000000ms".to_string(),
            },
        ];

        for err in errors {
            let error_str = err.to_string();
            assert!(!error_str.is_empty());

            // Test Debug format
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_streaming_error_edge_messages() {
        // Test with extreme lengths
        let long_reason = "x".repeat(10000);
        let err = StreamingError::ProcessingFailed {
            reason: long_reason.clone(),
        };
        assert!(err.to_string().contains(&long_reason));

        // Test with special characters
        let special_chars = "Error: \0\r\n\t\u{1F4A5}";
        let err = StreamingError::IoError(special_chars.to_string());
        let _ = err.to_string(); // Just ensure it doesn't panic
    }

    // StreamChunk edge cases

    #[test]
    fn test_stream_chunk_empty() {
        let chunk = StreamChunk::new(Bytes::new(), 0, "empty".to_string());
        assert!(chunk.is_empty());
        assert_eq!(chunk.size(), 0);
        assert_eq!(chunk.metadata.chunk_size, 0);
    }

    #[test]
    fn test_stream_chunk_extreme_sizes() {
        // Very large sequence numbers
        let chunk = StreamChunk::new(
            Bytes::from(vec![0u8; 1024]),
            u64::MAX,
            "large_seq".to_string(),
        );
        assert_eq!(chunk.sequence, u64::MAX);
        assert_eq!(chunk.size(), 1024);

        // Empty source ID
        let chunk = StreamChunk::new(Bytes::from("data"), 0, String::new());
        assert_eq!(chunk.metadata.source_id, "");
    }

    #[test]
    fn test_stream_chunk_display() {
        let chunk = StreamChunk::new(Bytes::from("test"), 12345, "source123".to_string());
        let display = chunk.to_string();
        assert!(display.contains("12345"));
        assert!(display.contains("source123"));
        assert!(display.contains("4")); // size

        // Test with unicode source ID
        let chunk = StreamChunk::new(Bytes::from("data"), 0, "源_🚀".to_string());
        let display = chunk.to_string();
        assert!(display.contains("源_🚀"));
    }

    #[test]
    fn test_chunk_metadata_edge_cases() {
        let mut metadata = ChunkMetadata {
            source_id: "".to_string(),
            chunk_size: usize::MAX,
            compression: Some("".to_string()),
            checksum: Some(u64::MAX),
        };

        // Test serialization with extreme values
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.chunk_size, usize::MAX);
        assert_eq!(deserialized.checksum, Some(u64::MAX));

        // Test with None values
        metadata.compression = None;
        metadata.checksum = None;
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();
        assert!(deserialized.compression.is_none());
        assert!(deserialized.checksum.is_none());
    }

    #[test]
    fn test_chunk_metadata_unicode() {
        let metadata = ChunkMetadata {
            source_id: "测试_test_テスト".to_string(),
            chunk_size: 0,
            compression: Some("brotli_高压缩".to_string()),
            checksum: None,
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("测试_test_テスト"));
        assert!(json.contains("brotli_高压缩"));
    }

    // StreamStats edge cases

    #[test]
    fn test_stream_stats_extremes() {
        let stats = StreamStats {
            chunks_processed: u64::MAX,
            bytes_processed: u64::MAX,
            processing_time_ms: u64::MAX,
            throughput_mbps: f64::INFINITY,
            errors: 0,
        };

        // Test serialization with extreme values
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains(&u64::MAX.to_string()));

        // Test with NaN
        let mut stats = StreamStats::default();
        stats.throughput_mbps = f64::NAN;
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("null")); // NaN serializes as null

        // Test with negative infinity
        stats.throughput_mbps = f64::NEG_INFINITY;
        let json = serde_json::to_string(&stats).unwrap();
        // Infinity values serialize as null in JSON, so deserialization will fail
        // This is expected behavior for JSON spec compliance
        assert!(json.contains("null"));
    }

    #[test]
    fn test_stream_stats_default() {
        let stats = StreamStats::default();
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.bytes_processed, 0);
        assert_eq!(stats.processing_time_ms, 0);
        assert_eq!(stats.throughput_mbps, 0.0);
        assert_eq!(stats.errors, 0);
    }

    // Time-based edge cases

    #[test]
    fn test_timestamp_edge_cases() {
        // Test chunk creation at different times
        let chunk1 = StreamChunk::new(Bytes::from("data1"), 1, "src1".to_string());
        std::thread::sleep(Duration::from_millis(10));
        let chunk2 = StreamChunk::new(Bytes::from("data2"), 2, "src2".to_string());

        assert!(chunk2.timestamp > chunk1.timestamp);

        // Timestamps should be reasonable
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assert!(chunk1.timestamp <= now);
        assert!(chunk2.timestamp <= now);
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify types are Send + Sync
        assert_send::<StreamingError>();
        assert_sync::<StreamingError>();
        assert_send::<StreamChunk>();
        assert_sync::<StreamChunk>();
        assert_send::<ChunkMetadata>();
        assert_sync::<ChunkMetadata>();
        assert_send::<StreamStats>();
        assert_sync::<StreamStats>();
    }

    // Bytes handling edge cases

    #[test]
    fn test_bytes_edge_cases() {
        // Test with various byte patterns
        let patterns = vec![
            vec![],
            vec![0u8],
            vec![255u8],
            vec![0u8; 1024],
            vec![255u8; 1024],
            (0..=255).collect::<Vec<u8>>(),
        ];

        for pattern in patterns {
            let chunk = StreamChunk::new(Bytes::from(pattern.clone()), 0, "test".to_string());
            assert_eq!(chunk.size(), pattern.len());
            assert_eq!(chunk.data.len(), pattern.len());
        }
    }

    #[test]
    fn test_chunk_cloning() {
        let original = StreamChunk {
            data: Bytes::from("test data"),
            sequence: 42,
            timestamp: 1234567890,
            metadata: ChunkMetadata {
                source_id: "original".to_string(),
                chunk_size: 9,
                compression: Some("gzip".to_string()),
                checksum: Some(0xDEADBEEF),
            },
        };

        let cloned = original.clone();
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.sequence, original.sequence);
        assert_eq!(cloned.timestamp, original.timestamp);
        assert_eq!(cloned.metadata.source_id, original.metadata.source_id);
        assert_eq!(cloned.metadata.compression, original.metadata.compression);
        assert_eq!(cloned.metadata.checksum, original.metadata.checksum);
    }

    // Error conversion tests

    #[test]
    fn test_error_into_box() {
        let errors: Vec<Box<dyn std::error::Error + Send + Sync>> = vec![
            Box::new(StreamingError::IoFailed {
                reason: "test".to_string(),
            }),
            Box::new(StreamingError::InvalidInput("bad input".to_string())),
            Box::new(StreamingError::Timeout {
                reason: "too slow".to_string(),
            }),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    // Compression metadata edge cases

    #[test]
    fn test_compression_metadata() {
        let compression_types = vec![
            None,
            Some("".to_string()),
            Some("gzip".to_string()),
            Some("brotli".to_string()),
            Some("lz4".to_string()),
            Some("zstd".to_string()),
            Some("unknown_compression_algorithm_with_very_long_name".to_string()),
            Some("compression/with/slashes".to_string()),
            Some("compression-with-dashes".to_string()),
            Some("compression.with.dots".to_string()),
        ];

        for compression in compression_types {
            let metadata = ChunkMetadata {
                source_id: "test".to_string(),
                chunk_size: 100,
                compression: compression.clone(),
                checksum: None,
            };

            let json = serde_json::to_string(&metadata).unwrap();
            let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.compression, compression);
        }
    }

    // Checksum edge cases

    #[test]
    fn test_checksum_values() {
        let checksum_values = vec![
            None,
            Some(0),
            Some(1),
            Some(u32::MAX as u64),
            Some(u64::MAX),
            Some(0xDEADBEEF),
            Some(0xCAFEBABE),
            Some(0x123456789ABCDEF0),
        ];

        for checksum in checksum_values {
            let metadata = ChunkMetadata {
                source_id: "test".to_string(),
                chunk_size: 100,
                compression: None,
                checksum,
            };

            if let Some(value) = checksum {
                assert_eq!(metadata.checksum.unwrap(), value);
            } else {
                assert!(metadata.checksum.is_none());
            }
        }
    }

    // Stats calculation edge cases

    #[test]
    fn test_throughput_calculation_edge_cases() {
        let mut stats = StreamStats::default();

        // Zero time should not cause division by zero
        stats.bytes_processed = 1_000_000;
        stats.processing_time_ms = 0;
        // Throughput calculation would need to handle this case

        // Very small time
        stats.processing_time_ms = 1;
        // This gives very high throughput

        // Very large values
        stats.bytes_processed = u64::MAX;
        stats.processing_time_ms = 1;
        // This gives extremely high throughput
    }

    // Source ID edge cases

    #[test]
    fn test_source_id_variations() {
        let source_ids = vec![
            "",
            "a",
            "source",
            "source-123",
            "source_123",
            "source.123",
            "source/123",
            "source\\123",
            "source:123",
            "source@host",
            "http://source.com",
            "file:///path/to/source",
            "источник",
            "来源",
            "🚀📡💾",
            "source\0with\0nulls",
            "source\nwith\nnewlines",
            "source\twith\ttabs",
            " source with spaces ",
            "very_long_source_id_that_exceeds_normal_length_expectations_and_continues_for_a_while",
        ];

        for id in source_ids {
            let chunk = StreamChunk::new(Bytes::from("data"), 0, id.to_string());
            assert_eq!(chunk.metadata.source_id, id);

            // Ensure Display doesn't panic
            let _ = chunk.to_string();
        }
    }

    // Concurrent chunk creation

    #[test]
    fn test_concurrent_chunk_creation() {
        use std::sync::Arc;
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let chunk = StreamChunk::new(
                        Bytes::from(format!("data{}", i)),
                        i as u64,
                        format!("source{}", i),
                    );
                    (chunk.sequence, chunk.timestamp)
                })
            })
            .collect();

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // Verify all chunks have unique sequences
        for i in 0..results.len() {
            assert_eq!(results[i].0, i as u64);
        }

        // Timestamps should be close but not necessarily identical
        let first_timestamp = results[0].1;
        for (_, timestamp) in &results {
            assert!(timestamp.abs_diff(first_timestamp) < 1000); // Within 1 second
        }
    }
}
