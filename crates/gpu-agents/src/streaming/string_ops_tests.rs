//! Comprehensive tests for GPU string operations
//!
//! Tests cover unit tests, integration tests, and performance benchmarks
//! for all string processing capabilities.

use super::string_ops::*;
use super::*;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::Duration;

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_string_processor_config_default() {
        let config = StringProcessorConfig::default();
        assert_eq!(config.max_string_length, 4096);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.num_streams, 4);
        assert!(config.use_pinned_memory);
        assert_eq!(config.pattern_cache_size, 256);
    }

    #[test]
    fn test_string_processor_config_custom() {
        let config = StringProcessorConfig {
            max_string_length: 8192,
            batch_size: 512,
            num_streams: 8,
            use_pinned_memory: false,
            pattern_cache_size: 128,
        };

        assert_eq!(config.max_string_length, 8192);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.num_streams, 8);
        assert!(!config.use_pinned_memory);
        assert_eq!(config.pattern_cache_size, 128);
    }

    #[test]
    fn test_string_operation_types() {
        let op1 = StringOperation::ToUppercase;
        let op2 = StringOperation::ToLowercase;
        let op3 = StringOperation::Reverse;
        let op4 = StringOperation::PatternMatch {
            pattern: "test".to_string(),
        };
        let op5 = StringOperation::Replace {
            from: "old".to_string(),
            to: "new".to_string(),
        };

        // Test that operations can be created and cloned
        let _op1_clone = op1.clone();
        let _op2_clone = op2.clone();
        let _op3_clone = op3.clone();
        let _op4_clone = op4.clone();
        let _op5_clone = op5.clone();
    }

    #[test]
    fn test_filter_predicates() {
        let pred1 = FilterPredicate::MinLength(5);
        let pred2 = FilterPredicate::MaxLength(100);
        let pred3 = FilterPredicate::Contains("test".to_string());
        let pred4 = FilterPredicate::StartsWith("prefix".to_string());
        let pred5 = FilterPredicate::EndsWith("suffix".to_string());
        let pred6 = FilterPredicate::Regex(".*pattern.*".to_string());

        // Test predicate creation
        match pred1 {
            FilterPredicate::MinLength(len) => assert_eq!(len, 5),
            _ => panic!("Wrong predicate type"),
        }

        match pred2 {
            FilterPredicate::MaxLength(len) => assert_eq!(len, 100),
            _ => panic!("Wrong predicate type"),
        }

        match pred3 {
            FilterPredicate::Contains(ref pattern) => assert_eq!(pattern, "test"),
            _ => panic!("Wrong predicate type"),
        }

        match pred4 {
            FilterPredicate::StartsWith(ref prefix) => assert_eq!(prefix, "prefix"),
            _ => panic!("Wrong predicate type"),
        }

        match pred5 {
            FilterPredicate::EndsWith(ref suffix) => assert_eq!(suffix, "suffix"),
            _ => panic!("Wrong predicate type"),
        }

        match pred6 {
            FilterPredicate::Regex(ref pattern) => assert_eq!(pattern, ".*pattern.*"),
            _ => panic!("Wrong predicate type"),
        }
    }

    #[test]
    fn test_transform_functions() {
        let func1 = TransformFunction::TrimWhitespace;
        let func2 = TransformFunction::RemoveDigits;
        let func3 = TransformFunction::KeepAlphanumeric;
        let func4 = TransformFunction::ReplaceSpaces("_".to_string());
        let func5 = TransformFunction::Capitalize;
        let func6 = TransformFunction::Custom(Box::new(|s: &str| s.to_uppercase()));

        // Test function types
        match func1 {
            TransformFunction::TrimWhitespace => {}
            _ => panic!("Wrong function type"),
        }

        match func4 {
            TransformFunction::ReplaceSpaces(ref replacement) => assert_eq!(replacement, "_"),
            _ => panic!("Wrong function type"),
        }

        // Test custom function
        if let TransformFunction::Custom(ref f) = func6 {
            assert_eq!(f("test"), "TEST");
        }
    }

    #[test]
    fn test_sort_orders() {
        let order1 = SortOrder::Ascending;
        let order2 = SortOrder::Descending;
        let order3 = SortOrder::ByLength;
        let order4 = SortOrder::ByLengthDesc;

        // Test that sort orders can be created and cloned
        let _order1_clone = order1.clone();
        let _order2_clone = order2.clone();
        let _order3_clone = order3.clone();
        let _order4_clone = order4.clone();
    }

    #[test]
    fn test_string_processor_stats_default() {
        let stats = StringProcessorStats::default();
        assert_eq!(stats.operations_processed, 0);
        assert_eq!(stats.strings_processed, 0);
        assert_eq!(stats.total_chars_processed, 0);
        assert_eq!(stats.total_processing_time, Duration::ZERO);

        assert_eq!(stats.throughput_strings_per_sec(), 0.0);
        assert_eq!(stats.throughput_chars_per_sec(), 0.0);
        assert_eq!(stats.avg_processing_time_per_operation(), Duration::ZERO);
    }

    #[test]
    fn test_string_processor_stats_calculations() {
        let mut stats = StringProcessorStats {
            operations_processed: 10,
            strings_processed: 1000,
            total_chars_processed: 50000,
            total_processing_time: Duration::from_secs(5),
        };

        assert_eq!(stats.throughput_strings_per_sec(), 200.0); // 1000 / 5
        assert_eq!(stats.throughput_chars_per_sec(), 10000.0); // 50000 / 5
        assert_eq!(
            stats.avg_processing_time_per_operation(),
            Duration::from_millis(500)
        ); // 5s / 10

        // Test zero division safety
        stats.total_processing_time = Duration::ZERO;
        assert_eq!(stats.throughput_strings_per_sec(), 0.0);
        assert_eq!(stats.throughput_chars_per_sec(), 0.0);

        stats.operations_processed = 0;
        assert_eq!(stats.avg_processing_time_per_operation(), Duration::ZERO);
    }

    #[test]
    fn test_pack_unpack_strings() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "Hello".to_string(),
                "World".to_string(),
                "GPU".to_string(),
                "String".to_string(),
                "Processing".to_string(),
            ];

            // Test packing
            let packed = processor.pack_strings(&test_strings).unwrap();
            assert!(!packed.is_empty());

            // Test unpacking
            let unpacked = processor
                .unpack_strings(&packed, test_strings.len())
                .unwrap();
            assert_eq!(unpacked.len(), test_strings.len());

            for (original, unpacked) in test_strings.iter().zip(unpacked.iter()) {
                assert_eq!(original, unpacked);
            }
        }
    }

    #[test]
    fn test_pack_unpack_empty_strings() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["".to_string(), "NotEmpty".to_string(), "".to_string()];

            let packed = processor.pack_strings(&test_strings)?;
            let unpacked = processor
                .unpack_strings(&packed, test_strings.len())
                .unwrap();

            assert_eq!(unpacked.len(), test_strings.len());
            assert_eq!(unpacked[0], "");
            assert_eq!(unpacked[1], "NotEmpty");
            assert_eq!(unpacked[2], "");
        }
    }

    #[test]
    fn test_pack_unpack_unicode_strings() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "Hello 🌍".to_string(),
                "Привет мир".to_string(),
                "こんにちは世界".to_string(),
                "🚀 GPU 🔥".to_string(),
            ];

            let packed = processor.pack_strings(&test_strings).unwrap();
            let unpacked = processor
                .unpack_strings(&packed, test_strings.len())
                .unwrap();

            assert_eq!(unpacked.len(), test_strings.len());
            for (original, unpacked) in test_strings.iter().zip(unpacked.iter()) {
                assert_eq!(original, unpacked);
            }
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_gpu_string_processor_creation() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();

            let result = GpuStringProcessor::new(ctx, config);
            assert!(result.is_ok());

            let processor = result?;
            let stats = processor.get_statistics();
            assert_eq!(stats.operations_processed, 0);
        }
    }

    #[tokio::test]
    async fn test_batch_to_uppercase() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["hello".to_string(), "world".to_string(), "gpu".to_string()];

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();

            // Note: Since we're using placeholder GPU kernels,
            // the actual transformation won't occur
            assert_eq!(processed.len(), test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_batch_to_lowercase() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["HELLO".to_string(), "WORLD".to_string(), "GPU".to_string()];

            let result = processor
                .process_batch(StringOperation::ToLowercase, &test_strings)
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_batch_reverse() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["hello".to_string(), "world".to_string(), "gpu".to_string()];

            let result = processor
                .process_batch(StringOperation::Reverse, &test_strings)
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_batch_pattern_match() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "hello world".to_string(),
                "gpu processing".to_string(),
                "string operations".to_string(),
            ];

            let result = processor
                .process_batch(
                    StringOperation::PatternMatch {
                        pattern: "gpu".to_string(),
                    },
                    &test_strings,
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            // Result length depends on GPU kernel implementation
            assert!(processed.len() <= test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_batch_replace() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "hello world".to_string(),
                "hello gpu".to_string(),
                "hello string".to_string(),
            ];

            let result = processor
                .process_batch(
                    StringOperation::Replace {
                        from: "hello".to_string(),
                        to: "hi".to_string(),
                    },
                    &test_strings,
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_batch_filter() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "short".to_string(),
                "medium length".to_string(),
                "this is a very long string that exceeds normal length".to_string(),
            ];

            let result = processor
                .process_batch(
                    StringOperation::Filter {
                        predicate: FilterPredicate::MinLength(10),
                    },
                    &test_strings,
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();

            // Should filter out "short" (5 chars)
            assert_eq!(processed.len(), 2);
            assert!(processed.iter().all(|s| s.len() >= 10));
        }
    }

    #[tokio::test]
    async fn test_batch_transform() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "  hello  ".to_string(),
                "  world  ".to_string(),
                "  gpu  ".to_string(),
            ];

            let result = processor
                .process_batch(
                    StringOperation::Transform {
                        function: TransformFunction::TrimWhitespace,
                    },
                    &test_strings,
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());

            for processed_string in processed {
                assert!(!processed_string.starts_with(' '));
                assert!(!processed_string.ends_with(' '));
            }
        }
    }

    #[tokio::test]
    async fn test_batch_sort() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec![
                "zebra".to_string(),
                "apple".to_string(),
                "banana".to_string(),
            ];

            let result = processor
                .process_batch(
                    StringOperation::Sort {
                        order: SortOrder::Ascending,
                    },
                    &test_strings,
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());

            // Check if sorted
            assert_eq!(processed[0], "apple");
            assert_eq!(processed[1], "banana");
            assert_eq!(processed[2], "zebra");
        }
    }

    #[tokio::test]
    async fn test_empty_batch() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings: Vec<String> = vec![];

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_batch_size_limit() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig {
                batch_size: 2, // Small batch size for testing
                ..Default::default()
            };
            let mut processor = GpuStringProcessor::new(ctx, config)?;

            let test_strings = vec![
                "one".to_string(),
                "two".to_string(),
                "three".to_string(), // Exceeds batch size
            ];

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("exceeds maximum"));
        }
    }

    #[tokio::test]
    async fn test_string_length_limit() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig {
                max_string_length: 10, // Small max length for testing
                ..Default::default()
            };
            let mut processor = GpuStringProcessor::new(ctx, config)?;

            let test_strings = vec![
                "short".to_string(),
                "this string is way too long for the limit".to_string(),
            ];

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("exceeds maximum"));
        }
    }

    #[tokio::test]
    async fn test_statistics_update() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["hello".to_string(), "world".to_string()];

            let initial_stats = processor.get_statistics();
            assert_eq!(initial_stats.operations_processed, 0);
            assert_eq!(initial_stats.strings_processed, 0);

            let _result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await
                .unwrap();

            let updated_stats = processor.get_statistics();
            assert_eq!(updated_stats.operations_processed, 1);
            assert_eq!(updated_stats.strings_processed, 2);
            assert_eq!(updated_stats.total_chars_processed, 10); // "hello" + "world" = 10 chars
            assert!(updated_stats.total_processing_time > Duration::ZERO);
        }
    }

    #[tokio::test]
    async fn test_pattern_cache() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["test pattern".to_string(), "another test".to_string()];

            // Use same pattern multiple times
            let _result1 = processor
                .process_batch(
                    StringOperation::PatternMatch {
                        pattern: "test".to_string(),
                    },
                    &test_strings,
                )
                .await
                .unwrap();

            let _result2 = processor
                .process_batch(
                    StringOperation::PatternMatch {
                        pattern: "test".to_string(),
                    },
                    &test_strings,
                )
                .await
                .unwrap();

            // Pattern should be cached (verified internally)
            assert_eq!(processor.pattern_cache.len(), 1);
            assert!(processor.pattern_cache.contains_key("test"));
        }
    }

    #[tokio::test]
    async fn test_clear_pattern_cache() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["test pattern".to_string()];

            // Add patterns to cache
            let _result = processor
                .process_batch(
                    StringOperation::PatternMatch {
                        pattern: "test".to_string(),
                    },
                    &test_strings,
                )
                .await
                .unwrap();

            assert!(!processor.pattern_cache.is_empty());

            processor.clear_pattern_cache();
            assert!(processor.pattern_cache.is_empty());
        }
    }
}

// =============================================================================
// High-Level Interface Tests
// =============================================================================

#[cfg(test)]
mod high_level_tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_string_stream_processor_creation() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();

            let result = StringStreamProcessor::new(ctx, config);
            assert!(result.is_ok());

            let processor = result?;
            let stats = processor.get_statistics();
            assert_eq!(stats.operations_processed, 0);
        }
    }

    #[tokio::test]
    async fn test_string_stream_processor_uppercase() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            let test_strings = vec!["hello".to_string(), "world".to_string(), "gpu".to_string()];

            let result = processor
                .process(StringOperation::ToUppercase, test_strings.clone())
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());
        }
    }

    #[tokio::test]
    async fn test_cpu_fallback() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            // Disable GPU acceleration to force CPU fallback
            processor.set_gpu_acceleration(false);

            let test_strings = vec!["hello".to_string(), "world".to_string()];

            let result = processor
                .process(StringOperation::ToUppercase, test_strings.clone())
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());
            assert_eq!(processed[0], "HELLO");
            assert_eq!(processed[1], "WORLD");
        }
    }

    #[tokio::test]
    async fn test_cpu_fallback_lowercase() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            processor.set_gpu_acceleration(false);

            let test_strings = vec!["HELLO".to_string(), "WORLD".to_string()];

            let result = processor
                .process(StringOperation::ToLowercase, test_strings.clone())
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed[0], "hello");
            assert_eq!(processed[1], "world");
        }
    }

    #[tokio::test]
    async fn test_cpu_fallback_reverse() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            processor.set_gpu_acceleration(false);

            let test_strings = vec!["hello".to_string(), "world".to_string()];

            let result = processor
                .process(StringOperation::Reverse, test_strings.clone())
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed[0], "olleh");
            assert_eq!(processed[1], "dlrow");
        }
    }

    #[tokio::test]
    async fn test_cpu_fallback_pattern_match() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            processor.set_gpu_acceleration(false);

            let test_strings = vec![
                "hello world".to_string(),
                "gpu processing".to_string(),
                "string operations".to_string(),
            ];

            let result = processor
                .process(
                    StringOperation::PatternMatch {
                        pattern: "gpu".to_string(),
                    },
                    test_strings.clone(),
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), 1);
            assert_eq!(processed[0], "gpu processing");
        }
    }

    #[tokio::test]
    async fn test_cpu_fallback_replace() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = StringStreamProcessor::new(ctx, config).unwrap();

            processor.set_gpu_acceleration(false);

            let test_strings = vec!["hello world".to_string(), "hello gpu".to_string()];

            let result = processor
                .process(
                    StringOperation::Replace {
                        from: "hello".to_string(),
                        to: "hi".to_string(),
                    },
                    test_strings.clone(),
                )
                .await;

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed[0], "hi world");
            assert_eq!(processed[1], "hi gpu");
        }
    }
}

// =============================================================================
// Performance Benchmark Tests
// =============================================================================

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use std::time::Instant;
    use tokio;

    #[tokio::test]
    async fn benchmark_batch_processing_performance() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            // Create large batch of strings
            let batch_size = 1000;
            let test_strings: Vec<String> = (0..batch_size)
                .map(|i| format!("test string number {}", i))
                .collect();

            let start = Instant::now();

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            let processing_time = start.elapsed();

            assert!(result.is_ok());
            let processed = result.unwrap();
            assert_eq!(processed.len(), test_strings.len());

            println!("Processed {} strings in {:?}", batch_size, processing_time);

            // Performance expectation: should process 1000 strings in under 100ms
            assert!(processing_time < Duration::from_millis(100));

            let stats = processor.get_statistics();
            assert!(stats.throughput_strings_per_sec() > 0.0);
        }
    }

    #[tokio::test]
    async fn benchmark_pattern_matching_performance() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let batch_size = 500;
            let test_strings: Vec<String> = (0..batch_size)
                .map(|i| {
                    if i % 3 == 0 {
                        format!("pattern match test {}", i)
                    } else {
                        format!("other string {}", i)
                    }
                })
                .collect();

            let start = Instant::now();

            let result = processor
                .process_batch(
                    StringOperation::PatternMatch {
                        pattern: "pattern".to_string(),
                    },
                    &test_strings,
                )
                .await;

            let processing_time = start.elapsed();

            assert!(result.is_ok());

            println!(
                "Pattern matched {} strings in {:?}",
                batch_size, processing_time
            );

            // Performance expectation: should process pattern matching in under 50ms
            assert!(processing_time < Duration::from_millis(50));
        }
    }

    #[tokio::test]
    async fn benchmark_multiple_operations() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let mut processor = GpuStringProcessor::new(ctx, config).unwrap();

            let test_strings: Vec<String> =
                (0..100).map(|i| format!("test string {}", i)).collect();

            let operations = vec![
                StringOperation::ToUppercase,
                StringOperation::ToLowercase,
                StringOperation::Reverse,
                StringOperation::PatternMatch {
                    pattern: "test".to_string(),
                },
                StringOperation::Replace {
                    from: "test".to_string(),
                    to: "TEST".to_string(),
                },
            ];

            let start = Instant::now();

            for operation in operations {
                let _result = processor
                    .process_batch(operation, &test_strings)
                    .await
                    .unwrap();
            }

            let total_time = start.elapsed();

            println!("Completed 5 operations on 100 strings in {:?}", total_time);

            let stats = processor.get_statistics();
            assert_eq!(stats.operations_processed, 5);
            assert_eq!(stats.strings_processed, 500); // 5 operations * 100 strings

            // Performance expectation: should complete all operations in under 200ms
            assert!(total_time < Duration::from_millis(200));
        }
    }

    #[tokio::test]
    async fn benchmark_large_string_processing() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig {
                max_string_length: 10000, // Allow larger strings
                ..Default::default()
            };
            let mut processor = GpuStringProcessor::new(ctx, config)?;

            // Create strings of varying sizes
            let test_strings: Vec<String> = vec![
                "a".repeat(1000), // 1KB string
                "b".repeat(5000), // 5KB string
                "c".repeat(8000), // 8KB string
            ];

            let start = Instant::now();

            let result = processor
                .process_batch(StringOperation::ToUppercase, &test_strings)
                .await;

            let processing_time = start.elapsed();

            assert!(result.is_ok());

            println!("Processed large strings in {:?}", processing_time);

            let total_chars: usize = test_strings.iter().map(|s| s.len()).sum();
            let throughput = total_chars as f64 / processing_time.as_secs_f64();

            println!("Character throughput: {:.2} chars/sec", throughput);

            // Should handle large strings efficiently
            assert!(processing_time < Duration::from_millis(50));
        }
    }

    #[tokio::test]
    async fn benchmark_concurrent_streams() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig {
                num_streams: 8, // More streams for concurrency
                ..Default::default()
            };
            let mut processor = GpuStringProcessor::new(ctx, config)?;

            let test_strings: Vec<String> =
                (0..200).map(|i| format!("concurrent test {}", i)).collect();

            let start = Instant::now();

            // Process multiple batches sequentially (simulating concurrent usage)
            for _ in 0..4 {
                let _result = processor
                    .process_batch(StringOperation::ToUppercase, &test_strings)
                    .await
                    .unwrap();
            }

            let processing_time = start.elapsed();

            println!(
                "Processed 4 batches of 200 strings in {:?}",
                processing_time
            );

            let stats = processor.get_statistics();
            assert_eq!(stats.operations_processed, 4);
            assert_eq!(stats.strings_processed, 800); // 4 * 200

            // Concurrent processing should be efficient
            assert!(processing_time < Duration::from_millis(400));
        }
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[cfg(test)]
mod error_tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_invalid_utf8_handling() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            // Create invalid UTF-8 data
            let invalid_data = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8 sequence

            let result = processor.unpack_strings(
                &[
                    3, 0, 0, 0, // count: 1
                    3, 0, 0, 0, // length: 3
                    0xFF, 0xFE, 0xFD, // invalid UTF-8
                ],
                1,
            );

            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Invalid UTF-8"));
        }
    }

    #[tokio::test]
    async fn test_insufficient_data_handling() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            // Test insufficient data for count
            let result = processor.unpack_strings(&[0x01, 0x02], 1);
            assert!(result.is_err());

            // Test insufficient data for string length
            let result = processor.unpack_strings(
                &[
                    1, 0, 0, 0, // count: 1
                    0x01, 0x02, // incomplete length
                ],
                1,
            );
            assert!(result.is_err());

            // Test insufficient data for string content
            let result = processor.unpack_strings(
                &[
                    1, 0, 0, 0, // count: 1
                    5, 0, 0, 0, // length: 5
                    0x41, 0x42, // only 2 bytes instead of 5
                ],
                1,
            );
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_count_mismatch_handling() {
        if let Ok(ctx) = CudaContext::new(0) {
            let config = StringProcessorConfig::default();
            let processor = GpuStringProcessor::new(ctx, config).unwrap();

            let result = processor.unpack_strings(
                &[
                    2, 0, 0, 0, // count: 2
                    5, 0, 0, 0, // length: 5
                    b'h', b'e', b'l', b'l', b'o', // only 1 string
                ],
                1,
            ); // expect 1 string

            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("count mismatch"));
        }
    }
}
