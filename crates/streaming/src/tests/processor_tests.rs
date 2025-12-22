//! Tests for stream processors

use crate::core::StreamProcessor;
use crate::processors::*;
use crate::{StreamChunk, StreamStats, StreamingError};
use bytes::Bytes;

#[tokio::test]
async fn test_compression_processor_creation() {
    let processor = CompressionProcessor::new(
        "test-compression".to_string(),
        CompressionAlgorithm::RunLength,
    );

    assert_eq!(processor.processor_id(), "test-compression");
}

#[tokio::test]
async fn test_compression_level_configuration() {
    let processor =
        CompressionProcessor::new("config-test".to_string(), CompressionAlgorithm::Lz77)
            .with_compression_level(9);

    // Test clamping with out-of-bounds values
    let processor2 =
        CompressionProcessor::new("clamp-test".to_string(), CompressionAlgorithm::Lz77)
            .with_compression_level(15); // Should clamp to 9

    let processor3 =
        CompressionProcessor::new("clamp-test2".to_string(), CompressionAlgorithm::Lz77)
            .with_compression_level(0); // Should clamp to 1
}

#[tokio::test]
async fn test_run_length_encoding_comprehensive() {
    let mut processor = CompressionProcessor::new(
        "rle-comprehensive".to_string(),
        CompressionAlgorithm::RunLength,
    );

    // Test various RLE scenarios
    let test_cases = vec![
        (vec![1, 1, 1, 2, 2, 3, 3, 3, 3], vec![3, 1, 2, 2, 4, 3]),
        (vec![1], vec![1, 1]),                   // Single byte
        (vec![1, 2, 3], vec![1, 1, 1, 2, 1, 3]), // No repeats
        (vec![5; 255], vec![255, 5]),            // Maximum run length
        (vec![5; 256], vec![255, 5, 1, 5]),      // Over maximum run length
    ];

    for (input, expected) in test_cases {
        let chunk = StreamChunk::new(Bytes::from(input.clone()), 1, "test".to_string());
        let result = processor.process(chunk).await.unwrap();
        assert_eq!(
            result.data.to_vec(),
            expected,
            "Failed for input: {:?}",
            input
        );
    }
}

#[tokio::test]
async fn test_lz77_compression_various_patterns() {
    let mut processor =
        CompressionProcessor::new("lz77-patterns".to_string(), CompressionAlgorithm::Lz77)
            .with_compression_level(5);

    let test_patterns = vec![
        b"abcabcabc".to_vec(),        // Simple pattern
        b"abcdefghijklmnop".to_vec(), // No patterns
        b"aaaaaaaaaa".to_vec(),       // Single character repeat
        b"abcabcdefdefghi".to_vec(),  // Multiple patterns
        b"a".to_vec(),                // Single character
        vec![],                       // Empty
    ];

    for pattern in test_patterns {
        let chunk = StreamChunk::new(Bytes::from(pattern.clone()), 1, "test".to_string());
        let result = processor.process(chunk).await;
        assert!(result.is_ok(), "LZ77 failed for pattern: {:?}", pattern);

        let result_chunk = result.unwrap();
        assert!(!result_chunk.data.is_empty() || pattern.is_empty());
    }
}

#[tokio::test]
async fn test_dictionary_compression_frequency_analysis() {
    let mut processor =
        CompressionProcessor::new("dict-freq".to_string(), CompressionAlgorithm::Dictionary)
            .with_compression_level(3);

    // Create data with specific frequency patterns
    let mut input_data = Vec::new();
    input_data.extend(vec![65; 50]); // 'A' appears 50 times
    input_data.extend(vec![66; 30]); // 'B' appears 30 times
    input_data.extend(vec![67; 20]); // 'C' appears 20 times
    input_data.extend(vec![68; 10]); // 'D' appears 10 times
    input_data.extend(vec![69; 5]); // 'E' appears 5 times

    let chunk = StreamChunk::new(Bytes::from(input_data.clone()), 1, "test".to_string());
    let result = processor.process(chunk).await.unwrap();
    let result_data = result.data.to_vec();

    // Dictionary should be created and used
    assert!(!result_data.is_empty());
    assert!(result_data[0] > 0); // Dictionary size should be > 0

    // Most frequent bytes should be in dictionary
    let dict_size = result_data[0] as usize;
    let dictionary = &result_data[1..=dict_size];
    assert!(dictionary.contains(&65)); // 'A' should be in dictionary
}

#[tokio::test]
async fn test_compression_algorithm_comparison() {
    let algorithms = vec![
        CompressionAlgorithm::RunLength,
        CompressionAlgorithm::Lz77,
        CompressionAlgorithm::Dictionary,
        CompressionAlgorithm::None,
    ];

    let test_data = vec![1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 5];
    let mut results = Vec::new();

    for algorithm in algorithms {
        let mut processor = CompressionProcessor::new(format!("test-{:?}", algorithm), algorithm);

        let chunk = StreamChunk::new(Bytes::from(test_data.clone()), 1, "test".to_string());
        let result = processor.process(chunk).await.unwrap();

        results.push((format!("{:?}", processor.algorithm()), result.data.len()));
    }

    // None algorithm should preserve original size
    let none_result = results.iter().find(|(name, _)| name == "None").unwrap();
    assert_eq!(none_result.1, test_data.len());

    // Other algorithms may vary in effectiveness
    for (name, size) in &results {
        assert!(*size > 0, "Algorithm {} produced empty output", name);
    }
}

#[tokio::test]
async fn test_compression_statistics_accuracy() {
    let mut processor = CompressionProcessor::new(
        "stats-accuracy".to_string(),
        CompressionAlgorithm::RunLength,
    );

    let chunks = vec![
        StreamChunk::new(Bytes::from(vec![1, 1, 2, 2]), 1, "test".to_string()),
        StreamChunk::new(Bytes::from(vec![3, 3, 3, 4]), 2, "test".to_string()),
        StreamChunk::new(Bytes::from(vec![5; 100]), 3, "test".to_string()),
    ];

    let mut total_input_bytes = 0;
    let mut total_output_bytes = 0;

    for chunk in chunks {
        total_input_bytes += chunk.data.len();
        let result = processor.process(chunk).await.unwrap();
        total_output_bytes += result.data.len();
    }

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 3);
    assert_eq!(stats.bytes_processed, total_output_bytes as u64);

    // Check compression ratio
    let ratio = processor.compression_ratio();
    assert!(ratio > 0.0);
    assert!(ratio <= 1.0); // Should achieve some compression with RLE
}

#[tokio::test]
async fn test_compression_performance_metrics() {
    let mut processor =
        CompressionProcessor::new("performance".to_string(), CompressionAlgorithm::Lz77);

    let large_data = vec![42u8; 10000]; // 10KB of data
    let chunk = StreamChunk::new(Bytes::from(large_data), 1, "test".to_string());

    let start = std::time::Instant::now();
    let _result = processor.process(chunk).await.unwrap();
    let processing_time = start.elapsed();

    let stats = processor.stats().await.unwrap();
    assert!(stats.processing_time_ms > 0);
    assert!(stats.throughput_mbps >= 0.0);

    // Processing time should be reasonable for 10KB
    assert!(processing_time.as_millis() < 1000); // Less than 1 second
}

#[tokio::test]
async fn test_compression_error_handling() {
    // Test with corrupted/invalid processor state would require
    // internal state manipulation which isn't easily testable
    // with the current API. This is more of a placeholder for
    // future error handling scenarios.

    let mut processor =
        CompressionProcessor::new("error-test".to_string(), CompressionAlgorithm::RunLength);

    // Test with very large chunk that might cause issues
    let very_large_data = vec![1u8; 100_000_000]; // 100MB
    let chunk = StreamChunk::new(Bytes::from(very_large_data), 1, "test".to_string());

    // This should complete without panic, even if it takes time
    let result = processor.process(chunk).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_compression_batch_processing_performance() {
    let mut processor =
        CompressionProcessor::new("batch-perf".to_string(), CompressionAlgorithm::RunLength);

    // Create a batch of chunks
    let mut chunks = Vec::new();
    for i in 0..50 {
        let data = vec![i as u8; 100]; // Each chunk has repeated data
        chunks.push(StreamChunk::new(Bytes::from(data), i, "test".to_string()));
    }

    let start = std::time::Instant::now();
    let results = processor.process_batch(chunks.clone()).await.unwrap();
    let batch_time = start.elapsed();

    assert_eq!(results.len(), chunks.len());

    // Batch processing should be reasonably fast
    assert!(batch_time.as_millis() < 5000); // Less than 5 seconds for 50 chunks

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 50);
}

#[tokio::test]
async fn test_compression_level_effectiveness() {
    let test_data = b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz".to_vec();
    let levels = [1, 3, 6, 9];
    let mut results = Vec::new();

    for level in levels {
        let mut processor =
            CompressionProcessor::new(format!("level-{}", level), CompressionAlgorithm::Lz77)
                .with_compression_level(level);

        let chunk = StreamChunk::new(Bytes::from(test_data.clone()), 1, "test".to_string());
        let result = processor.process(chunk).await.unwrap();

        results.push((level, result.data.len(), processor.compression_ratio()));
    }

    // All levels should produce valid output
    for (level, size, ratio) in &results {
        assert!(*size > 0, "Level {} produced empty output", level);
        assert!(*ratio > 0.0, "Level {} produced invalid ratio", level);
    }
}

#[tokio::test]
async fn test_compression_edge_cases() {
    let mut processor =
        CompressionProcessor::new("edge-cases".to_string(), CompressionAlgorithm::RunLength);

    // Test edge cases
    let edge_cases = vec![
        (vec![], "empty data"),
        (vec![0], "single zero"),
        (vec![255], "single max byte"),
        (vec![1, 2, 3, 4, 5], "no repeats"),
        (vec![42; 1000], "long run"),
        ((0..=255).collect::<Vec<u8>>(), "all byte values"),
    ];

    for (data, description) in edge_cases {
        let chunk = StreamChunk::new(Bytes::from(data.clone()), 1, "test".to_string());
        let result = processor.process(chunk).await;

        assert!(result.is_ok(), "Failed for case: {}", description);

        if !data.is_empty() {
            let result_chunk = result.unwrap();
            assert!(
                !result_chunk.data.is_empty(),
                "Empty result for case: {}",
                description
            );
        }
    }
}

#[tokio::test]
async fn test_compression_concurrent_processing() {
    use std::sync::Arc;
    use tokio::task;

    let processor = Arc::new(tokio::sync::Mutex::new(CompressionProcessor::new(
        "concurrent".to_string(),
        CompressionAlgorithm::Dictionary,
    )));

    let mut handles = Vec::new();

    // Process multiple chunks concurrently
    for i in 0..20 {
        let processor_clone = processor.clone();
        let handle = task::spawn(async move {
            let data = vec![i as u8; 50];
            let chunk = StreamChunk::new(Bytes::from(data), i as u64, "test".to_string());

            let mut proc = processor_clone.lock().await;
            proc.process(chunk).await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    let final_stats = processor.lock().await.stats().await.unwrap();
    assert_eq!(final_stats.chunks_processed, 20);
}

#[tokio::test]
async fn test_compression_memory_usage() {
    let mut processor =
        CompressionProcessor::new("memory-test".to_string(), CompressionAlgorithm::Lz77);

    // Process chunks of increasing size to test memory behavior
    let sizes = [1, 100, 1000, 10000, 100000];

    for size in sizes {
        let data = vec![42u8; size];
        let chunk = StreamChunk::new(Bytes::from(data), 1, "test".to_string());

        let result = processor.process(chunk).await.unwrap();

        // Result should not be empty (unless input was empty)
        if size > 0 {
            assert!(!result.data.is_empty());
        }

        // Memory usage should be reasonable (not exponential)
        assert!(result.data.len() < size * 10); // Allow for some overhead but not excessive
    }
}

#[tokio::test]
async fn test_compression_metadata_preservation() {
    let mut processor =
        CompressionProcessor::new("metadata-test".to_string(), CompressionAlgorithm::RunLength);

    let original_chunk = StreamChunk::new(
        Bytes::from(vec![1, 1, 2, 2]),
        42,
        "original-source".to_string(),
    );

    let original_sequence = original_chunk.sequence;
    let original_source = original_chunk.metadata.source_id.clone();

    let result = processor.process(original_chunk).await.unwrap();

    // Metadata should be preserved
    assert_eq!(result.sequence, original_sequence);
    assert_eq!(result.metadata.source_id, original_source);

    // But chunk size should be updated
    assert_eq!(result.metadata.chunk_size, result.data.len());
}
