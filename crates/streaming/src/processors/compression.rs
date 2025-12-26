//! Compression stream processor for data size optimization

use crate::{StreamChunk, StreamProcessor, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Compression algorithm types
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// Simple run-length encoding
    RunLength,
    /// LZ77-style compression
    Lz77,
    /// Dictionary-based compression
    Dictionary,
    /// No compression (passthrough)
    None,
}

/// Compression stream processor for reducing data size
pub struct CompressionProcessor {
    id: String,
    algorithm: CompressionAlgorithm,
    compression_level: u8,
    stats: Arc<CompressionStats>,
}

/// Thread-safe statistics for compression processor
#[derive(Debug, Default)]
struct CompressionStats {
    chunks_processed: AtomicU64,
    bytes_input: AtomicU64,
    bytes_output: AtomicU64,
    compression_time_ns: AtomicU64,
    errors: AtomicU64,
}

impl CompressionProcessor {
    /// Create a new compression processor
    pub fn new(id: String, algorithm: CompressionAlgorithm) -> Self {
        Self {
            id,
            algorithm,
            compression_level: 6, // Default medium compression
            stats: Arc::new(CompressionStats::default()),
        }
    }

    /// Configure compression level (1-9, where 9 is highest compression)
    pub fn with_compression_level(mut self, level: u8) -> Self {
        self.compression_level = level.clamp(1, 9);
        self
    }

    /// Get the compression algorithm
    pub fn algorithm(&self) -> &CompressionAlgorithm {
        &self.algorithm
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        // Relaxed: independent statistics counters with no ordering dependencies
        let chunks = self.stats.chunks_processed.load(Ordering::Relaxed);
        let input_bytes = self.stats.bytes_input.load(Ordering::Relaxed);
        let output_bytes = self.stats.bytes_output.load(Ordering::Relaxed);
        let time_ns = self.stats.compression_time_ns.load(Ordering::Relaxed);
        let errors = self.stats.errors.load(Ordering::Relaxed);

        let throughput_mbps = if time_ns > 0 {
            (input_bytes as f64) / ((time_ns as f64) / 1_000_000_000.0) / (1024.0 * 1024.0)
        } else {
            0.0
        };

        StreamStats {
            chunks_processed: chunks,
            bytes_processed: output_bytes, // Use output bytes for processed
            processing_time_ms: time_ns / 1_000_000,
            throughput_mbps,
            errors,
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        // Relaxed: approximate ratio sufficient for statistics
        let input = self.stats.bytes_input.load(Ordering::Relaxed);
        let output = self.stats.bytes_output.load(Ordering::Relaxed);

        if input > 0 {
            output as f64 / input as f64
        } else {
            1.0
        }
    }

    /// Compress data using the configured algorithm
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, StreamingError> {
        match &self.algorithm {
            CompressionAlgorithm::RunLength => self.run_length_encode(data),
            CompressionAlgorithm::Lz77 => self.lz77_compress(data),
            CompressionAlgorithm::Dictionary => self.dictionary_compress(data),
            CompressionAlgorithm::None => Ok(data.to_vec()),
        }
    }

    /// Simple run-length encoding
    fn run_length_encode(&self, data: &[u8]) -> Result<Vec<u8>, StreamingError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed = Vec::new();
        let mut current_byte = data[0];
        let mut count = 1u8;

        for &byte in &data[1..] {
            if byte == current_byte && count < 255 {
                count += 1;
            } else {
                compressed.push(count);
                compressed.push(current_byte);
                current_byte = byte;
                count = 1;
            }
        }

        // Add the last run
        compressed.push(count);
        compressed.push(current_byte);

        Ok(compressed)
    }

    /// Simple LZ77-style compression
    fn lz77_compress(&self, data: &[u8]) -> Result<Vec<u8>, StreamingError> {
        let mut compressed = Vec::new();
        let window_size = (self.compression_level as usize) * 32; // Variable window size
        let mut i = 0;

        while i < data.len() {
            let mut best_length = 0;
            let mut best_distance = 0;

            // Look for matches in the sliding window
            let window_start = i.saturating_sub(window_size);
            for j in window_start..i {
                let mut length = 0;
                while i + length < data.len()
                    && j + length < i
                    && data[j + length] == data[i + length]
                    && length < 255
                {
                    length += 1;
                }

                if length > best_length {
                    best_length = length;
                    best_distance = i - j;
                }
            }

            if best_length >= 3 {
                // Encode as (distance, length, next_char)
                compressed.push(best_distance as u8);
                compressed.push(best_length as u8);
                if i + best_length < data.len() {
                    compressed.push(data[i + best_length]);
                } else {
                    compressed.push(0);
                }
                i += best_length + 1;
            } else {
                // No good match, encode literal
                compressed.push(0); // 0 distance means literal
                compressed.push(1); // length 1
                compressed.push(data[i]);
                i += 1;
            }
        }

        Ok(compressed)
    }

    /// Simple dictionary-based compression
    fn dictionary_compress(&self, data: &[u8]) -> Result<Vec<u8>, StreamingError> {
        // Build frequency table
        let mut freq = [0u32; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }

        // Find most common bytes for dictionary
        let mut dict_entries: Vec<(u8, u32)> = freq
            .iter()
            .enumerate()
            .map(|(i, &count)| (i as u8, count))
            .collect();
        dict_entries.sort_by(|a, b| b.1.cmp(&a.1));

        let dict_size = std::cmp::min(16, self.compression_level as usize * 2);
        let mut dictionary = Vec::new();
        let mut dict_map = std::collections::HashMap::new();

        for (i, &(byte, _)) in dict_entries.iter().enumerate().take(dict_size) {
            dictionary.push(byte);
            dict_map.insert(byte, i as u8);
        }

        // Compress using dictionary
        let mut compressed = Vec::new();

        // Write dictionary size and entries
        compressed.push(dict_size as u8);
        compressed.extend_from_slice(&dictionary);

        // Encode data
        for &byte in data {
            if let Some(&dict_index) = dict_map.get(&byte) {
                compressed.push(0x80 | dict_index); // High bit indicates dictionary entry
            } else {
                compressed.push(byte); // Literal byte
            }
        }

        Ok(compressed)
    }
}

#[async_trait]
impl StreamProcessor for CompressionProcessor {
    async fn process(&mut self, chunk: StreamChunk) -> Result<StreamChunk, StreamingError> {
        let start_time = Instant::now();
        let input_size = chunk.data.len();

        let compressed_data = self.compress_data(&chunk.data)?;
        let output_size = compressed_data.len();

        let processing_time = start_time.elapsed().as_nanos() as u64;
        // Relaxed: independent statistics counters
        self.stats
            .compression_time_ns
            .fetch_add(processing_time, Ordering::Relaxed);
        self.stats.chunks_processed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_input
            .fetch_add(input_size as u64, Ordering::Relaxed);
        self.stats
            .bytes_output
            .fetch_add(output_size as u64, Ordering::Relaxed);

        Ok(StreamChunk::new(
            Bytes::from(compressed_data),
            chunk.sequence,
            chunk.metadata.source_id,
        ))
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn processor_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compression_processor_creation() {
        let processor = CompressionProcessor::new(
            "test-compression".to_string(),
            CompressionAlgorithm::RunLength,
        );

        assert_eq!(processor.processor_id(), "test-compression");
        assert_eq!(processor.compression_level, 6);
    }

    #[tokio::test]
    async fn test_compression_level_configuration() {
        let processor =
            CompressionProcessor::new("config-test".to_string(), CompressionAlgorithm::Lz77)
                .with_compression_level(9);

        assert_eq!(processor.compression_level, 9);

        // Test clamping
        let processor2 =
            CompressionProcessor::new("clamp-test".to_string(), CompressionAlgorithm::Lz77)
                .with_compression_level(15); // Should clamp to 9

        assert_eq!(processor2.compression_level, 9);
    }

    #[tokio::test]
    async fn test_run_length_encoding() {
        let mut processor =
            CompressionProcessor::new("rle-test".to_string(), CompressionAlgorithm::RunLength);

        // Test with repeated data
        let input_data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let chunk = StreamChunk::new(Bytes::from(input_data), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        let result_data = result.data.to_vec();

        // RLE should encode as [count, value] pairs
        assert_eq!(result_data, vec![3, 1, 2, 2, 4, 3]);
        assert_eq!(result.sequence, 1);
        assert_eq!(result.metadata.source_id, "test-source");

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert!(processor.compression_ratio() < 1.0); // Should achieve compression
    }

    #[tokio::test]
    async fn test_lz77_compression() {
        let mut processor =
            CompressionProcessor::new("lz77-test".to_string(), CompressionAlgorithm::Lz77)
                .with_compression_level(3);

        // Test with repeated patterns
        let input_data = b"abcabcabc".to_vec();
        let chunk = StreamChunk::new(Bytes::from(input_data), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        let result_data = result.data.to_vec();

        // LZ77 should find repeated patterns
        assert!(!result_data.is_empty());
        // LZ77 might expand small data due to overhead, but should work
        assert!(result_data.len() <= 30); // Allow for encoding overhead

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert_eq!(stats.bytes_processed, result_data.len() as u64);
    }

    #[tokio::test]
    async fn test_dictionary_compression() {
        let mut processor =
            CompressionProcessor::new("dict-test".to_string(), CompressionAlgorithm::Dictionary)
                .with_compression_level(4);

        // Test with data that has frequent bytes
        let input_data = vec![65, 65, 65, 66, 66, 67, 65, 66, 65]; // AAABBCABA
        let chunk = StreamChunk::new(
            Bytes::from(input_data.clone()),
            1,
            "test-source".to_string(),
        );

        let result = processor.process(chunk).await.unwrap();
        let result_data = result.data.to_vec();

        // Dictionary compression should start with dictionary size
        assert!(!result_data.is_empty());
        assert!(result_data[0] <= 8); // Dictionary size should be reasonable

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert_eq!(
            processor.stats.bytes_input.load(Ordering::Relaxed),
            input_data.len() as u64
        );
    }

    #[tokio::test]
    async fn test_no_compression() {
        let mut processor =
            CompressionProcessor::new("none-test".to_string(), CompressionAlgorithm::None);

        let input_data = vec![1, 2, 3, 4, 5];
        let chunk = StreamChunk::new(
            Bytes::from(input_data.clone()),
            1,
            "test-source".to_string(),
        );

        let result = processor.process(chunk).await.unwrap();
        let result_data = result.data.to_vec();

        // No compression should return identical data
        assert_eq!(result_data, input_data);
        assert_eq!(processor.compression_ratio(), 1.0); // No compression
    }

    #[tokio::test]
    async fn test_empty_data_compression() {
        let mut processor =
            CompressionProcessor::new("empty-test".to_string(), CompressionAlgorithm::RunLength);

        let chunk = StreamChunk::new(Bytes::new(), 1, "test-source".to_string());

        let result = processor.process(chunk).await.unwrap();
        assert!(result.data.is_empty());
    }

    #[tokio::test]
    async fn test_compression_stats() {
        let mut processor =
            CompressionProcessor::new("stats-test".to_string(), CompressionAlgorithm::RunLength);

        // Initial stats should be zero
        let initial_stats = processor.stats().await.unwrap();
        assert_eq!(initial_stats.chunks_processed, 0);
        assert_eq!(initial_stats.bytes_processed, 0);
        assert_eq!(processor.compression_ratio(), 1.0);

        // Process some data
        let input_data = vec![1, 1, 1, 1, 2, 2, 2]; // Good for RLE
        let chunk = StreamChunk::new(Bytes::from(input_data.clone()), 1, "test".to_string());
        processor.process(chunk).await.unwrap();

        let final_stats = processor.stats().await.unwrap();
        assert_eq!(final_stats.chunks_processed, 1);
        assert!(final_stats.processing_time_ms >= 0);
        assert!(final_stats.throughput_mbps >= 0.0);

        // Check compression ratio
        let ratio = processor.compression_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0);

        // Input bytes should be tracked separately
        assert_eq!(
            processor.stats.bytes_input.load(Ordering::Relaxed),
            input_data.len() as u64
        );
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let mut processor =
            CompressionProcessor::new("batch-test".to_string(), CompressionAlgorithm::RunLength);

        let chunks = vec![
            StreamChunk::new(Bytes::from(vec![1, 1, 2, 2]), 1, "test".to_string()),
            StreamChunk::new(Bytes::from(vec![3, 3, 3, 4]), 2, "test".to_string()),
        ];

        let results = processor.process_batch(chunks).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data.to_vec(), vec![2, 1, 2, 2]); // RLE of [1,1,2,2]
        assert_eq!(results[1].data.to_vec(), vec![3, 3, 1, 4]); // RLE of [3,3,3,4]

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 2);
    }
}
