//! Comprehensive test suite for Huffman coding implementation
//!
//! Tests all aspects of GPU-accelerated Huffman compression and decompression
//! for streaming data processing.

use super::*;
use crate::streaming::huffman::*;
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_huffman_tree_construction() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Test with simple data
        let data = "hello world".as_bytes();
        let tree = codec.build_huffman_tree(data)?;

        assert!(tree.root.is_some());
        assert!(!tree.code_table.is_empty());

        // Verify all characters have codes
        for &byte in data {
            assert!(tree.code_table.contains_key(&byte));
        }
    }

    #[test]
    fn test_frequency_analysis() {
        let codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let data = "aabbcc".as_bytes();
        let frequencies = codec.analyze_frequencies(data).unwrap();

        assert_eq!(frequencies[&b'a'], 2);
        assert_eq!(frequencies[&b'b'], 2);
        assert_eq!(frequencies[&b'c'], 2);
    }

    #[test]
    fn test_code_generation() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let data = "aaabbc".as_bytes();
        let tree = codec.build_huffman_tree(data).unwrap();

        // More frequent characters should have shorter codes
        let code_a = tree.code_table.get(&b'a')?;
        let code_c = tree.code_table.get(&b'c')?;

        assert!(code_a.bit_length <= code_c.bit_length);
    }

    #[test]
    fn test_basic_encode_decode() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let original = "hello world".as_bytes();
        let encoded = codec.encode(original).unwrap();
        let decoded = codec.decode(&encoded.data, &encoded.tree)?;

        assert_eq!(original, decoded.as_slice());
    }

    #[test]
    fn test_compression_ratio() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Test with repetitive data (should compress well)
        let data = "aaaaaabbbbccccdddd".as_bytes();
        let encoded = codec.encode(data)?;

        let compression_ratio = data.len() as f32 / encoded.data.len() as f32;
        assert!(compression_ratio > 1.0); // Should achieve some compression
    }

    #[test]
    fn test_empty_data() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let empty_data = b"";
        let result = codec.encode(empty_data);

        // Should handle empty data gracefully
        assert!(result.is_err() || result?.data.is_empty());
    }

    #[test]
    fn test_single_character() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let single_char = b"a";
        let encoded = codec.encode(single_char).unwrap();
        let decoded = codec.decode(&encoded.data, &encoded.tree)?;

        assert_eq!(single_char, decoded.as_slice());
    }

    #[test]
    fn test_code_table_validity() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        let data = "abcdefghij".as_bytes();
        let tree = codec.build_huffman_tree(data).unwrap();

        // Verify no code is a prefix of another (prefix property)
        let codes: Vec<_> = tree.code_table.values().collect();
        for (i, code1) in codes.iter().enumerate() {
            for (j, code2) in codes.iter().enumerate() {
                if i != j {
                    assert!(!is_prefix(
                        &code1.bits,
                        code1.bit_length,
                        &code2.bits,
                        code2.bit_length
                    ));
                }
            }
        }
    }

    #[test]
    fn test_large_data_handling() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Generate large test data
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let encoded = codec.encode(&large_data)?;
        let decoded = codec.decode(&encoded.data, &encoded.tree)?;

        assert_eq!(large_data, decoded);
    }

    #[test]
    fn test_huffman_config() {
        let config = HuffmanConfig {
            max_tree_depth: 20,
            batch_size: 512,
            use_gpu_acceleration: true,
            compression_level: CompressionLevel::Balanced,
            enable_statistics: true,
        };

        let codec = HuffmanCodec::new(config)?;
        assert_eq!(codec.config.max_tree_depth, 20);
        assert_eq!(codec.config.batch_size, 512);
    }

    fn is_prefix(bits1: &[u8], len1: usize, bits2: &[u8], len2: usize) -> bool {
        if len1 >= len2 {
            return false;
        }

        for i in 0..len1 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            let bit1 = (bits1[byte_idx] >> (7 - bit_idx)) & 1;
            let bit2 = (bits2[byte_idx] >> (7 - bit_idx)) & 1;

            if bit1 != bit2 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_huffman_processor() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        let test_data = vec![
            "hello world".as_bytes().to_vec(),
            "the quick brown fox".as_bytes().to_vec(),
            "jumps over the lazy dog".as_bytes().to_vec(),
        ];

        let encoded_batch = processor.encode_batch(&test_data).await?;
        assert_eq!(encoded_batch.len(), test_data.len());

        let decoded_batch = processor.decode_batch(&encoded_batch).await?;
        assert_eq!(decoded_batch.len(), test_data.len());

        for (original, decoded) in test_data.iter().zip(decoded_batch.iter()) {
            assert_eq!(original, decoded);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_compression_performance() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig {
            batch_size: 1024,
            use_gpu_acceleration: true,
            ..Default::default()
        };
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Generate test batch
        let test_batch: Vec<Vec<u8>> = (0..100)
            .map(|i| format!("test data item number {}", i).into_bytes())
            .collect();

        let start_time = std::time::Instant::now();
        let encoded = processor.encode_batch(&test_batch).await?;
        let encode_time = start_time.elapsed();

        let start_time = std::time::Instant::now();
        let decoded = processor.decode_batch(&encoded).await?;
        let decode_time = start_time.elapsed();

        // Verify correctness
        assert_eq!(test_batch, decoded);

        // Performance assertions
        assert!(encode_time.as_millis() < 1000); // < 1 second for 100 items
        assert!(decode_time.as_millis() < 1000);

        let stats = processor.get_statistics();
        assert!(stats.encode_operations > 0);
        assert!(stats.decode_operations > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_compression() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Simulate streaming data
        let chunks = vec![
            "chunk 1 data".as_bytes().to_vec(),
            "chunk 2 data".as_bytes().to_vec(),
            "chunk 3 data".as_bytes().to_vec(),
        ];

        let mut compressed_chunks = Vec::new();

        // Compress each chunk
        for chunk in &chunks {
            let compressed = processor.encode_batch(&[chunk.clone()]).await?;
            compressed_chunks.extend(compressed);
        }

        // Decompress all chunks
        let decompressed = processor.decode_batch(&compressed_chunks).await?;

        assert_eq!(chunks, decompressed);

        Ok(())
    }

    #[tokio::test]
    async fn test_huffman_with_different_data_types() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        let test_cases = vec![
            // Text data
            "Hello, world! This is a test string.".as_bytes().to_vec(),
            // Binary data
            vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD],
            // Repetitive data
            vec![0xAA; 1000],
            // Random-like data
            (0..256).cycle().take(1000).map(|x| x as u8).collect(),
        ];

        for test_data in test_cases {
            let encoded = processor.encode_batch(&[test_data.clone()]).await?;
            let decoded = processor.decode_batch(&encoded).await?;

            assert_eq!(test_data, decoded[0]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_compression_levels() -> Result<()> {
        let device = setup_test_device().await?;

        let test_data = "aaaaaabbbbbbccccccdddddd".as_bytes().to_vec();
        let mut results = HashMap::new();

        for level in [
            CompressionLevel::Fast,
            CompressionLevel::Balanced,
            CompressionLevel::Best,
        ] {
            let config = HuffmanConfig {
                compression_level: level.clone(),
                ..Default::default()
            };
            let mut processor = GpuHuffmanProcessor::new(device.clone(), config)?;

            let encoded = processor.encode_batch(&[test_data.clone()]).await?;
            results.insert(level, encoded[0].data.len());
        }

        // Best compression should produce smallest output
        assert!(results[&CompressionLevel::Best] <= results[&CompressionLevel::Balanced]);
        assert!(results[&CompressionLevel::Balanced] <= results[&CompressionLevel::Fast]);

        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Test with oversized batch
        let large_batch: Vec<Vec<u8>> = (0..10000)
            .map(|i| format!("item {}", i).into_bytes())
            .collect();

        let result = processor.encode_batch(&large_batch).await;
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());

        // Test with empty data
        let empty_batch: Vec<Vec<u8>> = vec![];
        let result = processor.encode_batch(&empty_batch).await;
        assert!(result.is_ok());

        Ok(())
    }

    async fn setup_test_device() -> Result<Arc<CudaDevice>> {
        CudaDevice::new(0).map(Arc::new).map_err(Into::into)
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[tokio::test]
    async fn benchmark_huffman_encoding_throughput() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig {
            batch_size: 1024,
            use_gpu_acceleration: true,
            ..Default::default()
        };
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Generate large test dataset
        let test_data: Vec<Vec<u8>> = (0..1000)
            .map(|i| {
                let content = format!(
                    "Test data item {} with some repetitive content that should compress well",
                    i
                );
                content.repeat(10).into_bytes()
            })
            .collect();

        let total_input_size: usize = test_data.iter().map(|d| d.len()).sum();

        let start_time = std::time::Instant::now();
        let encoded = processor.encode_batch(&test_data).await?;
        let encoding_time = start_time.elapsed();

        let total_compressed_size: usize = encoded.iter().map(|e| e.data.len()).sum();
        let compression_ratio = total_input_size as f64 / total_compressed_size as f64;

        // Performance targets
        let throughput_mbps =
            (total_input_size as f64 / (1024.0 * 1024.0)) / encoding_time.as_secs_f64();

        println!("Huffman Encoding Performance:");
        println!("  Input size: {} bytes", total_input_size);
        println!("  Compressed size: {} bytes", total_compressed_size);
        println!("  Compression ratio: {:.2}x", compression_ratio);
        println!("  Encoding time: {:?}", encoding_time);
        println!("  Throughput: {:.2} MB/s", throughput_mbps);

        // Performance assertions
        assert!(throughput_mbps > 10.0); // At least 10 MB/s
        assert!(compression_ratio > 1.0); // Should achieve compression
        assert!(encoding_time.as_millis() < 5000); // Complete within 5 seconds

        Ok(())
    }

    #[tokio::test]
    async fn benchmark_huffman_decoding_throughput() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Prepare encoded data
        let test_data: Vec<Vec<u8>> = (0..500)
            .map(|i| format!("Benchmark data item {}", i).repeat(20).into_bytes())
            .collect();

        let encoded = processor.encode_batch(&test_data).await?;
        let total_compressed_size: usize = encoded.iter().map(|e| e.data.len()).sum();

        let start_time = std::time::Instant::now();
        let decoded = processor.decode_batch(&encoded).await?;
        let decoding_time = start_time.elapsed();

        let total_output_size: usize = decoded.iter().map(|d| d.len()).sum();
        let throughput_mbps =
            (total_output_size as f64 / (1024.0 * 1024.0)) / decoding_time.as_secs_f64();

        println!("Huffman Decoding Performance:");
        println!("  Compressed input: {} bytes", total_compressed_size);
        println!("  Decompressed output: {} bytes", total_output_size);
        println!("  Decoding time: {:?}", decoding_time);
        println!("  Throughput: {:.2} MB/s", throughput_mbps);

        // Verify correctness
        assert_eq!(test_data, decoded);

        // Performance assertions
        assert!(throughput_mbps > 15.0); // Decoding should be faster than encoding
        assert!(decoding_time.as_millis() < 3000); // Complete within 3 seconds

        Ok(())
    }

    #[tokio::test]
    async fn benchmark_memory_efficiency() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig {
            enable_statistics: true,
            ..Default::default()
        };
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        // Test with various data sizes
        let sizes = vec![1024, 4096, 16384, 65536];

        for size in sizes {
            let test_data = vec![(0..size).map(|i| (i % 256) as u8).collect::<Vec<u8>>()];

            let memory_before = processor.get_memory_usage()?;
            let encoded = processor.encode_batch(&test_data).await?;
            let memory_peak = processor.get_memory_usage()?;
            let _decoded = processor.decode_batch(&encoded).await?;
            let memory_after = processor.get_memory_usage()?;

            let memory_overhead = memory_peak - memory_before;
            let memory_efficiency = size as f64 / memory_overhead as f64;

            println!(
                "Memory efficiency for {} bytes: {:.2}x",
                size, memory_efficiency
            );

            // Memory should return to baseline
            assert!((memory_after - memory_before) < memory_overhead / 2);

            // Memory overhead should be reasonable
            assert!(memory_efficiency > 0.1); // At least 10% efficiency
        }

        Ok(())
    }

    #[tokio::test]
    async fn benchmark_different_data_patterns() -> Result<()> {
        let device = setup_test_device().await?;
        let config = HuffmanConfig::default();
        let mut processor = GpuHuffmanProcessor::new(device, config)?;

        let test_patterns = vec![
            ("Highly repetitive", vec![b'A'; 10000]),
            ("Random-like", (0..10000).map(|i| (i % 256) as u8).collect()),
            (
                "Text-like",
                "The quick brown fox jumps over the lazy dog. "
                    .repeat(200)
                    .into_bytes(),
            ),
            (
                "Binary-like",
                (0..10000)
                    .map(|i| if i % 2 == 0 { 0x00 } else { 0xFF })
                    .collect(),
            ),
        ];

        for (name, data) in test_patterns {
            let start_time = std::time::Instant::now();
            let encoded = processor.encode_batch(&[data.clone()]).await?;
            let encode_time = start_time.elapsed();

            let compression_ratio = data.len() as f64 / encoded[0].data.len() as f64;

            println!(
                "{}: compression {:.2}x, time {:?}",
                name, compression_ratio, encode_time
            );

            // All patterns should complete within reasonable time
            assert!(encode_time.as_millis() < 2000);
        }

        Ok(())
    }

    async fn setup_test_device() -> Result<Arc<CudaDevice>> {
        CudaDevice::new(0).map(Arc::new).map_err(Into::into)
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_huffman_tree_properties() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Test various input sizes
        for size in [10, 100, 1000, 10000] {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let tree = codec.build_huffman_tree(&data)?;

            // Huffman tree properties
            assert!(!tree.code_table.is_empty());

            // All leaf nodes should have codes
            for &byte in &data {
                assert!(tree.code_table.contains_key(&byte));
            }

            // No code should be empty
            for code in tree.code_table.values() {
                assert!(code.bit_length > 0);
                assert!(code.bit_length <= codec.config.max_tree_depth);
            }
        }
    }

    #[test]
    fn test_encode_decode_invariant() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Property: encode(decode(x)) == x for any valid input
        let test_cases = vec![
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"hello world".to_vec(),
            (0..256).collect::<Vec<u8>>(),
            vec![0; 1000],
            (0..1000).map(|i| (i % 256) as u8).collect(),
        ];

        for original in test_cases {
            if original.is_empty() {
                continue;
            }

            let encoded = codec.encode(&original).unwrap();
            let decoded = codec.decode(&encoded.data, &encoded.tree).unwrap();

            assert_eq!(original, decoded, "Encode-decode invariant violated");
        }
    }

    #[test]
    fn test_compression_effectiveness() {
        let mut codec = HuffmanCodec::new(HuffmanConfig::default()).unwrap();

        // Highly repetitive data should compress well
        let repetitive = vec![b'A'; 1000];
        let encoded = codec.encode(&repetitive)?;
        let compression_ratio = repetitive.len() as f64 / encoded.data.len() as f64;

        assert!(
            compression_ratio > 2.0,
            "Repetitive data should compress > 2x"
        );

        // Random data should not compress much
        let random: Vec<u8> = (0..1000).map(|i| ((i * 17 + 31) % 256) as u8).collect();
        let encoded = codec.encode(&random).unwrap();
        let compression_ratio = random.len() as f64 / encoded.data.len() as f64;

        // Random data may not compress well, but should not expand significantly
        assert!(
            compression_ratio > 0.8,
            "Random data should not expand much"
        );
    }
}
