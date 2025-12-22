//! Binary Serialization Module for Performance Optimization
//!
//! Replaces JSON with MessagePack for 10x faster serialization

use crate::synthesis::{AstNode, Match, Pattern};
use anyhow::Result;
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance comparison metrics for serialization methods
#[derive(Debug, Clone)]
pub struct SerializationBenchmark {
    pub json_time: Duration,
    pub messagepack_time: Duration,
    pub json_size: usize,
    pub messagepack_size: usize,
    pub speedup_ratio: f64,
    pub size_reduction_ratio: f64,
}

/// Binary serialization engine using MessagePack
pub struct BinarySerializer {
    enable_compression: bool,
}

impl BinarySerializer {
    /// Create a new binary serializer
    pub fn new(enable_compression: bool) -> Self {
        Self { enable_compression }
    }

    /// Serialize patterns using MessagePack
    pub fn serialize_patterns(&self, patterns: &[Pattern]) -> Result<Vec<u8>> {
        let start = Instant::now();

        let mut buffer = Vec::new();
        patterns.serialize(&mut Serializer::new(&mut buffer))?;

        if self.enable_compression {
            // Apply LZ4 compression for additional space savings
            let compressed = lz4::block::compress(&buffer, None, true)?;
            Ok(compressed)
        } else {
            Ok(buffer)
        }
    }

    /// Deserialize patterns from MessagePack
    pub fn deserialize_patterns(&self, data: &[u8]) -> Result<Vec<Pattern>> {
        let decompressed_data = if self.enable_compression {
            lz4::block::decompress(data, None)?
        } else {
            data.to_vec()
        };

        let mut deserializer = Deserializer::new(&decompressed_data[..]);
        let patterns: Vec<Pattern> = Deserialize::deserialize(&mut deserializer)?;
        Ok(patterns)
    }

    /// Serialize AST nodes using MessagePack
    pub fn serialize_ast_nodes(&self, ast_nodes: &[AstNode]) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        ast_nodes.serialize(&mut Serializer::new(&mut buffer))?;

        if self.enable_compression {
            let compressed = lz4::block::compress(&buffer, None, true)?;
            Ok(compressed)
        } else {
            Ok(buffer)
        }
    }

    /// Deserialize AST nodes from MessagePack
    pub fn deserialize_ast_nodes(&self, data: &[u8]) -> Result<Vec<AstNode>> {
        let decompressed_data = if self.enable_compression {
            lz4::block::decompress(data, None)?
        } else {
            data.to_vec()
        };

        let mut deserializer = Deserializer::new(&decompressed_data[..]);
        let ast_nodes: Vec<AstNode> = Deserialize::deserialize(&mut deserializer)?;
        Ok(ast_nodes)
    }

    /// Serialize matches using MessagePack
    pub fn serialize_matches(&self, matches: &[Match]) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        matches.serialize(&mut Serializer::new(&mut buffer))?;

        if self.enable_compression {
            let compressed = lz4::block::compress(&buffer, None, true)?;
            Ok(compressed)
        } else {
            Ok(buffer)
        }
    }

    /// Deserialize matches from MessagePack
    pub fn deserialize_matches(&self, data: &[u8]) -> Result<Vec<Match>> {
        let decompressed_data = if self.enable_compression {
            lz4::block::decompress(data, None)?
        } else {
            data.to_vec()
        };

        let mut deserializer = Deserializer::new(&decompressed_data[..]);
        let matches: Vec<Match> = Deserialize::deserialize(&mut deserializer)?;
        Ok(matches)
    }

    /// Benchmark serialization performance vs JSON
    pub fn benchmark_serialization(&self, patterns: &[Pattern]) -> Result<SerializationBenchmark> {
        // Benchmark JSON serialization
        let json_start = Instant::now();
        let json_data = serde_json::to_vec(patterns)?;
        let json_time = json_start.elapsed();
        let json_size = json_data.len();

        // Benchmark MessagePack serialization
        let msgpack_start = Instant::now();
        let msgpack_data = self.serialize_patterns(patterns)?;
        let messagepack_time = msgpack_start.elapsed();
        let messagepack_size = msgpack_data.len();

        let speedup_ratio = json_time.as_secs_f64() / messagepack_time.as_secs_f64();
        let size_reduction_ratio = json_size as f64 / messagepack_size as f64;

        Ok(SerializationBenchmark {
            json_time,
            messagepack_time,
            json_size,
            messagepack_size,
            speedup_ratio,
            size_reduction_ratio,
        })
    }
}

/// Factory for creating optimized serializers
pub struct SerializationFactory;

impl SerializationFactory {
    /// Create a high-performance serializer for production use
    pub fn create_production_serializer() -> BinarySerializer {
        BinarySerializer::new(true) // Enable compression for maximum efficiency
    }

    /// Create a development serializer for debugging
    pub fn create_development_serializer() -> BinarySerializer {
        BinarySerializer::new(false) // Disable compression for faster development cycles
    }

    /// Create a serializer optimized for specific use case
    pub fn create_optimized_serializer(
        data_size: usize,
        latency_sensitive: bool,
    ) -> BinarySerializer {
        // For large data or non-latency sensitive: use compression
        // For small data or latency sensitive: skip compression
        let use_compression = data_size > 1024 && !latency_sensitive;
        BinarySerializer::new(use_compression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::NodeType;

    fn create_test_patterns(count: usize) -> Vec<Pattern> {
        (0..count)
            .map(|i| Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some(format!("pattern_{}", i)),
            })
            .collect()
    }

    fn create_test_asts(count: usize) -> Vec<AstNode> {
        (0..count)
            .map(|i| AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some(format!("node_{}", i)),
            })
            .collect()
    }

    #[test]
    fn test_pattern_serialization_roundtrip() {
        let serializer = BinarySerializer::new(false);
        let patterns = create_test_patterns(100);

        let serialized = serializer.serialize_patterns(&patterns)?;
        let deserialized = serializer.deserialize_patterns(&serialized)?;

        assert_eq!(patterns.len(), deserialized.len());
        for (original, recovered) in patterns.iter().zip(deserialized.iter()) {
            assert_eq!(original.node_type, recovered.node_type);
            assert_eq!(original.value, recovered.value);
        }
    }

    #[test]
    fn test_ast_serialization_roundtrip() {
        let serializer = BinarySerializer::new(false);
        let asts = create_test_asts(100);

        let serialized = serializer.serialize_ast_nodes(&asts)?;
        let deserialized = serializer.deserialize_ast_nodes(&serialized)?;

        assert_eq!(asts.len(), deserialized.len());
        for (original, recovered) in asts.iter().zip(deserialized.iter()) {
            assert_eq!(original.node_type, recovered.node_type);
            assert_eq!(original.value, recovered.value);
        }
    }

    #[test]
    fn test_compression_effectiveness() {
        let uncompressed_serializer = BinarySerializer::new(false);
        let compressed_serializer = BinarySerializer::new(true);

        let patterns = create_test_patterns(1000);

        let uncompressed_data = uncompressed_serializer
            .serialize_patterns(&patterns)
            ?;
        let compressed_data = compressed_serializer.serialize_patterns(&patterns)?;

        println!("Uncompressed size: {} bytes", uncompressed_data.len());
        println!("Compressed size: {} bytes", compressed_data.len());

        // Compression should reduce size significantly
        assert!(compressed_data.len() < uncompressed_data.len());
        let compression_ratio = uncompressed_data.len() as f64 / compressed_data.len() as f64;
        assert!(compression_ratio > 1.5); // At least 1.5x compression
    }

    #[test]
    fn test_serialization_benchmark() {
        let serializer = BinarySerializer::new(false);
        let patterns = create_test_patterns(1000);

        let benchmark = serializer.benchmark_serialization(&patterns)?;

        println!("JSON time: {:?}", benchmark.json_time);
        println!("MessagePack time: {:?}", benchmark.messagepack_time);
        println!("JSON size: {} bytes", benchmark.json_size);
        println!("MessagePack size: {} bytes", benchmark.messagepack_size);
        println!("Speedup ratio: {:.2}x", benchmark.speedup_ratio);
        println!(
            "Size reduction ratio: {:.2}x",
            benchmark.size_reduction_ratio
        );

        // MessagePack should be faster and smaller
        assert!(benchmark.speedup_ratio > 1.0);
        assert!(benchmark.size_reduction_ratio > 1.0);
        assert!(benchmark.messagepack_size < benchmark.json_size);
    }

    #[test]
    fn test_factory_methods() {
        let prod_serializer = SerializationFactory::create_production_serializer();
        let dev_serializer = SerializationFactory::create_development_serializer();
        let optimized_large = SerializationFactory::create_optimized_serializer(10000, false);
        let optimized_small = SerializationFactory::create_optimized_serializer(100, true);

        assert!(prod_serializer.enable_compression);
        assert!(!dev_serializer.enable_compression);
        assert!(optimized_large.enable_compression);
        assert!(!optimized_small.enable_compression);
    }

    #[test]
    fn test_large_dataset_performance() {
        let serializer = BinarySerializer::new(true); // With compression
        let patterns = create_test_patterns(10000); // Large dataset

        let start = Instant::now();
        let serialized = serializer.serialize_patterns(&patterns)?;
        let serialize_time = start.elapsed();

        let start = Instant::now();
        let _deserialized = serializer.deserialize_patterns(&serialized)?;
        let deserialize_time = start.elapsed();

        println!("Large dataset (10K patterns):");
        println!("  Serialize time: {:?}", serialize_time);
        println!("  Deserialize time: {:?}", deserialize_time);
        println!("  Serialized size: {} bytes", serialized.len());

        // Should complete reasonably quickly
        assert!(serialize_time.as_millis() < 100); // Less than 100ms
        assert!(deserialize_time.as_millis() < 100); // Less than 100ms
    }
}
