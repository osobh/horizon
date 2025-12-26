//! Consensus Message Compression Test Suite
//!
//! This module tests advanced message compression techniques for achieving
//! high compression ratios while maintaining consensus protocol integrity.
//! Tests cover real-world scenarios with millions of nodes and various
//! message patterns.
//!
//! ALL TESTS IN THIS FILE ARE DESIGNED TO FAIL INITIALLY (RED PHASE)
//! They represent compression functionality that needs to be implemented.

use stratoswarm_consensus::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Message compression configuration for testing
#[derive(Debug, Clone)]
struct CompressionConfig {
    /// Target compression ratio (0.0 = no compression, 1.0 = perfect compression)
    target_ratio: f32,
    /// Compression algorithm to use
    algorithm: CompressionAlgorithm,
    /// Maximum compression latency allowed
    max_compression_latency: Duration,
    /// Enable adaptive compression based on message patterns
    adaptive_compression: bool,
    /// Batch size for bulk compression
    batch_size: usize,
}

/// Supported compression algorithms for consensus messages
#[derive(Debug, Clone)]
enum CompressionAlgorithm {
    /// LZ4 for speed-optimized compression
    LZ4,
    /// ZSTD for high-ratio compression
    ZSTD,
    /// Custom consensus-aware compression
    ConsensusAware,
    /// GPU-accelerated compression
    GpuAccelerated,
    /// Adaptive algorithm selection
    Adaptive,
}

/// Compressed message wrapper
#[derive(Debug, Clone)]
struct CompressedMessage {
    /// Original message type identifier
    message_type: u8,
    /// Compressed payload
    compressed_data: Vec<u8>,
    /// Original size before compression
    original_size: usize,
    /// Compression algorithm used
    algorithm: CompressionAlgorithm,
    /// Compression timestamp
    compressed_at: u64,
    /// Checksum for integrity verification
    checksum: u32,
}

/// Compression statistics for analysis
#[derive(Debug, Clone)]
struct CompressionStats {
    /// Achieved compression ratio
    compression_ratio: f32,
    /// Compression latency
    compression_time: Duration,
    /// Decompression latency
    decompression_time: Duration,
    /// Messages processed
    messages_processed: usize,
    /// Bytes saved through compression
    bytes_saved: usize,
    /// Compression errors encountered
    errors_encountered: usize,
    /// GPU utilization during compression
    gpu_utilization: f32,
}

/// Message compression engine for consensus protocols
struct ConsensusCompressor {
    config: CompressionConfig,
    stats: CompressionStats,
    compression_cache: HashMap<String, Vec<u8>>,
    pattern_analyzer: MessagePatternAnalyzer,
    gpu_compressor: Option<GpuCompressor>,
}

impl ConsensusCompressor {
    fn new(config: CompressionConfig) -> Result<Self, ConsensusError> {
        // This will fail - compression engine not implemented
        Err(ConsensusError::NotImplemented(
            "ConsensusCompressor not yet implemented".to_string()
        ))
    }

    async fn compress_message(
        &mut self, 
        message: &ConsensusMessage
    ) -> Result<CompressedMessage, ConsensusError> {
        // This will fail - message compression not implemented
        Err(ConsensusError::NotImplemented(
            "Message compression not yet implemented".to_string()
        ))
    }

    async fn decompress_message(
        &mut self, 
        compressed: &CompressedMessage
    ) -> Result<ConsensusMessage, ConsensusError> {
        // This will fail - message decompression not implemented
        Err(ConsensusError::NotImplemented(
            "Message decompression not yet implemented".to_string()
        ))
    }

    async fn compress_batch(
        &mut self, 
        messages: &[ConsensusMessage]
    ) -> Result<Vec<CompressedMessage>, ConsensusError> {
        // This will fail - batch compression not implemented
        Err(ConsensusError::NotImplemented(
            "Batch compression not yet implemented".to_string()
        ))
    }

    fn get_stats(&self) -> &CompressionStats {
        &self.stats
    }

    async fn optimize_for_pattern(&mut self, pattern: MessagePattern) -> Result<(), ConsensusError> {
        // This will fail - pattern optimization not implemented
        Err(ConsensusError::NotImplemented(
            "Pattern optimization not yet implemented".to_string()
        ))
    }
}

/// Message pattern analyzer for adaptive compression
struct MessagePatternAnalyzer {
    patterns: HashMap<String, PatternStats>,
    current_pattern: Option<MessagePattern>,
}

#[derive(Debug, Clone)]
struct PatternStats {
    frequency: usize,
    average_size: usize,
    compression_efficiency: f32,
}

#[derive(Debug, Clone)]
enum MessagePattern {
    VotingRound,
    GpuComputation,
    HeartbeatBurst,
    ElectionCampaign,
    StateSync,
    Mixed,
}

/// GPU-accelerated compression engine
struct GpuCompressor {
    device_id: u32,
    compression_kernels: Vec<String>,
    memory_pool: Option<Vec<u8>>,
}

impl GpuCompressor {
    fn new(device_id: u32) -> Result<Self, ConsensusError> {
        // This will fail - GPU compression not implemented
        Err(ConsensusError::NotImplemented(
            "GPU compressor not yet implemented".to_string()
        ))
    }

    async fn compress_gpu(
        &mut self, 
        data: &[u8]
    ) -> Result<Vec<u8>, ConsensusError> {
        // This will fail - GPU compression not implemented
        Err(ConsensusError::NotImplemented(
            "GPU compression not yet implemented".to_string()
        ))
    }
}

// ===== FAILING TESTS (RED PHASE) =====

#[tokio::test]
async fn test_high_ratio_lz4_compression() {
    let config = CompressionConfig {
        target_ratio: 0.85, // 85% compression target
        algorithm: CompressionAlgorithm::LZ4,
        max_compression_latency: Duration::from_micros(100),
        adaptive_compression: false,
        batch_size: 1000,
    };

    // This test WILL FAIL - compression engine not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create compressor");

    // Create a large consensus message with redundant data
    let validator_id = ValidatorId::new();
    let large_signature = vec![42u8; 2048]; // 2KB signature
    let large_value = "consensus_value_".repeat(1000); // ~15KB value

    let message = ConsensusMessage::Proposal {
        round_id: RoundId::new(),
        height: 1000000,
        proposer_id: validator_id,
        value: large_value.clone(),
        signature: large_signature,
    };

    let original_size = bincode::serialize(&message).unwrap().len();
    
    let start_time = Instant::now();
    let compressed = compressor.compress_message(&message).await
        .expect("Compression should succeed");
    let compression_time = start_time.elapsed();

    // Compression ratio verification
    let compression_ratio = 1.0 - (compressed.compressed_data.len() as f32 / original_size as f32);
    assert!(compression_ratio >= config.target_ratio,
           "Compression ratio should meet target: {} >= {}", 
           compression_ratio, config.target_ratio);

    // Latency verification
    assert!(compression_time <= config.max_compression_latency,
           "Compression should be fast: {:?} <= {:?}", 
           compression_time, config.max_compression_latency);

    // Verify decompression integrity
    let decompressed = compressor.decompress_message(&compressed).await
        .expect("Decompression should succeed");
    
    match (message, decompressed) {
        (
            ConsensusMessage::Proposal { value: v1, height: h1, .. },
            ConsensusMessage::Proposal { value: v2, height: h2, .. }
        ) => {
            assert_eq!(v1, v2, "Decompressed value should match original");
            assert_eq!(h1, h2, "Decompressed height should match original");
        }
        _ => panic!("Decompressed message type should match original"),
    }
}

#[tokio::test]
async fn test_zstd_maximum_compression() {
    let config = CompressionConfig {
        target_ratio: 0.95, // Aggressive 95% compression target
        algorithm: CompressionAlgorithm::ZSTD,
        max_compression_latency: Duration::from_millis(1), // Allow more time for high compression
        adaptive_compression: false,
        batch_size: 100,
    };

    // This test WILL FAIL - ZSTD compression not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create ZSTD compressor");

    // Create multiple similar messages (high redundancy scenario)
    let validator_id = ValidatorId::new();
    let mut messages = Vec::new();
    
    for i in 0..1000 {
        let message = ConsensusMessage::Vote(Vote::new(
            RoundId::new(),
            VoteType::PreVote,
            validator_id.clone(),
            Some(format!("block_hash_{}", i % 10)), // Only 10 unique hashes
            vec![1, 2, 3, 4, 5, 6, 7, 8], // Repeated signature pattern
        ));
        messages.push(message);
    }

    let original_total_size: usize = messages.iter()
        .map(|m| bincode::serialize(m).unwrap().len())
        .sum();

    // Test batch compression
    let compressed_batch = compressor.compress_batch(&messages).await
        .expect("Batch compression should succeed");

    let compressed_total_size: usize = compressed_batch.iter()
        .map(|c| c.compressed_data.len())
        .sum();

    let batch_compression_ratio = 1.0 - (compressed_total_size as f32 / original_total_size as f32);
    
    // High compression ratio should be achievable with redundant data
    assert!(batch_compression_ratio >= config.target_ratio,
           "Batch compression should achieve high ratio: {} >= {}", 
           batch_compression_ratio, config.target_ratio);

    // Verify all messages can be decompressed correctly
    for (i, compressed) in compressed_batch.iter().enumerate() {
        let decompressed = compressor.decompress_message(compressed).await
            .expect(&format!("Decompression should succeed for message {}", i));
        
        // Verify message integrity
        assert!(matches!(decompressed, ConsensusMessage::Vote(_)),
               "Decompressed message should be a vote");
    }

    let stats = compressor.get_stats();
    assert_eq!(stats.messages_processed, messages.len(),
              "Should process all messages");
    assert_eq!(stats.errors_encountered, 0,
              "Should have no compression errors");
}

#[tokio::test]
async fn test_gpu_accelerated_compression() {
    let config = CompressionConfig {
        target_ratio: 0.90,
        algorithm: CompressionAlgorithm::GpuAccelerated,
        max_compression_latency: Duration::from_micros(50), // Very fast GPU compression
        adaptive_compression: false,
        batch_size: 10000, // Large batch for GPU efficiency
    };

    // This test WILL FAIL - GPU compression not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create GPU compressor");

    // Create GPU computation messages (typical for consensus workload)
    let mut gpu_messages = Vec::new();
    let validator_id = ValidatorId::new();
    
    for i in 0..10000 {
        let message = ConsensusMessage::GpuCompute {
            task_id: format!("gpu_task_{}", i),
            computation_type: "matrix_multiply".to_string(), // Repeated type
            data: vec![i as u8; 1024], // 1KB data per task
            priority: (i % 5) as u8, // 5 priority levels
        };
        gpu_messages.push(message);
    }

    let start_time = Instant::now();
    let compressed_batch = compressor.compress_batch(&gpu_messages).await
        .expect("GPU batch compression should succeed");
    let gpu_compression_time = start_time.elapsed();

    // GPU compression should be very fast for large batches
    assert!(gpu_compression_time < config.max_compression_latency * gpu_messages.len() as u32,
           "GPU compression should be faster than sequential: {:?}", gpu_compression_time);

    let original_size: usize = gpu_messages.iter()
        .map(|m| bincode::serialize(m).unwrap().len())
        .sum();
    
    let compressed_size: usize = compressed_batch.iter()
        .map(|c| c.compressed_data.len())
        .sum();

    let gpu_compression_ratio = 1.0 - (compressed_size as f32 / original_size as f32);
    
    assert!(gpu_compression_ratio >= config.target_ratio,
           "GPU compression should achieve target ratio: {} >= {}", 
           gpu_compression_ratio, config.target_ratio);

    let stats = compressor.get_stats();
    assert!(stats.gpu_utilization > 80.0,
           "GPU should be well utilized: {}%", stats.gpu_utilization);
    assert!(stats.compression_time < Duration::from_millis(10),
           "GPU compression should be very fast: {:?}", stats.compression_time);
}

#[tokio::test]
async fn test_adaptive_compression_patterns() {
    let config = CompressionConfig {
        target_ratio: 0.80,
        algorithm: CompressionAlgorithm::Adaptive,
        max_compression_latency: Duration::from_micros(200),
        adaptive_compression: true,
        batch_size: 5000,
    };

    // This test WILL FAIL - adaptive compression not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create adaptive compressor");

    // Test different message patterns
    let patterns = vec![
        (MessagePattern::VotingRound, create_voting_messages(1000)),
        (MessagePattern::GpuComputation, create_gpu_compute_messages(1000)),
        (MessagePattern::HeartbeatBurst, create_heartbeat_messages(1000)),
        (MessagePattern::StateSync, create_sync_messages(1000)),
    ];

    for (pattern, messages) in patterns {
        // Optimize compressor for current pattern
        compressor.optimize_for_pattern(pattern.clone()).await
            .expect("Pattern optimization should succeed");

        let original_size: usize = messages.iter()
            .map(|m| bincode::serialize(m).unwrap().len())
            .sum();

        let start_time = Instant::now();
        let compressed_batch = compressor.compress_batch(&messages).await
            .expect("Adaptive compression should succeed");
        let pattern_compression_time = start_time.elapsed();

        let compressed_size: usize = compressed_batch.iter()
            .map(|c| c.compressed_data.len())
            .sum();

        let pattern_compression_ratio = 1.0 - (compressed_size as f32 / original_size as f32);

        // Adaptive compression should optimize for each pattern
        assert!(pattern_compression_ratio >= config.target_ratio,
               "Pattern {:?} compression should meet target: {} >= {}", 
               pattern, pattern_compression_ratio, config.target_ratio);

        assert!(pattern_compression_time < config.max_compression_latency * messages.len() as u32,
               "Pattern {:?} compression should be fast: {:?}", pattern, pattern_compression_time);

        // Verify all messages decompress correctly
        for compressed in &compressed_batch {
            let decompressed = compressor.decompress_message(compressed).await
                .expect("Pattern-optimized decompression should succeed");
            
            // Basic integrity check
            assert!(matches!(
                decompressed,
                ConsensusMessage::Vote(_) | 
                ConsensusMessage::GpuCompute { .. } | 
                ConsensusMessage::Heartbeat { .. } |
                ConsensusMessage::SyncRequest { .. }
            ), "Decompressed message should have correct type");
        }
    }

    let final_stats = compressor.get_stats();
    assert!(final_stats.messages_processed >= 4000,
           "Should process all pattern messages: {}", final_stats.messages_processed);
    assert!(final_stats.compression_ratio >= 0.75,
           "Overall adaptive compression should be effective: {}", final_stats.compression_ratio);
}

#[tokio::test]
async fn test_consensus_aware_compression() {
    let config = CompressionConfig {
        target_ratio: 0.88,
        algorithm: CompressionAlgorithm::ConsensusAware,
        max_compression_latency: Duration::from_micros(150),
        adaptive_compression: true,
        batch_size: 2000,
    };

    // This test WILL FAIL - consensus-aware compression not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create consensus-aware compressor");

    // Create messages with consensus-specific patterns
    let validator_ids: Vec<ValidatorId> = (0..100).map(|_| ValidatorId::new()).collect();
    let round_ids: Vec<RoundId> = (0..10).map(|_| RoundId::new()).collect();
    
    let mut consensus_messages = Vec::new();
    
    // Create realistic consensus message patterns
    for round_id in &round_ids {
        for validator_id in &validator_ids {
            // Proposals (few per round)
            if validator_ids.iter().position(|v| v == validator_id).unwrap() < 5 {
                consensus_messages.push(ConsensusMessage::Proposal {
                    round_id: round_id.clone(),
                    height: 1000 + (round_ids.iter().position(|r| r == round_id).unwrap() as u64),
                    proposer_id: validator_id.clone(),
                    value: format!("block_data_round_{}", round_ids.iter().position(|r| r == round_id).unwrap()),
                    signature: vec![1, 2, 3, 4, 5], // Common signature pattern
                });
            }
            
            // Votes (many per round)
            consensus_messages.push(ConsensusMessage::Vote(Vote::new(
                round_id.clone(),
                VoteType::PreVote,
                validator_id.clone(),
                Some("block_hash".to_string()), // Same hash for same round
                vec![10, 20, 30, 40], // Common signature pattern
            )));
            
            consensus_messages.push(ConsensusMessage::Vote(Vote::new(
                round_id.clone(),
                VoteType::PreCommit,
                validator_id.clone(),
                Some("block_hash".to_string()),
                vec![10, 20, 30, 40],
            )));
        }
    }

    let original_size: usize = consensus_messages.iter()
        .map(|m| bincode::serialize(m).unwrap().len())
        .sum();

    // Consensus-aware compression should exploit protocol structure
    let compressed_batch = compressor.compress_batch(&consensus_messages).await
        .expect("Consensus-aware compression should succeed");

    let compressed_size: usize = compressed_batch.iter()
        .map(|c| c.compressed_data.len())
        .sum();

    let consensus_compression_ratio = 1.0 - (compressed_size as f32 / original_size as f32);
    
    // Should achieve high compression due to consensus protocol patterns
    assert!(consensus_compression_ratio >= config.target_ratio,
           "Consensus-aware compression should be highly effective: {} >= {}", 
           consensus_compression_ratio, config.target_ratio);

    // Verify consensus semantics are preserved
    let mut decompressed_proposals = 0;
    let mut decompressed_votes = 0;
    
    for compressed in &compressed_batch {
        let decompressed = compressor.decompress_message(compressed).await
            .expect("Consensus-aware decompression should succeed");
        
        match decompressed {
            ConsensusMessage::Proposal { .. } => decompressed_proposals += 1,
            ConsensusMessage::Vote(_) => decompressed_votes += 1,
            _ => panic!("Unexpected message type in consensus batch"),
        }
    }
    
    // Should preserve all consensus messages
    assert_eq!(decompressed_proposals + decompressed_votes, consensus_messages.len(),
              "All consensus messages should be preserved");
    
    let stats = compressor.get_stats();
    assert!(stats.bytes_saved > (original_size as f32 * config.target_ratio) as usize,
           "Should save significant bytes: {} > {}", 
           stats.bytes_saved, (original_size as f32 * config.target_ratio) as usize);
}

#[tokio::test]
async fn test_million_message_compression_stress() {
    let config = CompressionConfig {
        target_ratio: 0.85,
        algorithm: CompressionAlgorithm::Adaptive,
        max_compression_latency: Duration::from_micros(100),
        adaptive_compression: true,
        batch_size: 50000, // Large batches for efficiency
    };

    // This test WILL FAIL - large-scale compression not implemented
    let mut compressor = ConsensusCompressor::new(config.clone())
        .expect("Failed to create stress test compressor");

    let million_messages = 1_000_000;
    let batch_count = million_messages / config.batch_size;
    
    let mut total_original_size = 0usize;
    let mut total_compressed_size = 0usize;
    let mut total_compression_time = Duration::from_nanos(0);
    
    for batch_idx in 0..batch_count {
        // Create mixed message batch
        let batch = create_mixed_message_batch(config.batch_size, batch_idx);
        
        let batch_original_size: usize = batch.iter()
            .map(|m| bincode::serialize(m).unwrap().len())
            .sum();
        total_original_size += batch_original_size;
        
        let start_time = Instant::now();
        let compressed_batch = compressor.compress_batch(&batch).await
            .expect(&format!("Batch {} compression should succeed", batch_idx));
        let batch_compression_time = start_time.elapsed();
        
        total_compression_time += batch_compression_time;
        
        let batch_compressed_size: usize = compressed_batch.iter()
            .map(|c| c.compressed_data.len())
            .sum();
        total_compressed_size += batch_compressed_size;
        
        // Verify compression effectiveness per batch
        let batch_ratio = 1.0 - (batch_compressed_size as f32 / batch_original_size as f32);
        assert!(batch_ratio >= 0.75,
               "Each batch should achieve reasonable compression: {} >= 0.75", batch_ratio);
        
        // Spot check decompression (not all messages for performance)
        if batch_idx % 10 == 0 {
            let sample_decompressed = compressor.decompress_message(&compressed_batch[0]).await
                .expect("Sample decompression should succeed");
            assert!(matches!(sample_decompressed, ConsensusMessage::Vote(_) | 
                           ConsensusMessage::Proposal { .. } | 
                           ConsensusMessage::GpuCompute { .. }));
        }
    }
    
    let overall_compression_ratio = 1.0 - (total_compressed_size as f32 / total_original_size as f32);
    let average_batch_time = total_compression_time / batch_count as u32;
    
    // Million message performance requirements
    assert!(overall_compression_ratio >= config.target_ratio,
           "Million message compression should meet target: {} >= {}", 
           overall_compression_ratio, config.target_ratio);
    
    assert!(average_batch_time < Duration::from_millis(100),
           "Average batch compression should be fast: {:?}", average_batch_time);
    
    let final_stats = compressor.get_stats();
    assert_eq!(final_stats.messages_processed, million_messages,
              "Should process all million messages");
    
    // Memory efficiency check
    assert!(final_stats.bytes_saved > (total_original_size / 2),
           "Should save significant storage: {} > {}", 
           final_stats.bytes_saved, total_original_size / 2);
}

// Helper functions for test message creation

fn create_voting_messages(count: usize) -> Vec<ConsensusMessage> {
    let mut messages = Vec::with_capacity(count);
    let validator_id = ValidatorId::new();
    let round_id = RoundId::new();
    
    for i in 0..count {
        let vote_type = if i % 2 == 0 { VoteType::PreVote } else { VoteType::PreCommit };
        messages.push(ConsensusMessage::Vote(Vote::new(
            round_id.clone(),
            vote_type,
            validator_id.clone(),
            Some("block_hash".to_string()),
            vec![1, 2, 3, 4],
        )));
    }
    
    messages
}

fn create_gpu_compute_messages(count: usize) -> Vec<ConsensusMessage> {
    let mut messages = Vec::with_capacity(count);
    
    for i in 0..count {
        messages.push(ConsensusMessage::GpuCompute {
            task_id: format!("task_{}", i),
            computation_type: "neural_network".to_string(),
            data: vec![i as u8; 512],
            priority: (i % 3) as u8,
        });
    }
    
    messages
}

fn create_heartbeat_messages(count: usize) -> Vec<ConsensusMessage> {
    let mut messages = Vec::with_capacity(count);
    let validator_id = ValidatorId::new();
    
    for i in 0..count {
        messages.push(ConsensusMessage::Heartbeat {
            validator_id: validator_id.clone(),
            timestamp: 1000000 + i as u64,
            gpu_status: GpuStatus {
                available_memory: 8000 - (i % 1000) as u64,
                utilization: 50 + (i % 50) as u8,
                active_kernels: (i % 10) as u32,
                temperature: 65 + (i % 20) as u16,
            },
        });
    }
    
    messages
}

fn create_sync_messages(count: usize) -> Vec<ConsensusMessage> {
    let mut messages = Vec::with_capacity(count);
    let validator_id = ValidatorId::new();
    
    for i in 0..count {
        if i % 2 == 0 {
            messages.push(ConsensusMessage::SyncRequest {
                from_height: i as u64,
                to_height: (i + 100) as u64,
                requester_id: validator_id.clone(),
            });
        } else {
            messages.push(ConsensusMessage::SyncResponse {
                height: i as u64,
                state_data: vec![42u8; 256],
                proof: vec![1, 2, 3, 4],
            });
        }
    }
    
    messages
}

fn create_mixed_message_batch(count: usize, batch_idx: usize) -> Vec<ConsensusMessage> {
    let mut messages = Vec::with_capacity(count);
    
    let vote_count = count * 40 / 100;      // 40% votes
    let compute_count = count * 30 / 100;   // 30% GPU compute
    let heartbeat_count = count * 20 / 100; // 20% heartbeats
    let other_count = count - vote_count - compute_count - heartbeat_count; // 10% other
    
    messages.extend(create_voting_messages(vote_count));
    messages.extend(create_gpu_compute_messages(compute_count));
    messages.extend(create_heartbeat_messages(heartbeat_count));
    messages.extend(create_sync_messages(other_count));
    
    // Add some proposals
    let validator_id = ValidatorId::new();
    for i in 0..5 {
        messages.push(ConsensusMessage::Proposal {
            round_id: RoundId::new(),
            height: (batch_idx * 1000 + i) as u64,
            proposer_id: validator_id.clone(),
            value: format!("batch_{}_proposal_{}", batch_idx, i),
            signature: vec![i as u8; 64],
        });
    }
    
    messages
}

// Extension for missing functionality
impl ConsensusError {
    fn NotImplemented(msg: String) -> Self {
        ConsensusError::ValidationFailed(format!("Not implemented: {}", msg))
    }
}