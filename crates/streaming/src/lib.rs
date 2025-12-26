//! High-performance streaming pipeline framework for GPU-native data processing
//!
//! This crate provides the core abstractions and implementations for building
//! efficient streaming data pipelines with zero-copy operations and GPU acceleration.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

pub mod core;
pub mod pipeline;
pub mod processors;
pub mod sinks;
pub mod sources;

// Re-export main types
pub use core::*;
pub use pipeline::*;

/// Streaming errors
#[derive(Error, Debug)]
pub enum StreamingError {
    #[error("IO operation failed: {reason}")]
    IoFailed { reason: String },

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Pipeline configuration invalid: {reason}")]
    InvalidConfig { reason: String },

    #[error("Processing failed: {reason}")]
    ProcessingFailed { reason: String },

    #[error("Resource exhausted: {reason}")]
    ResourceExhausted { reason: String },

    #[error("Timeout occurred: {reason}")]
    Timeout { reason: String },
}

/// Streaming data chunk with metadata
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub data: Bytes,
    pub sequence: u64,
    pub timestamp: u64,
    pub metadata: ChunkMetadata,
}

/// Metadata associated with a stream chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub source_id: String,
    pub chunk_size: usize,
    pub compression: Option<String>,
    pub checksum: Option<u64>,
}

/// Stream statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamStats {
    pub chunks_processed: u64,
    pub bytes_processed: u64,
    pub processing_time_ms: u64,
    pub throughput_mbps: f64,
    pub errors: u64,
}

impl StreamChunk {
    /// Create a new stream chunk
    pub fn new(data: Bytes, sequence: u64, source_id: String) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let metadata = ChunkMetadata {
            source_id,
            chunk_size: data.len(),
            compression: None,
            checksum: None,
        };

        Self {
            data,
            sequence,
            timestamp,
            metadata,
        }
    }

    /// Get the size of the chunk in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if this chunk is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl fmt::Display for StreamChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StreamChunk(seq={}, size={}, source={})",
            self.sequence,
            self.size(),
            self.metadata.source_id
        )
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod edge_case_tests;
