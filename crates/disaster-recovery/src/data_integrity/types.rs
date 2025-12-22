//! Core data integrity types and structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Data object being monitored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataObject {
    /// Object ID
    pub id: Uuid,
    /// Object path or identifier
    pub path: String,
    /// Object type
    pub object_type: ObjectType,
    /// Size in bytes
    pub size_bytes: u64,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Checksum information
    pub checksums: HashMap<ChecksumAlgorithm, ChecksumInfo>,
    /// Integrity status
    pub integrity_status: IntegrityStatus,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Object type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectType {
    /// Database file
    Database,
    /// Configuration file
    Configuration,
    /// Log file
    LogFile,
    /// Binary executable
    Binary,
    /// Archive file
    Archive,
    /// Media file
    Media,
    /// Document
    Document,
    /// Source code
    SourceCode,
    /// Other type
    Other,
}

/// Checksum algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// SHA-256 hash
    SHA256,
    /// SHA-512 hash
    SHA512,
    /// CRC32 checksum
    CRC32,
    /// MD5 hash (legacy, not recommended)
    MD5,
    /// Blake2b hash
    Blake2b,
    /// xxHash
    XXHash,
}

impl ChecksumAlgorithm {
    /// Get the expected output length in bytes
    pub fn output_length(&self) -> usize {
        match self {
            ChecksumAlgorithm::SHA256 => 32,
            ChecksumAlgorithm::SHA512 => 64,
            ChecksumAlgorithm::CRC32 => 4,
            ChecksumAlgorithm::MD5 => 16,
            ChecksumAlgorithm::Blake2b => 64,
            ChecksumAlgorithm::XXHash => 8,
        }
    }
}

/// Checksum information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumInfo {
    /// Algorithm used
    pub algorithm: ChecksumAlgorithm,
    /// Checksum value
    pub value: Vec<u8>,
    /// Calculated timestamp
    pub calculated_at: DateTime<Utc>,
    /// Block checksums if applicable
    pub block_checksums: Option<Vec<BlockChecksum>>,
}

/// Block-level checksum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockChecksum {
    /// Block index
    pub block_index: u64,
    /// Block offset in bytes
    pub offset: u64,
    /// Block size in bytes
    pub size: u64,
    /// Checksum value
    pub checksum: Vec<u8>,
}

/// Integrity status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrityStatus {
    /// Verified and intact
    Verified,
    /// Corruption detected
    Corrupted,
    /// Repair in progress
    Repairing,
    /// Successfully repaired
    Repaired,
    /// Verification pending
    Pending,
    /// Verification in progress
    Verifying,
    /// Unknown status
    Unknown,
}

/// Integrity check type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegrityCheckType {
    /// Full file checksum verification
    Full,
    /// Block-level checksum verification
    Block,
    /// Real-time verification during I/O
    Realtime,
    /// Periodic background verification
    Periodic,
    /// On-demand verification
    OnDemand,
}

/// Verification schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationSchedule {
    /// Continuous verification
    Continuous,
    /// Hourly verification
    Hourly,
    /// Daily verification at specific hour
    Daily { hour: u32 },
    /// Weekly verification
    Weekly { day_of_week: u32, hour: u32 },
    /// Monthly verification
    Monthly { day_of_month: u32, hour: u32 },
    /// Custom cron schedule
    Cron(String),
}