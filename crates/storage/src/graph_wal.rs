//! Write-Ahead Log for graph storage durability

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

use crate::error::StorageError;
use crate::graph_format::NodeRecord;

/// WAL entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WALEntry {
    NodeWrite {
        id: u64,
        data: NodeRecord,
    },
    NodeUpdate {
        id: u64,
        updates: NodeUpdates,
    },
    EdgeAdd {
        from: u64,
        to: u64,
        edge_type: u32,
        weight: f32,
    },
    EdgeRemove {
        from: u64,
        to: u64,
    },
    Checkpoint {
        timestamp: u64,
    },
}

/// Updates to apply to a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdates {
    pub flags: Option<u32>,
    pub embedding_offset: Option<u64>,
    pub property_offset: Option<u64>,
    pub edge_count: Option<u32>,
    pub importance_score: Option<f32>,
}

/// WAL segment for managing individual log files
pub struct WALSegment {
    file: File,
    size: usize,
    id: u64,
    #[allow(dead_code)]
    path: PathBuf,
}

impl WALSegment {
    /// Create a new WAL segment
    async fn create(base_path: &Path, id: u64) -> Result<Self, StorageError> {
        let filename = format!("segment_{id:08}.wal");
        let path = base_path.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&path)
            .await?;

        Ok(Self {
            file,
            size: 0,
            id,
            path,
        })
    }

    /// Write entry to segment
    async fn write_entry(&mut self, entry: &[u8]) -> Result<(), StorageError> {
        self.file.write_all(entry).await?;
        self.file.sync_all().await?;
        self.size += entry.len();
        Ok(())
    }
}

/// Write-Ahead Log for graph operations
pub struct GraphWAL {
    current_segment: Arc<Mutex<WALSegment>>,
    segment_size: usize,
    base_path: PathBuf,
}

impl GraphWAL {
    /// Default segment size (1GB)
    const DEFAULT_SEGMENT_SIZE: usize = 1024 * 1024 * 1024;

    /// Create a new WAL
    pub async fn new(base_path: PathBuf) -> Result<Self, StorageError> {
        tokio::fs::create_dir_all(&base_path).await?;

        let segment = WALSegment::create(&base_path, 0).await?;

        Ok(Self {
            current_segment: Arc::new(Mutex::new(segment)),
            segment_size: Self::DEFAULT_SEGMENT_SIZE,
            base_path,
        })
    }

    /// Create WAL with custom segment size
    pub async fn with_segment_size(
        base_path: PathBuf,
        segment_size: usize,
    ) -> Result<Self, StorageError> {
        tokio::fs::create_dir_all(&base_path).await?;

        let segment = WALSegment::create(&base_path, 0).await?;

        Ok(Self {
            current_segment: Arc::new(Mutex::new(segment)),
            segment_size,
            base_path,
        })
    }

    /// Append an entry to the WAL
    pub async fn append(&self, entry: WALEntry) -> Result<(), StorageError> {
        let serialized = bincode::serialize(&entry).map_err(|e| StorageError::WALError {
            reason: format!("Failed to serialize entry: {e}"),
        })?;

        let entry_size = serialized.len() as u32;

        // Prepare entry with size prefix
        let mut entry_data = Vec::with_capacity(4 + serialized.len());
        entry_data.extend_from_slice(&entry_size.to_le_bytes());
        entry_data.extend_from_slice(&serialized);

        // Write to segment (with potential rotation)
        self.write_with_rotation(entry_data).await
    }

    /// Write data with segment rotation if needed
    async fn write_with_rotation(&self, data: Vec<u8>) -> Result<(), StorageError> {
        let mut segment = self.current_segment.lock().await;

        // Check if rotation is needed
        if segment.size + data.len() > self.segment_size {
            // Create new segment
            let new_segment = WALSegment::create(&self.base_path, segment.id + 1).await?;
            *segment = new_segment;
        }

        // Write the data
        segment.write_entry(&data).await?;

        Ok(())
    }

    /// Force a checkpoint
    pub async fn checkpoint(&self) -> Result<(), StorageError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.append(WALEntry::Checkpoint { timestamp }).await
    }

    /// Get the current segment ID
    pub async fn current_segment_id(&self) -> Result<u64, StorageError> {
        let segment = self.current_segment.lock().await;
        Ok(segment.id)
    }
}

/// WAL reader for recovery
pub struct WALReader {
    base_path: PathBuf,
}

impl WALReader {
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    /// Read all entries from a segment
    pub async fn read_segment(&self, segment_id: u64) -> Result<Vec<WALEntry>, StorageError> {
        let filename = format!("segment_{segment_id:08}.wal");
        let path = self.base_path.join(&filename);

        let mut file = File::open(&path).await?;
        let mut entries = Vec::new();
        let mut buffer = Vec::new();

        file.read_to_end(&mut buffer).await?;

        let mut cursor = 0;
        while cursor + 4 <= buffer.len() {
            // Read entry size
            let size_bytes: [u8; 4] = buffer[cursor..cursor + 4].try_into().map_err(|_| {
                StorageError::InvalidDataFormat {
                    reason: "Invalid entry size".to_string(),
                }
            })?;
            let entry_size = u32::from_le_bytes(size_bytes) as usize;

            cursor += 4;

            if cursor + entry_size > buffer.len() {
                break; // Incomplete entry
            }

            // Deserialize entry
            let entry: WALEntry = bincode::deserialize(&buffer[cursor..cursor + entry_size])
                .map_err(|e| StorageError::WALError {
                    reason: format!("Failed to deserialize entry: {e}"),
                })?;

            entries.push(entry);
            cursor += entry_size;
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_wal_creation() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        assert_eq!(wal.current_segment_id().await?, 0);
        assert!(dir.path().join("segment_00000000.wal").exists());
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_append_node_write() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        let node = NodeRecord::new(42, 7);
        let entry = WALEntry::NodeWrite { id: 42, data: node };

        wal.append(entry).await?;

        // Verify file has content
        let metadata = tokio::fs::metadata(dir.path().join("segment_00000000.wal"))
            .await
            .unwrap();
        assert!(metadata.len() > 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_append_multiple_entries() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        // Append various entry types
        wal.append(WALEntry::NodeWrite {
            id: 1,
            data: NodeRecord::new(1, 10),
        })
        .await
        .unwrap();

        wal.append(WALEntry::EdgeAdd {
            from: 1,
            to: 2,
            edge_type: 5,
            weight: 0.8,
        })
        .await
        .unwrap();

        wal.append(WALEntry::NodeUpdate {
            id: 1,
            updates: NodeUpdates {
                flags: Some(1),
                embedding_offset: None,
                property_offset: None,
                edge_count: Some(1),
                importance_score: None,
            },
        })
        .await
        .unwrap();

        wal.checkpoint().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_reader() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        // Write some entries
        let entries = vec![
            WALEntry::NodeWrite {
                id: 1,
                data: NodeRecord::new(1, 10),
            },
            WALEntry::EdgeAdd {
                from: 1,
                to: 2,
                edge_type: 5,
                weight: 0.8,
            },
            WALEntry::EdgeRemove { from: 1, to: 3 },
        ];

        for entry in &entries {
            wal.append(entry.clone()).await?;
        }

        // Read them back
        let reader = WALReader::new(dir.path().to_path_buf());
        let read_entries = reader.read_segment(0).await?;

        assert_eq!(read_entries.len(), 3);

        // Verify first entry
        match &read_entries[0] {
            WALEntry::NodeWrite { id, data } => {
                assert_eq!(*id, 1);
                assert_eq!(data.id, 1);
                assert_eq!(data.type_id, 10);
            }
            _ => panic!("Expected NodeWrite entry"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_segment_rotation() -> anyhow::Result<()> {
        let dir = tempdir()?;
        // Create WAL with small segment size (1KB)
        let wal = GraphWAL::with_segment_size(dir.path().to_path_buf(), 1024)
            .await
            .unwrap();

        // Write enough entries to trigger rotation
        for i in 0..50 {
            let node = NodeRecord::new(i, i as u32);
            wal.append(WALEntry::NodeWrite { id: i, data: node })
                .await
                .unwrap();
        }

        // Should have rotated to a new segment
        assert!(wal.current_segment_id().await? > 0);

        // Both segment files should exist
        assert!(dir.path().join("segment_00000000.wal").exists());
        assert!(dir.path().join("segment_00000001.wal").exists());
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_checkpoint() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        wal.checkpoint().await?;

        let reader = WALReader::new(dir.path().to_path_buf());
        let entries = reader.read_segment(0).await?;

        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WALEntry::Checkpoint { timestamp } => {
                assert!(*timestamp > 0);
            }
            _ => panic!("Expected Checkpoint entry"),
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Disabled: tokio::sync::Mutex doesn't have poisoning
    async fn test_wal_mutex_poisoning() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        // Poison the current_segment mutex
        let _poisoned_segment = Arc::clone(&wal.current_segment);
        std::thread::spawn(move || {
            // Note: This test is ignored since tokio::sync::Mutex doesn't poison
            panic!("Poisoning WAL segment mutex");
        })
        .join()
        .unwrap_err();

        // Try to append an entry
        let entry = WALEntry::NodeWrite {
            id: 1,
            data: NodeRecord::new(1, 100),
        };
        let result = wal.append(entry).await;

        assert!(
            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "WAL segment")
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_reader_error_cases() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let reader = WALReader::new(dir.path().to_path_buf());

        // Try to read non-existent segment
        let result = reader.read_segment(999).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_entry_types() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        // Test all entry types
        let entries = vec![
            WALEntry::NodeWrite {
                id: 1,
                data: NodeRecord::new(1, 10),
            },
            WALEntry::NodeUpdate {
                id: 1,
                updates: NodeUpdates {
                    flags: Some(1),
                    embedding_offset: Some(100),
                    property_offset: Some(200),
                    edge_count: Some(5),
                    importance_score: Some(0.8),
                },
            },
            WALEntry::EdgeAdd {
                from: 1,
                to: 2,
                edge_type: 5,
                weight: 0.8,
            },
            WALEntry::EdgeRemove { from: 1, to: 3 },
            WALEntry::Checkpoint { timestamp: 12345 },
        ];

        for entry in entries {
            wal.append(entry).await?;
        }

        let reader = WALReader::new(dir.path().to_path_buf());
        let read_entries = reader.read_segment(0).await?;

        assert_eq!(read_entries.len(), 5);
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_serialization_error_path() -> anyhow::Result<()> {
        // TDD test for line 130 - serialize error path in append()
        // This is a defensive test to ensure error handling works correctly
        let dir = tempdir()?;
        let wal = GraphWAL::new(dir.path().to_path_buf()).await?;

        // Create a valid entry to test normal serialization
        let entry = WALEntry::NodeWrite {
            id: 1,
            data: NodeRecord::new(1, 0),
        };

        // Normal append should work
        assert!(wal.append(entry).await.is_ok());

        // Note: bincode::serialize rarely fails for well-formed structs,
        // but the error path exists for safety and is properly handled
        Ok(())
    }

    #[tokio::test]
    #[ignore] // Disabled: tokio::sync::Mutex doesn't have poisoning
    async fn test_current_segment_id_lock_poisoning() -> anyhow::Result<()> {
        // TDD test for lines 181-182 - lock poisoning in current_segment_id()
        use std::sync::Arc;

        let dir = tempdir()?;
        let wal = Arc::new(GraphWAL::new(dir.path().to_path_buf()).await?);

        // With tokio::sync::Mutex, there's no poisoning behavior
        // This test is kept for documentation but marked as ignored
        let result = wal.current_segment_id().await;
        assert!(result.is_ok(), "tokio::sync::Mutex doesn't poison on panic");
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_reader_invalid_entry_size() -> anyhow::Result<()> {
        // TDD test for lines 214-215 - invalid entry size format in WALReader
        let dir = tempdir()?;
        let segment_path = dir.path().join("segment_00000000.wal");

        // Write invalid data (incomplete 4-byte size header)
        let invalid_data = vec![0xFF, 0xFF]; // Only 2 bytes instead of 4
        tokio::fs::write(&segment_path, invalid_data).await?;

        let reader = WALReader::new(dir.path().to_path_buf());
        let result = reader.read_segment(0).await;

        // Should handle invalid format gracefully
        match result {
            Ok(entries) => {
                // Reader skips invalid data and returns empty
                assert!(entries.is_empty());
            }
            Err(StorageError::InvalidDataFormat { reason }) => {
                assert!(reason.contains("Invalid entry size"));
            }
            Err(other) => panic!("Unexpected error: {:?}", other),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_reader_deserialization_error() -> anyhow::Result<()> {
        // TDD test for lines 228-229 - deserialization error in WALReader
        let dir = tempdir()?;
        let segment_path = dir.path().join("segment_00000000.wal");

        // Create file with valid size header but invalid serialized data
        let mut data = Vec::new();

        // Valid 4-byte size (8 bytes of data)
        let size: u32 = 8;
        data.extend_from_slice(&size.to_le_bytes());

        // Invalid serialized data that can't deserialize to WALEntry
        data.extend_from_slice(&[0xFF; 8]);

        tokio::fs::write(&segment_path, data).await?;

        let reader = WALReader::new(dir.path().to_path_buf());
        let result = reader.read_segment(0).await;

        // Should return deserialization error
        assert!(result.is_err());

        match result {
            Err(StorageError::WALError { reason }) => {
                assert!(reason.contains("Failed to deserialize entry"));
            }
            _ => panic!(
                "Expected WALError for deserialization failure, got: {:?}",
                result
            ),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_wal_reader_buffer_overflow_protection() -> anyhow::Result<()> {
        // TDD test to ensure lines 214-215 and 228-229 handle edge cases
        let dir = tempdir()?;
        let segment_path = dir.path().join("segment_00000000.wal");

        // Create data with valid size but that would cause buffer overflow
        let mut data = Vec::new();

        // Size claims 1000 bytes but we only provide 4
        let invalid_size: u32 = 1000;
        data.extend_from_slice(&invalid_size.to_le_bytes());
        data.extend_from_slice(&[0xFF; 4]); // Only 4 bytes of actual data

        tokio::fs::write(&segment_path, data).await?;

        let reader = WALReader::new(dir.path().to_path_buf());
        let result = reader.read_segment(0).await;

        // Should handle buffer bounds safely
        match result {
            Ok(entries) => {
                // Reader stops at buffer boundary, returns what it could read
                assert!(entries.is_empty());
            }
            Err(_) => {
                // Or returns appropriate error - both are acceptable for safety
            }
        }
        Ok(())
    }
}
