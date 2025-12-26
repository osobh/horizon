//! Storage-based stream sink for persistent data output

use crate::{StreamChunk, StreamSink, StreamStats, StreamingError};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

/// Storage backend types
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// Local file system
    FileSystem,
    /// NVMe optimized storage
    NvmeOptimized,
    /// Object storage (S3-compatible)
    ObjectStorage { bucket: String, prefix: String },
    /// Database storage
    Database { table: String },
}

/// Storage stream sink for persistent data output
pub struct StorageStreamSink {
    id: String,
    backend: StorageBackend,
    output_path: PathBuf,
    append_mode: bool,
    sync_writes: bool,
    buffer_size: usize,
    stats: Arc<StorageSinkStats>,
}

/// Thread-safe statistics for storage sink
#[derive(Debug, Default)]
struct StorageSinkStats {
    chunks_written: AtomicU64,
    bytes_written: AtomicU64,
    write_time_ns: AtomicU64,
    sync_operations: AtomicU64,
    errors: AtomicU64,
}

impl StorageStreamSink {
    /// Create a new storage stream sink
    pub fn new(id: String, backend: StorageBackend, output_path: PathBuf) -> Self {
        Self {
            id,
            backend,
            output_path,
            append_mode: true,      // Default to append mode
            sync_writes: false,     // Default async writes for performance
            buffer_size: 64 * 1024, // 64KB buffer
            stats: Arc::new(StorageSinkStats::default()),
        }
    }

    /// Configure append vs overwrite mode
    pub fn with_append_mode(mut self, append: bool) -> Self {
        self.append_mode = append;
        self
    }

    /// Configure synchronous writes for durability
    pub fn with_sync_writes(mut self, sync: bool) -> Self {
        self.sync_writes = sync;
        self
    }

    /// Configure write buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        // Relaxed: independent statistics counters with no ordering dependencies
        let chunks = self.stats.chunks_written.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_written.load(Ordering::Relaxed);
        let time_ns = self.stats.write_time_ns.load(Ordering::Relaxed);
        let errors = self.stats.errors.load(Ordering::Relaxed);

        let throughput_mbps = if time_ns > 0 {
            (bytes as f64) / ((time_ns as f64) / 1_000_000_000.0) / (1024.0 * 1024.0)
        } else {
            0.0
        };

        StreamStats {
            chunks_processed: chunks,
            bytes_processed: bytes,
            processing_time_ms: time_ns / 1_000_000,
            throughput_mbps,
            errors,
        }
    }

    /// Open file for writing based on configuration
    async fn open_file(&self) -> Result<File, StreamingError> {
        let file = if self.append_mode {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.output_path)
                .await
        } else {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.output_path)
                .await
        };

        file.map_err(|e| {
            StreamingError::IoError(format!(
                "Failed to open storage file {:?}: {}",
                self.output_path, e
            ))
        })
    }

    /// Write chunk data to storage with backend-specific optimizations
    async fn write_chunk_data(
        &self,
        file: &mut File,
        chunk: &StreamChunk,
    ) -> Result<(), StreamingError> {
        let start_time = Instant::now();

        match &self.backend {
            StorageBackend::FileSystem => {
                // Standard file system write
                file.write_all(&chunk.data)
                    .await
                    .map_err(|e| StreamingError::IoError(format!("File write failed: {e}")))?;
            }
            StorageBackend::NvmeOptimized => {
                // NVMe optimized writes with alignment
                let mut padded_data = chunk.data.to_vec();

                // Pad to 4KB boundary for NVMe efficiency
                let alignment = 4096;
                let remainder = padded_data.len() % alignment;
                if remainder != 0 {
                    padded_data.resize(padded_data.len() + (alignment - remainder), 0);
                }

                file.write_all(&padded_data)
                    .await
                    .map_err(|e| StreamingError::IoError(format!("NVMe write failed: {e}")))?;
            }
            StorageBackend::ObjectStorage {
                bucket: _,
                prefix: _,
            } => {
                // For now, write to local file (S3 integration would go here)
                file.write_all(&chunk.data).await.map_err(|e| {
                    StreamingError::IoError(format!("Object storage write failed: {e}"))
                })?;
            }
            StorageBackend::Database { table: _ } => {
                // For now, write as binary (database integration would go here)
                file.write_all(&chunk.data)
                    .await
                    .map_err(|e| StreamingError::IoError(format!("Database write failed: {e}")))?;
            }
        }

        if self.sync_writes {
            file.sync_all()
                .await
                .map_err(|e| StreamingError::IoError(format!("Sync failed: {e}")))?;
            // Relaxed: independent sync counter
            self.stats.sync_operations.fetch_add(1, Ordering::Relaxed);
        }

        let write_time = start_time.elapsed().as_nanos() as u64;
        // Relaxed: independent statistics counters
        self.stats
            .write_time_ns
            .fetch_add(write_time, Ordering::Relaxed);
        self.stats.chunks_written.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_written
            .fetch_add(chunk.data.len() as u64, Ordering::Relaxed);

        Ok(())
    }
}

#[async_trait]
impl StreamSink for StorageStreamSink {
    async fn write(&mut self, chunk: StreamChunk) -> Result<(), StreamingError> {
        let mut file = self.open_file().await?;
        self.write_chunk_data(&mut file, &chunk).await?;
        Ok(())
    }

    async fn write_batch(&mut self, chunks: Vec<StreamChunk>) -> Result<(), StreamingError> {
        let mut file = self.open_file().await?;

        for chunk in chunks {
            self.write_chunk_data(&mut file, &chunk).await?;
        }

        Ok(())
    }

    async fn flush(&mut self) -> Result<(), StreamingError> {
        // For file-based storage, ensure all data is written
        let file = self.open_file().await?;
        file.sync_all()
            .await
            .map_err(|e| StreamingError::IoError(format!("Flush failed: {e}")))?;
        // Relaxed: independent sync counter
        self.stats.sync_operations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn sink_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use tempfile::NamedTempFile;
    use tokio::fs;

    async fn create_temp_sink(backend: StorageBackend) -> (StorageStreamSink, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let sink = StorageStreamSink::new(
            "test-storage".to_string(),
            backend,
            temp_file.path().to_path_buf(),
        );
        (sink, temp_file)
    }

    #[tokio::test]
    async fn test_storage_sink_creation() {
        let (sink, _temp_file) = create_temp_sink(StorageBackend::FileSystem).await;

        assert_eq!(sink.sink_id(), "test-storage");
        assert!(sink.append_mode);
        assert!(!sink.sync_writes);
        assert_eq!(sink.buffer_size, 64 * 1024);
    }

    #[tokio::test]
    async fn test_storage_sink_configuration() {
        let temp_file = NamedTempFile::new().unwrap();
        let sink = StorageStreamSink::new(
            "config-test".to_string(),
            StorageBackend::NvmeOptimized,
            temp_file.path().to_path_buf(),
        )
        .with_append_mode(false)
        .with_sync_writes(true)
        .with_buffer_size(128 * 1024);

        assert!(!sink.append_mode);
        assert!(sink.sync_writes);
        assert_eq!(sink.buffer_size, 128 * 1024);
    }

    #[tokio::test]
    async fn test_filesystem_write() {
        let (mut sink, temp_file) = create_temp_sink(StorageBackend::FileSystem).await;

        let test_data = b"Hello, Storage World!";
        let chunk = StreamChunk::new(
            Bytes::from(test_data.to_vec()),
            1,
            "test-source".to_string(),
        );

        sink.write(chunk).await.unwrap();
        sink.flush().await.unwrap();

        // Verify data was written
        let written_data = fs::read(temp_file.path()).await.unwrap();
        assert_eq!(written_data, test_data);

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert_eq!(stats.bytes_processed, test_data.len() as u64);
    }

    #[tokio::test]
    async fn test_nvme_optimized_write() {
        let (mut sink, temp_file) = create_temp_sink(StorageBackend::NvmeOptimized).await;

        let test_data = b"NVMe optimized data";
        let chunk = StreamChunk::new(
            Bytes::from(test_data.to_vec()),
            1,
            "nvme-source".to_string(),
        );

        sink.write(chunk).await.unwrap();
        sink.flush().await.unwrap();

        // Verify data was written (may be padded for alignment)
        let written_data = fs::read(temp_file.path()).await.unwrap();
        assert!(written_data.starts_with(test_data));

        // NVMe optimization should align to 4KB boundary
        assert!(written_data.len() >= test_data.len());
        assert_eq!(written_data.len() % 4096, 0);

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_append_vs_overwrite_mode() {
        let temp_file = NamedTempFile::new().unwrap();

        // First, write in overwrite mode
        {
            let mut sink = StorageStreamSink::new(
                "overwrite-test".to_string(),
                StorageBackend::FileSystem,
                temp_file.path().to_path_buf(),
            )
            .with_append_mode(false);

            let chunk1 = StreamChunk::new(Bytes::from("first write"), 1, "test".to_string());
            sink.write(chunk1).await.unwrap();
        }

        // Then write in append mode
        {
            let mut sink = StorageStreamSink::new(
                "append-test".to_string(),
                StorageBackend::FileSystem,
                temp_file.path().to_path_buf(),
            )
            .with_append_mode(true);

            let chunk2 = StreamChunk::new(Bytes::from("second write"), 2, "test".to_string());
            sink.write(chunk2).await.unwrap();
        }

        // Verify both writes are present
        let content = fs::read_to_string(temp_file.path()).await.unwrap();
        assert!(content.contains("first write"));
        assert!(content.contains("second write"));
    }

    #[tokio::test]
    async fn test_batch_write() {
        let (mut sink, temp_file) = create_temp_sink(StorageBackend::FileSystem).await;

        let chunks = vec![
            StreamChunk::new(Bytes::from("chunk1"), 1, "test".to_string()),
            StreamChunk::new(Bytes::from("chunk2"), 2, "test".to_string()),
            StreamChunk::new(Bytes::from("chunk3"), 3, "test".to_string()),
        ];

        sink.write_batch(chunks).await.unwrap();
        sink.flush().await.unwrap();

        let content = fs::read_to_string(temp_file.path()).await.unwrap();
        assert_eq!(content, "chunk1chunk2chunk3");

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
        assert_eq!(stats.bytes_processed, 18); // 6 + 6 + 6
    }

    #[tokio::test]
    async fn test_sync_writes() {
        let (_temp_file1, temp_file) = create_temp_sink(StorageBackend::FileSystem).await;
        let mut sink = StorageStreamSink::new(
            "sync-test".to_string(),
            StorageBackend::FileSystem,
            temp_file.path().to_path_buf(),
        )
        .with_sync_writes(true);

        let chunk = StreamChunk::new(Bytes::from("sync test"), 1, "test".to_string());

        sink.write(chunk).await.unwrap();

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert!(sink.stats.sync_operations.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_object_storage_backend() {
        let (mut sink, _temp_file) = create_temp_sink(StorageBackend::ObjectStorage {
            bucket: "test-bucket".to_string(),
            prefix: "data/".to_string(),
        })
        .await;

        let chunk = StreamChunk::new(Bytes::from("object storage test"), 1, "test".to_string());

        // Should work with fallback to file storage
        sink.write(chunk).await.unwrap();

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_database_backend() {
        let (mut sink, _temp_file) = create_temp_sink(StorageBackend::Database {
            table: "stream_data".to_string(),
        })
        .await;

        let chunk = StreamChunk::new(Bytes::from("database test"), 1, "test".to_string());

        // Should work with fallback to file storage
        sink.write(chunk).await.unwrap();

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_storage_sink_stats() {
        let (mut sink, _temp_file) = create_temp_sink(StorageBackend::FileSystem).await;

        // Initial stats should be zero
        let initial_stats = sink.stats().await.unwrap();
        assert_eq!(initial_stats.chunks_processed, 0);
        assert_eq!(initial_stats.bytes_processed, 0);

        // Write some data
        for i in 0..3 {
            let chunk = StreamChunk::new(
                Bytes::from(format!("data-{i}")),
                i as u64,
                "test".to_string(),
            );
            sink.write(chunk).await.unwrap();
        }

        let final_stats = sink.stats().await.unwrap();
        assert_eq!(final_stats.chunks_processed, 3);
        assert_eq!(final_stats.bytes_processed, 18); // "data-0" + "data-1" + "data-2"
        assert!(final_stats.processing_time_ms >= 0);
        assert!(final_stats.throughput_mbps >= 0.0);
    }

    #[tokio::test]
    async fn test_write_error_handling() {
        // Try to write to an invalid path
        let sink = StorageStreamSink::new(
            "error-test".to_string(),
            StorageBackend::FileSystem,
            PathBuf::from("/invalid/path/file.dat"),
        );

        let chunk = StreamChunk::new(Bytes::from("test data"), 1, "test".to_string());

        let mut sink = sink;
        let result = sink.write(chunk).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::IoError(msg) => assert!(msg.contains("Failed to open storage file")),
            _ => panic!("Expected IoError"),
        }
    }
}
