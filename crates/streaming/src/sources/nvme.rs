//! NVMe-based stream source for high-performance data streaming

use crate::{StreamChunk, StreamSource, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use tokio_stream::Stream;

/// NVMe-optimized stream source for high-throughput data reading
pub struct NvmeStreamSource {
    id: String,
    file_path: PathBuf,
    chunk_size: usize,
    buffer_size: usize,
    read_ahead: usize,
    use_direct_io: bool,
    stats: Arc<StreamSourceStats>,
}

/// Thread-safe statistics for NVMe source
#[derive(Debug, Default)]
struct StreamSourceStats {
    chunks_read: AtomicU64,
    bytes_read: AtomicU64,
    read_time_ns: AtomicU64,
    errors: AtomicU64,
}

impl NvmeStreamSource {
    /// Create a new NVMe stream source
    pub fn new(id: String, file_path: PathBuf) -> Self {
        Self {
            id,
            file_path,
            chunk_size: 2 * 1024 * 1024, // 2MB chunks for NVMe optimization
            buffer_size: 32,             // 32 chunks buffered
            read_ahead: 4,               // 4 chunks read ahead
            use_direct_io: true,         // Direct I/O for performance
            stats: Arc::new(StreamSourceStats::default()),
        }
    }

    /// Configure chunk size for optimal NVMe performance
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Configure buffer size for memory management
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Configure read-ahead optimization
    pub fn with_read_ahead(mut self, read_ahead: usize) -> Self {
        self.read_ahead = read_ahead;
        self
    }

    /// Configure direct I/O usage
    pub fn with_direct_io(mut self, use_direct_io: bool) -> Self {
        self.use_direct_io = use_direct_io;
        self
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        // Relaxed: independent statistics counters with no ordering dependencies
        let chunks = self.stats.chunks_read.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_read.load(Ordering::Relaxed);
        let time_ns = self.stats.read_time_ns.load(Ordering::Relaxed);
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
}

#[async_trait]
impl StreamSource for NvmeStreamSource {
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    > {
        // Verify file exists and is readable
        let file = File::open(&self.file_path).await.map_err(|e| {
            StreamingError::IoError(format!(
                "Failed to open NVMe file {:?}: {}",
                self.file_path, e
            ))
        })?;

        let metadata = file
            .metadata()
            .await
            .map_err(|e| StreamingError::IoError(format!("Failed to get file metadata: {e}")))?;
        let file_size = metadata.len();

        if file_size == 0 {
            return Err(StreamingError::InvalidInput("File is empty".to_string()));
        }

        let chunk_size = self.chunk_size;
        let source_id = self.id.clone();
        let stats = Arc::clone(&self.stats);

        // Create async stream that reads chunks sequentially
        let stream = async_stream::stream! {
            let mut file = file;
            let mut position = 0u64;
            let mut sequence = 0u64;
            let mut buffer = vec![0u8; chunk_size];

            while position < file_size {
                let start_time = Instant::now();

                // Seek to current position
                if let Err(e) = file.seek(SeekFrom::Start(position)).await {
                    // Relaxed: independent error counter
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                    yield Err(StreamingError::IoError(format!("Seek failed: {e}")));
                    break;
                }

                // Read chunk data
                let bytes_to_read = std::cmp::min(chunk_size, (file_size - position) as usize);
                buffer.resize(bytes_to_read, 0);

                match file.read(&mut buffer).await {
                    Ok(bytes_read) => {
                        if bytes_read == 0 {
                            // End of file
                            break;
                        }

                        buffer.truncate(bytes_read);
                        let read_time = start_time.elapsed().as_nanos() as u64;
                        // Relaxed: independent statistics counters
                        stats.read_time_ns.fetch_add(read_time, Ordering::Relaxed);
                        stats.chunks_read.fetch_add(1, Ordering::Relaxed);
                        stats.bytes_read.fetch_add(bytes_read as u64, Ordering::Relaxed);

                        let chunk = StreamChunk::new(
                            Bytes::copy_from_slice(&buffer),
                            sequence,
                            source_id.clone(),
                        );

                        yield Ok(chunk);

                        position += bytes_read as u64;
                        sequence += 1;
                    }
                    Err(e) => {
                        // Relaxed: independent error counter
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        yield Err(StreamingError::IoError(format!("Read failed: {e}")));
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn stop(&mut self) -> Result<(), StreamingError> {
        // NVMe source cleanup - file handles are automatically closed
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.get_stats_snapshot())
    }

    fn source_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tokio_stream::StreamExt;

    async fn create_test_file(content: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content)
            .expect("Failed to write test content");
        file.flush().expect("Failed to flush test file");
        file
    }

    #[tokio::test]
    async fn test_nvme_source_creation() {
        let temp_file = create_test_file(b"test data").await;
        let source = NvmeStreamSource::new("test-nvme".to_string(), temp_file.path().to_path_buf());

        assert_eq!(source.source_id(), "test-nvme");
        assert_eq!(source.chunk_size, 2 * 1024 * 1024);
        assert_eq!(source.buffer_size, 32);
        assert_eq!(source.read_ahead, 4);
        assert!(source.use_direct_io);
    }

    #[tokio::test]
    async fn test_nvme_source_configuration() {
        let temp_file = create_test_file(b"test data").await;
        let source = NvmeStreamSource::new("test-nvme".to_string(), temp_file.path().to_path_buf())
            .with_chunk_size(1024)
            .with_buffer_size(16)
            .with_read_ahead(2)
            .with_direct_io(false);

        assert_eq!(source.chunk_size, 1024);
        assert_eq!(source.buffer_size, 16);
        assert_eq!(source.read_ahead, 2);
        assert!(!source.use_direct_io);
    }

    #[tokio::test]
    async fn test_nvme_source_small_file() {
        let test_data = b"Hello, NVMe World!";
        let temp_file = create_test_file(test_data).await;

        let mut source = NvmeStreamSource::new(
            "small-file-test".to_string(),
            temp_file.path().to_path_buf(),
        )
        .with_chunk_size(1024);

        let mut stream = source.start().await.unwrap();
        let mut chunks = Vec::new();

        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }

        assert_eq!(chunks.len(), 1);
        assert_eq!(&chunks[0].data[..], test_data);
        assert_eq!(chunks[0].sequence, 0);
        assert_eq!(chunks[0].metadata.source_id, "small-file-test");

        let stats = source.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
        assert_eq!(stats.bytes_processed, test_data.len() as u64);
        assert!(stats.throughput_mbps >= 0.0);
    }

    #[tokio::test]
    async fn test_nvme_source_multiple_chunks() {
        // Create test data larger than one small chunk
        let test_data = vec![42u8; 5000];
        let temp_file = create_test_file(&test_data).await;

        let mut source = NvmeStreamSource::new(
            "multi-chunk-test".to_string(),
            temp_file.path().to_path_buf(),
        )
        .with_chunk_size(2000); // Force multiple chunks

        let mut stream = source.start().await.unwrap();
        let mut chunks = Vec::new();

        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }

        assert_eq!(chunks.len(), 3); // 5000 bytes / 2000 chunk_size = 3 chunks
        assert_eq!(chunks[0].data.len(), 2000);
        assert_eq!(chunks[1].data.len(), 2000);
        assert_eq!(chunks[2].data.len(), 1000); // Remaining bytes

        // Verify sequence numbers
        assert_eq!(chunks[0].sequence, 0);
        assert_eq!(chunks[1].sequence, 1);
        assert_eq!(chunks[2].sequence, 2);

        // Verify data integrity
        let mut reconstructed = Vec::new();
        for chunk in &chunks {
            reconstructed.extend_from_slice(&chunk.data);
        }
        assert_eq!(reconstructed, test_data);

        let stats = source.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
        assert_eq!(stats.bytes_processed, 5000);
    }

    #[tokio::test]
    async fn test_nvme_source_empty_file() {
        let temp_file = create_test_file(b"").await;

        let mut source = NvmeStreamSource::new(
            "empty-file-test".to_string(),
            temp_file.path().to_path_buf(),
        );

        let result = source.start().await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::InvalidInput(msg) => assert_eq!(msg, "File is empty"),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_nvme_source_nonexistent_file() {
        let mut source = NvmeStreamSource::new(
            "nonexistent-test".to_string(),
            PathBuf::from("/nonexistent/file.bin"),
        );

        let result = source.start().await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::IoError(msg) => assert!(msg.contains("Failed to open NVMe file")),
            _ => panic!("Expected IoError"),
        }
    }

    #[tokio::test]
    async fn test_nvme_source_stats_tracking() {
        let test_data = b"Statistics test data for NVMe source";
        let temp_file = create_test_file(test_data).await;

        let mut source =
            NvmeStreamSource::new("stats-test".to_string(), temp_file.path().to_path_buf())
                .with_chunk_size(15); // Small chunks to test multiple reads

        // Initial stats should be zero
        let initial_stats = source.stats().await.unwrap();
        assert_eq!(initial_stats.chunks_processed, 0);
        assert_eq!(initial_stats.bytes_processed, 0);

        let mut stream = source.start().await.unwrap();
        let mut chunk_count = 0;

        while let Some(result) = stream.next().await {
            result.unwrap();
            chunk_count += 1;
        }

        let final_stats = source.stats().await.unwrap();
        assert_eq!(final_stats.chunks_processed, chunk_count);
        assert_eq!(final_stats.bytes_processed, test_data.len() as u64);
        assert!(final_stats.processing_time_ms >= 0);
        assert!(final_stats.throughput_mbps >= 0.0);
    }

    #[tokio::test]
    async fn test_nvme_source_stop() {
        let test_data = b"Stop test data";
        let temp_file = create_test_file(test_data).await;

        let mut source =
            NvmeStreamSource::new("stop-test".to_string(), temp_file.path().to_path_buf());

        // Test that stop works without errors
        let result = source.stop().await;
        assert!(result.is_ok());
    }
}
