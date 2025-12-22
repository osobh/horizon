//! Core streaming abstractions and traits

use crate::{StreamChunk, StreamStats, StreamingError};
use async_trait::async_trait;
use std::pin::Pin;
use tokio_stream::Stream;

/// Stream source trait for reading data chunks
#[async_trait]
pub trait StreamSource: Send + Sync {
    /// Start the stream and get an async iterator of chunks
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    >;

    /// Stop the stream
    async fn stop(&mut self) -> Result<(), StreamingError>;

    /// Get stream statistics
    async fn stats(&self) -> Result<StreamStats, StreamingError>;

    /// Get the source identifier
    fn source_id(&self) -> &str;
}

/// Stream processor trait for transforming data chunks
#[async_trait]
pub trait StreamProcessor: Send + Sync {
    /// Process a single chunk
    async fn process(&mut self, chunk: StreamChunk) -> Result<StreamChunk, StreamingError>;

    /// Process a batch of chunks (for GPU operations)
    async fn process_batch(
        &mut self,
        chunks: Vec<StreamChunk>,
    ) -> Result<Vec<StreamChunk>, StreamingError> {
        let mut results = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            results.push(self.process(chunk).await?);
        }
        Ok(results)
    }

    /// Get processor statistics
    async fn stats(&self) -> Result<StreamStats, StreamingError>;

    /// Get the processor identifier
    fn processor_id(&self) -> &str;
}

/// Stream sink trait for writing processed chunks
#[async_trait]
pub trait StreamSink: Send + Sync {
    /// Write a single chunk
    async fn write(&mut self, chunk: StreamChunk) -> Result<(), StreamingError>;

    /// Write a batch of chunks
    async fn write_batch(&mut self, chunks: Vec<StreamChunk>) -> Result<(), StreamingError> {
        for chunk in chunks {
            self.write(chunk).await?;
        }
        Ok(())
    }

    /// Flush any pending writes
    async fn flush(&mut self) -> Result<(), StreamingError>;

    /// Get sink statistics
    async fn stats(&self) -> Result<StreamStats, StreamingError>;

    /// Get the sink identifier
    fn sink_id(&self) -> &str;
}

/// Configuration for streaming operations
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub chunk_size: usize,
    pub buffer_size: usize,
    pub batch_size: usize,
    pub timeout_ms: u64,
    pub enable_compression: bool,
    pub enable_checksum: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1MB
            buffer_size: 16,         // 16 chunks
            batch_size: 8,           // 8 chunks per batch
            timeout_ms: 5000,        // 5 seconds
            enable_compression: false,
            enable_checksum: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamChunk;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio_stream::iter;

    // Mock implementations for testing

    struct MockSource {
        id: String,
        chunks: Vec<StreamChunk>,
        stats: Arc<AtomicU64>,
    }

    impl MockSource {
        fn new(id: String, data_chunks: Vec<&str>) -> Self {
            let chunks = data_chunks
                .into_iter()
                .enumerate()
                .map(|(i, data)| {
                    StreamChunk::new(Bytes::from(data.to_string()), i as u64, id.clone())
                })
                .collect();

            Self {
                id,
                chunks,
                stats: Arc::new(AtomicU64::new(0)),
            }
        }
    }

    #[async_trait]
    impl StreamSource for MockSource {
        async fn start(
            &mut self,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
            StreamingError,
        > {
            self.stats.store(self.chunks.len() as u64, Ordering::SeqCst);
            let chunks = self.chunks.clone();
            let stream = iter(chunks.into_iter().map(Ok));
            Ok(Box::pin(stream))
        }

        async fn stop(&mut self) -> Result<(), StreamingError> {
            Ok(())
        }

        async fn stats(&self) -> Result<StreamStats, StreamingError> {
            Ok(StreamStats {
                chunks_processed: self.stats.load(Ordering::SeqCst),
                bytes_processed: self.chunks.iter().map(|c| c.size() as u64).sum(),
                processing_time_ms: 100,
                throughput_mbps: 10.0,
                errors: 0,
            })
        }

        fn source_id(&self) -> &str {
            &self.id
        }
    }

    struct MockProcessor {
        id: String,
        transform: fn(&str) -> String,
        processed_count: Arc<AtomicU64>,
    }

    impl MockProcessor {
        fn new(id: String, transform: fn(&str) -> String) -> Self {
            Self {
                id,
                transform,
                processed_count: Arc::new(AtomicU64::new(0)),
            }
        }
    }

    #[async_trait]
    impl StreamProcessor for MockProcessor {
        async fn process(&mut self, chunk: StreamChunk) -> Result<StreamChunk, StreamingError> {
            let input = String::from_utf8_lossy(&chunk.data);
            let output = (self.transform)(&input);
            self.processed_count.fetch_add(1, Ordering::SeqCst);

            Ok(StreamChunk::new(
                Bytes::from(output),
                chunk.sequence,
                chunk.metadata.source_id,
            ))
        }

        async fn stats(&self) -> Result<StreamStats, StreamingError> {
            Ok(StreamStats {
                chunks_processed: self.processed_count.load(Ordering::SeqCst),
                bytes_processed: 0,
                processing_time_ms: 50,
                throughput_mbps: 20.0,
                errors: 0,
            })
        }

        fn processor_id(&self) -> &str {
            &self.id
        }
    }

    struct MockSink {
        id: String,
        written_chunks: Arc<tokio::sync::Mutex<Vec<StreamChunk>>>,
    }

    impl MockSink {
        fn new(id: String) -> Self {
            Self {
                id,
                written_chunks: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            }
        }

        async fn get_written_chunks(&self) -> Vec<StreamChunk> {
            self.written_chunks.lock().await.clone()
        }
    }

    #[async_trait]
    impl StreamSink for MockSink {
        async fn write(&mut self, chunk: StreamChunk) -> Result<(), StreamingError> {
            self.written_chunks.lock().await.push(chunk);
            Ok(())
        }

        async fn flush(&mut self) -> Result<(), StreamingError> {
            Ok(())
        }

        async fn stats(&self) -> Result<StreamStats, StreamingError> {
            let chunks = self.written_chunks.lock().await;
            Ok(StreamStats {
                chunks_processed: chunks.len() as u64,
                bytes_processed: chunks.iter().map(|c| c.size() as u64).sum(),
                processing_time_ms: 25,
                throughput_mbps: 15.0,
                errors: 0,
            })
        }

        fn sink_id(&self) -> &str {
            &self.id
        }
    }

    #[tokio::test]
    async fn test_stream_source_mock() {
        let mut source =
            MockSource::new("test-source".to_string(), vec!["data1", "data2", "data3"]);

        assert_eq!(source.source_id(), "test-source");

        let mut stream = source.start().await.unwrap();

        // Collect all chunks
        let mut chunks = Vec::new();
        while let Some(result) = tokio_stream::StreamExt::next(&mut stream).await {
            chunks.push(result.unwrap());
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(String::from_utf8_lossy(&chunks[0].data), "data1");
        assert_eq!(String::from_utf8_lossy(&chunks[1].data), "data2");
        assert_eq!(String::from_utf8_lossy(&chunks[2].data), "data3");

        let stats = source.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
    }

    #[tokio::test]
    async fn test_stream_processor_mock() {
        let mut processor = MockProcessor::new("test-processor".to_string(), |s| s.to_uppercase());

        assert_eq!(processor.processor_id(), "test-processor");

        let chunk = StreamChunk::new(Bytes::from("hello"), 1, "test".to_string());
        let result = processor.process(chunk).await.unwrap();

        assert_eq!(String::from_utf8_lossy(&result.data), "HELLO");

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 1);
    }

    #[tokio::test]
    async fn test_stream_processor_batch() {
        let mut processor =
            MockProcessor::new("batch-processor".to_string(), |s| format!("processed_{s}"));

        let chunks = vec![
            StreamChunk::new(Bytes::from("a"), 1, "test".to_string()),
            StreamChunk::new(Bytes::from("b"), 2, "test".to_string()),
            StreamChunk::new(Bytes::from("c"), 3, "test".to_string()),
        ];

        let results = processor.process_batch(chunks).await.unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(String::from_utf8_lossy(&results[0].data), "processed_a");
        assert_eq!(String::from_utf8_lossy(&results[1].data), "processed_b");
        assert_eq!(String::from_utf8_lossy(&results[2].data), "processed_c");

        let stats = processor.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
    }

    #[tokio::test]
    async fn test_stream_sink_mock() {
        let mut sink = MockSink::new("test-sink".to_string());

        assert_eq!(sink.sink_id(), "test-sink");

        let chunk1 = StreamChunk::new(Bytes::from("chunk1"), 1, "test".to_string());
        let chunk2 = StreamChunk::new(Bytes::from("chunk2"), 2, "test".to_string());

        sink.write(chunk1).await.unwrap();
        sink.write(chunk2).await.unwrap();
        sink.flush().await.unwrap();

        let written = sink.get_written_chunks().await;
        assert_eq!(written.len(), 2);
        assert_eq!(String::from_utf8_lossy(&written[0].data), "chunk1");
        assert_eq!(String::from_utf8_lossy(&written[1].data), "chunk2");

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 2);
        assert_eq!(stats.bytes_processed, 12); // "chunk1" + "chunk2" = 6 + 6 = 12
    }

    #[tokio::test]
    async fn test_stream_sink_batch() {
        let mut sink = MockSink::new("batch-sink".to_string());

        let chunks = vec![
            StreamChunk::new(Bytes::from("batch1"), 1, "test".to_string()),
            StreamChunk::new(Bytes::from("batch2"), 2, "test".to_string()),
            StreamChunk::new(Bytes::from("batch3"), 3, "test".to_string()),
        ];

        sink.write_batch(chunks).await.unwrap();

        let written = sink.get_written_chunks().await;
        assert_eq!(written.len(), 3);

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();

        assert_eq!(config.chunk_size, 1024 * 1024);
        assert_eq!(config.buffer_size, 16);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.timeout_ms, 5000);
        assert!(!config.enable_compression);
        assert!(!config.enable_checksum);
    }
}
