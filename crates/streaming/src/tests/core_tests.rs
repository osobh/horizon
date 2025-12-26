//! Tests for core streaming traits and types

use crate::core::*;
use crate::{StreamChunk, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio_stream::{iter, Stream, StreamExt};

// Test implementations

struct TestSource {
    id: String,
    chunks: Vec<StreamChunk>,
    started: bool,
    stats: StreamStats,
}

impl TestSource {
    fn new(id: String, data_pieces: Vec<&str>) -> Self {
        let chunks = data_pieces
            .into_iter()
            .enumerate()
            .map(|(i, data)| StreamChunk::new(Bytes::from(data.to_string()), i as u64, id.clone()))
            .collect();

        Self {
            id,
            chunks,
            started: false,
            stats: StreamStats::default(),
        }
    }

    fn with_empty_chunks(id: String, count: usize) -> Self {
        let chunks = (0..count)
            .map(|i| StreamChunk::new(Bytes::new(), i as u64, id.clone()))
            .collect();

        Self {
            id,
            chunks,
            started: false,
            stats: StreamStats::default(),
        }
    }
}

#[async_trait]
impl StreamSource for TestSource {
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    > {
        self.started = true;
        self.stats.chunks_processed = self.chunks.len() as u64;

        let chunks = self.chunks.clone();
        let stream = iter(chunks.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn stop(&mut self) -> Result<(), StreamingError> {
        self.started = false;
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.stats.clone())
    }

    fn source_id(&self) -> &str {
        &self.id
    }
}

struct FailingSource {
    id: String,
    fail_on_start: bool,
}

impl FailingSource {
    fn new(id: String, fail_on_start: bool) -> Self {
        Self { id, fail_on_start }
    }
}

#[async_trait]
impl StreamSource for FailingSource {
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    > {
        if self.fail_on_start {
            return Err(StreamingError::IoFailed {
                reason: "Simulated start failure".to_string(),
            });
        }

        let error_stream = iter(vec![Err(StreamingError::ProcessingFailed {
            reason: "Simulated chunk error".to_string(),
        })]);
        Ok(Box::pin(error_stream))
    }

    async fn stop(&mut self) -> Result<(), StreamingError> {
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Err(StreamingError::IoFailed {
            reason: "Stats unavailable".to_string(),
        })
    }

    fn source_id(&self) -> &str {
        &self.id
    }
}

struct TestProcessor {
    id: String,
    transform_fn: fn(&str) -> Result<String, &'static str>,
    processed_count: Arc<AtomicU64>,
    process_time_ms: u64,
}

impl TestProcessor {
    fn new(id: String, transform_fn: fn(&str) -> Result<String, &'static str>) -> Self {
        Self {
            id,
            transform_fn,
            processed_count: Arc::new(AtomicU64::new(0)),
            process_time_ms: 10,
        }
    }

    fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.process_time_ms = time_ms;
        self
    }
}

#[async_trait]
impl StreamProcessor for TestProcessor {
    async fn process(&mut self, chunk: StreamChunk) -> Result<StreamChunk, StreamingError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.process_time_ms)).await;

        let input = String::from_utf8_lossy(&chunk.data);
        let output = (self.transform_fn)(&input).map_err(|e| StreamingError::ProcessingFailed {
            reason: e.to_string(),
        })?;

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
            processing_time_ms: self.process_time_ms,
            throughput_mbps: 10.0,
            errors: 0,
        })
    }

    fn processor_id(&self) -> &str {
        &self.id
    }
}

struct TestSink {
    id: String,
    chunks: Arc<tokio::sync::Mutex<Vec<StreamChunk>>>,
    fail_on_write: bool,
    write_delay_ms: u64,
}

impl TestSink {
    fn new(id: String) -> Self {
        Self {
            id,
            chunks: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            fail_on_write: false,
            write_delay_ms: 0,
        }
    }

    fn with_failure(mut self) -> Self {
        self.fail_on_write = true;
        self
    }

    fn with_write_delay(mut self, delay_ms: u64) -> Self {
        self.write_delay_ms = delay_ms;
        self
    }

    async fn get_chunks(&self) -> Vec<StreamChunk> {
        self.chunks.lock().await.clone()
    }

    async fn chunk_count(&self) -> usize {
        self.chunks.lock().await.len()
    }
}

#[async_trait]
impl StreamSink for TestSink {
    async fn write(&mut self, chunk: StreamChunk) -> Result<(), StreamingError> {
        if self.fail_on_write {
            return Err(StreamingError::IoFailed {
                reason: "Simulated write failure".to_string(),
            });
        }

        if self.write_delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.write_delay_ms)).await;
        }

        self.chunks.lock().await.push(chunk);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), StreamingError> {
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        let chunks = self.chunks.lock().await;
        Ok(StreamStats {
            chunks_processed: chunks.len() as u64,
            bytes_processed: chunks.iter().map(|c| c.size() as u64).sum(),
            processing_time_ms: 5,
            throughput_mbps: 15.0,
            errors: 0,
        })
    }

    fn sink_id(&self) -> &str {
        &self.id
    }
}

// Core functionality tests

#[tokio::test]
async fn test_stream_source_basic_functionality() {
    let mut source = TestSource::new(
        "basic-source".to_string(),
        vec!["chunk1", "chunk2", "chunk3"],
    );

    assert_eq!(source.source_id(), "basic-source");

    let mut stream = source.start().await.unwrap();
    let mut chunks = Vec::new();

    while let Some(result) = stream.next().await {
        chunks.push(result.unwrap());
    }

    assert_eq!(chunks.len(), 3);
    assert_eq!(String::from_utf8_lossy(&chunks[0].data), "chunk1");
    assert_eq!(String::from_utf8_lossy(&chunks[1].data), "chunk2");
    assert_eq!(String::from_utf8_lossy(&chunks[2].data), "chunk3");

    // Test sequence numbers
    assert_eq!(chunks[0].sequence, 0);
    assert_eq!(chunks[1].sequence, 1);
    assert_eq!(chunks[2].sequence, 2);

    let stats = source.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 3);
}

#[tokio::test]
async fn test_stream_source_empty_stream() {
    let mut source = TestSource::new("empty-source".to_string(), vec![]);

    let mut stream = source.start().await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    assert!(chunks.is_empty());

    let stats = source.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 0);
}

#[tokio::test]
async fn test_stream_source_with_empty_chunks() {
    let mut source = TestSource::with_empty_chunks("empty-chunks".to_string(), 5);

    let mut stream = source.start().await.unwrap();
    let chunks: Vec<_> = stream.collect().await;

    assert_eq!(chunks.len(), 5);
    for (i, result) in chunks.into_iter().enumerate() {
        let chunk = result.unwrap();
        assert!(chunk.is_empty());
        assert_eq!(chunk.sequence, i as u64);
    }
}

#[tokio::test]
async fn test_stream_source_stop() {
    let mut source = TestSource::new("stop-test".to_string(), vec!["data"]);

    let _stream = source.start().await.unwrap();
    let stop_result = source.stop().await;

    assert!(stop_result.is_ok());
}

#[tokio::test]
async fn test_failing_stream_source() {
    let mut failing_source = FailingSource::new("failing-source".to_string(), true);

    let start_result = failing_source.start().await;
    assert!(start_result.is_err());

    let stats_result = failing_source.stats().await;
    assert!(stats_result.is_err());
}

#[tokio::test]
async fn test_stream_source_with_chunk_errors() {
    let mut failing_source = FailingSource::new("chunk-error-source".to_string(), false);

    let mut stream = failing_source.start().await.unwrap();
    let first_result = stream.next().await.unwrap();

    assert!(first_result.is_err());
}

#[tokio::test]
async fn test_stream_processor_basic_functionality() {
    let mut processor = TestProcessor::new("test-processor".to_string(), |s| Ok(s.to_uppercase()));

    assert_eq!(processor.processor_id(), "test-processor");

    let chunk = StreamChunk::new(Bytes::from("hello world"), 1, "test".to_string());
    let result = processor.process(chunk).await.unwrap();

    assert_eq!(String::from_utf8_lossy(&result.data), "HELLO WORLD");
    assert_eq!(result.sequence, 1);
    assert_eq!(result.metadata.source_id, "test");

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 1);
}

#[tokio::test]
async fn test_stream_processor_batch_processing() {
    let mut processor = TestProcessor::new("batch-processor".to_string(), |s| {
        Ok(format!("processed_{}", s))
    });

    let chunks = vec![
        StreamChunk::new(Bytes::from("data1"), 1, "test".to_string()),
        StreamChunk::new(Bytes::from("data2"), 2, "test".to_string()),
        StreamChunk::new(Bytes::from("data3"), 3, "test".to_string()),
    ];

    let results = processor.process_batch(chunks).await.unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(String::from_utf8_lossy(&results[0].data), "processed_data1");
    assert_eq!(String::from_utf8_lossy(&results[1].data), "processed_data2");
    assert_eq!(String::from_utf8_lossy(&results[2].data), "processed_data3");

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 3);
}

#[tokio::test]
async fn test_stream_processor_error_handling() {
    let mut processor =
        TestProcessor::new("error-processor".to_string(), |_| Err("Processing failed"));

    let chunk = StreamChunk::new(Bytes::from("test"), 1, "test".to_string());
    let result = processor.process(chunk).await;

    assert!(result.is_err());

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 0);
}

#[tokio::test]
async fn test_stream_processor_batch_error_handling() {
    let mut processor = TestProcessor::new("batch-error-processor".to_string(), |s| {
        if s == "error" {
            Err("Processing failed")
        } else {
            Ok(s.to_string())
        }
    });

    let chunks = vec![
        StreamChunk::new(Bytes::from("good"), 1, "test".to_string()),
        StreamChunk::new(Bytes::from("error"), 2, "test".to_string()),
        StreamChunk::new(Bytes::from("good2"), 3, "test".to_string()),
    ];

    let result = processor.process_batch(chunks).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_stream_processor_processing_time() {
    let mut processor = TestProcessor::new("timed-processor".to_string(), |s| Ok(s.to_string()))
        .with_processing_time(50);

    let chunk = StreamChunk::new(Bytes::from("test"), 1, "test".to_string());

    let start = std::time::Instant::now();
    let _result = processor.process(chunk).await.unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed >= std::time::Duration::from_millis(45));
    assert!(elapsed <= std::time::Duration::from_millis(100));
}

#[tokio::test]
async fn test_stream_sink_basic_functionality() {
    let mut sink = TestSink::new("test-sink".to_string());

    assert_eq!(sink.sink_id(), "test-sink");

    let chunk1 = StreamChunk::new(Bytes::from("chunk1"), 1, "test".to_string());
    let chunk2 = StreamChunk::new(Bytes::from("chunk2"), 2, "test".to_string());

    sink.write(chunk1.clone()).await.unwrap();
    sink.write(chunk2.clone()).await.unwrap();
    sink.flush().await.unwrap();

    let written_chunks = sink.get_chunks().await;
    assert_eq!(written_chunks.len(), 2);
    assert_eq!(String::from_utf8_lossy(&written_chunks[0].data), "chunk1");
    assert_eq!(String::from_utf8_lossy(&written_chunks[1].data), "chunk2");

    let stats = sink.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 2);
    assert_eq!(stats.bytes_processed, 12); // 6 + 6 bytes
}

#[tokio::test]
async fn test_stream_sink_batch_writing() {
    let mut sink = TestSink::new("batch-sink".to_string());

    let chunks = vec![
        StreamChunk::new(Bytes::from("batch1"), 1, "test".to_string()),
        StreamChunk::new(Bytes::from("batch2"), 2, "test".to_string()),
        StreamChunk::new(Bytes::from("batch3"), 3, "test".to_string()),
    ];

    sink.write_batch(chunks.clone()).await.unwrap();

    let written_chunks = sink.get_chunks().await;
    assert_eq!(written_chunks.len(), 3);

    for (i, chunk) in written_chunks.iter().enumerate() {
        assert_eq!(chunk.data, chunks[i].data);
        assert_eq!(chunk.sequence, chunks[i].sequence);
    }
}

#[tokio::test]
async fn test_stream_sink_error_handling() {
    let mut sink = TestSink::new("error-sink".to_string()).with_failure();

    let chunk = StreamChunk::new(Bytes::from("test"), 1, "test".to_string());
    let result = sink.write(chunk).await;

    assert!(result.is_err());

    let chunk_count = sink.chunk_count().await;
    assert_eq!(chunk_count, 0);
}

#[tokio::test]
async fn test_stream_sink_batch_error_handling() {
    let mut sink = TestSink::new("batch-error-sink".to_string()).with_failure();

    let chunks = vec![
        StreamChunk::new(Bytes::from("chunk1"), 1, "test".to_string()),
        StreamChunk::new(Bytes::from("chunk2"), 2, "test".to_string()),
    ];

    let result = sink.write_batch(chunks).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_stream_sink_write_delay() {
    let mut sink = TestSink::new("delayed-sink".to_string()).with_write_delay(30);

    let chunk = StreamChunk::new(Bytes::from("test"), 1, "test".to_string());

    let start = std::time::Instant::now();
    sink.write(chunk).await.unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed >= std::time::Duration::from_millis(25));
    assert!(elapsed <= std::time::Duration::from_millis(50));
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

#[test]
fn test_stream_config_custom() {
    let config = StreamConfig {
        chunk_size: 512 * 1024,
        buffer_size: 32,
        batch_size: 16,
        timeout_ms: 10000,
        enable_compression: true,
        enable_checksum: true,
    };

    assert_eq!(config.chunk_size, 512 * 1024);
    assert_eq!(config.buffer_size, 32);
    assert_eq!(config.batch_size, 16);
    assert_eq!(config.timeout_ms, 10000);
    assert!(config.enable_compression);
    assert!(config.enable_checksum);
}

#[tokio::test]
async fn test_concurrent_processing() {
    use std::sync::Arc;
    use tokio::task;

    let processor = Arc::new(tokio::sync::Mutex::new(TestProcessor::new(
        "concurrent-processor".to_string(),
        |s| Ok(s.to_uppercase()),
    )));

    let mut handles = Vec::new();

    // Process multiple chunks concurrently
    for i in 0..10 {
        let processor_clone = processor.clone();
        let handle = task::spawn(async move {
            let chunk = StreamChunk::new(
                Bytes::from(format!("chunk{}", i)),
                i as u64,
                "test".to_string(),
            );

            let mut proc = processor_clone.lock().await;
            proc.process(chunk).await
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    let final_stats = processor.lock().await.stats().await.unwrap();
    assert_eq!(final_stats.chunks_processed, 10);
}

#[tokio::test]
async fn test_large_chunk_handling() {
    let large_data = "x".repeat(10 * 1024 * 1024); // 10MB
    let mut source = TestSource::new("large-source".to_string(), vec![&large_data]);

    let mut stream = source.start().await.unwrap();
    let chunk_result = stream.next().await.unwrap();
    let chunk = chunk_result.unwrap();

    assert_eq!(chunk.size(), 10 * 1024 * 1024);
    assert_eq!(String::from_utf8_lossy(&chunk.data), large_data);
}

#[tokio::test]
async fn test_chunk_metadata_preservation() {
    let mut processor = TestProcessor::new("metadata-processor".to_string(), |s| {
        Ok(format!("processed_{}", s))
    });

    let original_chunk = StreamChunk::new(Bytes::from("test"), 42, "original-source".to_string());
    let original_timestamp = original_chunk.timestamp;
    let original_metadata = original_chunk.metadata.clone();

    let processed_chunk = processor.process(original_chunk).await.unwrap();

    assert_eq!(processed_chunk.sequence, 42);
    assert_eq!(processed_chunk.metadata.source_id, "original-source");
    // Note: timestamp might be different as it's set during creation
    assert_eq!(processed_chunk.metadata.chunk_size, "processed_test".len());
}

#[tokio::test]
async fn test_stats_aggregation() {
    let mut processor = TestProcessor::new("stats-processor".to_string(), |s| Ok(s.to_string()));

    // Process multiple chunks
    for i in 0..5 {
        let chunk = StreamChunk::new(
            Bytes::from(format!("chunk{}", i)),
            i as u64,
            "test".to_string(),
        );
        let _result = processor.process(chunk).await.unwrap();
    }

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 5);
    assert!(stats.processing_time_ms > 0);
    assert!(stats.throughput_mbps > 0.0);
}

#[tokio::test]
async fn test_empty_batch_processing() {
    let mut processor =
        TestProcessor::new("empty-batch-processor".to_string(), |s| Ok(s.to_string()));

    let results = processor.process_batch(vec![]).await.unwrap();
    assert!(results.is_empty());

    let stats = processor.stats().await.unwrap();
    assert_eq!(stats.chunks_processed, 0);
}

#[tokio::test]
async fn test_sink_flush_behavior() {
    let mut sink = TestSink::new("flush-test".to_string());

    // Write some chunks
    let chunk1 = StreamChunk::new(Bytes::from("chunk1"), 1, "test".to_string());
    let chunk2 = StreamChunk::new(Bytes::from("chunk2"), 2, "test".to_string());

    sink.write(chunk1).await.unwrap();
    sink.write(chunk2).await.unwrap();

    // Flush should not affect the written chunks
    sink.flush().await.unwrap();

    let written_chunks = sink.get_chunks().await;
    assert_eq!(written_chunks.len(), 2);
}

#[tokio::test]
async fn test_stream_component_ids() {
    let source = TestSource::new("unique-source-id".to_string(), vec!["data"]);
    let processor = TestProcessor::new("unique-processor-id".to_string(), |s| Ok(s.to_string()));
    let sink = TestSink::new("unique-sink-id".to_string());

    assert_eq!(source.source_id(), "unique-source-id");
    assert_eq!(processor.processor_id(), "unique-processor-id");
    assert_eq!(sink.sink_id(), "unique-sink-id");
}
