//! Memory-based stream sink

use crate::{StreamChunk, StreamSink, StreamStats, StreamingError};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

/// In-memory stream sink for testing
pub struct MemoryStreamSink {
    id: String,
    chunks: Arc<Mutex<Vec<StreamChunk>>>,
    stats: StreamStats,
}

impl MemoryStreamSink {
    /// Create a new memory stream sink
    pub fn new(id: String) -> Self {
        Self {
            id,
            chunks: Arc::new(Mutex::new(Vec::new())),
            stats: StreamStats::default(),
        }
    }

    /// Get all written chunks
    pub async fn get_chunks(&self) -> Vec<StreamChunk> {
        self.chunks.lock().await.clone()
    }
}

#[async_trait]
impl StreamSink for MemoryStreamSink {
    async fn write(&mut self, chunk: StreamChunk) -> Result<(), StreamingError> {
        self.stats.chunks_processed += 1;
        self.stats.bytes_processed += chunk.size() as u64;
        self.chunks.lock().await.push(chunk);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), StreamingError> {
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.stats.clone())
    }

    fn sink_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_memory_stream_sink() {
        let mut sink = MemoryStreamSink::new("test-sink".to_string());
        assert_eq!(sink.sink_id(), "test-sink");

        let chunk1 = StreamChunk::new(Bytes::from("test1"), 1, "source".to_string());
        let chunk2 = StreamChunk::new(Bytes::from("test2"), 2, "source".to_string());

        sink.write(chunk1).await.unwrap();
        sink.write(chunk2).await.unwrap();
        sink.flush().await.unwrap();

        let chunks = sink.get_chunks().await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(String::from_utf8_lossy(&chunks[0].data), "test1");
        assert_eq!(String::from_utf8_lossy(&chunks[1].data), "test2");

        let stats = sink.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 2);
        assert_eq!(stats.bytes_processed, 10); // 5 + 5
    }
}
