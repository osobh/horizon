//! Memory-based stream source

use crate::{StreamChunk, StreamSource, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::pin::Pin;
use tokio_stream::{iter, Stream};

/// In-memory stream source for testing
pub struct MemoryStreamSource {
    id: String,
    chunks: Vec<StreamChunk>,
    stats: StreamStats,
}

impl MemoryStreamSource {
    /// Create a new memory stream source
    pub fn new(id: String, data: Vec<Bytes>) -> Self {
        let chunks = data
            .into_iter()
            .enumerate()
            .map(|(i, bytes)| StreamChunk::new(bytes, i as u64, id.clone()))
            .collect();

        Self {
            id,
            chunks,
            stats: StreamStats::default(),
        }
    }
}

#[async_trait]
impl StreamSource for MemoryStreamSource {
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    > {
        let chunks = self.chunks.clone();
        self.stats.chunks_processed = chunks.len() as u64;
        self.stats.bytes_processed = chunks.iter().map(|c| c.size() as u64).sum();

        let stream = iter(chunks.into_iter().map(Ok));
        Ok(Box::pin(stream))
    }

    async fn stop(&mut self) -> Result<(), StreamingError> {
        Ok(())
    }

    async fn stats(&self) -> Result<StreamStats, StreamingError> {
        Ok(self.stats.clone())
    }

    fn source_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_memory_stream_source() {
        let data = vec![
            Bytes::from("chunk1"),
            Bytes::from("chunk2"),
            Bytes::from("chunk3"),
        ];

        let mut source = MemoryStreamSource::new("test-memory".to_string(), data);
        assert_eq!(source.source_id(), "test-memory");

        let mut stream = source.start().await.unwrap();

        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }

        assert_eq!(chunks.len(), 3);
        assert_eq!(String::from_utf8_lossy(&chunks[0].data), "chunk1");
        assert_eq!(String::from_utf8_lossy(&chunks[1].data), "chunk2");
        assert_eq!(String::from_utf8_lossy(&chunks[2].data), "chunk3");

        let stats = source.stats().await.unwrap();
        assert_eq!(stats.chunks_processed, 3);
        assert_eq!(stats.bytes_processed, 18); // 6 + 6 + 6
    }
}
