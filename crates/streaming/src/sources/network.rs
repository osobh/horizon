//! Network-based stream source for distributed data streaming

use crate::{StreamChunk, StreamSource, StreamStats, StreamingError};
use async_trait::async_trait;
use bytes::Bytes;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, BufReader};
use tokio::net::{TcpStream, UdpSocket};
use tokio_stream::Stream;

/// Network protocol types
#[derive(Debug, Clone)]
pub enum NetworkProtocol {
    Tcp,
    Udp,
    Http,
    WebSocket,
}

/// Network stream source for receiving data over the network
pub struct NetworkStreamSource {
    id: String,
    address: SocketAddr,
    protocol: NetworkProtocol,
    chunk_size: usize,
    timeout_ms: u64,
    stats: Arc<NetworkSourceStats>,
}

/// Thread-safe statistics for network source
///
/// Cache-line aligned (64 bytes) to prevent false sharing when
/// multiple network threads update counters concurrently.
#[repr(C, align(64))]
#[derive(Debug, Default)]
struct NetworkSourceStats {
    chunks_received: AtomicU64,
    bytes_received: AtomicU64,
    network_time_ns: AtomicU64,
    connection_errors: AtomicU64,
    // Padding to fill cache line (4 * 8 = 32 bytes, need 32 more)
    _padding: [u8; 32],
}

impl NetworkStreamSource {
    /// Create a new network stream source
    pub fn new(id: String, address: SocketAddr, protocol: NetworkProtocol) -> Self {
        Self {
            id,
            address,
            protocol,
            chunk_size: 64 * 1024, // 64KB chunks for network
            timeout_ms: 5000,      // 5 second timeout
            stats: Arc::new(NetworkSourceStats::default()),
        }
    }

    /// Configure chunk size for network operations
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Configure network timeout
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Get current statistics snapshot
    pub fn get_stats_snapshot(&self) -> StreamStats {
        // Relaxed: independent statistics counters with no ordering dependencies
        let chunks = self.stats.chunks_received.load(Ordering::Relaxed);
        let bytes = self.stats.bytes_received.load(Ordering::Relaxed);
        let time_ns = self.stats.network_time_ns.load(Ordering::Relaxed);
        let errors = self.stats.connection_errors.load(Ordering::Relaxed);

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

    /// Create TCP stream
    async fn create_tcp_stream(
        &self,
    ) -> Result<impl Stream<Item = Result<StreamChunk, StreamingError>>, StreamingError> {
        let stream = TcpStream::connect(self.address).await.map_err(|e| {
            StreamingError::IoError(format!("Failed to connect to TCP {}: {}", self.address, e))
        })?;

        let chunk_size = self.chunk_size;
        let source_id = self.id.clone();
        let stats = Arc::clone(&self.stats);

        let stream = async_stream::stream! {
            let mut reader = BufReader::new(stream);
            let mut sequence = 0u64;
            let mut buffer = vec![0u8; chunk_size];

            loop {
                let start_time = Instant::now();

                match reader.read(&mut buffer).await {
                    Ok(bytes_read) => {
                        if bytes_read == 0 {
                            // Connection closed
                            break;
                        }

                        buffer.truncate(bytes_read);
                        let network_time = start_time.elapsed().as_nanos() as u64;

                        // Relaxed: independent statistics counters
                        stats.network_time_ns.fetch_add(network_time, Ordering::Relaxed);
                        stats.chunks_received.fetch_add(1, Ordering::Relaxed);
                        stats.bytes_received.fetch_add(bytes_read as u64, Ordering::Relaxed);

                        let chunk = StreamChunk::new(
                            Bytes::copy_from_slice(&buffer),
                            sequence,
                            source_id.clone(),
                        );

                        yield Ok(chunk);

                        sequence += 1;
                        buffer.resize(chunk_size, 0);
                    }
                    Err(e) => {
                        // Relaxed: independent error counter
                        stats.connection_errors.fetch_add(1, Ordering::Relaxed);
                        yield Err(StreamingError::IoError(format!("TCP read failed: {e}")));
                        break;
                    }
                }
            }
        };

        Ok(stream)
    }

    /// Create UDP stream
    async fn create_udp_stream(
        &self,
    ) -> Result<impl Stream<Item = Result<StreamChunk, StreamingError>>, StreamingError> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| StreamingError::IoError(format!("Failed to bind UDP socket: {e}")))?;

        socket.connect(self.address).await.map_err(|e| {
            StreamingError::IoError(format!("Failed to connect UDP to {}: {}", self.address, e))
        })?;

        let chunk_size = self.chunk_size;
        let source_id = self.id.clone();
        let stats = Arc::clone(&self.stats);

        let stream = async_stream::stream! {
            let mut sequence = 0u64;
            let mut buffer = vec![0u8; chunk_size];

            loop {
                let start_time = Instant::now();

                match socket.recv(&mut buffer).await {
                    Ok(bytes_read) => {
                        let network_time = start_time.elapsed().as_nanos() as u64;

                        // Relaxed: independent statistics counters
                        stats.network_time_ns.fetch_add(network_time, Ordering::Relaxed);
                        stats.chunks_received.fetch_add(1, Ordering::Relaxed);
                        stats.bytes_received.fetch_add(bytes_read as u64, Ordering::Relaxed);

                        let chunk = StreamChunk::new(
                            Bytes::copy_from_slice(&buffer[..bytes_read]),
                            sequence,
                            source_id.clone(),
                        );

                        yield Ok(chunk);
                        sequence += 1;
                    }
                    Err(e) => {
                        // Relaxed: independent error counter
                        stats.connection_errors.fetch_add(1, Ordering::Relaxed);
                        yield Err(StreamingError::IoError(format!("UDP recv failed: {e}")));
                        break;
                    }
                }
            }
        };

        Ok(stream)
    }
}

#[async_trait]
impl StreamSource for NetworkStreamSource {
    async fn start(
        &mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamingError>> + Send>>,
        StreamingError,
    > {
        match self.protocol {
            NetworkProtocol::Tcp => {
                let stream = self.create_tcp_stream().await?;
                Ok(Box::pin(stream))
            }
            NetworkProtocol::Udp => {
                let stream = self.create_udp_stream().await?;
                Ok(Box::pin(stream))
            }
            NetworkProtocol::Http | NetworkProtocol::WebSocket => {
                Err(StreamingError::InvalidInput(format!(
                    "Protocol {:?} not yet implemented",
                    self.protocol
                )))
            }
        }
    }

    async fn stop(&mut self) -> Result<(), StreamingError> {
        // Network connections are automatically closed when dropped
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
    use std::net::{IpAddr, Ipv4Addr};
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio_stream::StreamExt;

    async fn create_test_tcp_server(data: Vec<u8>) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let _ = stream.write_all(&data).await;
                let _ = stream.shutdown().await;
            }
        });

        addr
    }

    #[tokio::test]
    async fn test_network_source_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let source =
            NetworkStreamSource::new("test-network".to_string(), addr, NetworkProtocol::Tcp);

        assert_eq!(source.source_id(), "test-network");
        assert_eq!(source.address, addr);
        assert_eq!(source.chunk_size, 64 * 1024);
        assert_eq!(source.timeout_ms, 5000);
    }

    #[tokio::test]
    async fn test_network_source_configuration() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let source =
            NetworkStreamSource::new("test-network".to_string(), addr, NetworkProtocol::Tcp)
                .with_chunk_size(1024)
                .with_timeout_ms(10000);

        assert_eq!(source.chunk_size, 1024);
        assert_eq!(source.timeout_ms, 10000);
    }

    #[tokio::test]
    async fn test_network_source_tcp_stream() {
        let test_data = b"Hello from TCP server!".to_vec();
        let server_addr = create_test_tcp_server(test_data.clone()).await;

        let mut source =
            NetworkStreamSource::new("tcp-test".to_string(), server_addr, NetworkProtocol::Tcp)
                .with_chunk_size(1024);

        let mut stream = source.start().await.unwrap();
        let mut chunks = Vec::new();

        // Read chunks with timeout
        let timeout_duration = std::time::Duration::from_millis(1000);
        while let Ok(Some(result)) = tokio::time::timeout(timeout_duration, stream.next()).await {
            match result {
                Ok(chunk) => chunks.push(chunk),
                Err(_) => break,
            }
        }

        assert!(!chunks.is_empty());

        // Reconstruct data
        let mut received_data = Vec::new();
        for chunk in &chunks {
            received_data.extend_from_slice(&chunk.data);
        }

        assert_eq!(received_data, test_data);
        assert_eq!(chunks[0].metadata.source_id, "tcp-test");

        let stats = source.stats().await.unwrap();
        assert!(stats.chunks_processed > 0);
        assert_eq!(stats.bytes_processed, test_data.len() as u64);
    }

    #[tokio::test]
    async fn test_network_source_unsupported_protocol() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut source =
            NetworkStreamSource::new("http-test".to_string(), addr, NetworkProtocol::Http);

        let result = source.start().await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::InvalidInput(msg) => assert!(msg.contains("not yet implemented")),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_network_source_connection_failure() {
        // Use a non-existent address
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9999);
        let mut source =
            NetworkStreamSource::new("failed-connection".to_string(), addr, NetworkProtocol::Tcp);

        let result = source.start().await;
        assert!(result.is_err());
        match result.err().unwrap() {
            StreamingError::IoError(msg) => assert!(msg.contains("Failed to connect to TCP")),
            _ => panic!("Expected IoError"),
        }
    }

    #[tokio::test]
    async fn test_network_source_stop() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut source =
            NetworkStreamSource::new("stop-test".to_string(), addr, NetworkProtocol::Tcp);

        let result = source.stop().await;
        assert!(result.is_ok());
    }
}
