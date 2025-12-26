//! QUIC stream helpers for bidirectional and unidirectional communication

use crate::error::{RpcError, Result};
use quinn::{RecvStream, SendStream};
use std::io;
use tracing::debug;

/// Bidirectional QUIC stream
///
/// Provides helpers for reading and writing on a bidirectional stream.
pub struct QuicBiStream {
    send: SendStream,
    recv: RecvStream,
}

impl QuicBiStream {
    /// Create a new bidirectional stream wrapper
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self { send, recv }
    }

    /// Split into separate send and receive streams
    pub fn split(self) -> (SendStream, RecvStream) {
        (self.send, self.recv)
    }

    /// Write all data to the send stream
    pub async fn write_all(&mut self, data: &[u8]) -> Result<()> {
        debug!(bytes = data.len(), "Writing to bidirectional stream");
        self.send
            .write_all(data)
            .await
            .map_err(RpcError::QuicWrite)?;
        Ok(())
    }

    /// Read data from the receive stream
    ///
    /// Returns `Ok(Some(n))` where n is the number of bytes read,
    /// or `Ok(None)` if the stream has been closed.
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<Option<usize>> {
        match self.recv.read(buf).await {
            Ok(Some(n)) => {
                debug!(bytes = n, "Read from bidirectional stream");
                Ok(Some(n))
            }
            Ok(None) => {
                debug!("Bidirectional stream closed");
                Ok(None)
            }
            Err(e) => {
                debug!(?e, "Read error on bidirectional stream");
                Err(RpcError::QuicRead(io::Error::other(e.to_string())))
            }
        }
    }

    /// Finish the send stream
    pub async fn finish(&mut self) -> Result<()> {
        debug!("Finishing bidirectional stream");
        self.send
            .finish()
            .await
            .map_err(RpcError::QuicWrite)?;
        Ok(())
    }

    /// Read all remaining data from the stream until EOF
    pub async fn read_to_end(&mut self, max_size: usize) -> Result<Vec<u8>> {
        debug!(max_size, "Reading to end of bidirectional stream");
        let data = self
            .recv
            .read_to_end(max_size)
            .await
            .map_err(|e| RpcError::QuicRead(io::Error::other(e.to_string())))?;
        Ok(data)
    }
}

/// Unidirectional QUIC stream (send-only)
pub struct QuicUniStream {
    send: SendStream,
}

impl QuicUniStream {
    /// Create a new unidirectional stream wrapper
    pub fn new(send: SendStream) -> Self {
        Self { send }
    }

    /// Write all data to the stream
    pub async fn write_all(&mut self, data: &[u8]) -> Result<()> {
        debug!(bytes = data.len(), "Writing to unidirectional stream");
        self.send
            .write_all(data)
            .await
            .map_err(RpcError::QuicWrite)?;
        Ok(())
    }

    /// Finish the stream
    pub async fn finish(&mut self) -> Result<()> {
        debug!("Finishing unidirectional stream");
        self.send
            .finish()
            .await
            .map_err(RpcError::QuicWrite)?;
        Ok(())
    }

    /// Set stream priority
    pub fn set_priority(&self, priority: i32) {
        let _ = self.send.set_priority(priority);
    }
}

/// Extension trait for Connection to provide stream helpers
#[allow(dead_code)]
pub trait ConnectionExt {
    /// Open a bidirectional stream with helpers
    async fn open_bi_stream(&self) -> Result<QuicBiStream>;

    /// Open a unidirectional stream with helpers
    async fn open_uni_stream(&self) -> Result<QuicUniStream>;

    /// Accept a bidirectional stream with helpers
    async fn accept_bi_stream(&self) -> Result<QuicBiStream>;

    /// Accept a unidirectional receive stream
    async fn accept_uni_stream(&self) -> Result<RecvStream>;
}

impl ConnectionExt for quinn::Connection {
    async fn open_bi_stream(&self) -> Result<QuicBiStream> {
        let (send, recv) = self.open_bi().await?;
        Ok(QuicBiStream::new(send, recv))
    }

    async fn open_uni_stream(&self) -> Result<QuicUniStream> {
        let send = self.open_uni().await?;
        Ok(QuicUniStream::new(send))
    }

    async fn accept_bi_stream(&self) -> Result<QuicBiStream> {
        let (send, recv) = self.accept_bi().await?;
        Ok(QuicBiStream::new(send, recv))
    }

    async fn accept_uni_stream(&self) -> Result<RecvStream> {
        let recv = self.accept_uni().await?;
        Ok(recv)
    }
}

#[cfg(test)]
mod tests {

    // Unit tests for stream wrappers
    // Integration tests are in tests/rpc_tests.rs

    #[test]
    fn test_bi_stream_creation() {
        // Basic type checking test
        // Actual functionality tested in integration tests
    }

    #[test]
    fn test_uni_stream_creation() {
        // Basic type checking test
        // Actual functionality tested in integration tests
    }
}
