//! Error types for RPC operations

use thiserror::Error;

/// Errors that can occur during RPC operations
#[derive(Error, Debug)]
pub enum RpcError {
    /// gRPC transport error
    #[error("gRPC transport error: {0}")]
    GrpcTransport(#[from] tonic::transport::Error),

    /// gRPC status error
    #[error("gRPC status error: {0}")]
    GrpcStatus(Box<tonic::Status>),

    /// QUIC connection error
    #[error("QUIC connection error: {0}")]
    QuicConnection(#[from] quinn::ConnectionError),

    /// QUIC connect error
    #[error("QUIC connect error: {0}")]
    QuicConnect(#[from] quinn::ConnectError),

    /// QUIC write error
    #[error("QUIC write error: {0}")]
    QuicWrite(#[from] quinn::WriteError),

    /// QUIC read error
    #[error("QUIC read error: {0}")]
    QuicRead(#[source] std::io::Error),

    /// TLS/Certificate error
    #[error("TLS error: {0}")]
    Tls(#[from] hpc_auth::AuthError),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Connection timeout
    #[error("Connection timeout after {0:?}")]
    Timeout(std::time::Duration),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for RPC operations
pub type Result<T> = std::result::Result<T, RpcError>;

impl From<tonic::Status> for RpcError {
    fn from(status: tonic::Status) -> Self {
        RpcError::GrpcStatus(Box::new(status))
    }
}
