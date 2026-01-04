//! # Horizon RPC Abstractions (rpcx)
//!
//! High-performance RPC primitives for Horizon services, providing unified
//! interfaces for gRPC (via tonic) and QUIC (via quinn) communication.
//!
//! ## Features
//!
//! - **gRPC Server/Client**: Type-safe wrappers with optional mTLS
//! - **QUIC Endpoints**: Low-latency streaming with multiplexing
//! - **Connection Pooling**: Automatic connection management for gRPC clients
//! - **Backpressure**: Flow control for QUIC streams
//! - **TLS Integration**: Seamless integration with horizon-authx
//! - **Zero-Copy**: Efficient data transfer where possible
//!
//! ## Design Principles
//!
//! 1. **Type Safety**: Leverage Rust's type system for protocol correctness
//! 2. **Performance**: Sub-millisecond latency for local calls
//! 3. **Security**: mTLS by default in production
//! 4. **Observability**: Structured tracing for all RPC calls
//! 5. **Reliability**: Automatic reconnection and backpressure handling
//!
//! ## Examples
//!
//! ### gRPC Server (Plaintext)
//!
//! ```rust,no_run
//! use hpc_rpc::grpc::GrpcServerBuilder;
//! use std::net::SocketAddr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let addr: SocketAddr = "127.0.0.1:50051".parse()?;
//!
//!     let server = GrpcServerBuilder::new(addr)
//!         // .add_service(MyServiceServer::new(service))
//!         .build()?;
//!
//!     server.serve().await?;
//!     Ok(())
//! }
//! ```
//!
//! ### gRPC Server (mTLS)
//!
//! ```rust,no_run
//! use hpc_rpc::grpc::GrpcServerBuilder;
//! use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
//! use std::net::SocketAddr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let ca = generate_ca_cert("Horizon CA")?;
//!     let server_identity = ServiceIdentity::new("my-service");
//!     let server_cert = generate_signed_cert(&server_identity, &ca)?;
//!
//!     let addr: SocketAddr = "127.0.0.1:50051".parse()?;
//!
//!     let server = GrpcServerBuilder::new(addr)
//!         .with_tls(server_cert)
//!         .with_client_auth(ca)
//!         // .add_service(MyServiceServer::new(service))
//!         .build()?;
//!
//!     server.serve().await?;
//!     Ok(())
//! }
//! ```
//!
//! ### gRPC Client
//!
//! ```rust,no_run
//! use hpc_rpc::grpc::GrpcClientBuilder;
//! use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let ca = generate_ca_cert("Horizon CA")?;
//!     let client_identity = ServiceIdentity::new("my-client");
//!     let client_cert = generate_signed_cert(&client_identity, &ca)?;
//!
//!     // let client: MyServiceClient<_> = GrpcClientBuilder::new("https://localhost:50051")
//!     //     .with_server_ca(ca)
//!     //     .with_client_cert(client_cert)
//!     //     .with_pool_size(5)
//!     //     .connect()
//!     //     .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### QUIC Streaming
//!
//! ```rust,no_run
//! use hpc_rpc::quic::QuicEndpoint;
//! use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
//! use std::net::SocketAddr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let ca = generate_ca_cert("Horizon CA")?;
//!     let server_identity = ServiceIdentity::new("quic-server");
//!     let server_cert = generate_signed_cert(&server_identity, &ca)?;
//!
//!     let addr: SocketAddr = "127.0.0.1:6000".parse()?;
//!     let endpoint = QuicEndpoint::server(addr, server_cert)?;
//!
//!     // Accept connections
//!     let incoming = endpoint.accept().await.expect("No incoming connection");
//!     let conn = incoming.await?;
//!
//!     // Accept bidirectional stream
//!     let (mut send, mut recv) = conn.accept_bi().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod grpc;
pub mod quic;

pub use error::{Result, RpcError};

// Re-export key types for convenience
pub use grpc::{GrpcClientBuilder, GrpcServer, GrpcServerBuilder};
pub use quic::{QuicBiStream, QuicEndpoint, QuicUniStream};
