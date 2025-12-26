//! Simple gRPC echo server example
//!
//! This example demonstrates a basic gRPC server that echoes messages back to clients.
//!
//! Run with: cargo run --example grpc_echo_server
//!
//! Test with: cargo run --example grpc_echo_client

use hpc_rpc::grpc::GrpcServerBuilder;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// Test proto schema
mod test_proto {
    tonic::include_proto!("test");
}

use test_proto::{
    echo_server::{Echo, EchoServer},
    EchoRequest, EchoResponse,
};

#[derive(Clone)]
struct EchoService {
    counter: Arc<AtomicU64>,
}

impl EchoService {
    fn new() -> Self {
        Self {
            counter: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[tonic::async_trait]
impl Echo for EchoService {
    async fn echo(
        &self,
        request: tonic::Request<EchoRequest>,
    ) -> Result<tonic::Response<EchoResponse>, tonic::Status> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);
        let msg = request.into_inner().message;

        println!("[Request #{}] Received: {}", count + 1, msg);

        Ok(tonic::Response::new(EchoResponse { message: msg }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let addr: SocketAddr = "127.0.0.1:50051".parse()?;
    let service = EchoService::new();

    println!("Starting gRPC echo server on {}", addr);
    println!("Test with: grpcurl -plaintext -d '{{\"message\": \"hello\"}}' localhost:50051 test.Echo/Echo");

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(service))
        .build()?;

    server.serve().await?;

    Ok(())
}
