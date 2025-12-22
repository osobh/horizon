//! mTLS gRPC server and client demonstration
//!
//! This example shows how to set up mutual TLS authentication between
//! a gRPC server and client using horizon-authx certificates.
//!
//! Run with: cargo run --example grpc_mtls_demo

use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_rpc::grpc::{GrpcClientBuilder, GrpcServerBuilder};
use std::net::SocketAddr;
use std::time::Duration;

mod test_proto {
    tonic::include_proto!("test");
}

use test_proto::{
    echo_client::EchoClient,
    echo_server::{Echo, EchoServer},
    EchoRequest, EchoResponse,
};

#[derive(Default)]
struct EchoService;

#[tonic::async_trait]
impl Echo for EchoService {
    async fn echo(
        &self,
        request: tonic::Request<EchoRequest>,
    ) -> Result<tonic::Response<EchoResponse>, tonic::Status> {
        let msg = request.into_inner().message;
        println!("[Server] Received authenticated message: {}", msg);
        Ok(tonic::Response::new(EchoResponse { message: msg }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== mTLS gRPC Demonstration ===\n");

    // 1. Generate certificates
    println!("1. Generating certificates...");
    let ca = generate_ca_cert("Horizon Demo CA")?;
    println!("   - CA certificate generated");

    let server_identity = ServiceIdentity::new("demo-server");
    let server_cert = generate_signed_cert(&server_identity, &ca)?;
    println!("   - Server certificate generated");

    let client_identity = ServiceIdentity::new("demo-client");
    let client_cert = generate_signed_cert(&client_identity, &ca)?;
    println!("   - Client certificate generated\n");

    // 2. Start mTLS server
    println!("2. Starting mTLS gRPC server...");
    let addr: SocketAddr = "127.0.0.1:50052".parse()?;

    let server = GrpcServerBuilder::new(addr)
        .with_tls(server_cert)?
        .with_client_auth(ca.clone())?
        .add_service(EchoServer::new(EchoService))
        .build()?;

    let server_handle = tokio::spawn(async move {
        if let Err(e) = server.serve().await {
            eprintln!("Server error: {}", e);
        }
    });

    println!("   - Server listening on {}\n", addr);

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 3. Create mTLS client
    println!("3. Creating mTLS gRPC client...");
    let channel: tonic::transport::Channel = GrpcClientBuilder::new("https://127.0.0.1:50052")
        .with_server_ca(ca)
        .with_client_cert(client_cert)
        .connect()
        .await?;
    let client = EchoClient::new(channel);

    println!("   - Client connected with mTLS\n");

    // 4. Send test requests
    println!("4. Sending test requests...\n");

    for i in 1..=5 {
        let request = tonic::Request::new(EchoRequest {
            message: format!("mTLS message #{}", i),
        });

        let response = client.clone().echo(request).await?;
        println!("[Client] Response: {}", response.into_inner().message);

        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    println!("\n=== mTLS demonstration complete ===");
    println!("All requests authenticated successfully!");

    server_handle.abort();

    Ok(())
}
