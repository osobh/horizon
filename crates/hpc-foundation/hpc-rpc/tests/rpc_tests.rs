//! Comprehensive RPC integration tests (TDD RED phase)
//!
//! These tests drive the implementation of gRPC and QUIC abstractions.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

// Test proto schema for gRPC tests
mod test_proto {
    tonic::include_proto!("test");
}

use test_proto::{
    echo_server::{Echo, EchoServer},
    EchoRequest, EchoResponse,
};

#[derive(Default, Clone)]
struct EchoService {
    counter: Arc<AtomicU64>,
}

#[tonic::async_trait]
impl Echo for EchoService {
    async fn echo(
        &self,
        request: tonic::Request<EchoRequest>,
    ) -> Result<tonic::Response<EchoResponse>, tonic::Status> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        let msg = request.into_inner().message;
        Ok(tonic::Response::new(EchoResponse { message: msg }))
    }
}

// ============================================================================
// gRPC Server Builder Tests
// ============================================================================

#[tokio::test]
async fn test_grpc_server_plaintext() {
    use hpc_rpc::grpc::GrpcServerBuilder;

    let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
    let service = EchoService::default();

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(service))
        .build()
        .expect("Failed to build server");

    let handle = tokio::spawn(async move {
        server.serve().await.expect("Server failed");
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect with client
    use test_proto::echo_client::EchoClient;
    let mut client = EchoClient::connect("http://127.0.0.1:50051")
        .await
        .expect("Failed to connect");

    let response = client
        .echo(tonic::Request::new(EchoRequest {
            message: "hello".to_string(),
        }))
        .await
        .expect("Request failed");

    assert_eq!(response.into_inner().message, "hello");

    handle.abort();
}

#[tokio::test]
async fn test_grpc_server_with_tls() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::grpc::GrpcServerBuilder;

    // Generate test certificates with proper SANs
    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("test-server")
        .with_dns_names(vec!["localhost", "127.0.0.1"]);
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let addr: SocketAddr = "127.0.0.1:50052".parse().unwrap();
    let service = EchoService::default();

    let server = GrpcServerBuilder::new(addr)
        .with_tls(server_cert)
        .expect("TLS config failed")
        .add_service(EchoServer::new(service))
        .build()
        .expect("Failed to build TLS server");

    let handle = tokio::spawn(async move {
        server.serve().await.expect("Server failed");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Connect with TLS client
    use hpc_rpc::grpc::GrpcClientBuilder;
    use test_proto::echo_client::EchoClient;

    let channel = GrpcClientBuilder::new("https://127.0.0.1:50052")
        .with_server_ca(ca)
        .connect::<tonic::transport::Channel>()
        .await
        .expect("Failed to connect with TLS");

    let client = EchoClient::new(channel);

    let response = client
        .clone()
        .echo(tonic::Request::new(EchoRequest {
            message: "secure hello".to_string(),
        }))
        .await
        .expect("TLS request failed");

    assert_eq!(response.into_inner().message, "secure hello");

    handle.abort();
}

#[tokio::test]
async fn test_grpc_server_with_mtls() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::grpc::GrpcServerBuilder;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("test-server")
        .with_dns_names(vec!["localhost", "127.0.0.1"]);
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();
    let client_identity = ServiceIdentity::new("test-client");
    let client_cert = generate_signed_cert(&client_identity, &ca).unwrap();

    // Clone CA for client before consuming in server
    let ca_for_client = ca.clone();

    let addr: SocketAddr = "127.0.0.1:50053".parse().unwrap();
    let service = EchoService::default();

    let server = GrpcServerBuilder::new(addr)
        .with_tls(server_cert)
        .expect("TLS config failed")
        .with_client_auth(ca)
        .expect("mTLS config failed")
        .add_service(EchoServer::new(service))
        .build()
        .expect("Failed to build mTLS server");

    let handle = tokio::spawn(async move {
        server.serve().await.expect("Server failed");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    use hpc_rpc::grpc::GrpcClientBuilder;
    use test_proto::echo_client::EchoClient;

    let channel = GrpcClientBuilder::new("https://127.0.0.1:50053")
        .with_server_ca(ca_for_client)
        .with_client_cert(client_cert)
        .connect::<tonic::transport::Channel>()
        .await
        .expect("Failed to connect with mTLS");

    let client = EchoClient::new(channel);

    let response = client
        .clone()
        .echo(tonic::Request::new(EchoRequest {
            message: "mtls hello".to_string(),
        }))
        .await
        .expect("mTLS request failed");

    assert_eq!(response.into_inner().message, "mtls hello");

    handle.abort();
}

// ============================================================================
// gRPC Client Tests
// ============================================================================

#[tokio::test]
async fn test_grpc_client_connection_pooling() {
    use hpc_rpc::grpc::{GrpcClientBuilder, GrpcServerBuilder};
    use test_proto::echo_client::EchoClient;

    let addr: SocketAddr = "127.0.0.1:50054".parse().unwrap();
    let service = EchoService::default();
    let counter = service.counter.clone();

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(service))
        .build()
        .unwrap();

    let handle = tokio::spawn(async move {
        server.serve().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create client with connection pool
    let channel = GrpcClientBuilder::new("http://127.0.0.1:50054")
        .with_pool_size(5)
        .connect::<tonic::transport::Channel>()
        .await
        .unwrap();

    let client = EchoClient::new(channel);

    // Send multiple concurrent requests
    let mut handles = vec![];
    for i in 0..10 {
        let mut c = client.clone();
        handles.push(tokio::spawn(async move {
            c.echo(tonic::Request::new(EchoRequest {
                message: format!("msg-{}", i),
            }))
            .await
        }));
    }

    for h in handles {
        h.await.unwrap().unwrap();
    }

    assert_eq!(counter.load(Ordering::SeqCst), 10);

    handle.abort();
}

#[tokio::test]
async fn test_grpc_client_timeout() {
    use hpc_rpc::grpc::{GrpcClientBuilder, GrpcServerBuilder};
    use test_proto::echo_client::EchoClient;

    let addr: SocketAddr = "127.0.0.1:50055".parse().unwrap();

    // Slow service that delays responses
    #[derive(Default)]
    struct SlowEchoService;

    #[tonic::async_trait]
    impl Echo for SlowEchoService {
        async fn echo(
            &self,
            request: tonic::Request<EchoRequest>,
        ) -> Result<tonic::Response<EchoResponse>, tonic::Status> {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let msg = request.into_inner().message;
            Ok(tonic::Response::new(EchoResponse { message: msg }))
        }
    }

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(SlowEchoService))
        .build()
        .unwrap();

    let handle = tokio::spawn(async move {
        server.serve().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let channel = GrpcClientBuilder::new("http://127.0.0.1:50055")
        .with_timeout(Duration::from_millis(500))
        .connect::<tonic::transport::Channel>()
        .await
        .unwrap();

    let client = EchoClient::new(channel);

    // Should timeout
    let result = timeout(
        Duration::from_secs(1),
        client.clone().echo(tonic::Request::new(EchoRequest {
            message: "test".to_string(),
        })),
    )
    .await;

    assert!(result.is_ok()); // Didn't hang
    assert!(result.unwrap().is_err()); // But request failed (timeout)

    handle.abort();
}

// ============================================================================
// QUIC Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_quic_endpoint_creation() {
    use hpc_rpc::quic::QuicEndpoint;

    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint = QuicEndpoint::client(addr).expect("Failed to create client endpoint");

    assert!(endpoint.local_addr().is_ok());
}

#[tokio::test]
async fn test_quic_server_endpoint_with_tls() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("quic-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint = QuicEndpoint::server(addr, server_cert)
        .expect("Failed to create server endpoint");

    assert!(endpoint.local_addr().is_ok());
}

#[tokio::test]
async fn test_quic_bidirectional_stream() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("quic-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6000".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert)
        .expect("Failed to create server");

    // Server task
    let server_handle = tokio::spawn(async move {
        let incoming = server_endpoint.accept().await.expect("No incoming connection");
        let conn = incoming.await.expect("Connection failed");

        let (mut send, mut recv) = conn.accept_bi().await.expect("Failed to accept stream");

        let mut buf = vec![0u8; 1024];
        let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
        let msg = String::from_utf8(buf[..n].to_vec()).unwrap();

        assert_eq!(msg, "ping");

        send.write_all(b"pong").await.expect("Write failed");
        send.finish().await.expect("Finish failed");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Client
    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client_endpoint = QuicEndpoint::client(client_addr)
        .expect("Failed to create client endpoint");

    let conn = client_endpoint
        .connect(server_addr, "quic-server", ca)
        .await
        .expect("Client connection failed");

    let (mut send, mut recv) = conn.open_bi().await.expect("Failed to open stream");

    send.write_all(b"ping").await.expect("Write failed");
    send.finish().await.expect("Finish failed");

    let mut buf = vec![0u8; 1024];
    let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
    let msg = String::from_utf8(buf[..n].to_vec()).unwrap();

    assert_eq!(msg, "pong");

    server_handle.await.unwrap();
}

#[tokio::test]
async fn test_quic_unidirectional_stream() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("quic-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6001".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert)
        .expect("Failed to create server");

    let server_handle = tokio::spawn(async move {
        let incoming = server_endpoint.accept().await.expect("No incoming connection");
        let conn = incoming.await.expect("Connection failed");

        let mut recv = conn.accept_uni().await.expect("Failed to accept uni stream");

        let mut buf = vec![0u8; 1024];
        let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
        let msg = String::from_utf8(buf[..n].to_vec()).unwrap();

        assert_eq!(msg, "one-way message");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client_endpoint = QuicEndpoint::client(client_addr)
        .expect("Failed to create client endpoint");

    let conn = client_endpoint
        .connect(server_addr, "quic-server", ca)
        .await
        .expect("Client connection failed");

    let mut send = conn.open_uni().await.expect("Failed to open uni stream");

    send.write_all(b"one-way message").await.expect("Write failed");
    send.finish().await.expect("Finish failed");

    server_handle.await.unwrap();
}

// ============================================================================
// Backpressure and Flow Control Tests
// ============================================================================

#[tokio::test]
async fn test_quic_stream_backpressure() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("quic-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6002".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert)
        .expect("Failed to create server");

    let server_handle = tokio::spawn(async move {
        let incoming = server_endpoint.accept().await.expect("No incoming connection");
        let conn = incoming.await.expect("Connection failed");

        let (_, mut recv) = conn.accept_bi().await.expect("Failed to accept stream");

        // Slow consumer - simulate backpressure
        let mut total = 0;
        let mut buf = vec![0u8; 1024];
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            match recv.read(&mut buf).await {
                Ok(Some(n)) => total += n,
                Ok(None) => break,
                Err(e) => panic!("Read error: {}", e),
            }
        }

        assert!(total > 0);
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client_endpoint = QuicEndpoint::client(client_addr)
        .expect("Failed to create client endpoint");

    let conn = client_endpoint
        .connect(server_addr, "quic-server", ca)
        .await
        .expect("Client connection failed");

    let (mut send, _) = conn.open_bi().await.expect("Failed to open stream");

    // Send large amount of data
    let data = vec![42u8; 1024 * 1024]; // 1MB
    send.write_all(&data).await.expect("Write failed");
    send.finish().await.expect("Finish failed");

    server_handle.await.unwrap();
}

// ============================================================================
// Connection Management Tests
// ============================================================================

#[tokio::test]
async fn test_quic_multiple_streams_on_connection() {
    use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();
    let server_identity = ServiceIdentity::new("quic-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6003".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert)
        .expect("Failed to create server");

    let server_handle = tokio::spawn(async move {
        let incoming = server_endpoint.accept().await.expect("No incoming connection");
        let conn = incoming.await.expect("Connection failed");

        // Accept multiple streams
        for _ in 0..5 {
            let (mut send, mut recv) = conn.accept_bi().await.expect("Failed to accept stream");

            tokio::spawn(async move {
                let mut buf = vec![0u8; 1024];
                let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
                let msg = String::from_utf8(buf[..n].to_vec()).unwrap();

                send.write_all(format!("echo: {}", msg).as_bytes())
                    .await
                    .expect("Write failed");
                send.finish().await.expect("Finish failed");
            });
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client_endpoint = QuicEndpoint::client(client_addr)
        .expect("Failed to create client endpoint");

    let conn = client_endpoint
        .connect(server_addr, "quic-server", ca)
        .await
        .expect("Client connection failed");

    // Open multiple streams concurrently
    let mut handles = vec![];
    for i in 0..5 {
        let c = conn.clone();
        handles.push(tokio::spawn(async move {
            let (mut send, mut recv) = c.open_bi().await.expect("Failed to open stream");

            send.write_all(format!("stream-{}", i).as_bytes())
                .await
                .expect("Write failed");
            send.finish().await.expect("Finish failed");

            let mut buf = vec![0u8; 1024];
            let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
            String::from_utf8(buf[..n].to_vec()).unwrap()
        }));
    }

    for h in handles {
        let response = h.await.unwrap();
        assert!(response.starts_with("echo: stream-"));
    }

    server_handle.await.unwrap();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_grpc_connection_refused() {
    use hpc_rpc::grpc::GrpcClientBuilder;

    // Try to connect to non-existent server
    let result = GrpcClientBuilder::new("http://127.0.0.1:9999")
        .connect::<tonic::transport::Channel>()
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_quic_connection_timeout() {
    use hpc_auth::cert::generate_ca_cert;
    use hpc_rpc::quic::QuicEndpoint;

    let ca = generate_ca_cert("Test CA").unwrap();

    let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let client_endpoint = QuicEndpoint::client(client_addr)
        .expect("Failed to create client endpoint");

    // Try to connect to non-existent server with timeout
    let server_addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
    let result = timeout(
        Duration::from_millis(500),
        client_endpoint.connect(server_addr, "nonexistent", ca),
    )
    .await;

    assert!(result.is_err() || result.unwrap().is_err());
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_grpc_echo_property(msg in "\\PC*") {
            let result = tokio::runtime::Runtime::new().unwrap().block_on(async {
                use hpc_rpc::grpc::{GrpcServerBuilder, GrpcClientBuilder};
                use test_proto::echo_client::EchoClient;

                let addr: SocketAddr = "127.0.0.1:50060".parse().unwrap();
                let service = EchoService::default();

                let server = GrpcServerBuilder::new(addr)
                    .add_service(EchoServer::new(service))
                    .build()
                    .unwrap();

                let handle = tokio::spawn(async move {
                    server.serve().await.unwrap();
                });

                tokio::time::sleep(Duration::from_millis(100)).await;

                let channel = GrpcClientBuilder::new("http://127.0.0.1:50060")
                    .connect::<tonic::transport::Channel>()
                    .await
                    .unwrap();

                let client = EchoClient::new(channel);

                let response = client
                    .clone()
                    .echo(tonic::Request::new(EchoRequest {
                        message: msg.clone(),
                    }))
                    .await
                    .unwrap();

                handle.abort();

                response.into_inner().message
            });

            prop_assert_eq!(result, msg);
        }
    }
}
