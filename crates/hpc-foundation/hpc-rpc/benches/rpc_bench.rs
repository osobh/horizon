//! Performance benchmarks for RPC operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_rpc::grpc::{GrpcClientBuilder, GrpcServerBuilder};
use hpc_rpc::quic::QuicEndpoint;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::runtime::Runtime;

// Test proto schema
mod test_proto {
    tonic::include_proto!("test");
}

use test_proto::{
    echo_client::EchoClient,
    echo_server::{Echo, EchoServer},
    EchoRequest, EchoResponse,
};

#[derive(Default, Clone)]
struct EchoService;

#[tonic::async_trait]
impl Echo for EchoService {
    async fn echo(
        &self,
        request: tonic::Request<EchoRequest>,
    ) -> Result<tonic::Response<EchoResponse>, tonic::Status> {
        let msg = request.into_inner().message;
        Ok(tonic::Response::new(EchoResponse { message: msg }))
    }
}

fn grpc_latency_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Start server
    let addr: SocketAddr = "127.0.0.1:50100".parse().unwrap();
    let service = EchoService::default();

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(service))
        .build()
        .unwrap();

    rt.spawn(async move {
        server.serve().await.unwrap();
    });

    // Wait for server to start
    std::thread::sleep(Duration::from_millis(200));

    // Create client
    let client = rt.block_on(async {
        let c: EchoClient<_> = GrpcClientBuilder::new("http://127.0.0.1:50100")
            .connect()
            .await
            .unwrap();
        c
    });

    let mut group = c.benchmark_group("grpc_latency");

    for size in [16, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let msg = "x".repeat(size);
            let mut client_clone = client.clone();

            b.to_async(&rt).iter(|| async {
                let req = tonic::Request::new(EchoRequest {
                    message: msg.clone(),
                });
                black_box(client_clone.echo(req).await.unwrap());
            });
        });
    }

    group.finish();
}

fn grpc_throughput_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Start server
    let addr: SocketAddr = "127.0.0.1:50101".parse().unwrap();
    let service = EchoService::default();

    let server = GrpcServerBuilder::new(addr)
        .add_service(EchoServer::new(service))
        .build()
        .unwrap();

    rt.spawn(async move {
        server.serve().await.unwrap();
    });

    std::thread::sleep(Duration::from_millis(200));

    let client = rt.block_on(async {
        let c: EchoClient<_> = GrpcClientBuilder::new("http://127.0.0.1:50101")
            .with_pool_size(10)
            .connect()
            .await
            .unwrap();
        c
    });

    c.bench_function("grpc_throughput_concurrent", |b| {
        let msg = "benchmark message".to_string();
        let client_clone = client.clone();

        b.to_async(&rt).iter(|| async {
            let mut handles = vec![];
            for _ in 0..100 {
                let mut c = client_clone.clone();
                let m = msg.clone();
                handles.push(tokio::spawn(async move {
                    c.echo(tonic::Request::new(EchoRequest { message: m }))
                        .await
                        .unwrap();
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        });
    });
}

fn quic_stream_latency_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let ca = generate_ca_cert("Bench CA").unwrap();
    let server_identity = ServiceIdentity::new("bench-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6100".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert).unwrap();

    // Server task
    let ca_clone = ca.clone();
    rt.spawn(async move {
        loop {
            if let Some(incoming) = server_endpoint.accept().await {
                let conn = incoming.await.unwrap();
                tokio::spawn(async move {
                    loop {
                        match conn.accept_bi().await {
                            Ok((mut send, mut recv)) => {
                                let mut buf = vec![0u8; 4096];
                                if let Ok(Some(n)) = recv.read(&mut buf).await {
                                    send.write_all(&buf[..n]).await.unwrap();
                                    send.finish().await.unwrap();
                                }
                            }
                            Err(_) => break,
                        }
                    }
                });
            }
        }
    });

    std::thread::sleep(Duration::from_millis(200));

    let client_endpoint = QuicEndpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
    let conn = rt.block_on(async {
        client_endpoint
            .connect(server_addr, "bench-server", ca_clone)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("quic_stream_latency");

    for size in [16, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let data = vec![42u8; size];
            let conn_clone = conn.clone();

            b.to_async(&rt).iter(|| async {
                let (mut send, mut recv) = conn_clone.open_bi().await.unwrap();
                send.write_all(&data).await.unwrap();
                send.finish().await.unwrap();

                let mut buf = vec![0u8; 4096];
                recv.read(&mut buf).await.unwrap();
            });
        });
    }

    group.finish();
}

fn quic_stream_creation_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let ca = generate_ca_cert("Bench CA").unwrap();
    let server_identity = ServiceIdentity::new("bench-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6101".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert).unwrap();

    rt.spawn(async move {
        loop {
            if let Some(incoming) = server_endpoint.accept().await {
                let conn = incoming.await.unwrap();
                tokio::spawn(async move {
                    loop {
                        if conn.accept_bi().await.is_err() {
                            break;
                        }
                    }
                });
            }
        }
    });

    std::thread::sleep(Duration::from_millis(200));

    let client_endpoint = QuicEndpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
    let conn = rt.block_on(async {
        client_endpoint
            .connect(server_addr, "bench-server", ca)
            .await
            .unwrap()
    });

    c.bench_function("quic_stream_creation", |b| {
        let conn_clone = conn.clone();

        b.to_async(&rt).iter(|| async {
            let (send, recv) = conn_clone.open_bi().await.unwrap();
            black_box((send, recv));
        });
    });
}

fn quic_throughput_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let ca = generate_ca_cert("Bench CA").unwrap();
    let server_identity = ServiceIdentity::new("bench-server");
    let server_cert = generate_signed_cert(&server_identity, &ca).unwrap();

    let server_addr: SocketAddr = "127.0.0.1:6102".parse().unwrap();
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert).unwrap();

    rt.spawn(async move {
        loop {
            if let Some(incoming) = server_endpoint.accept().await {
                let conn = incoming.await.unwrap();
                tokio::spawn(async move {
                    loop {
                        match conn.accept_uni().await {
                            Ok(mut recv) => {
                                let mut buf = vec![0u8; 1024 * 1024];
                                while recv.read(&mut buf).await.is_ok() {}
                            }
                            Err(_) => break,
                        }
                    }
                });
            }
        }
    });

    std::thread::sleep(Duration::from_millis(200));

    let client_endpoint = QuicEndpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
    let conn = rt.block_on(async {
        client_endpoint
            .connect(server_addr, "bench-server", ca)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("quic_throughput");
    group.throughput(Throughput::Bytes(1024 * 1024));

    group.bench_function("1MB_transfer", |b| {
        let data = vec![42u8; 1024 * 1024];
        let conn_clone = conn.clone();

        b.to_async(&rt).iter(|| async {
            let mut send = conn_clone.open_uni().await.unwrap();
            send.write_all(&data).await.unwrap();
            send.finish().await.unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    grpc_latency_benchmark,
    grpc_throughput_benchmark,
    quic_stream_latency_benchmark,
    quic_stream_creation_benchmark,
    quic_throughput_benchmark,
);

criterion_main!(benches);
