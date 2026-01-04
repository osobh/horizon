//! QUIC bidirectional streaming demonstration
//!
//! This example shows how to use QUIC for low-latency bidirectional streaming.
//!
//! Run with: cargo run --example quic_stream_demo

use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_rpc::quic::QuicEndpoint;
use std::net::SocketAddr;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== QUIC Bidirectional Streaming Demo ===\n");

    // 1. Generate certificates
    println!("1. Generating certificates...");
    let ca = generate_ca_cert("QUIC Demo CA")?;
    let server_identity = ServiceIdentity::new("quic-demo-server");
    let server_cert = generate_signed_cert(&server_identity, &ca)?;
    println!("   - Certificates ready\n");

    // 2. Start QUIC server
    println!("2. Starting QUIC server...");
    let server_addr: SocketAddr = "127.0.0.1:6000".parse()?;
    let server_endpoint = QuicEndpoint::server(server_addr, server_cert)?;

    let server_handle = tokio::spawn(async move {
        println!("   - Server listening on {}\n", server_addr);

        let incoming = server_endpoint.accept().await.expect("No connection");
        let conn = incoming.await.expect("Connection failed");

        println!("[Server] Client connected");

        // Handle multiple streams
        for stream_num in 1..=5 {
            let (mut send, mut recv) = conn.accept_bi().await.expect("Stream failed");

            let mut buf = vec![0u8; 1024];
            let n = recv.read(&mut buf).await.expect("Read failed").unwrap();
            let msg = String::from_utf8(buf[..n].to_vec()).unwrap();

            println!("[Server] Stream #{}: Received: {}", stream_num, msg);

            // Echo back with modification
            let response = format!("Echo: {}", msg);
            send.write_all(response.as_bytes())
                .await
                .expect("Write failed");
            send.finish().await.expect("Finish failed");

            println!("[Server] Stream #{}: Sent response", stream_num);
        }

        println!("[Server] All streams processed");
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    // 3. Create QUIC client
    println!("3. Creating QUIC client...");
    let client_addr: SocketAddr = "127.0.0.1:0".parse()?;
    let client_endpoint = QuicEndpoint::client(client_addr)?;

    let conn = client_endpoint
        .connect(server_addr, "quic-demo-server", ca)
        .await?;

    println!("   - Client connected\n");

    // 4. Open multiple bidirectional streams
    println!("4. Opening bidirectional streams...\n");

    for i in 1..=5 {
        let (mut send, mut recv) = conn.open_bi().await?;

        let message = format!("Message #{}", i);
        println!("[Client] Stream #{}: Sending: {}", i, message);

        send.write_all(message.as_bytes()).await?;
        send.finish().await?;

        let mut buf = vec![0u8; 1024];
        let n = recv.read(&mut buf).await?.unwrap();
        let response = String::from_utf8(buf[..n].to_vec())?;

        println!("[Client] Stream #{}: Received: {}", i, response);

        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    println!("\n=== Demo complete ===");
    println!(
        "Demonstrated {} bidirectional streams over a single QUIC connection!",
        5
    );

    server_handle.await?;

    Ok(())
}
