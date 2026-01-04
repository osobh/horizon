//! QUIC telemetry streaming simulation
//!
//! This example simulates a telemetry agent sending metrics to a collector
//! via QUIC unidirectional streams, demonstrating the pattern used in
//! Horizon's telemetry system.
//!
//! Run with: cargo run --example quic_telemetry_sim

use hpc_auth::cert::{generate_ca_cert, generate_signed_cert, ServiceIdentity};
use hpc_rpc::quic::QuicEndpoint;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};

#[derive(Debug)]
struct MetricBatch {
    timestamp: SystemTime,
    host_id: String,
    gpu_util: f32,
    mem_used_gb: f32,
    power_watts: f32,
}

impl MetricBatch {
    fn serialize(&self) -> Vec<u8> {
        // Simplified serialization (in production, use protobuf or flatbuffers)
        format!(
            "{:?}|{}|{:.2}|{:.2}|{:.2}",
            self.timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            self.host_id,
            self.gpu_util,
            self.mem_used_gb,
            self.power_watts
        )
        .into_bytes()
    }

    fn deserialize(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let s = String::from_utf8(data.to_vec())?;
        let parts: Vec<&str> = s.split('|').collect();

        Ok(Self {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(parts[0].parse()?),
            host_id: parts[1].to_string(),
            gpu_util: parts[2].parse()?,
            mem_used_gb: parts[3].parse()?,
            power_watts: parts[4].parse()?,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== QUIC Telemetry Streaming Simulation ===\n");

    // Setup certificates
    let ca = generate_ca_cert("Horizon CA")?;
    let collector_identity = ServiceIdentity::new("telemetry-collector");
    let collector_cert = generate_signed_cert(&collector_identity, &ca)?;

    // Start collector (server)
    let collector_addr: SocketAddr = "127.0.0.1:6001".parse()?;
    let collector_endpoint = QuicEndpoint::server(collector_addr, collector_cert)?;

    println!("Telemetry Collector started on {}\n", collector_addr);

    let collector_handle = tokio::spawn(async move {
        let mut total_batches = 0;
        let mut total_bytes = 0;

        let incoming = collector_endpoint.accept().await.expect("No connection");
        let conn = incoming.await.expect("Connection failed");

        println!("[Collector] Agent connected\n");

        // Accept unidirectional streams (metrics)
        loop {
            match conn.accept_uni().await {
                Ok(mut recv) => {
                    let mut buf = vec![0u8; 4096];
                    match recv.read(&mut buf).await {
                        Ok(Some(n)) => {
                            total_bytes += n;
                            total_batches += 1;

                            let batch = MetricBatch::deserialize(&buf[..n]).unwrap();
                            println!(
                                "[Collector] Batch #{}: host={}, gpu_util={:.1}%, mem={:.2}GB, power={:.1}W",
                                total_batches, batch.host_id, batch.gpu_util, batch.mem_used_gb, batch.power_watts
                            );
                        }
                        Ok(None) => break,
                        Err(e) => {
                            eprintln!("[Collector] Read error: {}", e);
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }

        println!("\n[Collector] Summary:");
        println!("  - Total batches: {}", total_batches);
        println!("  - Total bytes: {}", total_bytes);
        println!(
            "  - Avg batch size: {} bytes",
            total_bytes / total_batches.max(1)
        );
    });

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Start agent (client)
    println!("Starting telemetry agent...\n");
    let agent_addr: SocketAddr = "127.0.0.1:0".parse()?;
    let agent_endpoint = QuicEndpoint::client(agent_addr)?;

    let conn = agent_endpoint
        .connect(collector_addr, "telemetry-collector", ca)
        .await?;

    println!("[Agent] Connected to collector\n");

    // Simulate sending metrics every second
    for i in 1..=10 {
        let batch = MetricBatch {
            timestamp: SystemTime::now(),
            host_id: "gpu-host-001".to_string(),
            gpu_util: 75.0 + (i as f32 * 2.0),
            mem_used_gb: 32.0 + (i as f32 * 0.5),
            power_watts: 350.0 + (i as f32 * 5.0),
        };

        let data = batch.serialize();

        // Open unidirectional stream for this batch
        let mut send = conn.open_uni().await?;
        send.write_all(&data).await?;
        send.finish().await?;

        println!(
            "[Agent] Sent batch #{}: gpu_util={:.1}%, {} bytes",
            i,
            batch.gpu_util,
            data.len()
        );

        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    println!("\n[Agent] All metrics sent, closing connection");

    // Close connection gracefully
    conn.close(0u32.into(), b"done");

    collector_handle.await?;

    println!("\n=== Telemetry simulation complete ===");
    println!("Demonstrated QUIC's efficiency for streaming telemetry data!");

    Ok(())
}
