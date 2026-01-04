//! Comprehensive demonstration of the tracingx crate.
//!
//! This example shows how to:
//! - Initialize tracing with and without OTLP
//! - Set up Prometheus metrics endpoint
//! - Use instrumented functions
//! - Record metrics
//! - Handle graceful shutdown

use hpc_tracing::{init, init_metrics, TracingConfig};
use std::net::SocketAddr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, warn};

/// Simulates a data processing operation with tracing
#[instrument(skip(data))]
async fn process_data(id: u64, data_size: usize, data: Vec<u8>) -> Result<u64, String> {
    info!(id, data_size, "Starting data processing");

    // Simulate some work
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Record metrics
    metrics::increment_counter!("data_processed_total");
    metrics::histogram!("data_size_bytes", data_size as f64);

    if data.is_empty() {
        warn!(id, "Empty data received");
        return Err("empty data".to_string());
    }

    // Simulate processing
    let checksum = data.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));

    info!(id, checksum, "Data processing completed");
    metrics::histogram!("processing_duration_ms", 100.0);

    Ok(checksum)
}

/// Simulates an API request handler
#[instrument]
async fn handle_request(method: &str, path: &str) -> Result<u16, String> {
    info!(method, path, "Handling request");

    // Record request metrics
    metrics::increment_counter!("requests_total");

    // Simulate request processing
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Simulate response
    let status = if path == "/health" {
        200
    } else if path == "/error" {
        error!(method, path, "Request failed");
        metrics::increment_counter!("requests_errors_total");
        500
    } else {
        200
    };

    Ok(status)
}

/// Background task that updates metrics
#[instrument]
async fn background_metrics_updater() {
    info!("Starting background metrics updater");

    for i in 0..10 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Update gauges
        metrics::gauge!("active_connections", 10.0 + i as f64);
        metrics::gauge!("memory_usage_mb", 256.0 + (i * 10) as f64);

        debug!(iteration = i, "Updated background metrics");
    }

    info!("Background metrics updater finished");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Horizon TracingX Demo ===\n");

    // Step 1: Initialize metrics endpoint
    println!("1. Initializing Prometheus metrics endpoint on 0.0.0.0:9090...");
    let metrics_addr: SocketAddr = "0.0.0.0:9090".parse()?;
    init_metrics(metrics_addr)?;
    println!("   ✓ Metrics available at http://0.0.0.0:9090/metrics\n");

    // Step 2: Initialize tracing
    println!("2. Initializing tracing subsystem...");

    // Configure with optional OTLP endpoint
    // To use OTLP, start a Jaeger instance:
    //   docker run -d -p 4317:4317 jaegertracing/all-in-one:latest
    let otlp_endpoint = std::env::var("OTLP_ENDPOINT").ok();

    let config = TracingConfig {
        service_name: "tracingx-demo".to_string(),
        log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        otlp_endpoint: otlp_endpoint.clone(),
    };

    let _guard = init(config)?;

    if let Some(endpoint) = otlp_endpoint {
        println!("   ✓ Tracing initialized with OTLP endpoint: {}", endpoint);
    } else {
        println!("   ✓ Tracing initialized (local logging only)");
        println!("   ℹ Set OTLP_ENDPOINT environment variable to enable distributed tracing");
    }
    println!();

    // Step 3: Log some messages at different levels
    println!("3. Logging messages at different levels...");
    info!("Service started successfully");
    debug!(version = env!("CARGO_PKG_VERSION"), "Debug information");
    warn!("This is a warning message");
    println!();

    // Step 4: Process some data
    println!("4. Processing data with instrumented functions...");
    let data1 = vec![1, 2, 3, 4, 5];
    match process_data(1, data1.len(), data1).await {
        Ok(checksum) => info!(checksum, "Data 1 processed successfully"),
        Err(e) => error!(error = ?e, "Data 1 processing failed"),
    }

    let data2 = vec![10, 20, 30];
    match process_data(2, data2.len(), data2).await {
        Ok(checksum) => info!(checksum, "Data 2 processed successfully"),
        Err(e) => error!(error = ?e, "Data 2 processing failed"),
    }

    // Test error case
    let empty_data = vec![];
    match process_data(3, empty_data.len(), empty_data).await {
        Ok(checksum) => info!(checksum, "Empty data processed"),
        Err(e) => warn!(error = ?e, "Empty data processing failed (expected)"),
    }
    println!();

    // Step 5: Handle some API requests
    println!("5. Simulating API requests...");
    let _ = handle_request("GET", "/health").await;
    let _ = handle_request("GET", "/api/users").await;
    let _ = handle_request("POST", "/api/data").await;
    let _ = handle_request("GET", "/error").await;
    println!();

    // Step 6: Start background task
    println!("6. Starting background metrics updater (runs for 10 seconds)...");
    let bg_task = tokio::spawn(background_metrics_updater());

    // Step 7: Record some custom metrics
    println!("7. Recording custom metrics...");
    for i in 0..5 {
        metrics::increment_counter!("demo_counter");
        metrics::histogram!("demo_latency_ms", (i * 10) as f64);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    println!();

    // Step 8: Wait for background task
    println!("8. Waiting for background task to complete...");
    let _ = bg_task.await;
    println!();

    // Step 9: Final metrics summary
    println!("9. Final metrics summary:");
    println!("   - Check http://0.0.0.0:9090/metrics for full metrics");
    println!("   - Look for:");
    println!("     * data_processed_total");
    println!("     * requests_total");
    println!("     * active_connections");
    println!("     * processing_duration_ms");
    println!();

    // Step 10: Graceful shutdown
    println!("10. Shutting down...");
    info!("Demo completed successfully");
    println!();

    // Give time for final logs to flush
    tokio::time::sleep(Duration::from_millis(100)).await;

    // When _guard is dropped here, the tracer provider will be shut down
    println!("=== Demo Complete ===");
    println!("\nTips:");
    println!("- Run with RUST_LOG=debug for more verbose output");
    println!("- Set OTLP_ENDPOINT=http://localhost:4317 to enable distributed tracing");
    println!("- Visit http://localhost:9090/metrics to see Prometheus metrics");
    println!();

    Ok(())
}
