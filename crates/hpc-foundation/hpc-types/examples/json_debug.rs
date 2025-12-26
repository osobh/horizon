//! Example: Converting messages to JSON for debugging
//!
//! This example shows how to use JSON serialization for debugging
//! and human-readable inspection of protobuf messages.

use hpc_types::telemetry_helpers::{GpuMetricBuilder, MetricBatchBuilder};

fn main() {
    println!("=== JSON Debugging Example ===\n");

    // Create a sample GPU metric
    let metric = GpuMetricBuilder::new("host-001", "gpu-0")
        .utilization(87.5)
        .sm_occupancy(92.3)
        .memory(64.5, 80.0)
        .pcie_bandwidth(12.5, 11.8)
        .nvlink_bandwidth(300.0)
        .temperature(68.5)
        .power(350.0)
        .mig_profile("1g.10gb")
        .build();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&metric).expect("Failed to serialize to JSON");

    println!("GPU Metric as JSON:");
    println!("{}\n", json);

    // Demonstrate compact JSON
    let compact_json = serde_json::to_string(&metric).expect("Failed to serialize");
    println!("Compact JSON (single line):");
    println!("{}\n", compact_json);

    // Create a batch with multiple metrics
    let batch = MetricBatchBuilder::new()
        .add_gpu_metric(metric)
        .add_gpu_metric(
            GpuMetricBuilder::new("host-001", "gpu-1")
                .utilization(92.1)
                .memory(72.0, 80.0)
                .temperature(70.2)
                .power(365.0)
                .build(),
        )
        .build();

    let batch_json = serde_json::to_string_pretty(&batch).expect("Failed to serialize batch");

    println!("Metric Batch as JSON:");
    println!("{}\n", batch_json);

    // Calculate size comparison
    use prost::Message;
    let mut protobuf_buf = Vec::new();
    batch.encode(&mut protobuf_buf).expect("Failed to encode");

    println!("Size comparison:");
    println!("  Protobuf: {} bytes", protobuf_buf.len());
    println!("  JSON (pretty): {} bytes", batch_json.len());
    println!("  JSON (compact): {} bytes", compact_json.len());
    println!(
        "  JSON overhead: {:.1}x protobuf size",
        batch_json.len() as f32 / protobuf_buf.len() as f32
    );

    println!("\nJSON debugging example completed!");
}
