//! Example: Creating large metric batches
//!
//! This example demonstrates creating and serializing large batches
//! of metrics, simulating a multi-GPU host reporting telemetry.

use prost::Message;
use hpc_types::create_timestamp;
use hpc_types::telemetry_helpers::{GpuMetricBuilder, MetricBatchBuilder};

fn main() {
    println!("=== Large Batch Creation Demo ===\n");

    // Simulate a host with 8 GPUs
    let host_id = "gpu-host-001";
    let num_gpus = 8;
    let timestamp = create_timestamp(1234567890, 0);

    println!(
        "Creating metrics for host {} with {} GPUs...",
        host_id, num_gpus
    );

    let mut batch_builder = MetricBatchBuilder::new();

    for gpu_idx in 0..num_gpus {
        let utilization = 50.0 + (gpu_idx as f32 * 5.0);
        let memory_used = 40.0 + (gpu_idx as f32 * 2.5);
        let temperature = 60.0 + (gpu_idx as f32 * 1.5);

        let metric = GpuMetricBuilder::new(host_id, format!("gpu-{}", gpu_idx))
            .timestamp(timestamp.clone())
            .utilization(utilization)
            .sm_occupancy(90.0)
            .memory(memory_used, 80.0)
            .pcie_bandwidth(10.0, 10.0)
            .nvlink_bandwidth(300.0)
            .temperature(temperature)
            .power(300.0 + (gpu_idx as f32 * 10.0))
            .ecc_errors(false)
            .build();

        batch_builder = batch_builder.add_gpu_metric(metric);
    }

    let batch = batch_builder.build();

    println!("\nBatch created:");
    println!("  Total GPU metrics: {}", batch.gpu_metrics.len());

    // Display summary statistics
    let total_util: f32 = batch.gpu_metrics.iter().map(|m| m.utilization).sum();
    let avg_util = total_util / batch.gpu_metrics.len() as f32;
    println!("  Average GPU utilization: {:.2}%", avg_util);

    let total_memory_used: f32 = batch.gpu_metrics.iter().map(|m| m.memory_used_gb).sum();
    let total_memory_total: f32 = batch.gpu_metrics.iter().map(|m| m.memory_total_gb).sum();
    println!(
        "  Total memory: {:.2}/{:.2} GB ({:.1}%)",
        total_memory_used,
        total_memory_total,
        (total_memory_used / total_memory_total) * 100.0
    );

    let avg_temp: f32 = batch
        .gpu_metrics
        .iter()
        .map(|m| m.temperature_celsius)
        .sum::<f32>()
        / batch.gpu_metrics.len() as f32;
    println!("  Average temperature: {:.1}°C", avg_temp);

    let total_power: f32 = batch.gpu_metrics.iter().map(|m| m.power_watts).sum();
    println!("  Total power consumption: {:.0}W", total_power);

    // Serialize and analyze size
    let mut buf = Vec::new();
    batch.encode(&mut buf).expect("Failed to encode batch");

    println!("\nSerialization results:");
    println!("  Serialized size: {} bytes", buf.len());
    println!(
        "  Bytes per metric: {:.1}",
        buf.len() as f32 / batch.gpu_metrics.len() as f32
    );

    // Test deserialization
    use hpc_types::telemetry::v1::MetricBatch;
    let decoded = MetricBatch::decode(&buf[..]).expect("Failed to decode");
    println!(
        "  Deserialization successful: {} metrics",
        decoded.gpu_metrics.len()
    );

    // Simulate 1000 batches (1000 seconds of data at 1Hz)
    println!("\n=== Simulating 1000 batches (1000s at 1Hz) ===");
    let single_batch_size = buf.len();
    let total_size = single_batch_size * 1000;
    println!(
        "  Total data size: {} bytes ({:.2} MB)",
        total_size,
        total_size as f32 / 1_000_000.0
    );
    println!(
        "  Bandwidth required: {:.2} KB/s",
        single_batch_size as f32 / 1000.0
    );

    println!("\nDemo completed successfully!");
}
