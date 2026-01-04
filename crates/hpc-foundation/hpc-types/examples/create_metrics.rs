//! Example: Creating and serializing various metric types
//!
//! This example demonstrates how to create GPU, CPU, and NIC metrics
//! using both the raw structs and the builder patterns.

use hpc_types::telemetry_helpers::{
    CpuMetricBuilder, GpuMetricBuilder, MetricBatchBuilder, NicMetricBuilder,
};
use hpc_types::{create_timestamp, current_timestamp};
use prost::Message;

fn main() {
    println!("=== Schemax Metrics Creation Example ===\n");

    // Example 1: Create a GPU metric using the builder
    println!("1. Creating GPU metric using builder:");
    let gpu_metric = GpuMetricBuilder::new("host-001", "gpu-0")
        .utilization(87.5)
        .sm_occupancy(92.3)
        .memory(64.5, 80.0)
        .pcie_bandwidth(12.5, 11.8)
        .nvlink_bandwidth(300.0)
        .temperature(68.5)
        .power(350.0)
        .mig_profile("1g.10gb")
        .build();

    println!("  Host: {}", gpu_metric.host_id);
    println!("  GPU: {}", gpu_metric.gpu_id);
    println!("  Utilization: {}%", gpu_metric.utilization);
    println!(
        "  Memory: {}/{} GB",
        gpu_metric.memory_used_gb, gpu_metric.memory_total_gb
    );
    println!("  Temperature: {}°C", gpu_metric.temperature_celsius);
    println!("  Power: {}W", gpu_metric.power_watts);

    // Serialize to protobuf
    let mut buf = Vec::new();
    gpu_metric.encode(&mut buf).expect("Failed to encode");
    println!("  Serialized size: {} bytes\n", buf.len());

    // Example 2: Create a CPU metric
    println!("2. Creating CPU metric:");
    let cpu_metric = CpuMetricBuilder::new("host-001", 0)
        .utilization(45.6)
        .ipc(2.3)
        .cache_misses(1_234_567)
        .build();

    println!("  Host: {}", cpu_metric.host_id);
    println!("  Socket: {}", cpu_metric.socket);
    println!("  Utilization: {}%", cpu_metric.utilization);
    println!("  IPC: {}", cpu_metric.ipc);
    println!("  Cache misses: {}\n", cpu_metric.cache_misses);

    // Example 3: Create a NIC metric
    println!("3. Creating NIC metric:");
    let nic_metric = NicMetricBuilder::new("host-001", "eth0")
        .bandwidth(8.5, 7.2)
        .errors_and_drops(0, 0)
        .build();

    println!("  Host: {}", nic_metric.host_id);
    println!("  Interface: {}", nic_metric.interface);
    println!("  RX: {} Gbps", nic_metric.rx_gbps);
    println!("  TX: {} Gbps", nic_metric.tx_gbps);
    println!(
        "  Errors: {}, Drops: {}\n",
        nic_metric.errors, nic_metric.drops
    );

    // Example 4: Create a metric batch
    println!("4. Creating metric batch:");
    let batch = MetricBatchBuilder::new()
        .add_gpu_metric(gpu_metric)
        .add_cpu_metric(cpu_metric)
        .add_nic_metric(nic_metric)
        .build();

    println!("  GPU metrics: {}", batch.gpu_metrics.len());
    println!("  CPU metrics: {}", batch.cpu_metrics.len());
    println!("  NIC metrics: {}", batch.nic_metrics.len());

    // Serialize batch
    let mut batch_buf = Vec::new();
    batch
        .encode(&mut batch_buf)
        .expect("Failed to encode batch");
    println!("  Batch serialized size: {} bytes\n", batch_buf.len());

    // Example 5: Working with timestamps
    println!("5. Working with timestamps:");
    let ts1 = create_timestamp(1234567890, 123456789);
    println!(
        "  Manual timestamp: {} seconds, {} nanos",
        ts1.seconds, ts1.nanos
    );

    let ts2 = current_timestamp();
    println!(
        "  Current timestamp: {} seconds, {} nanos",
        ts2.seconds, ts2.nanos
    );

    println!("\nExample completed successfully!");
}
