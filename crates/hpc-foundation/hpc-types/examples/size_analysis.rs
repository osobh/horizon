//! Example: Analyzing message sizes
//!
//! This example analyzes the serialized sizes of different message types
//! and configurations to help optimize network bandwidth usage.

use prost::Message;
use hpc_types::telemetry::v1::MetricBatch;
use hpc_types::telemetry_helpers::{
    CpuMetricBuilder, GpuMetricBuilder, MetricBatchBuilder, NicMetricBuilder,
};

fn analyze_gpu_metric_size() {
    println!("=== GPU Metric Size Analysis ===");

    // Minimal GPU metric
    let minimal = GpuMetricBuilder::new("h", "g").build();
    let minimal_size = minimal.encoded_len();
    println!("  Minimal GPU metric: {} bytes", minimal_size);

    // Typical GPU metric
    let typical = GpuMetricBuilder::new("host-001", "gpu-0")
        .utilization(87.5)
        .memory(64.5, 80.0)
        .temperature(68.5)
        .power(350.0)
        .build();
    let typical_size = typical.encoded_len();
    println!("  Typical GPU metric: {} bytes", typical_size);

    // Full GPU metric with all fields
    let full = GpuMetricBuilder::new("host-001-long-name", "gpu-0")
        .utilization(87.5)
        .sm_occupancy(92.3)
        .memory(64.5, 80.0)
        .pcie_bandwidth(12.5, 11.8)
        .nvlink_bandwidth(300.0)
        .temperature(68.5)
        .power(350.0)
        .ecc_errors(false)
        .mig_profile("1g.10gb")
        .build();
    let full_size = full.encoded_len();
    println!("  Full GPU metric: {} bytes", full_size);
    println!();
}

fn analyze_cpu_metric_size() {
    println!("=== CPU Metric Size Analysis ===");

    let minimal = CpuMetricBuilder::new("h", 0).build();
    println!("  Minimal CPU metric: {} bytes", minimal.encoded_len());

    let full = CpuMetricBuilder::new("host-001", 0)
        .utilization(45.6)
        .ipc(2.3)
        .cache_misses(1_234_567)
        .build();
    println!("  Full CPU metric: {} bytes", full.encoded_len());
    println!();
}

fn analyze_nic_metric_size() {
    println!("=== NIC Metric Size Analysis ===");

    let minimal = NicMetricBuilder::new("h", "e").build();
    println!("  Minimal NIC metric: {} bytes", minimal.encoded_len());

    let full = NicMetricBuilder::new("host-001", "eth0")
        .bandwidth(8.5, 7.2)
        .errors_and_drops(123, 456)
        .build();
    println!("  Full NIC metric: {} bytes", full.encoded_len());
    println!();
}

fn analyze_batch_sizes() {
    println!("=== Batch Size Analysis ===");

    for num_metrics in [1, 10, 100, 1000] {
        let mut batch_builder = MetricBatchBuilder::new();

        for i in 0..num_metrics {
            let metric =
                GpuMetricBuilder::new(format!("host-{:03}", i / 8), format!("gpu-{}", i % 8))
                    .utilization(50.0 + (i % 50) as f32)
                    .memory(40.0, 80.0)
                    .temperature(65.0)
                    .power(300.0)
                    .build();

            batch_builder = batch_builder.add_gpu_metric(metric);
        }

        let batch = batch_builder.build();
        let batch_size = batch.encoded_len();

        println!("  {} GPU metrics:", num_metrics);
        println!(
            "    Total size: {} bytes ({:.2} KB)",
            batch_size,
            batch_size as f32 / 1024.0
        );
        println!(
            "    Per metric: {:.1} bytes",
            batch_size as f32 / num_metrics as f32
        );
        println!(
            "    Metrics/KB: {:.1}",
            (num_metrics as f32) / (batch_size as f32 / 1024.0)
        );
    }
    println!();
}

fn analyze_bandwidth_requirements() {
    println!("=== Bandwidth Requirements ===");

    // Scenario: 1000 GPUs, 1 Hz reporting
    let gpus = 1000;
    let frequency_hz = 1.0;

    let typical_metric = GpuMetricBuilder::new("host-001", "gpu-0")
        .utilization(87.5)
        .memory(64.5, 80.0)
        .temperature(68.5)
        .power(350.0)
        .build();

    let metric_size = typical_metric.encoded_len();
    let total_bytes_per_second = metric_size * gpus;
    let total_mb_per_hour = (total_bytes_per_second as f32 * 3600.0) / 1_000_000.0;

    println!("  Scenario: {} GPUs @ {} Hz", gpus, frequency_hz);
    println!("  Metric size: {} bytes", metric_size);
    println!("  Total bandwidth: {} KB/s", total_bytes_per_second / 1024);
    println!("  Data per hour: {:.2} MB", total_mb_per_hour);
    println!(
        "  Data per day: {:.2} GB",
        total_mb_per_hour * 24.0 / 1000.0
    );

    // With batching (8 GPUs per host)
    let hosts = gpus / 8;
    let batch = create_sample_batch(8);
    let batch_size = batch.encoded_len();

    println!("\n  With batching (8 GPUs/host):");
    println!("  Hosts: {}", hosts);
    println!("  Batch size: {} bytes", batch_size);
    println!(
        "  Per-metric overhead reduction: {:.1}%",
        (1.0 - (batch_size as f32 / 8.0) / metric_size as f32) * 100.0
    );
    println!();
}

fn create_sample_batch(num_gpus: usize) -> MetricBatch {
    let mut builder = MetricBatchBuilder::new();

    for i in 0..num_gpus {
        let metric = GpuMetricBuilder::new("host-001", format!("gpu-{}", i))
            .utilization(50.0 + i as f32)
            .memory(40.0, 80.0)
            .temperature(65.0)
            .power(300.0)
            .build();
        builder = builder.add_gpu_metric(metric);
    }

    builder.build()
}

fn main() {
    println!("=== Message Size Analysis ===\n");

    analyze_gpu_metric_size();
    analyze_cpu_metric_size();
    analyze_nic_metric_size();
    analyze_batch_sizes();
    analyze_bandwidth_requirements();

    println!("Analysis completed!");
}
