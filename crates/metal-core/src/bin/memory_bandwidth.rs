//! Memory Bandwidth Benchmark for Metal GPU
//!
//! Measures unified memory performance on Apple Silicon:
//! - Buffer allocation speed
//! - CPU write throughput
//! - CPU read throughput
//! - GPU kernel read/write throughput
//!
//! Run with: cargo run --release -p stratoswarm-metal-core --bin memory_bandwidth

use std::time::{Duration, Instant};
use stratoswarm_metal_core::backend::MetalBackend;
use stratoswarm_metal_core::buffer::MetalBuffer;
use stratoswarm_metal_core::command::{MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder};
use stratoswarm_metal_core::metal3::Metal3Backend;

fn main() -> anyhow::Result<()> {
    println!("Metal Memory Bandwidth Benchmark (M4)");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize Metal backend
    let backend = Metal3Backend::new()?;
    let device = backend.device();
    let info = device.info();

    println!("Device: {}", info.name);
    println!("Unified Memory: {}", info.unified_memory);
    println!("Max Buffer Size: {} GB", info.max_buffer_length / (1024 * 1024 * 1024));
    println!();

    // Warmup phase
    println!("Warming up GPU...");
    for _ in 0..20 {
        let _ = backend.create_buffer::<f32>(1024)?;
    }
    println!();

    // ========================================================================
    // Benchmark 1: Buffer Allocation Time
    // ========================================================================
    println!("1. Buffer Allocation Time");
    println!("{}", "-".repeat(70));
    println!("{:>15} {:>12} {:>12} {:>12} {:>12}", "Size", "Avg", "P50", "P99", "Throughput");

    let allocation_sizes = [
        1024,                    // 1 KB
        64 * 1024,               // 64 KB
        1024 * 1024,             // 1 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        256 * 1024 * 1024,       // 256 MB
    ];

    for size in allocation_sizes {
        let mut times = Vec::with_capacity(50);

        for _ in 0..50 {
            let start = Instant::now();
            let _buffer = backend.create_buffer::<u8>(size)?;
            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() * 99 / 100];
        let throughput_gbps = (size as f64) / avg.as_secs_f64() / 1e9;

        println!(
            "{:>12} KB {:>10.2?} {:>10.2?} {:>10.2?} {:>9.2} GB/s",
            size / 1024,
            avg,
            p50,
            p99,
            throughput_gbps
        );
    }
    println!();

    // ========================================================================
    // Benchmark 2: CPU Write Throughput (Unified Memory)
    // ========================================================================
    println!("2. CPU Write Throughput (Unified Memory)");
    println!("{}", "-".repeat(70));
    println!("{:>15} {:>12} {:>12} {:>12} {:>12}", "Size", "Avg", "P50", "P99", "Throughput");

    let write_sizes = [
        1024 * 1024,             // 1 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        256 * 1024 * 1024,       // 256 MB
        512 * 1024 * 1024,       // 512 MB
    ];

    for size in write_sizes {
        let element_count = size / std::mem::size_of::<f32>();
        let mut buffer = backend.create_buffer::<f32>(element_count)?;
        let mut times = Vec::with_capacity(20);

        for iteration in 0..20 {
            let start = Instant::now();
            {
                let data = buffer.contents_mut::<f32>();
                // Write pattern that's hard to optimize away
                for (i, val) in data.iter_mut().enumerate() {
                    *val = (i as f32) + (iteration as f32);
                }
            }
            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() * 99 / 100];
        let throughput_gbps = (size as f64) / avg.as_secs_f64() / 1e9;

        println!(
            "{:>12} MB {:>10.2?} {:>10.2?} {:>10.2?} {:>9.2} GB/s",
            size / (1024 * 1024),
            avg,
            p50,
            p99,
            throughput_gbps
        );
    }
    println!();

    // ========================================================================
    // Benchmark 3: CPU Read Throughput (Unified Memory)
    // ========================================================================
    println!("3. CPU Read Throughput (Unified Memory)");
    println!("{}", "-".repeat(70));
    println!("{:>15} {:>12} {:>12} {:>12} {:>12}", "Size", "Avg", "P50", "P99", "Throughput");

    for size in write_sizes {
        let element_count = size / std::mem::size_of::<f32>();
        let mut buffer = backend.create_buffer::<f32>(element_count)?;

        // Initialize buffer
        {
            let data = buffer.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = i as f32;
            }
        }

        let mut times = Vec::with_capacity(20);
        let mut checksum: f64 = 0.0;

        for _ in 0..20 {
            let start = Instant::now();
            {
                let data = buffer.contents::<f32>();
                // Read pattern that's hard to optimize away
                let mut sum: f64 = 0.0;
                for val in data.iter() {
                    sum += *val as f64;
                }
                checksum += sum;
            }
            times.push(start.elapsed());
        }

        // Use checksum to prevent optimization
        std::hint::black_box(checksum);

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() * 99 / 100];
        let throughput_gbps = (size as f64) / avg.as_secs_f64() / 1e9;

        println!(
            "{:>12} MB {:>10.2?} {:>10.2?} {:>10.2?} {:>9.2} GB/s",
            size / (1024 * 1024),
            avg,
            p50,
            p99,
            throughput_gbps
        );
    }
    println!();

    // ========================================================================
    // Benchmark 4: GPU Kernel Throughput
    // ========================================================================
    println!("4. GPU Kernel Memory Throughput");
    println!("{}", "-".repeat(70));
    println!("{:>15} {:>12} {:>12} {:>12} {:>12}", "Size", "Avg", "P50", "P99", "Throughput");

    // Simple copy kernel for bandwidth measurement
    let copy_shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void copy_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint tid [[thread_position_in_grid]]
        ) {
            output[tid] = input[tid] * 1.001f;  // Small multiply to prevent optimization
        }

        kernel void sum_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint tid [[thread_position_in_grid]],
            uint threads [[threads_per_grid]]
        ) {
            // Each thread reads 4 floats
            float sum = 0.0f;
            for (uint i = tid; i < threads; i += threads) {
                sum += input[i];
            }
            output[tid] = sum;
        }
    "#;

    let pipeline = backend.create_compute_pipeline(copy_shader, "copy_kernel")?;
    let queue = backend.create_command_queue()?;

    let gpu_sizes = [
        1024 * 1024,             // 1 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        128 * 1024 * 1024,       // 128 MB
    ];

    for size in gpu_sizes {
        let element_count = size / std::mem::size_of::<f32>();

        // Create input and output buffers
        let input = backend.create_buffer::<f32>(element_count)?;
        let output = backend.create_buffer::<f32>(element_count)?;

        // Initialize input
        {
            let mut input_mut = backend.create_buffer_with_data(&vec![1.0f32; element_count])?;
            std::mem::swap(&mut input_mut, &mut backend.create_buffer::<f32>(element_count)?);
        }

        let mut times = Vec::with_capacity(50);

        // Warmup
        for _ in 0..5 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &output, 0)?;
                encoder.dispatch_threads(element_count as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit(&mut cmd)?;
        }

        // Benchmark
        for _ in 0..50 {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &output, 0)?;
                encoder.dispatch_threads(element_count as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() * 99 / 100];
        // Read + write = 2x size
        let effective_bandwidth = (size as f64 * 2.0) / avg.as_secs_f64() / 1e9;

        println!(
            "{:>12} MB {:>10.2?} {:>10.2?} {:>10.2?} {:>9.2} GB/s",
            size / (1024 * 1024),
            avg,
            p50,
            p99,
            effective_bandwidth
        );
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "-".repeat(70));
    println!("M4 Theoretical Peak: ~120 GB/s (unified memory)");
    println!("Benchmark complete!");
    println!();

    Ok(())
}
