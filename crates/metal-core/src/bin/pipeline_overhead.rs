//! Pipeline Overhead Benchmark for Metal GPU
//!
//! Measures Metal pipeline and command buffer overhead:
//! - Shader compilation time (cold vs warm)
//! - Command buffer creation overhead
//! - Dispatch latency (small vs large grids)
//! - Synchronization cost
//!
//! Run with: cargo run --release -p stratoswarm-metal-core --bin pipeline_overhead

use std::time::{Duration, Instant};
use stratoswarm_metal_core::backend::MetalBackend;
use stratoswarm_metal_core::command::{MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder};
use stratoswarm_metal_core::metal3::Metal3Backend;

fn main() -> anyhow::Result<()> {
    println!("Metal Pipeline Overhead Benchmark (M4)");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize Metal backend
    let backend = Metal3Backend::new()?;
    let device = backend.device();
    let info = device.info();

    println!("Device: {}", info.name);
    println!("Max Threads/Threadgroup: {}", info.max_threads_per_threadgroup);
    println!();

    // ========================================================================
    // Benchmark 1: Shader Compilation Time
    // ========================================================================
    println!("1. Shader Compilation Time");
    println!("{}", "-".repeat(70));

    // Simple shader
    let simple_shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void simple_kernel(
            device float* data [[buffer(0)]],
            uint tid [[thread_position_in_grid]]
        ) {
            data[tid] = data[tid] * 2.0f;
        }
    "#;

    // Medium shader with more operations
    let medium_shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void medium_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant float& scale [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            float x = input[tid];
            float y = x * scale;
            y = sin(y) + cos(y);
            y = exp(-y * y);
            y = sqrt(abs(y) + 1.0f);
            output[tid] = y;
        }
    "#;

    // Complex shader with loops and more ops
    let complex_shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void complex_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& iterations [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            float acc = input[tid];

            for (uint i = 0; i < iterations; i++) {
                acc = sin(acc) * cos(acc);
                acc = acc * acc + 1.0f;
                acc = sqrt(acc);
                acc = tanh(acc);
                acc = exp(-acc * 0.1f);
            }

            output[tid] = acc;
        }
    "#;

    // Cold compilation (first time)
    println!("\n  Cold Compilation (first time):");

    let start = Instant::now();
    let _simple_pipeline = backend.create_compute_pipeline(simple_shader, "simple_kernel")?;
    let simple_cold = start.elapsed();
    println!("    Simple shader:  {:>10.2?}", simple_cold);

    let start = Instant::now();
    let _medium_pipeline = backend.create_compute_pipeline(medium_shader, "medium_kernel")?;
    let medium_cold = start.elapsed();
    println!("    Medium shader:  {:>10.2?}", medium_cold);

    let start = Instant::now();
    let _complex_pipeline = backend.create_compute_pipeline(complex_shader, "complex_kernel")?;
    let complex_cold = start.elapsed();
    println!("    Complex shader: {:>10.2?}", complex_cold);

    // Warm compilation (shader cache)
    println!("\n  Warm Compilation (shader cache):");
    let mut simple_times = Vec::with_capacity(10);
    let mut medium_times = Vec::with_capacity(10);
    let mut complex_times = Vec::with_capacity(10);

    for _ in 0..10 {
        let start = Instant::now();
        let _p = backend.create_compute_pipeline(simple_shader, "simple_kernel")?;
        simple_times.push(start.elapsed());

        let start = Instant::now();
        let _p = backend.create_compute_pipeline(medium_shader, "medium_kernel")?;
        medium_times.push(start.elapsed());

        let start = Instant::now();
        let _p = backend.create_compute_pipeline(complex_shader, "complex_kernel")?;
        complex_times.push(start.elapsed());
    }

    simple_times.sort();
    medium_times.sort();
    complex_times.sort();

    println!(
        "    Simple shader:  {:>10.2?} (p50)",
        simple_times[simple_times.len() / 2]
    );
    println!(
        "    Medium shader:  {:>10.2?} (p50)",
        medium_times[medium_times.len() / 2]
    );
    println!(
        "    Complex shader: {:>10.2?} (p50)",
        complex_times[complex_times.len() / 2]
    );
    println!();

    // ========================================================================
    // Benchmark 2: Command Buffer Creation Overhead
    // ========================================================================
    println!("2. Command Buffer Creation Overhead");
    println!("{}", "-".repeat(70));

    let queue = backend.create_command_queue()?;
    let pipeline = backend.create_compute_pipeline(simple_shader, "simple_kernel")?;
    let buffer = backend.create_buffer::<f32>(1024)?;

    // Command buffer creation only
    let mut create_times = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        let _cmd = queue.create_command_buffer()?;
        create_times.push(start.elapsed());
    }

    create_times.sort();
    println!(
        "  Command buffer creation: {:>8.2?} avg, {:>8.2?} p50, {:>8.2?} p99",
        create_times.iter().sum::<Duration>() / create_times.len() as u32,
        create_times[create_times.len() / 2],
        create_times[create_times.len() * 99 / 100]
    );

    // Command buffer + encoder creation
    let mut encoder_times = Vec::with_capacity(500);
    for _ in 0..500 {
        let start = Instant::now();
        let mut cmd = queue.create_command_buffer()?;
        {
            let mut encoder = cmd.compute_encoder()?;
            encoder.set_pipeline(&pipeline)?;
            encoder.set_buffer(0, &buffer, 0)?;
            encoder.end_encoding()?;
        }
        encoder_times.push(start.elapsed());
    }

    encoder_times.sort();
    println!(
        "  With encoder setup:      {:>8.2?} avg, {:>8.2?} p50, {:>8.2?} p99",
        encoder_times.iter().sum::<Duration>() / encoder_times.len() as u32,
        encoder_times[encoder_times.len() / 2],
        encoder_times[encoder_times.len() * 99 / 100]
    );
    println!();

    // ========================================================================
    // Benchmark 3: Dispatch Latency
    // ========================================================================
    println!("3. Dispatch Latency (kernel execution overhead)");
    println!("{}", "-".repeat(70));
    println!("{:>15} {:>12} {:>12} {:>12} {:>15}", "Threads", "Avg", "P50", "P99", "Throughput");

    let dispatch_sizes: [u64; 7] = [
        1,           // Single thread
        32,          // Single SIMD group
        256,         // Single threadgroup
        1024,        // Multiple threadgroups
        65536,       // 64K threads
        1048576,     // 1M threads
        16777216,    // 16M threads
    ];

    for threads in dispatch_sizes {
        // Create buffer large enough for dispatch
        let dispatch_buffer = backend.create_buffer::<f32>(threads as usize)?;

        let mut times = Vec::with_capacity(100);

        // Warmup
        for _ in 0..5 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&pipeline)?;
                encoder.set_buffer(0, &dispatch_buffer, 0)?;
                encoder.dispatch_threads(threads)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        for _ in 0..100 {
            let start = Instant::now();
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&pipeline)?;
                encoder.set_buffer(0, &dispatch_buffer, 0)?;
                encoder.dispatch_threads(threads)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() * 99 / 100];
        let throughput = (threads as f64) / avg.as_secs_f64() / 1e9;

        println!(
            "{:>15} {:>10.2?} {:>10.2?} {:>10.2?} {:>12.2} GT/s",
            format_threads(threads),
            avg,
            p50,
            p99,
            throughput
        );
    }
    println!();

    // ========================================================================
    // Benchmark 4: Synchronization Cost
    // ========================================================================
    println!("4. Synchronization Cost");
    println!("{}", "-".repeat(70));

    let sync_buffer = backend.create_buffer::<f32>(1024 * 1024)?;

    // Measure submit overhead (async return)
    let mut submit_times = Vec::with_capacity(100);
    for _ in 0..100 {
        let mut cmd = queue.create_command_buffer()?;
        {
            let mut encoder = cmd.compute_encoder()?;
            encoder.set_pipeline(&pipeline)?;
            encoder.set_buffer(0, &sync_buffer, 0)?;
            encoder.dispatch_threads(1024 * 1024)?;
            encoder.end_encoding()?;
        }

        let start = Instant::now();
        queue.submit(&mut cmd)?;
        submit_times.push(start.elapsed());

        // Wait for completion before next iteration
        queue.wait_until_completed()?;
    }

    submit_times.sort();
    println!(
        "  Submit (async return):   {:>8.2?} avg, {:>8.2?} p50",
        submit_times.iter().sum::<Duration>() / submit_times.len() as u32,
        submit_times[submit_times.len() / 2]
    );

    // Measure wait overhead (after submit)
    let mut wait_times = Vec::with_capacity(100);
    for _ in 0..100 {
        let mut cmd = queue.create_command_buffer()?;
        {
            let mut encoder = cmd.compute_encoder()?;
            encoder.set_pipeline(&pipeline)?;
            encoder.set_buffer(0, &sync_buffer, 0)?;
            encoder.dispatch_threads(1024 * 1024)?;
            encoder.end_encoding()?;
        }
        queue.submit(&mut cmd)?;

        let start = Instant::now();
        queue.wait_until_completed()?;
        wait_times.push(start.elapsed());
    }

    wait_times.sort();
    println!(
        "  Wait (1M threads):       {:>8.2?} avg, {:>8.2?} p50",
        wait_times.iter().sum::<Duration>() / wait_times.len() as u32,
        wait_times[wait_times.len() / 2]
    );

    // Measure back-to-back dispatch (no sync between)
    let mut batch_times = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();

        for _ in 0..10 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&pipeline)?;
                encoder.set_buffer(0, &sync_buffer, 0)?;
                encoder.dispatch_threads(1024 * 1024)?;
                encoder.end_encoding()?;
            }
            queue.submit(&mut cmd)?;
        }

        // Only wait for the last one
        queue.wait_until_completed()?;
        batch_times.push(start.elapsed());
    }

    batch_times.sort();
    let avg_per_dispatch =
        batch_times.iter().sum::<Duration>() / batch_times.len() as u32 / 10;
    println!(
        "  Batched (10 dispatches): {:>8.2?} per dispatch",
        avg_per_dispatch
    );
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "-".repeat(70));
    println!("Target overhead: <10us for small kernels");
    println!("Target compilation: <100ms for complex shaders");
    println!("Benchmark complete!");
    println!();

    Ok(())
}

fn format_threads(threads: u64) -> String {
    if threads >= 1_000_000 {
        format!("{}M", threads / 1_000_000)
    } else if threads >= 1_000 {
        format!("{}K", threads / 1_000)
    } else {
        format!("{}", threads)
    }
}
