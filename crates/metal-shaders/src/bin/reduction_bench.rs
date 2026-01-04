//! Parallel Reduction Benchmark for Metal GPU
//!
//! Classic GPU benchmark measuring reduction efficiency:
//! - Sum reduction
//! - Max reduction with indices
//! - GPU vs CPU comparison
//!
//! Run with: cargo run --release -p stratoswarm-metal-shaders --bin reduction_bench

use std::time::{Duration, Instant};
use stratoswarm_metal_shaders::core::backend::MetalBackend;
use stratoswarm_metal_shaders::core::buffer::MetalBuffer;
use stratoswarm_metal_shaders::core::command::{
    MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder,
};
use stratoswarm_metal_shaders::core::metal3::Metal3Backend;
use stratoswarm_metal_shaders::{combine_shaders, common};

fn main() -> anyhow::Result<()> {
    println!("Metal Parallel Reduction Benchmark (M4)");
    println!("{}", "=".repeat(70));
    println!();

    // Initialize Metal backend
    let backend = Metal3Backend::new()?;
    let device = backend.device();
    let info = device.info();

    println!("Device: {}", info.name);
    println!();

    // Reduction shaders
    let reduction_shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        // First-level reduction: reduce within threadgroups
        kernel void sum_reduce_first(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            device atomic_uint* count [[buffer(2)]],
            constant uint& n [[buffer(3)]],
            threadgroup float* shared [[threadgroup(0)]],
            uint tid [[thread_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint gid [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]]
        ) {
            // Load into shared memory
            float sum = 0.0f;
            if (tid < n) {
                sum = input[tid];
            }
            shared[lid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce within threadgroup
            for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride && tid + stride < n) {
                    shared[lid] += shared[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write result
            if (lid == 0) {
                output[gid] = shared[0];
            }
        }

        // Second-level reduction: reduce the partial sums
        kernel void sum_reduce_final(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            threadgroup float* shared [[threadgroup(0)]],
            uint tid [[thread_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint group_size [[threads_per_threadgroup]]
        ) {
            float sum = 0.0f;
            for (uint i = tid; i < n; i += group_size) {
                sum += input[i];
            }
            shared[lid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    shared[lid] += shared[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                output[0] = shared[0];
            }
        }

        // Max reduction with index
        kernel void max_reduce(
            device const float* input [[buffer(0)]],
            device float* max_values [[buffer(1)]],
            device uint* max_indices [[buffer(2)]],
            constant uint& n [[buffer(3)]],
            threadgroup float* shared_vals [[threadgroup(0)]],
            threadgroup uint* shared_idx [[threadgroup(1)]],
            uint tid [[thread_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint gid [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]]
        ) {
            // Load into shared memory
            float val = -INFINITY;
            uint idx = 0;
            if (tid < n) {
                val = input[tid];
                idx = tid;
            }
            shared_vals[lid] = val;
            shared_idx[lid] = idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce within threadgroup
            for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    if (shared_vals[lid + stride] > shared_vals[lid]) {
                        shared_vals[lid] = shared_vals[lid + stride];
                        shared_idx[lid] = shared_idx[lid + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write result
            if (lid == 0) {
                max_values[gid] = shared_vals[0];
                max_indices[gid] = shared_idx[0];
            }
        }

        // Simple sum (no threadgroup reduction, for baseline)
        kernel void simple_sum(
            device const float* input [[buffer(0)]],
            device atomic_uint* output [[buffer(1)]],
            uint tid [[thread_position_in_grid]]
        ) {
            // Atomic add (slow but simple)
            uint val = as_type<uint>(input[tid]);
            atomic_fetch_add_explicit(output, val, memory_order_relaxed);
        }
    "#;

    // Combine with common utilities
    let shader_source = combine_shaders(&[common::ATOMICS, reduction_shader]);

    println!("Compiling reduction shaders...");
    let sum_first_pipeline = backend.create_compute_pipeline(&shader_source, "sum_reduce_first")?;
    let sum_final_pipeline = backend.create_compute_pipeline(&shader_source, "sum_reduce_final")?;
    let max_pipeline = backend.create_compute_pipeline(&shader_source, "max_reduce")?;
    println!("Compilation successful!");
    println!();

    let queue = backend.create_command_queue()?;

    // ========================================================================
    // Benchmark 1: Sum Reduction
    // ========================================================================
    println!("1. Sum Reduction Performance");
    println!("{}", "-".repeat(70));
    println!(
        "{:>15} {:>12} {:>12} {:>15} {:>15}",
        "Elements", "GPU Time", "CPU Time", "GPU GB/s", "Speedup"
    );

    let sizes: [usize; 7] = [
        1_000_000,   // 1M
        4_000_000,   // 4M
        16_000_000,  // 16M
        64_000_000,  // 64M
        128_000_000, // 128M
        256_000_000, // 256M
        512_000_000, // 512M (2GB)
    ];

    for n in sizes {
        // Create and initialize input buffer
        let input = backend.create_buffer::<f32>(n)?;
        {
            let mut input_init = backend.create_buffer::<f32>(n)?;
            let data = input_init.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = 1.0 + (i as f32) * 0.0000001; // Small variation to prevent optimization
            }
        }

        // Calculate number of threadgroups
        let group_size = 256u32;
        let num_groups = ((n as u32) + group_size - 1) / group_size;

        // Partial sums buffer
        let partial_sums = backend.create_buffer::<f32>(num_groups as usize)?;
        let result = backend.create_buffer::<f32>(1)?;
        let count = backend.create_buffer::<u32>(1)?;

        // GPU warmup
        for _ in 0..3 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_first_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &partial_sums, 0)?;
                encoder.set_buffer(2, &count, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_final_pipeline)?;
                encoder.set_buffer(0, &partial_sums, 0)?;
                encoder.set_buffer(1, &result, 0)?;
                encoder.set_bytes(2, &num_groups.to_ne_bytes())?;
                encoder.dispatch_threads(group_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // GPU benchmark
        let iterations = 10;
        let mut gpu_times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_first_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &partial_sums, 0)?;
                encoder.set_buffer(2, &count, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_final_pipeline)?;
                encoder.set_buffer(0, &partial_sums, 0)?;
                encoder.set_buffer(1, &result, 0)?;
                encoder.set_bytes(2, &num_groups.to_ne_bytes())?;
                encoder.dispatch_threads(group_size as u64)?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            gpu_times.push(start.elapsed());
        }

        gpu_times.sort();
        let gpu_avg = gpu_times.iter().sum::<Duration>() / gpu_times.len() as u32;

        // CPU benchmark (for comparison)
        let cpu_data: Vec<f32> = (0..n.min(10_000_000))
            .map(|i| 1.0 + (i as f32) * 0.0000001)
            .collect();

        let cpu_start = Instant::now();
        let _cpu_sum: f64 = cpu_data.iter().map(|&x| x as f64).sum();
        let cpu_time = cpu_start.elapsed();

        // Scale CPU time if we used smaller dataset
        let cpu_scaled = if n > 10_000_000 {
            Duration::from_secs_f64(cpu_time.as_secs_f64() * (n as f64 / 10_000_000.0))
        } else {
            cpu_time
        };

        let bytes = n * std::mem::size_of::<f32>();
        let gpu_bandwidth = (bytes as f64) / gpu_avg.as_secs_f64() / 1e9;
        let speedup = cpu_scaled.as_secs_f64() / gpu_avg.as_secs_f64();

        println!(
            "{:>15} {:>10.2?} {:>10.2?} {:>13.2} {:>13.1}x",
            format_count(n as u64),
            gpu_avg,
            cpu_scaled,
            gpu_bandwidth,
            speedup
        );
    }
    println!();

    // ========================================================================
    // Benchmark 2: Max Reduction with Index
    // ========================================================================
    println!("2. Max Reduction with Index");
    println!("{}", "-".repeat(70));
    println!(
        "{:>15} {:>12} {:>15} {:>15}",
        "Elements", "GPU Time", "GB/s", "M ops/sec"
    );

    for n in &sizes[..5] {
        // Max reduction needs more memory per element
        let n = *n;
        let group_size = 256u32;
        let num_groups = ((n as u32) + group_size - 1) / group_size;

        let input = backend.create_buffer::<f32>(n)?;
        let max_values = backend.create_buffer::<f32>(num_groups as usize)?;
        let max_indices = backend.create_buffer::<u32>(num_groups as usize)?;

        // Initialize with pattern that has max somewhere in the middle
        {
            let mut input_init = backend.create_buffer::<f32>(n)?;
            let data = input_init.contents_mut::<f32>();
            for (i, val) in data.iter_mut().enumerate() {
                *val = (i as f32).sin();
            }
        }

        // Warmup
        for _ in 0..3 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&max_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &max_values, 0)?;
                encoder.set_buffer(2, &max_indices, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        let iterations = 20;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&max_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &max_values, 0)?;
                encoder.set_buffer(2, &max_indices, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;

        let bytes = n * std::mem::size_of::<f32>();
        let bandwidth = (bytes as f64) / avg.as_secs_f64() / 1e9;
        let ops_per_sec = (n as f64) / avg.as_secs_f64() / 1e6;

        println!(
            "{:>15} {:>10.2?} {:>13.2} {:>13.1}",
            format_count(n as u64),
            avg,
            bandwidth,
            ops_per_sec
        );
    }
    println!();

    // ========================================================================
    // Benchmark 3: Reduction Efficiency Analysis
    // ========================================================================
    println!("3. Reduction Efficiency (% of Peak Bandwidth)");
    println!("{}", "-".repeat(70));
    println!(
        "{:>15} {:>12} {:>15} {:>15}",
        "Elements", "GB/s", "% of 120GB/s", "Efficiency"
    );

    let peak_bandwidth = 120.0; // M4 theoretical peak

    for n in &sizes[..5] {
        let n = *n;
        let group_size = 256u32;
        let num_groups = ((n as u32) + group_size - 1) / group_size;

        let input = backend.create_buffer::<f32>(n)?;
        let partial_sums = backend.create_buffer::<f32>(num_groups as usize)?;
        let result = backend.create_buffer::<f32>(1)?;
        let count = backend.create_buffer::<u32>(1)?;

        // Warmup
        for _ in 0..5 {
            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_first_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &partial_sums, 0)?;
                encoder.set_buffer(2, &count, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;
        }

        // Benchmark
        let iterations = 50;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            let mut cmd = queue.create_command_buffer()?;
            {
                let mut encoder = cmd.compute_encoder()?;
                encoder.set_pipeline(&sum_first_pipeline)?;
                encoder.set_buffer(0, &input, 0)?;
                encoder.set_buffer(1, &partial_sums, 0)?;
                encoder.set_buffer(2, &count, 0)?;
                encoder.set_bytes(3, &(n as u32).to_ne_bytes())?;
                encoder
                    .dispatch_threadgroups((num_groups as u64, 1, 1), (group_size as u64, 1, 1))?;
                encoder.end_encoding()?;
            }
            queue.submit_and_wait(cmd)?;

            times.push(start.elapsed());
        }

        times.sort();
        let avg = times.iter().sum::<Duration>() / times.len() as u32;

        let bytes = n * std::mem::size_of::<f32>();
        let bandwidth = (bytes as f64) / avg.as_secs_f64() / 1e9;
        let pct_peak = (bandwidth / peak_bandwidth) * 100.0;

        // Theoretical efficiency (reduction reads each element once)
        let efficiency = if bandwidth > 0.0 {
            format!("{:.1}%", pct_peak)
        } else {
            "N/A".to_string()
        };

        println!(
            "{:>15} {:>10.2} {:>13.1}% {:>15}",
            format_count(n as u64),
            bandwidth,
            pct_peak,
            efficiency
        );
    }
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "-".repeat(70));
    println!("Target: >50 GB/s effective bandwidth for reduction");
    println!("M4 Peak Memory Bandwidth: ~120 GB/s");
    println!("Benchmark complete!");
    println!();

    Ok(())
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{}B", n / 1_000_000_000)
    } else if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}
