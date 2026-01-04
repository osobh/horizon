//! Simple Synthesis Micro-benchmarks
//!
//! Simplified benchmarks to measure synthesis performance
//! Target: 2.6B operations/second

use anyhow::Result;
use cudarc::driver::{CudaDevice, DevicePtr};
use gpu_agents::synthesis::launch_match_patterns_fast;
use std::sync::Arc;
use std::time::Instant;

const TARGET_OPS_PER_SEC: f64 = 2.6e9; // 2.6 billion ops/sec

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Synthesis Micro-benchmarks (Simplified)");
    println!("==========================================");
    println!("Target: {:.1} billion ops/sec\n", TARGET_OPS_PER_SEC / 1e9);

    let device = CudaDevice::new(0)?;

    // Phase 1: Raw Kernel Performance
    println!("📊 Phase 1: Raw CUDA Kernel Performance");
    println!("--------------------------------------");
    let raw_ops = benchmark_raw_kernel(device.clone())?;

    // Phase 2: Pattern Matching at Scale
    println!("\n📊 Phase 2: Pattern Matching at Scale");
    println!("------------------------------------");
    let scale_ops = benchmark_pattern_scale(device.clone())?;

    // Phase 3: Optimization Analysis
    println!("\n📈 Performance Analysis");
    println!("======================");

    let best_ops = raw_ops.max(scale_ops);
    let percentage = (best_ops / TARGET_OPS_PER_SEC) * 100.0;

    println!(
        "\n  Best Performance: {:.2} million ops/sec ({:.1}% of target)",
        best_ops / 1e6,
        percentage
    );

    if percentage >= 100.0 {
        println!("\n🎉 SUCCESS: Target performance achieved!");
    } else {
        println!(
            "\n⚠️  Currently at {:.1}% of target performance",
            percentage
        );

        println!("\n💡 Recommendations to reach 2.6B ops/sec:");

        if best_ops < 1e8 {
            println!("  1. Enable GPU compiler optimizations (-O3)");
            println!("  2. Use shared memory for pattern data");
            println!("  3. Implement warp-level primitives");
            println!("  4. Increase block size to 256 or 512 threads");
        }

        if best_ops < 1e9 {
            println!("  5. Batch multiple patterns per thread");
            println!("  6. Use texture memory for read-only data");
            println!("  7. Implement kernel fusion");
            println!("  8. Enable persistent kernels");
        }

        println!("  9. Profile with Nsight Compute for bottlenecks");
        println!(" 10. Consider using Tensor Cores if available");
    }

    Ok(())
}

/// Benchmark raw kernel performance with minimal overhead
fn benchmark_raw_kernel(device: Arc<CudaDevice>) -> Result<f64> {
    const NODE_SIZE: usize = 64; // Aligned node size
    const BATCH_SIZE: usize = 10000;

    // Allocate aligned buffers
    let pattern_size = NODE_SIZE * 32; // 32 patterns
    let ast_size = NODE_SIZE * BATCH_SIZE; // Large AST forest

    // SAFETY: alloc returns uninitialized memory. pattern_buffer and ast_buffer
    // will be written via htod_copy_into before any kernel reads.
    let pattern_buffer = unsafe { device.alloc::<u8>(pattern_size)? };
    let ast_buffer = unsafe { device.alloc::<u8>(ast_size)? };
    let match_buffer = device.alloc_zeros::<u32>(BATCH_SIZE * 2)?;

    // Initialize with test data
    let pattern_data = vec![0u8; pattern_size];
    let ast_data = vec![0u8; ast_size];

    device.htod_copy_into(pattern_data, &mut pattern_buffer.clone())?;
    device.htod_copy_into(ast_data, &mut ast_buffer.clone())?;

    let warmup_iters = 100;
    let measure_iters = 1000;

    // Warmup
    // SAFETY: All pointers are valid device pointers. pattern_buffer and ast_buffer
    // were initialized via htod_copy_into. Parameters match allocation sizes.
    for _ in 0..warmup_iters {
        unsafe {
            launch_match_patterns_fast(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                32,
                BATCH_SIZE as u32,
            );
        }
    }
    device.synchronize()?;

    // Measure
    // SAFETY: Same as warmup - all pointers valid, buffers initialized.
    let start = Instant::now();
    for _ in 0..measure_iters {
        unsafe {
            launch_match_patterns_fast(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                32,
                BATCH_SIZE as u32,
            );
        }
    }
    device.synchronize()?;
    let elapsed = start.elapsed();

    let total_ops = (measure_iters * 32 * BATCH_SIZE) as f64;
    let ops_per_sec = total_ops / elapsed.as_secs_f64();

    println!(
        "  Raw kernel: {:.2} million ops/sec ({:.1}% of target)",
        ops_per_sec / 1e6,
        (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
    );

    Ok(ops_per_sec)
}

/// Benchmark pattern matching at scale
fn benchmark_pattern_scale(device: Arc<CudaDevice>) -> Result<f64> {
    const NODE_SIZE: usize = 64;
    const MAX_PATTERNS: usize = 1024;
    const MAX_NODES: usize = 100000;

    // Test different scales
    let scales = vec![
        (32, 1000),     // Small
        (128, 10000),   // Medium
        (512, 50000),   // Large
        (1024, 100000), // XLarge
    ];

    let mut best_ops: f64 = 0.0;

    for (num_patterns, num_nodes) in scales {
        println!(
            "\n  Testing {} patterns × {} nodes:",
            num_patterns, num_nodes
        );

        // Allocate buffers for this scale
        let pattern_size = NODE_SIZE * num_patterns;
        let ast_size = NODE_SIZE * num_nodes;

        // SAFETY: alloc returns uninitialized memory. Buffers will be written
        // via htod_copy_into before kernel execution.
        let pattern_buffer = unsafe { device.alloc::<u8>(pattern_size)? };
        let ast_buffer = unsafe { device.alloc::<u8>(ast_size)? };
        let match_buffer = device.alloc_zeros::<u32>(num_nodes * 2)?;

        // Initialize
        let pattern_data = vec![0u8; pattern_size];
        let ast_data = vec![0u8; ast_size];

        device.htod_copy_into(pattern_data, &mut pattern_buffer.clone())?;
        device.htod_copy_into(ast_data, &mut ast_buffer.clone())?;

        // Time a single run
        // SAFETY: All pointers are valid device pointers. Buffers were initialized
        // via htod_copy_into. Parameters match allocation sizes for this scale.
        let start = Instant::now();
        unsafe {
            launch_match_patterns_fast(
                *pattern_buffer.device_ptr() as *const u8,
                *ast_buffer.device_ptr() as *const u8,
                *match_buffer.device_ptr() as *mut u32,
                num_patterns as u32,
                num_nodes as u32,
            );
        }
        device.synchronize()?;
        let elapsed = start.elapsed();

        let ops = (num_patterns * num_nodes) as f64;
        let ops_per_sec = ops / elapsed.as_secs_f64();

        println!(
            "    → {:.2} million ops/sec ({:.1}% of target)",
            ops_per_sec / 1e6,
            (ops_per_sec / TARGET_OPS_PER_SEC) * 100.0
        );

        best_ops = best_ops.max(ops_per_sec);
    }

    Ok(best_ops)
}
