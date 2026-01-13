//! Simple Synthesis Micro-benchmarks
//!
//! Simplified benchmarks to measure synthesis performance
//! Target: 2.6B operations/second

use anyhow::Result;
use cudarc::driver::{CudaContext, DevicePtr};
use gpu_agents::synthesis::launch_match_patterns_fast;
use std::sync::Arc;
use std::time::Instant;

const TARGET_OPS_PER_SEC: f64 = 2.6e9; // 2.6 billion ops/sec

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Synthesis Micro-benchmarks (Simplified)");
    println!("==========================================");
    println!("Target: {:.1} billion ops/sec\n", TARGET_OPS_PER_SEC / 1e9);

    let ctx = CudaContext::new(0)?;

    // Phase 1: Raw Kernel Performance
    println!("📊 Phase 1: Raw CUDA Kernel Performance");
    println!("--------------------------------------");
    let raw_ops = benchmark_raw_kernel(ctx.clone())?;

    // Phase 2: Pattern Matching at Scale
    println!("\n📊 Phase 2: Pattern Matching at Scale");
    println!("------------------------------------");
    let scale_ops = benchmark_pattern_scale(ctx.clone())?;

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
fn benchmark_raw_kernel(ctx: Arc<CudaContext>) -> Result<f64> {
    const NODE_SIZE: usize = 64; // Aligned node size
    const BATCH_SIZE: usize = 10000;

    let stream = ctx.default_stream();

    // Allocate aligned buffers
    let pattern_size = NODE_SIZE * 32; // 32 patterns
    let ast_size = NODE_SIZE * BATCH_SIZE; // Large AST forest

    // Initialize with test data
    let pattern_data = vec![0u8; pattern_size];
    let ast_data = vec![0u8; ast_size];

    // Allocate and initialize buffers using stream
    let pattern_buffer = stream.clone_htod(&pattern_data)?;
    let ast_buffer = stream.clone_htod(&ast_data)?;
    let match_buffer = stream.alloc_zeros::<u32>(BATCH_SIZE * 2)?;

    let warmup_iters = 100;
    let measure_iters = 1000;

    // Warmup
    // SAFETY: All pointers are valid device pointers. pattern_buffer and ast_buffer
    // were initialized via clone_htod. Parameters match allocation sizes.
    for _ in 0..warmup_iters {
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_fast(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                32,
                BATCH_SIZE as u32,
            );
        }
    }
    stream.synchronize()?;

    // Measure
    // SAFETY: Same as warmup - all pointers valid, buffers initialized.
    let start = Instant::now();
    for _ in 0..measure_iters {
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_fast(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                32,
                BATCH_SIZE as u32,
            );
        }
    }
    stream.synchronize()?;
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
fn benchmark_pattern_scale(ctx: Arc<CudaContext>) -> Result<f64> {
    const NODE_SIZE: usize = 64;
    const MAX_PATTERNS: usize = 1024;
    const MAX_NODES: usize = 100000;

    let stream = ctx.default_stream();

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

        // Initialize with test data
        let pattern_data = vec![0u8; pattern_size];
        let ast_data = vec![0u8; ast_size];

        // Allocate and initialize buffers using stream
        let pattern_buffer = stream.clone_htod(&pattern_data)?;
        let ast_buffer = stream.clone_htod(&ast_data)?;
        let match_buffer = stream.alloc_zeros::<u32>(num_nodes * 2)?;

        // Time a single run
        // SAFETY: All pointers are valid device pointers. Buffers were initialized
        // via clone_htod. Parameters match allocation sizes for this scale.
        let start = Instant::now();
        let (pattern_ptr, _pattern_guard) = pattern_buffer.device_ptr(&stream);
        let (ast_ptr, _ast_guard) = ast_buffer.device_ptr(&stream);
        let (match_ptr, _match_guard) = match_buffer.device_ptr(&stream);
        unsafe {
            launch_match_patterns_fast(
                pattern_ptr as *const u8,
                ast_ptr as *const u8,
                match_ptr as *mut u32,
                num_patterns as u32,
                num_nodes as u32,
            );
        }
        stream.synchronize()?;
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
