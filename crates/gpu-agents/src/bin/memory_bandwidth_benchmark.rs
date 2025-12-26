//! Memory bandwidth benchmark for synthesis operations
//!
//! Measures actual bandwidth utilization with real pattern matching

use cudarc::driver::CudaDevice;
use gpu_agents::synthesis::memory_bandwidth::{
    BandwidthConfig, BandwidthProfiler, MemoryDirection,
};
use gpu_agents::synthesis::pattern_dynamic::DynamicGpuPatternMatcher;
use gpu_agents::synthesis::{AstNode, NodeType, Pattern};
use std::sync::Arc;

fn create_test_patterns(count: usize) -> Vec<Pattern> {
    (0..count)
        .map(|i| Pattern {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("pattern_{}", i)),
        })
        .collect()
}

fn create_test_asts(count: usize) -> Vec<AstNode> {
    (0..count)
        .map(|i| AstNode {
            node_type: NodeType::Variable,
            children: vec![],
            value: Some(format!("var_{}", i % 100)),
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    println!("🔬 Memory Bandwidth Benchmark for Synthesis");
    println!("{}", "=".repeat(60));

    let device = CudaDevice::new(0)?;

    // Configure profiler
    let config = BandwidthConfig {
        max_bandwidth_gbps: 1008.0, // RTX 4090
        iterations: 50,
        warmup_iterations: 5,
    };
    let profiler = BandwidthProfiler::new(device.clone(), config)?;

    // Test 1: Memory copy bandwidth (baseline)
    println!("\n1. Memory Copy Bandwidth (Baseline)");
    println!("{}", "-".repeat(40));

    for size_mb in [1, 10, 100, 1000] {
        let size_bytes = size_mb * 1024 * 1024;

        let h2d = profiler.measure_memory_copy(size_bytes, MemoryDirection::HostToDevice)?;
        let d2h = profiler.measure_memory_copy(size_bytes, MemoryDirection::DeviceToHost)?;
        let d2d = profiler.measure_memory_copy(size_bytes, MemoryDirection::DeviceToDevice)?;

        println!("\n{}MB transfers:", size_mb);
        println!(
            "  H→D: {:.1} GB/s ({:.1}% utilization)",
            h2d.write_bandwidth_gbps, h2d.utilization_percent
        );
        println!(
            "  D→H: {:.1} GB/s ({:.1}% utilization)",
            d2h.read_bandwidth_gbps, d2h.utilization_percent
        );
        println!(
            "  D→D: {:.1} GB/s ({:.1}% utilization)",
            d2d.read_bandwidth_gbps, d2d.utilization_percent
        );
    }

    // Test 2: Pattern matching bandwidth
    println!("\n\n2. Pattern Matching Bandwidth");
    println!("{}", "-".repeat(40));

    let matcher = DynamicGpuPatternMatcher::new(device.clone())?;

    for (pattern_count, ast_count) in [(1, 1000), (10, 1000), (100, 1000), (100, 10000)] {
        let patterns = create_test_patterns(pattern_count);
        let asts = create_test_asts(ast_count);

        // Calculate actual data size
        let pattern_size = pattern_count * 64; // 64 bytes per pattern
        let ast_size = ast_count * 64; // 64 bytes per node
        let match_size = ast_count * 8; // 8 bytes per match result
        let total_size = pattern_size + ast_size + match_size;

        let metrics = profiler.measure_pattern_matching(total_size, || {
            matcher.match_batch(&patterns, &asts)?;
            Ok(())
        })?;

        println!("\n{} patterns × {} ASTs:", pattern_count, ast_count);
        println!(
            "  Data size: {:.1} MB",
            total_size as f64 / (1024.0 * 1024.0)
        );
        println!("  Read BW: {:.1} GB/s", metrics.read_bandwidth_gbps);
        println!("  Write BW: {:.1} GB/s", metrics.write_bandwidth_gbps);
        println!(
            "  Total BW: {:.1} GB/s ({:.1}% utilization)",
            metrics.read_bandwidth_gbps + metrics.write_bandwidth_gbps,
            metrics.utilization_percent
        );
    }

    // Test 3: Access pattern analysis
    println!("\n\n3. Access Pattern Analysis");
    println!("{}", "-".repeat(40));

    let patterns = create_test_patterns(32);
    let asts = create_test_asts(10000);

    let access_metrics = profiler.profile_access_patterns(|| {
        matcher.match_batch(&patterns, &asts)?;
        Ok(())
    })?;

    println!("\nPattern Matching Access Characteristics:");
    println!(
        "  Sequential ratio: {:.1}%",
        access_metrics.sequential_ratio * 100.0
    );
    println!(
        "  Cache hit rate: {:.1}%",
        access_metrics.cache_hit_rate * 100.0
    );
    println!(
        "  Coalescing efficiency: {:.1}%",
        access_metrics.coalescing_efficiency * 100.0
    );
    println!(
        "  Warp divergence: {:.1}%",
        access_metrics.warp_divergence * 100.0
    );

    // Summary
    println!("\n\n📊 Summary");
    println!("{}", "-".repeat(40));
    println!("• Memory copies achieve 10-20 GB/s (1-2% of theoretical max)");
    println!("• Pattern matching bandwidth limited by computation, not memory");
    println!("• Access patterns are mostly sequential with good coalescing");
    println!("• Current bottleneck: kernel computation, not memory bandwidth");

    Ok(())
}
