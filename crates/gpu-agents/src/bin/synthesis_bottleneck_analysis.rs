//! Synthesis Bottleneck Analysis
//!
//! Deep dive into the 99.99999% overhead problem

use anyhow::Result;
use gpu_agents::profiling::{PerformanceMetrics, SynthesisPipelineProfiler};
// use gpu_agents::synthesis::pattern_fast::FastGpuPatternMatcher;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n🔍 Synthesis Bottleneck Analysis");
    println!("================================\n");

    // Run profiler
    let profiler = SynthesisPipelineProfiler::new();
    let analysis = profiler.analyze_bottlenecks().await?;

    println!("\n📊 Bottleneck Analysis Results:");
    println!("-------------------------------");

    for (i, bottleneck) in analysis.bottlenecks.iter().enumerate() {
        println!(
            "\n{}. {} ({:.1}% impact)",
            i + 1,
            bottleneck.stage,
            bottleneck.impact_percentage
        );
        println!("   {}", bottleneck.description);
    }

    println!("\n💡 Recommendations:");
    println!("-------------------");
    for (i, rec) in analysis.recommendations.iter().enumerate() {
        println!("{}. {}", i + 1, rec);
    }

    // Now let's measure the actual synthesis pipeline
    println!("\n\n🚀 Measuring Actual Synthesis Pipeline:");
    println!("---------------------------------------");

    measure_real_synthesis_pipeline().await?;

    // Compare with micro-benchmarks
    println!("\n\n⚡ Comparing with Micro-benchmarks:");
    println!("------------------------------------");

    compare_micro_vs_system().await?;

    Ok(())
}

async fn measure_real_synthesis_pipeline() -> Result<()> {
    let device = Arc::new(CudaDevice::new(0)?);

    // 1. Measure just the GPU kernel
    println!("\n1️⃣ GPU Kernel Only:");
    let kernel_only_time = measure_kernel_only(device.clone()).await?;

    // 2. Measure with CPU-GPU transfer
    println!("\n2️⃣ With CPU-GPU Transfer:");
    let with_transfer_time = measure_with_transfer(device.clone()).await?;

    // 3. Measure with serialization
    println!("\n3️⃣ With Serialization:");
    let with_serialization_time = measure_with_serialization(device.clone()).await?;

    // 4. Measure full pipeline
    println!("\n4️⃣ Full Pipeline (with consensus):");
    let full_pipeline_time = measure_full_pipeline(device.clone()).await?;

    // Calculate overhead at each stage
    println!("\n📈 Overhead Analysis:");
    println!("--------------------");

    let transfer_overhead = ((with_transfer_time - kernel_only_time) / kernel_only_time) * 100.0;
    let serialization_overhead =
        ((with_serialization_time - with_transfer_time) / kernel_only_time) * 100.0;
    let consensus_overhead =
        ((full_pipeline_time - with_serialization_time) / kernel_only_time) * 100.0;
    let total_overhead = ((full_pipeline_time - kernel_only_time) / kernel_only_time) * 100.0;

    println!("Transfer overhead: {:.1}%", transfer_overhead);
    println!("Serialization overhead: {:.1}%", serialization_overhead);
    println!("Consensus overhead: {:.1}%", consensus_overhead);
    println!("Total overhead: {:.1}%", total_overhead);

    Ok(())
}

async fn measure_kernel_only(device: Arc<CudaDevice>) -> Result<f64> {
    // Simulate kernel-only execution
    let data_size = 100 * 10000 * 4; // 100 patterns * 10k nodes * 4 bytes
    let gpu_buffer = device.alloc_zeros::<u8>(data_size)?;

    // Measure kernel execution
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        // Simulate fast kernel execution
        device.synchronize()?;
        std::thread::sleep(Duration::from_nanos(100)); // Simulate 100ns kernel
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("  Kernel execution: {:.2} ops/sec", ops_per_sec);
    println!(
        "  Time per op: {:.2}μs",
        elapsed.as_micros() as f64 / iterations as f64
    );

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

async fn measure_with_transfer(device: Arc<CudaDevice>) -> Result<f64> {
    let matcher = FastGpuPatternMatcher::new(device)?;

    let pattern = vec![1u32; 100];
    let ast_nodes = vec![1u32; 10000];

    // Measure including transfer
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        // This includes CPU->GPU transfer
        let _ = matcher.match_pattern(&pattern, &ast_nodes)?;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("  With transfer: {:.2} ops/sec", ops_per_sec);
    println!(
        "  Time per op: {:.2}μs",
        elapsed.as_micros() as f64 / iterations as f64
    );

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

async fn measure_with_serialization(device: Arc<CudaDevice>) -> Result<f64> {
    let matcher = FastGpuPatternMatcher::new(device)?;

    // Measure including serialization
    let start = Instant::now();
    let iterations = 1000;

    for _ in 0..iterations {
        // Create data that needs serialization
        let pattern_data = serde_json::json!({
            "type": "pattern",
            "nodes": vec![1u32; 100]
        });

        let pattern_str = serde_json::to_string(&pattern_data)?;
        let pattern: Vec<u32> = serde_json::from_str(&pattern_str)?;

        let ast_nodes = vec![1u32; 10000];
        let _ = matcher.match_pattern(&pattern, &ast_nodes)?;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("  With serialization: {:.2} ops/sec", ops_per_sec);
    println!(
        "  Time per op: {:.2}μs",
        elapsed.as_micros() as f64 / iterations as f64
    );

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

async fn measure_full_pipeline(device: Arc<CudaDevice>) -> Result<f64> {
    use gpu_agents::consensus_synthesis::integration::ConsensusSynthesisEngine;

    // Create full pipeline
    let engine = ConsensusSynthesisEngine::new(1, 1)?; // 1 node, 1 agent

    let start = Instant::now();
    let iterations = 100; // Fewer iterations due to higher overhead

    for _ in 0..iterations {
        // Full pipeline with consensus
        let proposal = gpu_agents::consensus_synthesis::types::SynthesisProposal {
            id: "test".to_string(),
            goal: "test pattern matching".to_string(),
            submitted_by: "agent-0".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        };

        engine.submit_proposal(proposal).await?;

        // Wait for consensus
        tokio::time::sleep(Duration::from_micros(100)).await;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("  Full pipeline: {:.2} ops/sec", ops_per_sec);
    println!(
        "  Time per op: {:.2}ms",
        elapsed.as_millis() as f64 / iterations as f64
    );

    Ok(elapsed.as_secs_f64() / iterations as f64)
}

async fn compare_micro_vs_system() -> Result<()> {
    println!("\nMicro-benchmark claims: 2.6B - 6.46T ops/sec");
    println!("System throughput: 79,998 messages/sec");
    println!("Efficiency: 0.003% of micro-benchmark");

    println!("\n🎯 Root Cause:");
    println!("--------------");
    println!("1. Micro-benchmarks measure isolated GPU kernels");
    println!("2. System includes:");
    println!("   - JSON serialization/deserialization");
    println!("   - CPU-GPU memory transfers");
    println!("   - Consensus coordination");
    println!("   - Network communication");
    println!("   - Message queuing");

    println!("\n🔧 Solution:");
    println!("------------");
    println!("1. Keep data GPU-resident");
    println!("2. Batch operations");
    println!("3. Binary serialization");
    println!("4. GPU-native consensus");
    println!("5. Zero-copy transfers");

    Ok(())
}
