//! Simplified GPU Optimization Test
//!
//! Demonstrates GPU utilization optimization without the Send trait issues

use anyhow::Result;
use cudarc::driver::CudaDevice;
use gpu_agents::utilization::{
    gpu_metrics::GpuMetricsCollector, gpu_workload_generator::GpuWorkloadGenerator,
    integrated_optimizer::IntegratedGpuOptimizer,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🚀 GPU Optimization Test - Simplified Version");
    println!("===========================================\n");

    let device = CudaDevice::new(0)?;

    // Phase 1: Setup
    println!("📋 Phase 1: Setting up optimization components");
    let mut workload_generator = GpuWorkloadGenerator::new(device.clone())?;
    let optimizer = IntegratedGpuOptimizer::new(device.clone()).await?;
    let metrics_collector = GpuMetricsCollector::new(device.clone());

    println!("  ✓ Components initialized");

    // Phase 2: Baseline measurement
    println!("\n📊 Phase 2: Baseline measurement");
    workload_generator.set_intensity(0.5);

    let baseline_start = Instant::now();
    let _baseline_stats = workload_generator.generate_burst_workload(100).await?;
    let baseline_time = baseline_start.elapsed();

    let baseline_metrics = metrics_collector.collect_metrics().await?;
    println!(
        "  Baseline GPU utilization: {:.1}%",
        baseline_metrics.compute_utilization * 100.0
    );
    println!(
        "  Baseline throughput: {:.0} kernels/sec",
        100.0 / baseline_time.as_secs_f64()
    );

    // Phase 3: Start optimization
    println!("\n⚡ Phase 3: Optimization");
    optimizer.start_optimization().await?;

    // Run optimized workload with increasing intensity
    let mut peak_utilization = 0.0f32;
    let test_duration = Duration::from_secs(20);
    let start_time = Instant::now();

    println!(
        "  Running optimized workload for {} seconds...",
        test_duration.as_secs()
    );

    while start_time.elapsed() < test_duration {
        // Gradually increase intensity
        let progress = start_time.elapsed().as_secs_f32() / test_duration.as_secs_f32();
        let intensity = 0.5 + (progress * 2.5); // 0.5 to 3.0
        workload_generator.set_intensity(intensity);

        // Generate workload
        let _stats = workload_generator.generate_burst_workload(50).await?;

        // Measure utilization
        let metrics = metrics_collector.collect_metrics().await?;
        let current_util = metrics.compute_utilization;

        if current_util > peak_utilization {
            peak_utilization = current_util;
        }

        // Progress update
        println!(
            "  [{:>3.0}s] Utilization: {:.1}% | Intensity: {:.1}x | Memory BW: {:.1}%",
            start_time.elapsed().as_secs(),
            current_util * 100.0,
            intensity,
            metrics.memory_bandwidth_utilization * 100.0
        );

        if current_util >= 0.90 {
            println!(
                "\n✅ Target achieved! GPU utilization: {:.1}%",
                current_util * 100.0
            );
            break;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Phase 4: Results
    println!("\n📈 Phase 4: Results");
    let final_report = optimizer.generate_report().await?;

    println!("  Peak GPU utilization: {:.1}%", peak_utilization * 100.0);
    println!(
        "  Improvement: +{:.1} percentage points",
        (peak_utilization - baseline_metrics.compute_utilization) * 100.0
    );

    println!("\n{}", final_report);

    if peak_utilization >= 0.90 {
        println!("\n🎉 SUCCESS: GPU optimization achieved target utilization!");
    } else {
        println!(
            "\n⚠️  Peak utilization {:.1}% - adjust parameters for 90%+ target",
            peak_utilization * 100.0
        );
    }

    println!("\n✅ Test completed!");
    Ok(())
}
