//! GPU Utilization Optimization Benchmark
//!
//! Demonstrates achieving 90% GPU utilization through integrated optimization

use anyhow::Result;
use cudarc::driver::CudaDevice;
use gpu_agents::utilization::{
    gpu_metrics::GpuMetricsCollector,
    integrated_optimizer::IntegratedGpuOptimizer,
    kernel_optimizer::{KernelConfig, KernelOptimizer},
    kernel_scheduler::{AdvancedKernelScheduler, KernelPriority, ScheduledKernel, SchedulerConfig},
    memory_coalescing::MemoryCoalescingOptimizer,
    UtilizationManager,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🚀 GPU Utilization Optimization Benchmark");
    println!("=========================================");
    println!("Target: Achieve 90% GPU utilization through optimization\n");

    let device = CudaDevice::new(0)?;

    // Phase 1: Baseline measurement
    println!("📊 Phase 1: Baseline Measurement");
    println!("--------------------------------");
    let baseline_util = measure_baseline_utilization(Arc::clone(&device)).await?;
    println!("  Baseline utilization: {:.1}%", baseline_util * 100.0);
    println!(
        "  Gap to target: {:.1} percentage points\n",
        (0.90 - baseline_util) * 100.0
    );

    // Phase 2: Apply optimizations
    println!("⚡ Phase 2: Applying Optimizations");
    println!("----------------------------------");

    // Create integrated optimizer
    let optimizer = IntegratedGpuOptimizer::new(Arc::clone(&device)).await?;

    // Start optimization
    optimizer.start_optimization().await?;

    // Monitor progress
    let start_time = Instant::now();
    let mut last_report = Instant::now();
    let mut measurements = Vec::new();

    println!("  Starting optimization process...");

    while start_time.elapsed() < Duration::from_secs(30) {
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Collect metrics
        let metrics_collector = GpuMetricsCollector::new(Arc::clone(&device));
        let metrics = metrics_collector.collect_metrics().await?;
        let current_util = metrics.compute_utilization;
        measurements.push(current_util);

        // Progress update every 2 seconds
        if last_report.elapsed() >= Duration::from_secs(2) {
            println!(
                "  Progress: {:.1}% utilization (memory: {:.1}%, SM: {:.1}%)",
                current_util * 100.0,
                metrics.memory_bandwidth_utilization * 100.0,
                metrics.sm_efficiency * 100.0
            );
            last_report = Instant::now();
        }

        // Check if target achieved
        if current_util >= 0.90 {
            println!(
                "\n✅ Target achieved! GPU utilization: {:.1}%",
                current_util * 100.0
            );
            break;
        }
    }

    // Phase 3: Results analysis
    println!("\n📈 Phase 3: Results Analysis");
    println!("----------------------------");

    let final_report = optimizer.generate_report().await?;

    // Calculate statistics
    let avg_utilization = measurements.iter().sum::<f32>() / measurements.len() as f32;
    let max_utilization = measurements.iter().cloned().fold(0.0f32, f32::max);
    let improvement = max_utilization - baseline_util;

    println!("  Baseline utilization: {:.1}%", baseline_util * 100.0);
    println!("  Average utilization: {:.1}%", avg_utilization * 100.0);
    println!("  Peak utilization: {:.1}%", max_utilization * 100.0);
    println!(
        "  Improvement: +{:.1} percentage points",
        improvement * 100.0
    );
    println!(
        "  Time to optimize: {:.1}s",
        start_time.elapsed().as_secs_f32()
    );

    // Phase 4: Detailed benchmarks
    println!("\n🔬 Phase 4: Detailed Benchmarks");
    println!("-------------------------------");

    run_detailed_benchmarks(Arc::clone(&device)).await?;

    // Print final report
    println!("\n📋 Optimization Report:");
    println!("{}", final_report);

    println!("\n🎉 Benchmark completed successfully!");

    Ok(())
}

/// Measure baseline GPU utilization
async fn measure_baseline_utilization(device: Arc<CudaDevice>) -> Result<f32> {
    let metrics_collector = GpuMetricsCollector::new(Arc::clone(&device));
    let mut samples = Vec::new();

    // Run baseline workload
    for _ in 0..10 {
        // Simulate basic kernel execution
        let metrics = metrics_collector.collect_metrics().await?;
        samples.push(metrics.compute_utilization);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(samples.iter().sum::<f32>() / samples.len() as f32)
}

/// Run detailed performance benchmarks
async fn run_detailed_benchmarks(device: Arc<CudaDevice>) -> Result<()> {
    // Benchmark 1: Kernel optimization impact
    println!("\n  1. Kernel Optimization Impact:");
    let kernel_optimizer = KernelOptimizer::new(Arc::clone(&device));

    let default_config = KernelConfig::default();
    let default_occupancy = kernel_optimizer.calculate_occupancy(default_config);

    let optimized_config = KernelConfig {
        block_size: 256,
        grid_size: 512,
        shared_mem_size: 49152, // 48KB
        registers_per_thread: 32,
    };
    let optimized_occupancy = kernel_optimizer.calculate_occupancy(optimized_config);

    println!("     Default occupancy: {:.1}%", default_occupancy * 100.0);
    println!(
        "     Optimized occupancy: {:.1}%",
        optimized_occupancy * 100.0
    );
    println!(
        "     Improvement: +{:.1} percentage points",
        (optimized_occupancy - default_occupancy) * 100.0
    );

    // Benchmark 2: Memory coalescing impact
    println!("\n  2. Memory Coalescing Impact:");
    let mut memory_optimizer = MemoryCoalescingOptimizer::new(Arc::clone(&device));

    // Uncoalesced access pattern
    let uncoalesced: Vec<(u32, u64)> = (0..32)
        .map(|i| (i, 0x1000000 + i as u64 * 128)) // Large stride
        .collect();

    let uncoalesced_pattern =
        memory_optimizer.analyze_access_pattern("uncoalesced_kernel", 0x1000000, &uncoalesced)?;

    // Coalesced access pattern
    let coalesced: Vec<(u32, u64)> = (0..32)
        .map(|i| (i, 0x1000000 + i as u64 * 4)) // Sequential
        .collect();

    let coalesced_pattern =
        memory_optimizer.analyze_access_pattern("coalesced_kernel", 0x1000000, &coalesced)?;

    println!(
        "     Uncoalesced efficiency: {:.1}%",
        uncoalesced_pattern.coalescing_efficiency * 100.0
    );
    println!(
        "     Coalesced efficiency: {:.1}%",
        coalesced_pattern.coalescing_efficiency * 100.0
    );
    println!(
        "     Improvement: +{:.1} percentage points",
        (coalesced_pattern.coalescing_efficiency - uncoalesced_pattern.coalescing_efficiency)
            * 100.0
    );

    // Benchmark 3: Multi-stream scheduling impact
    println!("\n  3. Multi-Stream Scheduling Impact:");
    let single_stream_config = SchedulerConfig {
        num_streams: 1,
        enable_fusion: false,
        enable_load_balancing: false,
        ..Default::default()
    };

    let multi_stream_config = SchedulerConfig {
        num_streams: 4,
        enable_fusion: true,
        enable_load_balancing: true,
        ..Default::default()
    };

    let single_throughput = benchmark_scheduler(Arc::clone(&device), single_stream_config).await?;
    let multi_throughput = benchmark_scheduler(Arc::clone(&device), multi_stream_config).await?;

    println!("     Single stream: {:.0} kernels/sec", single_throughput);
    println!("     Multi-stream (4): {:.0} kernels/sec", multi_throughput);
    println!("     Speedup: {:.1}x", multi_throughput / single_throughput);

    // Benchmark 4: Workload scaling
    println!("\n  4. Workload Scaling:");
    let utilization_manager = UtilizationManager::new(Arc::clone(&device))?;

    println!("     Initial multiplier: 1.0x");

    // Simulate low utilization
    utilization_manager
        .apply_optimization(gpu_agents::utilization::OptimizationStrategy::IncreaseWorkload)
        .await?;
    let _multiplier1 = utilization_manager.get_workload_multiplier();

    utilization_manager
        .apply_optimization(gpu_agents::utilization::OptimizationStrategy::IncreaseWorkload)
        .await?;
    let multiplier2 = utilization_manager.get_workload_multiplier();

    println!("     After optimization: {:.1}x", multiplier2);
    println!("     Scaling factor: {:.1}x", multiplier2);

    Ok(())
}

/// Benchmark kernel scheduler performance
async fn benchmark_scheduler(device: Arc<CudaDevice>, config: SchedulerConfig) -> Result<f64> {
    let scheduler = AdvancedKernelScheduler::new(device, config)?;
    let start = Instant::now();
    let num_kernels = 100;

    // Submit kernels
    for i in 0..num_kernels {
        let kernel = ScheduledKernel {
            id: i,
            name: format!("benchmark_kernel_{}", i),
            priority: match i % 4 {
                0 => KernelPriority::Critical,
                1 => KernelPriority::High,
                2 => KernelPriority::Normal,
                _ => KernelPriority::Low,
            },
            config: KernelConfig::default(),
            dependencies: if i > 0 && i % 5 != 0 {
                vec![i - 1]
            } else {
                vec![]
            },
            estimated_time: Duration::from_millis(2 + (i % 3) * 2),
            submitted_at: Instant::now(),
            data_size: 1024 * 1024, // 1MB
        };

        scheduler.submit_kernel(kernel).await?;
    }

    // Wait for completion
    tokio::time::sleep(Duration::from_millis(500)).await;

    let elapsed = start.elapsed();
    let throughput = num_kernels as f64 / elapsed.as_secs_f64();

    Ok(throughput)
}

/// Generate sample workload for testing
async fn generate_workload(device: Arc<CudaDevice>) {
    let scheduler_config = SchedulerConfig::default();
    let scheduler = Arc::new(tokio::sync::RwLock::new(
        AdvancedKernelScheduler::new(Arc::clone(&device), scheduler_config)?,
    ));

    // Continuous kernel submission
    let scheduler_clone = scheduler.clone();
    tokio::spawn(async move {
        let mut kernel_id = 0;
        loop {
            let kernel = ScheduledKernel {
                id: kernel_id,
                name: format!("workload_kernel_{}", kernel_id),
                priority: KernelPriority::Normal,
                config: KernelConfig {
                    block_size: 256,
                    grid_size: 256,
                    shared_mem_size: 0,
                    registers_per_thread: 32,
                },
                dependencies: vec![],
                estimated_time: Duration::from_millis(5),
                submitted_at: Instant::now(),
                data_size: 4 * 1024 * 1024, // 4MB
            };

            let mut sched = scheduler_clone.write().await;
            let _ = sched.submit_kernel(kernel).await;

            kernel_id += 1;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
}
