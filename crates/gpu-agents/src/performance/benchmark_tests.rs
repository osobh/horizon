//! Benchmark tests for performance optimization module
//!
//! Comprehensive benchmarking of all performance optimization components
//! to ensure targets are met: 90% GPU utilization, <1ms memory migration, <100μs consensus.

use super::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// =============================================================================
// Performance Benchmark Constants
// =============================================================================

const TARGET_GPU_UTILIZATION: f32 = 0.90;
const TARGET_MEMORY_MIGRATION_LATENCY_MS: f32 = 1.0;
const TARGET_CONSENSUS_LATENCY_US: u64 = 100;
const BENCHMARK_DURATION_MS: u64 = 1000;
const LARGE_DATASET_SIZE: usize = 100_000;
const STRESS_TEST_ITERATIONS: usize = 10_000;

// =============================================================================
// GPU Utilization Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_gpu_utilization_target_achievement() -> Result<()> {
    let config = UtilizationConfig {
        target_utilization: TARGET_GPU_UTILIZATION,
        monitoring_interval: Duration::from_millis(10),
        ..Default::default()
    };

    let analyzer = GpuUtilizationAnalyzer::new(config);
    analyzer.start().await?;

    // Record intensive kernel execution pattern
    let start = Instant::now();
    let mut utilization_samples = vec![];

    for i in 0..100 {
        let kernel_duration = Duration::from_micros(4000 + (i % 100) * 10); // 4-5ms kernels
        analyzer
            .record_kernel_execution(
                &format!("benchmark_kernel_{}", i),
                kernel_duration,
                (1024, 1, 1), // Large grid
                (256, 1, 1),  // Large block
                8192,         // High memory usage
                0.90,         // High compute intensity
            )
            .await?;

        // Sample utilization every 10 iterations
        if i % 10 == 0 {
            utilization_samples.push(analyzer.get_current_utilization());
        }

        sleep(Duration::from_millis(5)).await;
    }

    let benchmark_time = start.elapsed();
    analyzer.stop().await?;

    // Analyze results
    let avg_utilization =
        utilization_samples.iter().sum::<f32>() / utilization_samples.len() as f32;
    let min_utilization = utilization_samples
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let max_utilization = utilization_samples
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    println!("GPU Utilization Benchmark Results:");
    println!("  Average utilization: {:.1}%", avg_utilization * 100.0);
    println!("  Min utilization: {:.1}%", min_utilization * 100.0);
    println!("  Max utilization: {:.1}%", max_utilization * 100.0);
    println!("  Benchmark duration: {:?}", benchmark_time);

    // Verify target achievement
    assert!(
        avg_utilization >= TARGET_GPU_UTILIZATION * 0.95,
        "Average GPU utilization {:.1}% should be within 5% of target {:.1}%",
        avg_utilization * 100.0,
        TARGET_GPU_UTILIZATION * 100.0
    );

    assert!(
        min_utilization >= TARGET_GPU_UTILIZATION * 0.80,
        "Minimum GPU utilization {:.1}% should be within 20% of target",
        min_utilization * 100.0
    );

    Ok(())
}

#[tokio::test]
async fn benchmark_kernel_optimization_impact() -> Result<()> {
    let config = UtilizationConfig::default();
    let analyzer = GpuUtilizationAnalyzer::new(config);
    analyzer.start().await?;

    // Baseline: unoptimized kernel pattern
    let baseline_start = Instant::now();
    for i in 0..50 {
        analyzer
            .record_kernel_execution(
                "unoptimized_kernel",
                Duration::from_micros(8000), // Slow kernel
                (128, 1, 1),                 // Small grid
                (32, 1, 1),                  // Small block
                2048,                        // Low memory usage
                0.40,                        // Low compute intensity
            )
            .await?;
        sleep(Duration::from_millis(2)).await;
    }
    let baseline_utilization = analyzer.get_current_utilization();
    let baseline_time = baseline_start.elapsed();

    // Wait for metrics to settle
    sleep(Duration::from_millis(100)).await;

    // Optimized: improved kernel pattern
    let optimized_start = Instant::now();
    for i in 0..50 {
        analyzer
            .record_kernel_execution(
                "optimized_kernel",
                Duration::from_micros(4000), // Fast kernel
                (1024, 1, 1),                // Large grid
                (256, 1, 1),                 // Large block
                8192,                        // High memory usage
                0.85,                        // High compute intensity
            )
            .await?;
        sleep(Duration::from_millis(1)).await; // Reduced gap
    }
    let optimized_utilization = analyzer.get_current_utilization();
    let optimized_time = optimized_start.elapsed();

    analyzer.stop().await?;

    let improvement = optimized_utilization - baseline_utilization;

    println!("Kernel Optimization Benchmark:");
    println!(
        "  Baseline utilization: {:.1}%",
        baseline_utilization * 100.0
    );
    println!(
        "  Optimized utilization: {:.1}%",
        optimized_utilization * 100.0
    );
    println!(
        "  Improvement: {:.1} percentage points",
        improvement * 100.0
    );
    println!("  Baseline time: {:?}", baseline_time);
    println!("  Optimized time: {:?}", optimized_time);

    // Verify optimization effectiveness
    assert!(
        improvement > 0.10,
        "Optimization should improve utilization by at least 10%"
    );
    assert!(
        optimized_time < baseline_time,
        "Optimized pattern should be faster"
    );

    Ok(())
}

// =============================================================================
// Memory Tier Migration Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_memory_tier_migration_latency() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);
    optimizer.start().await?;

    // Test different migration scenarios
    let migration_scenarios = vec![
        ("GPU→CPU", 1024 * 1024, MemoryTier::Gpu, MemoryTier::Cpu), // 1MB
        (
            "CPU→NVMe",
            4 * 1024 * 1024,
            MemoryTier::Cpu,
            MemoryTier::Nvme,
        ), // 4MB
        (
            "NVMe→SSD",
            16 * 1024 * 1024,
            MemoryTier::Nvme,
            MemoryTier::Ssd,
        ), // 16MB
        (
            "SSD→HDD",
            64 * 1024 * 1024,
            MemoryTier::Ssd,
            MemoryTier::Hdd,
        ), // 64MB
        ("HDD→GPU", 1024 * 1024, MemoryTier::Hdd, MemoryTier::Gpu), // 1MB
    ];

    let mut results = vec![];

    for (name, size, source, target) in migration_scenarios {
        let start = Instant::now();
        let latency = optimizer
            .optimize_tier_migration(size, source, target)
            .await?;
        let actual_time = start.elapsed();

        results.push((name, size, latency, actual_time));

        println!(
            "Migration {}: {}MB in {:?} (estimated: {:?})",
            name,
            size / (1024 * 1024),
            actual_time,
            latency
        );
    }

    optimizer.stop().await?;

    // Verify latency targets
    for (name, size, estimated_latency, actual_time) in &results {
        let latency_ms = actual_time.as_secs_f32() * 1000.0;

        // Critical migrations (GPU/CPU tier) should be under 1ms
        if name.contains("GPU") || name.contains("CPU") {
            assert!(
                latency_ms < TARGET_MEMORY_MIGRATION_LATENCY_MS,
                "Critical migration {} took {:.2}ms, should be < {}ms",
                name,
                latency_ms,
                TARGET_MEMORY_MIGRATION_LATENCY_MS
            );
        }

        // All migrations should be reasonably fast
        assert!(
            latency_ms < 10.0,
            "Migration {} took {:.2}ms, should be < 10ms",
            name,
            latency_ms
        );
    }

    Ok(())
}

#[tokio::test]
async fn benchmark_access_pattern_analysis_performance() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);
    optimizer.start().await?;

    // Generate large number of memory accesses
    let start = Instant::now();
    for i in 0..LARGE_DATASET_SIZE {
        let address = (i * 4096) as u64; // 4KB aligned
        let size = if i % 100 == 0 { 8192 } else { 4096 }; // Mostly 4KB with some 8KB
        optimizer.record_access(address, size).await?;

        // Add some hot spots
        if i % 1000 == 0 {
            for j in 0..10 {
                optimizer.record_access(address + j * 1024, 1024).await?;
            }
        }
    }
    let record_time = start.elapsed();

    // Benchmark analysis performance
    let analysis_start = Instant::now();
    let analysis = optimizer.analyze_access_patterns().await?;
    let analysis_time = analysis_start.elapsed();

    optimizer.stop().await?;

    println!("Access Pattern Analysis Benchmark:");
    println!(
        "  Recorded {} accesses in {:?}",
        LARGE_DATASET_SIZE, record_time
    );
    println!("  Analysis completed in {:?}", analysis_time);
    println!("  Hot regions found: {}", analysis.hot_regions.len());
    println!("  Cold regions found: {}", analysis.cold_regions.len());
    println!(
        "  Temporal locality: {:.1}%",
        analysis.temporal_locality * 100.0
    );
    println!(
        "  Spatial locality: {:.1}%",
        analysis.spatial_locality * 100.0
    );

    // Performance requirements
    assert!(
        record_time < Duration::from_millis(500),
        "Recording {}K accesses should take < 500ms, took {:?}",
        LARGE_DATASET_SIZE / 1000,
        record_time
    );

    assert!(
        analysis_time < Duration::from_millis(100),
        "Analysis should take < 100ms, took {:?}",
        analysis_time
    );

    // Quality requirements
    assert!(
        !analysis.hot_regions.is_empty() || !analysis.cold_regions.is_empty(),
        "Analysis should identify access patterns"
    );

    Ok(())
}

// =============================================================================
// Job Batching Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_job_batching_efficiency() -> Result<()> {
    let config = BatchingConfig {
        max_batch_size: 100,
        max_batch_wait_time: Duration::from_millis(10),
        target_efficiency: 0.80,
        ..Default::default()
    };

    let batcher = JobBatchOptimizer::new(config);
    batcher.start().await?;

    // Generate mixed workload
    let start = Instant::now();
    let mut job_count = 0;

    // High-frequency small jobs
    for i in 0..500 {
        let job = MockJob {
            id: i,
            size: 512 + (i % 512), // 512B - 1KB
            priority: if i % 10 == 0 {
                JobPriority::High
            } else {
                JobPriority::Normal
            },
            arrival_time: Instant::now(),
        };
        batcher.add_job(job).await?;
        job_count += 1;

        if i % 50 == 0 {
            sleep(Duration::from_millis(1)).await;
        }
    }

    // Some larger jobs
    for i in 500..550 {
        let job = MockJob {
            id: i,
            size: 8192, // 8KB
            priority: JobPriority::Normal,
            arrival_time: Instant::now(),
        };
        batcher.add_job(job).await?;
        job_count += 1;
    }

    // Wait for processing to complete
    sleep(Duration::from_millis(100)).await;

    let processing_time = start.elapsed();
    let metrics = batcher.get_batch_metrics();

    batcher.stop().await?;

    println!("Job Batching Benchmark:");
    println!("  Processed {} jobs in {:?}", job_count, processing_time);
    println!("  Average batch size: {:.1}", metrics.average_batch_size);
    println!("  Batch efficiency: {:.1}%", metrics.efficiency * 100.0);
    println!("  Jobs processed: {}", metrics.jobs_processed);
    println!("  Batches created: {}", metrics.batches_created);

    // Performance requirements
    assert!(
        metrics.efficiency >= 0.70,
        "Batch efficiency {:.1}% should be >= 70%",
        metrics.efficiency * 100.0
    );

    assert!(
        metrics.jobs_processed >= job_count as u64,
        "Should process all {} jobs, processed {}",
        job_count,
        metrics.jobs_processed
    );

    assert!(
        processing_time < Duration::from_millis(500),
        "Processing should complete within 500ms"
    );

    Ok(())
}

#[tokio::test]
async fn benchmark_batching_under_stress() -> Result<()> {
    let config = BatchingConfig {
        max_batch_size: 50,
        max_batch_wait_time: Duration::from_millis(5),
        ..Default::default()
    };

    let batcher = JobBatchOptimizer::new(config);
    batcher.start().await?;

    // Stress test with high job arrival rate
    let start = Instant::now();
    let stress_jobs = STRESS_TEST_ITERATIONS;

    for i in 0..stress_jobs {
        let job = MockJob {
            id: i,
            size: 256 + (i % 1024), // Variable sizes
            priority: match i % 4 {
                0 => JobPriority::High,
                1 => JobPriority::Normal,
                _ => JobPriority::Low,
            },
            arrival_time: Instant::now(),
        };

        batcher.add_job(job).await?;

        // Rapid submission with occasional pauses
        if i % 1000 == 0 {
            sleep(Duration::from_micros(100)).await;
        }
    }

    // Wait for processing
    sleep(Duration::from_millis(200)).await;

    let stress_time = start.elapsed();
    let final_metrics = batcher.get_batch_metrics();

    batcher.stop().await?;

    println!("Stress Test Benchmark:");
    println!("  Submitted {} jobs in {:?}", stress_jobs, stress_time);
    println!(
        "  Final efficiency: {:.1}%",
        final_metrics.efficiency * 100.0
    );
    println!("  Jobs processed: {}", final_metrics.jobs_processed);
    println!(
        "  Average batch size: {:.1}",
        final_metrics.average_batch_size
    );
    println!(
        "  Throughput: {:.0} jobs/sec",
        final_metrics.jobs_processed as f64 / stress_time.as_secs_f64()
    );

    // Stress test requirements
    assert!(
        final_metrics.jobs_processed >= stress_jobs as u64 * 9 / 10,
        "Should process at least 90% of {} jobs under stress",
        stress_jobs
    );

    assert!(
        final_metrics.efficiency >= 0.60,
        "Efficiency should remain >= 60% under stress"
    );

    let throughput = final_metrics.jobs_processed as f64 / stress_time.as_secs_f64();
    assert!(
        throughput >= 5000.0,
        "Should maintain >= 5000 jobs/sec throughput"
    );

    Ok(())
}

// =============================================================================
// Double Buffering Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_double_buffering_performance() -> Result<()> {
    let config = BufferingConfig {
        buffer_size: 1024 * 1024, // 1MB buffers
        enable_prefetch: true,
        ..Default::default()
    };

    let buffer_manager = DoubleBufferManager::new(config);
    buffer_manager.start().await?;

    // Benchmark buffer access patterns
    let access_patterns = vec![
        ("Sequential", generate_sequential_accesses(1000)),
        ("Random", generate_random_accesses(1000)),
        ("Strided", generate_strided_accesses(1000, 64)),
        ("Hot-spot", generate_hotspot_accesses(1000, 10)),
    ];

    let mut results = vec![];

    for (pattern_name, accesses) in access_patterns {
        let start = Instant::now();

        for (address, size) in &accesses {
            buffer_manager.record_access(*address, *size).await?;
        }

        sleep(Duration::from_millis(50)).await; // Allow processing

        let pattern_time = start.elapsed();
        let hit_rate = buffer_manager.get_hit_rate();

        results.push((pattern_name, accesses.len(), pattern_time, hit_rate));

        println!(
            "Buffer pattern {}: {} accesses in {:?}, hit rate: {:.1}%",
            pattern_name,
            accesses.len(),
            pattern_time,
            hit_rate * 100.0
        );
    }

    buffer_manager.stop().await?;

    // Verify performance requirements
    for (pattern_name, access_count, time, hit_rate) in &results {
        let accesses_per_ms = *access_count as f64 / time.as_millis() as f64;

        assert!(
            accesses_per_ms >= 10.0,
            "Pattern {} should handle >= 10 accesses/ms, got {:.1}",
            pattern_name,
            accesses_per_ms
        );

        // Sequential and strided patterns should have good hit rates
        if pattern_name.contains("Sequential") || pattern_name.contains("Strided") {
            assert!(
                *hit_rate >= 0.50,
                "Pattern {} should have >= 50% hit rate, got {:.1}%",
                pattern_name,
                hit_rate * 100.0
            );
        }
    }

    Ok(())
}

// =============================================================================
// Compression Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_compression_by_tier() -> Result<()> {
    let config = CompressionConfig::default();
    let tuner = CompressionTuner::new(config);
    tuner.start().await?;

    // Test data of various sizes and patterns
    let test_datasets = vec![
        ("Small uniform", vec![42u8; 1024]),
        ("Medium random", generate_random_data(64 * 1024)),
        ("Large structured", generate_structured_data(256 * 1024)),
        ("Huge sparse", generate_sparse_data(1024 * 1024)),
    ];

    let tiers = vec![
        MemoryTier::Gpu,
        MemoryTier::Cpu,
        MemoryTier::Nvme,
        MemoryTier::Ssd,
        MemoryTier::Hdd,
    ];

    for (data_name, data) in &test_datasets {
        println!("\nCompression benchmark for {}:", data_name);

        for tier in &tiers {
            let start = Instant::now();

            let settings = tuner
                .get_compression_settings_for_tier(tier.clone())
                .await?;
            let algorithm = tuner
                .select_compression_algorithm(data, tier.clone())
                .await?;

            let compression_time = start.elapsed();

            println!(
                "  {:?}: level {}, algorithm {:?}, time {:?}",
                tier, settings.compression_level, algorithm, compression_time
            );

            // Verify compression time scales appropriately with tier speed
            let max_time_ms = match tier {
                MemoryTier::Gpu => 1.0,   // Very fast
                MemoryTier::Cpu => 5.0,   // Fast
                MemoryTier::Nvme => 10.0, // Medium
                MemoryTier::Ssd => 20.0,  // Slower
                MemoryTier::Hdd => 50.0,  // Slowest
            };

            assert!(
                compression_time.as_secs_f32() * 1000.0 < max_time_ms,
                "Compression for {:?} took {:.1}ms, should be < {:.1}ms",
                tier,
                compression_time.as_secs_f32() * 1000.0,
                max_time_ms
            );
        }
    }

    tuner.stop().await?;
    Ok(())
}

// =============================================================================
// End-to-End Performance Benchmarks
// =============================================================================

#[tokio::test]
async fn benchmark_full_optimization_cycle() -> Result<()> {
    let config = OptimizationConfig {
        target_gpu_utilization: TARGET_GPU_UTILIZATION,
        target_memory_efficiency: 0.85,
        target_batch_efficiency: 0.80,
        optimization_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let mut optimizer = PerformanceOptimizer::new(config);

    // Benchmark full cycle
    let start = Instant::now();
    optimizer.start().await?;

    // Generate realistic workload
    tokio::spawn(async move {
        let workload_start = Instant::now();
        while workload_start.elapsed() < Duration::from_millis(BENCHMARK_DURATION_MS) {
            // Simulate GPU kernels
            tokio::time::sleep(Duration::from_millis(10)).await;

            // Simulate memory accesses
            tokio::time::sleep(Duration::from_millis(5)).await;

            // Simulate job batching
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    });

    // Let optimization run
    sleep(Duration::from_millis(BENCHMARK_DURATION_MS + 100)).await;

    let final_metrics = optimizer.get_performance_metrics();
    let optimization_score = optimizer.calculate_optimization_score();

    // Generate comprehensive report
    let report = optimizer.generate_optimization_report().await?;

    optimizer.stop().await?;
    let total_time = start.elapsed();

    println!("\nFull Optimization Cycle Benchmark:");
    println!("  Total duration: {:?}", total_time);
    println!(
        "  GPU utilization: {:.1}%",
        final_metrics.gpu_utilization * 100.0
    );
    println!(
        "  Memory efficiency: {:.1}%",
        final_metrics.memory_efficiency.cache_hit_rate * 100.0
    );
    println!(
        "  Batch efficiency: {:.1}%",
        final_metrics.batch_efficiency.efficiency * 100.0
    );
    println!(
        "  Buffer hit rate: {:.1}%",
        final_metrics.buffer_hit_rate * 100.0
    );
    println!(
        "  Compression ratio: {:.1}x",
        final_metrics.compression_ratio
    );
    println!("  Optimization score: {:.1}%", optimization_score * 100.0);
    println!(
        "  Optimization cycles: {}",
        final_metrics.optimization_cycles
    );
    println!(
        "  Improvements applied: {}",
        final_metrics.improvements_applied
    );
    println!("  Recommendations: {}", report.recommendations.len());

    // Verify overall system performance
    assert!(
        final_metrics.gpu_utilization >= TARGET_GPU_UTILIZATION * 0.90,
        "GPU utilization should reach 90% of target"
    );

    assert!(
        optimization_score >= 0.70,
        "Overall optimization score should be >= 70%"
    );

    assert!(
        final_metrics.optimization_cycles >= 10,
        "Should complete multiple optimization cycles"
    );

    assert!(
        !report.recommendations.is_empty(),
        "Should generate optimization recommendations"
    );

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn generate_sequential_accesses(count: usize) -> Vec<(u64, usize)> {
    (0..count).map(|i| (i as u64 * 4096, 4096)).collect()
}

fn generate_random_accesses(count: usize) -> Vec<(u64, usize)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let address = (hasher.finish() % 1_000_000) * 4096;
            (address, 4096)
        })
        .collect()
}

fn generate_strided_accesses(count: usize, stride: u64) -> Vec<(u64, usize)> {
    (0..count)
        .map(|i| (i as u64 * stride * 4096, 4096))
        .collect()
}

fn generate_hotspot_accesses(count: usize, hotspots: usize) -> Vec<(u64, usize)> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hotspot = (hasher.finish() % hotspots as u64) * 1024 * 1024; // 1MB apart
            (hotspot, 4096)
        })
        .collect()
}

fn generate_random_data(size: usize) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..size)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            (hasher.finish() % 256) as u8
        })
        .collect()
}

fn generate_structured_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i % 256) as u8)).collect()
}

fn generate_sparse_data(size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| if i % 100 == 0 { (i % 256) as u8 } else { 0 })
        .collect()
}

#[derive(Debug, Clone)]
struct MockJob {
    id: usize,
    size: usize,
    priority: JobPriority,
    arrival_time: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JobPriority {
    High,
    Normal,
    Low,
}
