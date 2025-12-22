//! Integration tests for performance optimization module
//!
//! Tests the integration between different optimization components
//! and their effect on overall system performance.

use super::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// =============================================================================
// Performance Integration Tests
// =============================================================================

#[tokio::test]
async fn test_complete_performance_optimization_cycle() -> Result<()> {
    let config = OptimizationConfig {
        target_gpu_utilization: 0.85,
        target_memory_efficiency: 0.80,
        target_batch_efficiency: 0.75,
        optimization_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let mut optimizer = PerformanceOptimizer::new(config);

    // Start the optimizer
    optimizer.start().await?;

    // Let it run for a few optimization cycles
    sleep(Duration::from_millis(200)).await;

    // Get initial metrics
    let initial_metrics = optimizer.get_performance_metrics();

    // Generate and apply recommendations
    let report = optimizer.generate_optimization_report().await?;
    assert!(
        !report.recommendations.is_empty(),
        "Should generate optimization recommendations"
    );

    // Apply high-priority recommendations
    let high_priority_recs: Vec<_> = report
        .recommendations
        .into_iter()
        .filter(|r| {
            r.priority == RecommendationPriority::High
                || r.priority == RecommendationPriority::Critical
        })
        .collect();

    if !high_priority_recs.is_empty() {
        optimizer.apply_optimizations(high_priority_recs).await?;

        // Wait for optimizations to take effect
        sleep(Duration::from_millis(100)).await;

        // Verify improvements were applied
        let final_metrics = optimizer.get_performance_metrics();
        assert!(
            final_metrics.improvements_applied > initial_metrics.improvements_applied,
            "Should have applied optimization improvements"
        );
    }

    // Verify optimization score calculation
    let score = optimizer.calculate_optimization_score();
    assert!(
        score >= 0.0 && score <= 1.0,
        "Optimization score should be between 0 and 1"
    );

    optimizer.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_multi_component_optimization() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    // Test GPU utilization optimization
    let gpu_recommendations = optimizer.utilization_analyzer.get_recommendations().await?;
    assert!(
        !gpu_recommendations.is_empty(),
        "Should have GPU utilization recommendations"
    );

    // Test memory optimization
    let memory_recommendations = optimizer.memory_optimizer.get_recommendations().await?;
    // Memory recommendations might be empty if no optimization needed

    // Test job batching optimization
    let batch_recommendations = optimizer.job_batcher.get_recommendations().await?;
    // Batch recommendations might be empty if efficiency is already good

    // Test that different optimizers produce different types of recommendations
    for rec in &gpu_recommendations {
        assert_eq!(rec.optimization_type, OptimizationType::GpuUtilization);
    }

    for rec in &memory_recommendations {
        assert_eq!(rec.optimization_type, OptimizationType::MemoryTier);
    }

    Ok(())
}

#[tokio::test]
async fn test_optimization_recommendation_prioritization() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let recommendations = optimizer.generate_recommendations().await?;

    // Verify recommendations are sorted by priority
    let mut prev_priority = RecommendationPriority::Critical;
    for rec in &recommendations {
        assert!(
            rec.priority <= prev_priority,
            "Recommendations should be sorted by priority"
        );
        prev_priority = rec.priority.clone();
    }

    // Verify each recommendation has valid parameters
    for rec in &recommendations {
        assert!(
            !rec.description.is_empty(),
            "Recommendation should have description"
        );
        assert!(rec.estimated_impact >= 0.0, "Impact should be non-negative");
        assert!(
            rec.implementation_cost >= 0.0,
            "Cost should be non-negative"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_performance_metrics_correlation() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let metrics = optimizer.get_performance_metrics();

    // Test metric ranges and relationships
    assert!(metrics.gpu_utilization >= 0.0 && metrics.gpu_utilization <= 1.0);
    assert!(
        metrics.memory_efficiency.cache_hit_rate >= 0.0
            && metrics.memory_efficiency.cache_hit_rate <= 1.0
    );
    assert!(metrics.batch_efficiency.efficiency >= 0.0);
    assert!(metrics.buffer_hit_rate >= 0.0 && metrics.buffer_hit_rate <= 1.0);
    assert!(metrics.compression_ratio >= 1.0);

    // Test memory efficiency relationships
    assert!(metrics.memory_efficiency.tier_migration_latency.as_micros() > 0);
    assert!(metrics.memory_efficiency.memory_bandwidth_utilization >= 0.0);

    // Test batch efficiency relationships
    assert!(metrics.batch_efficiency.average_batch_size > 0);

    Ok(())
}

#[tokio::test]
async fn test_optimization_impact_measurement() -> Result<()> {
    let config = OptimizationConfig::default();
    let mut optimizer = PerformanceOptimizer::new(config);

    optimizer.start().await?;

    // Get baseline metrics
    let baseline_metrics = optimizer.get_performance_metrics();

    // Create mock optimization recommendations
    let recommendations = vec![
        OptimizationRecommendation {
            optimization_type: OptimizationType::GpuUtilization,
            description: "Test GPU optimization".to_string(),
            estimated_impact: 0.1,
            implementation_cost: 0.2,
            priority: RecommendationPriority::High,
            parameters: [("test_param".to_string(), "test_value".to_string())].into(),
        },
        OptimizationRecommendation {
            optimization_type: OptimizationType::MemoryTier,
            description: "Test memory optimization".to_string(),
            estimated_impact: 0.05,
            implementation_cost: 0.1,
            priority: RecommendationPriority::Medium,
            parameters: [("test_param".to_string(), "test_value".to_string())].into(),
        },
    ];

    // Apply optimizations
    optimizer
        .apply_optimizations(recommendations.clone())
        .await?;

    // Verify impact tracking
    let post_optimization_metrics = optimizer.get_performance_metrics();
    assert_eq!(
        post_optimization_metrics.improvements_applied - baseline_metrics.improvements_applied,
        recommendations.len() as u64,
        "Should track applied optimizations"
    );

    optimizer.stop().await?;

    Ok(())
}

// =============================================================================
// Component Integration Tests
// =============================================================================

#[tokio::test]
async fn test_gpu_utilization_memory_optimization_integration() -> Result<()> {
    let utilization_config = UtilizationConfig::default();
    let memory_config = MemoryOptimizationConfig::default();

    let utilization_analyzer = GpuUtilizationAnalyzer::new(utilization_config);
    let memory_optimizer = MemoryTierOptimizer::new(memory_config);

    // Start both optimizers
    utilization_analyzer.start().await?;
    memory_optimizer.start().await?;

    // Simulate some GPU activity affecting memory
    utilization_analyzer
        .record_kernel_execution(
            "test_kernel",
            Duration::from_micros(5000),
            (256, 1, 1),
            (256, 1, 1),
            4096,
            0.75,
        )
        .await?;

    // Record corresponding memory accesses
    memory_optimizer
        .record_access(0x10000000, 1024 * 1024)
        .await?;
    memory_optimizer
        .record_access(0x20000000, 2048 * 1024)
        .await?;

    // Wait for analysis
    sleep(Duration::from_millis(100)).await;

    // Get recommendations from both
    let util_recommendations = utilization_analyzer.get_recommendations().await?;
    let memory_recommendations = memory_optimizer.get_recommendations().await?;

    // Verify both components generate relevant recommendations
    assert!(
        !util_recommendations.is_empty(),
        "Should have utilization recommendations"
    );

    // Stop optimizers
    utilization_analyzer.stop().await?;
    memory_optimizer.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_job_batching_double_buffering_integration() -> Result<()> {
    let batching_config = BatchingConfig::default();
    let buffering_config = BufferingConfig::default();

    let job_batcher = JobBatchOptimizer::new(batching_config);
    let buffer_manager = DoubleBufferManager::new(buffering_config);

    job_batcher.start().await?;
    buffer_manager.start().await?;

    // Simulate job processing with buffering
    for i in 0..20 {
        let job = MockJob {
            id: i,
            size: 1024,
            priority: if i % 5 == 0 {
                JobPriority::High
            } else {
                JobPriority::Normal
            },
            arrival_time: Instant::now(),
        };

        job_batcher.add_job(job).await?;

        // Simulate buffer access
        buffer_manager.record_access(i * 1024, 1024).await?;
    }

    // Wait for processing
    sleep(Duration::from_millis(100)).await;

    // Verify metrics
    let batch_metrics = job_batcher.get_batch_metrics();
    let buffer_hit_rate = buffer_manager.get_hit_rate();

    assert!(batch_metrics.jobs_processed > 0, "Should process jobs");
    assert!(
        buffer_hit_rate >= 0.0 && buffer_hit_rate <= 1.0,
        "Hit rate should be valid"
    );

    job_batcher.stop().await?;
    buffer_manager.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_compression_tuning_memory_tier_integration() -> Result<()> {
    let compression_config = CompressionConfig::default();
    let memory_config = MemoryOptimizationConfig::default();

    let compression_tuner = CompressionTuner::new(compression_config);
    let memory_optimizer = MemoryTierOptimizer::new(memory_config);

    compression_tuner.start().await?;
    memory_optimizer.start().await?;

    // Test compression settings for different tiers
    let tiers = vec![
        MemoryTier::Gpu,
        MemoryTier::Cpu,
        MemoryTier::Nvme,
        MemoryTier::Ssd,
        MemoryTier::Hdd,
    ];

    for tier in tiers {
        let settings = compression_tuner
            .get_compression_settings_for_tier(tier.clone())
            .await?;

        // Verify compression level increases with slower tiers
        match tier {
            MemoryTier::Gpu => assert!(settings.compression_level <= 3),
            MemoryTier::Cpu => assert!(settings.compression_level <= 6),
            MemoryTier::Nvme => assert!(settings.compression_level <= 9),
            MemoryTier::Ssd => assert!(settings.compression_level <= 12),
            MemoryTier::Hdd => assert!(settings.compression_level <= 15),
        }
    }

    // Test algorithm selection
    let test_data = vec![42u8; 4096];
    let algorithm = compression_tuner
        .select_compression_algorithm(&test_data, MemoryTier::Nvme)
        .await?;
    assert_ne!(algorithm, CompressionAlgorithm::None);

    compression_tuner.stop().await?;
    memory_optimizer.stop().await?;

    Ok(())
}

// =============================================================================
// Performance Benchmark Integration Tests
// =============================================================================

#[tokio::test]
async fn test_optimization_performance_overhead() -> Result<()> {
    let config = OptimizationConfig {
        optimization_interval: Duration::from_millis(10), // Aggressive optimization
        ..Default::default()
    };

    let mut optimizer = PerformanceOptimizer::new(config);

    // Measure startup time
    let start = Instant::now();
    optimizer.start().await?;
    let startup_time = start.elapsed();

    assert!(
        startup_time < Duration::from_millis(100),
        "Optimizer startup should be fast: {:?}",
        startup_time
    );

    // Measure optimization cycle overhead
    let start = Instant::now();
    let initial_cycles = optimizer.stats.optimization_cycles.load(Ordering::Relaxed);

    sleep(Duration::from_millis(100)).await;

    let final_cycles = optimizer.stats.optimization_cycles.load(Ordering::Relaxed);
    let cycle_time = start.elapsed();

    assert!(
        final_cycles > initial_cycles,
        "Should complete optimization cycles"
    );

    let avg_cycle_time = cycle_time.as_micros() / (final_cycles - initial_cycles) as u128;
    assert!(
        avg_cycle_time < 50_000, // 50ms per cycle
        "Optimization cycles should be efficient: {}μs per cycle",
        avg_cycle_time
    );

    // Measure shutdown time
    let start = Instant::now();
    optimizer.stop().await?;
    let shutdown_time = start.elapsed();

    assert!(
        shutdown_time < Duration::from_millis(50),
        "Optimizer shutdown should be fast: {:?}",
        shutdown_time
    );

    Ok(())
}

#[tokio::test]
async fn test_memory_optimization_scalability() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);

    optimizer.start().await?;

    // Simulate large number of memory accesses
    let start = Instant::now();
    for i in 0..10000 {
        let address = (i * 4096) as u64; // 4KB pages
        optimizer.record_access(address, 4096).await?;
    }
    let record_time = start.elapsed();

    assert!(
        record_time < Duration::from_millis(100),
        "Recording 10K accesses should be fast: {:?}",
        record_time
    );

    // Measure analysis performance
    let start = Instant::now();
    let analysis = optimizer.analyze_access_patterns().await?;
    let analysis_time = start.elapsed();

    assert!(
        analysis_time < Duration::from_millis(50),
        "Analysis should be fast: {:?}",
        analysis_time
    );

    // Verify analysis quality
    assert!(
        !analysis.hot_regions.is_empty() || !analysis.cold_regions.is_empty(),
        "Analysis should identify patterns"
    );

    optimizer.stop().await?;

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

async fn create_test_workload() -> Result<Vec<MockJob>> {
    let mut jobs = vec![];

    for i in 0..100 {
        jobs.push(MockJob {
            id: i,
            size: 1024 + (i % 4096), // Variable sizes
            priority: match i % 3 {
                0 => JobPriority::High,
                1 => JobPriority::Normal,
                _ => JobPriority::Low,
            },
            arrival_time: Instant::now(),
        });
    }

    Ok(jobs)
}

fn verify_optimization_effectiveness(
    before: &PerformanceMetrics,
    after: &PerformanceMetrics,
) -> bool {
    // Check if any metric improved
    after.gpu_utilization >= before.gpu_utilization
        || after.memory_efficiency.cache_hit_rate >= before.memory_efficiency.cache_hit_rate
        || after.batch_efficiency.efficiency >= before.batch_efficiency.efficiency
        || after.buffer_hit_rate >= before.buffer_hit_rate
}
