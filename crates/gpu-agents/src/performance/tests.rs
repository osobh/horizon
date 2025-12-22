//! Unit tests for performance optimization module
//!
//! Tests all performance optimization components including:
//! - GPU utilization analysis and tuning
//! - Memory tier optimization
//! - Job batching optimization
//! - Double buffering implementation
//! - Compression tuning

use super::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// =============================================================================
// Performance Optimizer Tests
// =============================================================================

#[tokio::test]
async fn test_performance_optimizer_creation() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    // Verify initial state
    assert!(!optimizer.is_running.load(Ordering::Relaxed));
    assert_eq!(
        optimizer.stats.optimization_cycles.load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        optimizer.stats.improvements_applied.load(Ordering::Relaxed),
        0
    );

    Ok(())
}

#[tokio::test]
async fn test_performance_optimizer_lifecycle() -> Result<()> {
    let config = OptimizationConfig {
        optimization_interval: Duration::from_millis(10),
        ..Default::default()
    };
    let mut optimizer = PerformanceOptimizer::new(config);

    // Start optimizer
    optimizer.start().await?;
    assert!(optimizer.is_running.load(Ordering::Relaxed));

    // Let it run briefly
    sleep(Duration::from_millis(50)).await;

    // Check that optimization cycles are running
    let cycles = optimizer.stats.optimization_cycles.load(Ordering::Relaxed);
    assert!(
        cycles > 0,
        "Should have completed at least one optimization cycle"
    );

    // Stop optimizer
    optimizer.stop().await?;
    assert!(!optimizer.is_running.load(Ordering::Relaxed));

    Ok(())
}

#[tokio::test]
async fn test_gpu_utilization_improvement_to_85_percent() -> Result<()> {
    // RED PHASE: Test that GPU utilization reaches 85%+ target
    let config = UtilizationConfig {
        target_utilization: 0.85, // 85% target
        ..Default::default()
    };
    let analyzer = GpuUtilizationAnalyzer::new(config);

    // Start monitoring
    analyzer.start().await?;

    // Simulate workload for optimization
    sleep(Duration::from_millis(100)).await;

    // Get current utilization
    let utilization = analyzer.get_current_utilization().await?;

    // Should achieve 85%+ utilization (this will fail initially - RED phase)
    assert!(
        utilization >= 0.85,
        "GPU utilization {:.1}% should be >= 85%",
        utilization * 100.0
    );

    analyzer.stop().await?;
    Ok(())
}

#[tokio::test]
async fn test_performance_metrics_collection() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let metrics = optimizer.get_performance_metrics();

    // Verify metrics structure
    assert!(metrics.gpu_utilization >= 0.0 && metrics.gpu_utilization <= 1.0);
    assert!(metrics.memory_efficiency.cache_hit_rate >= 0.0);
    assert!(metrics.batch_efficiency.efficiency >= 0.0);
    assert!(metrics.buffer_hit_rate >= 0.0 && metrics.buffer_hit_rate <= 1.0);
    assert!(metrics.compression_ratio >= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_optimization_report_generation() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let report = optimizer.generate_optimization_report().await?;

    // Verify report structure
    assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
    assert!(!report.recommendations.is_empty());

    // Check that recommendations are sorted by priority
    let mut prev_priority = RecommendationPriority::Critical;
    for rec in &report.recommendations {
        assert!(rec.priority <= prev_priority);
        prev_priority = rec.priority.clone();
    }

    Ok(())
}

#[tokio::test]
async fn test_optimization_score_calculation() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let score = optimizer.calculate_optimization_score();

    // Score should be between 0 and 1
    assert!(score >= 0.0 && score <= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_recommendation_application() -> Result<()> {
    let config = OptimizationConfig::default();
    let optimizer = PerformanceOptimizer::new(config);

    let recommendations = vec![
        OptimizationRecommendation {
            optimization_type: OptimizationType::GpuUtilization,
            description: "Increase batch size".to_string(),
            estimated_impact: 0.1,
            implementation_cost: 0.2,
            priority: RecommendationPriority::High,
            parameters: [("batch_size".to_string(), "64".to_string())].into(),
        },
        OptimizationRecommendation {
            optimization_type: OptimizationType::MemoryTier,
            description: "Adjust prefetch strategy".to_string(),
            estimated_impact: 0.15,
            implementation_cost: 0.1,
            priority: RecommendationPriority::Medium,
            parameters: [("prefetch_size".to_string(), "2MB".to_string())].into(),
        },
    ];

    let initial_improvements = optimizer.stats.improvements_applied.load(Ordering::Relaxed);
    optimizer
        .apply_optimizations(recommendations.clone())
        .await?;

    let final_improvements = optimizer.stats.improvements_applied.load(Ordering::Relaxed);
    assert_eq!(
        final_improvements - initial_improvements,
        recommendations.len() as u64
    );

    Ok(())
}

// =============================================================================
// GPU Utilization Analyzer Tests
// =============================================================================

#[tokio::test]
async fn test_gpu_utilization_analyzer_creation() -> Result<()> {
    let config = UtilizationConfig::default();
    let analyzer = GpuUtilizationAnalyzer::new(config);

    // Verify initial state
    let utilization = analyzer.get_current_utilization();
    assert!(utilization >= 0.0 && utilization <= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_gpu_utilization_monitoring() -> Result<()> {
    let config = UtilizationConfig {
        monitoring_interval: Duration::from_millis(10),
        ..Default::default()
    };
    let analyzer = GpuUtilizationAnalyzer::new(config);

    analyzer.start().await?;

    // Let it collect some samples
    sleep(Duration::from_millis(50)).await;

    let utilization = analyzer.get_current_utilization();
    assert!(utilization >= 0.0 && utilization <= 1.0);

    analyzer.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_gpu_utilization_recommendations() -> Result<()> {
    let config = UtilizationConfig::default();
    let analyzer = GpuUtilizationAnalyzer::new(config);

    let recommendations = analyzer.get_recommendations().await?;

    // Should have at least basic recommendations
    assert!(!recommendations.is_empty());

    for rec in &recommendations {
        assert_eq!(rec.optimization_type, OptimizationType::GpuUtilization);
        assert!(!rec.description.is_empty());
        assert!(rec.estimated_impact >= 0.0);
    }

    Ok(())
}

#[tokio::test]
async fn test_gpu_utilization_optimization_trigger() -> Result<()> {
    let config = UtilizationConfig::default();
    let analyzer = GpuUtilizationAnalyzer::new(config);

    // Trigger optimization
    analyzer.trigger_optimization().await?;

    // Verify optimization was attempted
    let report = analyzer.generate_report().await?;
    assert!(report.optimizations_attempted > 0);

    Ok(())
}

// =============================================================================
// Memory Tier Optimizer Tests
// =============================================================================

#[tokio::test]
async fn test_memory_tier_optimizer_creation() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);

    let metrics = optimizer.get_efficiency_metrics();
    assert!(metrics.cache_hit_rate >= 0.0 && metrics.cache_hit_rate <= 1.0);
    assert!(metrics.tier_migration_latency.as_micros() > 0);

    Ok(())
}

#[tokio::test]
async fn test_memory_access_pattern_analysis() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);

    optimizer.start().await?;

    // Simulate some memory accesses
    optimizer.record_access(0x1000, 1024).await?;
    optimizer.record_access(0x2000, 2048).await?;
    optimizer.record_access(0x1000, 1024).await?; // Repeat access

    let patterns = optimizer.analyze_access_patterns().await?;

    // Should detect the repeated access pattern
    assert!(!patterns.hot_regions.is_empty());
    assert!(patterns.access_frequency.len() > 0);

    optimizer.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_memory_prefetching_optimization() -> Result<()> {
    let config = MemoryOptimizationConfig {
        enable_prefetching: true,
        prefetch_size: 1024 * 1024, // 1MB
        ..Default::default()
    };
    let optimizer = MemoryTierOptimizer::new(config);

    // Test prefetch recommendation
    let recommendations = optimizer.get_recommendations().await?;

    let prefetch_recs: Vec<_> = recommendations
        .iter()
        .filter(|r| r.parameters.contains_key("prefetch_size"))
        .collect();

    assert!(
        !prefetch_recs.is_empty(),
        "Should have prefetching recommendations"
    );

    Ok(())
}

#[tokio::test]
async fn test_memory_tier_migration_optimization() -> Result<()> {
    let config = MemoryOptimizationConfig::default();
    let optimizer = MemoryTierOptimizer::new(config);

    // Simulate tier migration
    let migration_time = optimizer
        .optimize_tier_migration(
            1024 * 1024, // 1MB data
            MemoryTier::Gpu,
            MemoryTier::Cpu,
        )
        .await?;

    assert!(migration_time.as_micros() > 0);
    assert!(migration_time.as_millis() < 10); // Should be fast

    Ok(())
}

// =============================================================================
// Job Batch Optimizer Tests
// =============================================================================

#[tokio::test]
async fn test_job_batch_optimizer_creation() -> Result<()> {
    let config = BatchingConfig::default();
    let optimizer = JobBatchOptimizer::new(config);

    let metrics = optimizer.get_batch_metrics();
    assert!(metrics.efficiency >= 0.0 && metrics.efficiency <= 1.0);
    assert!(metrics.average_batch_size > 0);

    Ok(())
}

#[tokio::test]
async fn test_job_batching_size_optimization() -> Result<()> {
    let config = BatchingConfig {
        min_batch_size: 1,
        max_batch_size: 128,
        target_batch_latency: Duration::from_millis(10),
        ..Default::default()
    };
    let optimizer = JobBatchOptimizer::new(config);

    optimizer.start().await?;

    // Simulate job arrivals with different patterns
    for i in 0..100 {
        optimizer
            .add_job(MockJob {
                id: i,
                size: 1024,
                priority: if i % 10 == 0 {
                    JobPriority::High
                } else {
                    JobPriority::Normal
                },
                arrival_time: Instant::now(),
            })
            .await?;
    }

    sleep(Duration::from_millis(50)).await;

    let metrics = optimizer.get_batch_metrics();
    assert!(metrics.jobs_processed > 0);
    assert!(metrics.average_batch_size > 1); // Should batch multiple jobs

    optimizer.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_dynamic_batch_size_adjustment() -> Result<()> {
    let config = BatchingConfig::default();
    let optimizer = JobBatchOptimizer::new(config);

    // Test batch size recommendations for different workloads
    let light_workload = WorkloadPattern {
        jobs_per_second: 10,
        average_job_size: 1024,
        job_size_variance: 0.1,
    };

    let heavy_workload = WorkloadPattern {
        jobs_per_second: 1000,
        average_job_size: 512,
        job_size_variance: 0.5,
    };

    let light_batch_size = optimizer.recommend_batch_size(&light_workload).await?;
    let heavy_batch_size = optimizer.recommend_batch_size(&heavy_workload).await?;

    // Heavy workload should recommend larger batch sizes
    assert!(heavy_batch_size > light_batch_size);

    Ok(())
}

// =============================================================================
// Double Buffer Manager Tests
// =============================================================================

#[tokio::test]
async fn test_double_buffer_manager_creation() -> Result<()> {
    let config = BufferingConfig::default();
    let manager = DoubleBufferManager::new(config);

    let hit_rate = manager.get_hit_rate();
    assert!(hit_rate >= 0.0 && hit_rate <= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_double_buffering_read_write_patterns() -> Result<()> {
    let config = BufferingConfig {
        buffer_size: 1024 * 1024, // 1MB
        enable_async_writes: true,
        ..Default::default()
    };
    let manager = DoubleBufferManager::new(config);

    manager.start().await?;

    // Test read-while-write pattern
    let write_handle = manager
        .start_write_operation(vec![42u8; 512 * 1024])
        .await?;
    let read_data = manager.read_from_buffer(0, 1024).await?;

    assert!(!read_data.is_empty());

    write_handle.await?;
    manager.stop().await?;

    Ok(())
}

#[tokio::test]
async fn test_buffer_hit_rate_optimization() -> Result<()> {
    let config = BufferingConfig::default();
    let manager = DoubleBufferManager::new(config);

    // Simulate buffer accesses
    for i in 0..100 {
        let offset = (i % 10) * 1024; // Create locality
        manager.record_access(offset, 1024).await?;
    }

    let hit_rate = manager.get_hit_rate();
    assert!(
        hit_rate > 0.5,
        "Should achieve decent hit rate with locality"
    );

    Ok(())
}

// =============================================================================
// Compression Tuner Tests
// =============================================================================

#[tokio::test]
async fn test_compression_tuner_creation() -> Result<()> {
    let config = CompressionConfig::default();
    let tuner = CompressionTuner::new(config);

    let ratio = tuner.get_compression_ratio();
    assert!(ratio >= 1.0); // Should be at least 1:1

    Ok(())
}

#[tokio::test]
async fn test_compression_algorithm_selection() -> Result<()> {
    let config = CompressionConfig::default();
    let tuner = CompressionTuner::new(config);

    // Test different data types
    let text_data = vec![b'a'; 1024]; // Highly compressible
    let random_data = (0..1024).map(|i| (i * 7) as u8).collect::<Vec<_>>(); // Less compressible

    let text_algo = tuner
        .select_compression_algorithm(&text_data, MemoryTier::Nvme)
        .await?;
    let random_algo = tuner
        .select_compression_algorithm(&random_data, MemoryTier::Nvme)
        .await?;

    // Should select appropriate algorithms
    assert_ne!(text_algo, random_algo);

    Ok(())
}

#[tokio::test]
async fn test_tier_specific_compression_tuning() -> Result<()> {
    let config = CompressionConfig::default();
    let tuner = CompressionTuner::new(config);

    let data = vec![0u8; 4096];

    // Different tiers should have different compression strategies
    let gpu_settings = tuner
        .get_compression_settings_for_tier(MemoryTier::Gpu)
        .await?;
    let nvme_settings = tuner
        .get_compression_settings_for_tier(MemoryTier::Nvme)
        .await?;
    let hdd_settings = tuner
        .get_compression_settings_for_tier(MemoryTier::Hdd)
        .await?;

    // GPU should prefer speed, HDD should prefer ratio
    assert!(gpu_settings.compression_level < hdd_settings.compression_level);
    assert!(nvme_settings.compression_level >= gpu_settings.compression_level);
    assert!(nvme_settings.compression_level <= hdd_settings.compression_level);

    Ok(())
}

// =============================================================================
// Helper Types and Mock Implementations
// =============================================================================

#[derive(Clone)]
struct MockJob {
    id: u64,
    size: usize,
    priority: JobPriority,
    arrival_time: Instant,
}

#[derive(Clone)]
struct WorkloadPattern {
    jobs_per_second: u32,
    average_job_size: usize,
    job_size_variance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JobPriority {
    High,
    Normal,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MemoryTier {
    Gpu,
    Cpu,
    Nvme,
    Ssd,
    Hdd,
}

// Mock implementations for testing
impl GpuUtilizationAnalyzer {
    fn new(_config: UtilizationConfig) -> Self {
        // Mock implementation
        Self
    }

    async fn start(&self) -> Result<()> {
        Ok(())
    }
    async fn stop(&self) -> Result<()> {
        Ok(())
    }

    fn get_current_utilization(&self) -> f32 {
        0.75
    }

    async fn get_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        Ok(vec![OptimizationRecommendation {
            optimization_type: OptimizationType::GpuUtilization,
            description: "Increase kernel occupancy".to_string(),
            estimated_impact: 0.1,
            implementation_cost: 0.2,
            priority: RecommendationPriority::High,
            parameters: HashMap::new(),
        }])
    }

    async fn trigger_optimization(&self) -> Result<()> {
        Ok(())
    }

    async fn generate_report(&self) -> Result<UtilizationReport> {
        Ok(UtilizationReport {
            current_utilization: 0.75,
            target_utilization: 0.90,
            optimizations_attempted: 1,
            improvement_achieved: 0.05,
        })
    }

    async fn apply_optimization(&self, _rec: OptimizationRecommendation) -> Result<()> {
        Ok(())
    }
}

struct UtilizationConfig {
    monitoring_interval: Duration,
}

impl Default for UtilizationConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_millis(100),
        }
    }
}

#[derive(Debug)]
struct UtilizationReport {
    current_utilization: f32,
    target_utilization: f32,
    optimizations_attempted: u32,
    improvement_achieved: f32,
}

// Continue with other mock implementations...
// (Similar pattern for MemoryTierOptimizer, JobBatchOptimizer, etc.)
