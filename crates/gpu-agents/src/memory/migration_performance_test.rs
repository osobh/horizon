//! Memory Migration Performance Tests
//! Target: <10ms migration time for real-world scenarios

use super::migration::*;
use super::{MigrationPriority, MigrationRequest, PageId, TierLevel};
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that memory migration completes within 10ms target
    #[test]
    fn test_memory_migration_under_10ms() {
        if let Ok(device) = CudaContext::new(0) {
            let mut engine = MigrationEngine::new(device).unwrap();

            // Start the migration workers
            engine.start().unwrap();

            // Create a batch of realistic migration requests
            let mut requests = Vec::new();
            for i in 0..100 {
                requests.push(MigrationRequest {
                    page_id: PageId::new(i as u64),
                    source_tier: TierLevel::Nvme,
                    target_tier: TierLevel::Gpu,
                    priority: MigrationPriority::High,
                    deadline: Some(Duration::from_millis(10)),
                });
            }

            // Measure migration time
            let start = Instant::now();
            engine.submit_batch(requests).unwrap();

            // Wait for migrations to complete
            std::thread::sleep(Duration::from_millis(50));
            let elapsed = start.elapsed();

            // Get stats to verify migrations completed
            let stats = engine.get_stats();

            // Stop the engine
            engine.stop();

            // Verify performance target
            assert!(stats.successful_migrations > 0, "No migrations completed");
            assert!(
                stats.average_migration_time < Duration::from_millis(10),
                "Average migration time {:?} exceeds 10ms target",
                stats.average_migration_time
            );
            assert!(
                elapsed < Duration::from_millis(100),
                "Total batch migration time {:?} too high",
                elapsed
            );
        }
    }

    /// Test zero-copy migration optimization
    #[test]
    fn test_zero_copy_migration() {
        if let Ok(device) = CudaContext::new(0) {
            let policy = MigrationPolicy {
                enable_prefetch: true,
                prefetch_distance: 32,
                batch_size: 4096,
                max_concurrent: 8,
                hot_threshold: 5.0,
                cold_threshold: Duration::from_secs(60),
                adaptive: true,
            };

            let mut engine = MigrationEngine::with_policy(device, policy).unwrap();
            engine.start().unwrap();

            // Test GPU<->CPU zero-copy migration
            let request = MigrationRequest {
                page_id: PageId::new(1),
                source_tier: TierLevel::Gpu,
                target_tier: TierLevel::Cpu,
                priority: MigrationPriority::Critical,
                deadline: Some(Duration::from_millis(5)),
            };

            let start = Instant::now();
            engine.submit_migration(request).unwrap();

            // Allow time for migration
            std::thread::sleep(Duration::from_millis(10));
            let elapsed = start.elapsed();

            engine.stop();

            // Zero-copy should be very fast
            assert!(
                elapsed < Duration::from_millis(5),
                "Zero-copy migration took {:?}, exceeding 5ms target",
                elapsed
            );
        }
    }

    /// Test predictive prefetching performance
    #[test]
    fn test_prefetch_performance() {
        if let Ok(device) = CudaContext::new(0) {
            let engine = MigrationEngine::new(device).unwrap();

            // Record sequential access pattern
            for i in 0..100 {
                engine.record_access(PageId::new(i));
            }

            // Test prefetch prediction speed
            let start = Instant::now();
            let predictions = engine.predict_prefetch(PageId::new(100));
            let elapsed = start.elapsed();

            // Prefetch prediction should be instant
            assert!(
                elapsed < Duration::from_micros(100),
                "Prefetch prediction took {:?}, should be <100μs",
                elapsed
            );
            assert!(!predictions.is_empty(), "No prefetch predictions made");
        }
    }

    /// Test hot/cold page detection performance
    #[test]
    fn test_hot_cold_detection_performance() {
        if let Ok(device) = CudaContext::new(0) {
            let engine = MigrationEngine::new(device).unwrap();

            // Simulate access patterns
            let hot_pages = vec![1, 2, 3, 4, 5];
            let cold_pages = vec![100, 101, 102];

            // Record hot page accesses
            for _ in 0..100 {
                for &page in &hot_pages {
                    engine.record_access(PageId::new(page));
                }
            }

            // Record cold page accesses (once)
            for &page in &cold_pages {
                engine.record_access(PageId::new(page));
            }

            // Test detection performance
            let start = Instant::now();
            let detected_hot = engine.get_hot_pages(TierLevel::Cpu);
            let detected_cold = engine.get_cold_pages(TierLevel::Cpu);
            let elapsed = start.elapsed();

            // Detection should be fast
            assert!(
                elapsed < Duration::from_millis(1),
                "Hot/cold detection took {:?}, should be <1ms",
                elapsed
            );
            assert!(!detected_hot.is_empty(), "No hot pages detected");
        }
    }

    /// Test batch optimization performance
    #[test]
    fn test_batch_optimization_performance() {
        let mut optimizer = BatchOptimizer::new(1000);
        let mut requests = Vec::new();

        // Create large batch with mixed priorities
        for i in 0..1000 {
            let priority = match i % 3 {
                0 => MigrationPriority::Critical,
                1 => MigrationPriority::High,
                _ => MigrationPriority::Normal,
            };

            requests.push(MigrationRequest {
                page_id: PageId::new(i as u64),
                source_tier: TierLevel::Nvme,
                target_tier: TierLevel::Gpu,
                priority,
                deadline: None,
            });
        }

        // Test optimization speed
        let start = Instant::now();
        let optimized = BatchOptimizer::optimize_batch(requests);
        let elapsed = start.elapsed();

        // Batch optimization should be fast
        assert!(
            elapsed < Duration::from_millis(1),
            "Batch optimization took {:?}, should be <1ms",
            elapsed
        );
        assert_eq!(optimized.len(), 1000);
        assert_eq!(optimized[0].priority, MigrationPriority::Critical);
    }
}
