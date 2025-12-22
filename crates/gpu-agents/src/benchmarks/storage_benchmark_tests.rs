//! TDD tests for storage benchmarks
//! Following rust.md TDD principles

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::time::Duration;
    
    #[test]
    fn test_storage_benchmark_metrics() {
        // RED: Test that we can collect storage metrics
        let metrics = super::StorageBenchmarkMetrics::default();
        
        assert_eq!(metrics.total_agents_stored, 0);
        assert_eq!(metrics.total_agents_retrieved, 0);
        assert_eq!(metrics.cache_hit_rate, 0.0);
        assert!(metrics.avg_store_latency_ms > 0.0);
        assert!(metrics.avg_retrieve_latency_ms > 0.0);
    }
    
    #[test]
    fn test_benchmark_scenarios() {
        // RED: Test different benchmark scenarios
        let scenarios = super::StorageBenchmarkScenario::all();
        
        assert!(scenarios.contains(&super::StorageBenchmarkScenario::BurstWrite));
        assert!(scenarios.contains(&super::StorageBenchmarkScenario::RandomAccess));
        assert!(scenarios.contains(&super::StorageBenchmarkScenario::HotColdPattern));
        assert!(scenarios.contains(&super::StorageBenchmarkScenario::SwarmCheckpoint));
        assert!(scenarios.contains(&super::StorageBenchmarkScenario::KnowledgeGraphUpdate));
    }
    
    #[tokio::test]
    async fn test_burst_write_scenario() -> Result<()> {
        // RED: Test burst write scenario
        let config = super::StorageBenchmarkConfig {
            scenario: super::StorageBenchmarkScenario::BurstWrite,
            agent_count: 100,
            iterations: 10,
            enable_gpu_cache: true,
            cache_size_mb: 128,
            measure_gpu_memory: true,
            warmup_iterations: 2,
        };
        
        let benchmark = super::StorageBenchmark::new(config)?;
        let results = benchmark.run().await?;
        
        assert!(results.metrics.total_agents_stored >= 100);
        assert!(results.metrics.avg_store_latency_ms > 0.0);
        assert!(results.metrics.store_throughput_agents_per_sec > 0.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_cache_effectiveness() -> Result<()> {
        // RED: Test cache hit rate measurement
        let config = super::StorageBenchmarkConfig {
            scenario: super::StorageBenchmarkScenario::HotColdPattern,
            agent_count: 1000,
            iterations: 100,
            enable_gpu_cache: true,
            cache_size_mb: 256,
            hot_agent_percentage: 20, // 20% of agents are "hot"
            ..Default::default()
        };
        
        let benchmark = super::StorageBenchmark::new(config)?;
        let results = benchmark.run().await?;
        
        // With 20% hot agents accessed frequently, cache hit rate should be good
        assert!(results.metrics.cache_hit_rate > 0.5);
        assert!(results.metrics.hot_agent_avg_latency_ms < results.metrics.cold_agent_avg_latency_ms);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_concurrent_access_benchmark() -> Result<()> {
        // RED: Test concurrent swarm access patterns
        let config = super::StorageBenchmarkConfig {
            scenario: super::StorageBenchmarkScenario::ConcurrentSwarmAccess,
            agent_count: 10_000,
            concurrent_tasks: 10,
            iterations: 50,
            enable_gpu_cache: true,
            ..Default::default()
        };
        
        let benchmark = super::StorageBenchmark::new(config)?;
        let results = benchmark.run().await?;
        
        assert!(results.metrics.concurrent_throughput > results.metrics.store_throughput_agents_per_sec);
        assert!(results.metrics.contention_rate < 0.1); // Low contention
        
        Ok(())
    }
    
    #[test]
    fn test_benchmark_report_generation() {
        // RED: Test benchmark report formatting
        let metrics = super::StorageBenchmarkMetrics {
            total_agents_stored: 1_000_000,
            total_agents_retrieved: 5_000_000,
            avg_store_latency_ms: 0.5,
            avg_retrieve_latency_ms: 0.1,
            cache_hit_rate: 0.85,
            store_throughput_agents_per_sec: 50_000.0,
            retrieve_throughput_agents_per_sec: 250_000.0,
            ..Default::default()
        };
        
        let report = super::StorageBenchmarkReport::from_metrics(metrics);
        
        assert!(report.summary.contains("Storage Benchmark Results"));
        assert!(report.summary.contains("1,000,000"));
        assert!(report.summary.contains("85.0%"));
        assert!(report.has_warnings() == false); // Good performance, no warnings
    }
    
    #[tokio::test]
    async fn test_gpu_memory_tracking() -> Result<()> {
        // RED: Test GPU memory usage tracking during storage ops
        if cudarc::driver::CudaDevice::new(0).is_err() {
            return Ok(()); // Skip if no GPU
        }
        
        let config = super::StorageBenchmarkConfig {
            scenario: super::StorageBenchmarkScenario::MemoryPressure,
            agent_count: 100_000,
            agent_state_size: 1024, // Large state vectors
            measure_gpu_memory: true,
            ..Default::default()
        };
        
        let benchmark = super::StorageBenchmark::new(config)?;
        let results = benchmark.run().await?;
        
        assert!(results.metrics.peak_gpu_memory_mb > 0.0);
        assert!(results.metrics.gpu_memory_efficiency > 0.7); // Good efficiency
        
        Ok(())
    }
    
    #[test]
    fn test_scenario_comparison() {
        // RED: Test comparing different scenarios
        let results1 = super::StorageBenchmarkResults {
            scenario: super::StorageBenchmarkScenario::BurstWrite,
            metrics: super::StorageBenchmarkMetrics {
                store_throughput_agents_per_sec: 50_000.0,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let results2 = super::StorageBenchmarkResults {
            scenario: super::StorageBenchmarkScenario::RandomAccess,
            metrics: super::StorageBenchmarkMetrics {
                retrieve_throughput_agents_per_sec: 100_000.0,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let comparison = super::compare_scenarios(vec![results1, results2]);
        
        assert_eq!(comparison.best_write_scenario, super::StorageBenchmarkScenario::BurstWrite);
        assert_eq!(comparison.best_read_scenario, super::StorageBenchmarkScenario::RandomAccess);
    }
}