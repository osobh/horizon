//! TDD Tests for CUDA Memory Management in Evolution Module
//! These tests define the expected behavior for GPU evolution operations

use super::*;
use anyhow::Result;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Test CUDA memory allocation for evolution operations
#[cfg(test)]
mod cuda_memory_tests {
    use super::*;

    /// Test that we can allocate GPU memory for small populations
    #[tokio::test]
    async fn test_small_population_allocation() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 32, // Small population
            genome_size: 64,     // Small genome
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 32,
        };

        // This should succeed without CUDA memory errors
        let result = GpuEvolutionEngine::new(device, config);
        assert!(result.is_ok(), "Small population allocation should succeed");

        Ok(())
    }

    /// Test that we can initialize random population without assertion failures
    #[tokio::test]
    async fn test_random_initialization() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 32,
            genome_size: 64,
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 32,
        };

        let mut engine = GpuEvolutionEngine::new(device, config)?;

        // This should not panic with CUDA memory assertion error
        let result = engine.initialize_random();
        assert!(result.is_ok(), "Random initialization should succeed");

        Ok(())
    }

    /// Test that we can run one evolution generation
    #[tokio::test]
    async fn test_single_generation() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 32,
            genome_size: 64,
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 32,
        };

        let mut engine = GpuEvolutionEngine::new(device, config)?;
        engine.initialize_random()?;

        // This should complete without CUDA errors
        let result = engine.evolve_generation();
        assert!(
            result.is_ok(),
            "Single generation should complete successfully"
        );

        Ok(())
    }

    /// Test GPU memory scaling - gradually increase population size
    #[tokio::test]
    async fn test_memory_scaling() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);

        // Test increasing population sizes
        let population_sizes = vec![32, 64, 128, 256, 512];

        for &pop_size in &population_sizes {
            let config = GpuEvolutionConfig {
                population_size: pop_size,
                genome_size: 64,
                fitness_objectives: 1,
                mutation_rate: 0.01,
                crossover_rate: 0.7,
                elite_percentage: 0.1,
                block_size: 32,
            };

            let result = GpuEvolutionEngine::new(device.clone(), config);
            assert!(
                result.is_ok(),
                "Population size {} should allocate successfully",
                pop_size
            );
        }

        Ok(())
    }

    /// Test consensus latency requirements (<100μs)
    #[tokio::test]
    async fn test_consensus_latency_target() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 1024, // Larger population for consensus
            genome_size: 32,       // Smaller genome for speed
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        };

        let mut engine = GpuEvolutionEngine::new(device, config)?;
        engine.initialize_random()?;

        // Measure consensus operation time
        let start = std::time::Instant::now();
        engine.evolve_generation()?;
        let duration = start.elapsed();

        // Should complete in under 100 microseconds
        assert!(
            duration.as_micros() < 100,
            "Consensus operation took {}μs, target is <100μs",
            duration.as_micros()
        );

        Ok(())
    }

    /// Test synthesis throughput requirements (2.6B ops/sec)
    #[tokio::test]
    async fn test_synthesis_throughput_target() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 10000, // Large population for throughput
            genome_size: 256,       // Large genome for operations
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        };

        let mut engine = GpuEvolutionEngine::new(device, config)?;
        engine.initialize_random()?;

        // Measure synthesis operations throughput
        let operations_per_generation = config.population_size * config.genome_size;
        let target_ops_per_sec = 2_600_000_000u64; // 2.6B ops/sec

        let start = std::time::Instant::now();
        engine.evolve_generation()?;
        let duration = start.elapsed();

        let actual_ops_per_sec = (operations_per_generation as f64) / duration.as_secs_f64();

        assert!(
            actual_ops_per_sec >= target_ops_per_sec as f64,
            "Synthesis throughput was {:.2}B ops/sec, target is 2.6B ops/sec",
            actual_ops_per_sec / 1_000_000_000.0
        );

        Ok(())
    }
}

/// Integration tests with benchmark scenarios
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test benchmark scenario that was failing
    #[tokio::test]
    async fn test_benchmark_scenario_10k_256() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            population_size: 10000, // Exact scenario that was failing
            genome_size: 256,       // Exact scenario that was failing
            fitness_objectives: 1,
            mutation_rate: 0.01,
            crossover_rate: 0.7,
            elite_percentage: 0.1,
            block_size: 256,
        };

        // This exact scenario was causing CUDA assertion failure
        let result = GpuEvolutionEngine::new(device, config);
        assert!(result.is_ok(), "Benchmark scenario 10k/256 should work");

        let mut engine = result.unwrap();
        let init_result = engine.initialize_random();
        assert!(init_result.is_ok(), "Benchmark initialization should work");

        Ok(())
    }
}
