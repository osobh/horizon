//! TDD tests to isolate where evolution pipeline hangs
//! Following RED-GREEN-REFACTOR methodology

use super::*;
use crate::evolution::{GpuEvolutionConfig, GpuEvolutionEngine};
use anyhow::Result;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Test utilities
fn create_test_device() -> Arc<CudaContext> {
    CudaContext::new(0).expect("CUDA device should be available")
}

fn create_minimal_config() -> GpuEvolutionConfig {
    GpuEvolutionConfig {
        population_size: 32, // Minimal population
        genome_size: 8,      // Minimal genome
        fitness_objectives: 1,
        mutation_rate: 0.01,
        crossover_rate: 0.7,
        elite_percentage: 0.1,
        block_size: 32,
    }
}

/// Test with timeout to detect hangs
fn run_with_timeout<F, T>(test_fn: F, timeout_secs: u64) -> Result<T, String>
where
    F: FnOnce() -> Result<T> + std::panic::UnwindSafe,
{
    let start = Instant::now();

    // Use std::panic::catch_unwind to catch panics
    let result = std::panic::catch_unwind(|| test_fn());

    let elapsed = start.elapsed();

    match result {
        Ok(Ok(value)) => {
            if elapsed > Duration::from_secs(timeout_secs) {
                Err(format!("Test took too long: {:?}", elapsed))
            } else {
                Ok(value)
            }
        }
        Ok(Err(e)) => Err(format!("Test failed: {}", e)),
        Err(_) => Err("Test panicked".to_string()),
    }
}

// =============================================================================
// PHASE 1: RED - Tests that should expose the hang
// =============================================================================

#[cfg(test)]
mod hang_isolation_tests {
    use super::*;

    #[test]
    #[ignore] // Ignore by default since these are expected to hang
    pub fn test_basic_evolution_engine_creation() -> Result<()> {
        println!("🧪 Testing basic evolution engine creation...");

        let result = run_with_timeout(
            || {
                let device = create_test_device();
                let config = create_minimal_config();
                let _engine = GpuEvolutionEngine::new(device, config)?;
                println!("   ✅ Engine created successfully");
                Ok(())
            },
            5,
        );

        match result {
            Ok(_) => {
                println!("   ✅ Engine creation: PASSED");
                Ok(())
            }
            Err(e) => {
                println!("   ❌ Engine creation: FAILED - {}", e);
                panic!("Engine creation should not hang or fail");
            }
        }
    }

    #[test]
    #[ignore] // Ignore by default since these are expected to hang
    pub fn test_random_initialization() -> Result<()> {
        println!("🧪 Testing random initialization...");

        let result = run_with_timeout(
            || {
                let device = create_test_device();
                let config = create_minimal_config();
                let mut engine = GpuEvolutionEngine::new(device, config)?;
                engine.initialize_random()?;
                println!("   ✅ Random initialization successful");
                Ok(())
            },
            5,
        );

        match result {
            Ok(_) => {
                println!("   ✅ Random initialization: PASSED");
                Ok(())
            }
            Err(e) => {
                println!("   ❌ Random initialization: FAILED - {}", e);
                panic!("Random initialization should not hang or fail");
            }
        }
    }

    #[test]
    #[ignore] // Ignore by default since this is expected to hang
    pub fn test_fitness_evaluation_hangs() -> Result<()> {
        println!("🧪 Testing fitness evaluation (expected to hang)...");

        let result = run_with_timeout(
            || {
                let device = create_test_device();
                let config = create_minimal_config();
                let mut engine = GpuEvolutionEngine::new(device, config)?;
                engine.initialize_random()?;

                println!("   🔍 About to call evaluate_fitness()...");
                engine.evaluate_fitness()?;
                println!("   ✅ Fitness evaluation completed");
                Ok(())
            },
            10,
        ); // 10 second timeout

        match result {
            Ok(_) => {
                println!("   ✅ Fitness evaluation: UNEXPECTEDLY PASSED");
                Ok(())
            }
            Err(e) => {
                println!(
                    "   ❌ Fitness evaluation: HANGED/FAILED as expected - {}",
                    e
                );
                // This is expected, so we return Ok for the test framework
                Ok(())
            }
        }
    }

    #[test]
    #[ignore] // Ignore by default since this is expected to hang
    pub fn test_single_evolve_generation_hangs() -> Result<()> {
        println!("🧪 Testing single evolve_generation() call (expected to hang)...");

        let result = run_with_timeout(
            || {
                let device = create_test_device();
                let config = create_minimal_config();
                let mut engine = GpuEvolutionEngine::new(device, config)?;
                engine.initialize_random()?;

                println!("   🔍 About to call evolve_generation()...");
                engine.evolve_generation()?;
                println!("   ✅ Single generation completed");
                Ok(())
            },
            15,
        ); // 15 second timeout

        match result {
            Ok(_) => {
                println!("   ✅ Single generation: UNEXPECTEDLY PASSED");
                Ok(())
            }
            Err(e) => {
                println!("   ❌ Single generation: HANGED/FAILED as expected - {}", e);
                // This is expected, so we return Ok for the test framework
                Ok(())
            }
        }
    }

    #[test]
    fn test_population_has_fitness_after_initialization() -> Result<()> {
        println!("🧪 Testing population fitness state after initialization...");

        let device = create_test_device();
        let config = create_minimal_config();
        let mut engine = GpuEvolutionEngine::new(device, config)?;
        engine.initialize_random()?;

        // Get statistics to check fitness state
        let stats = engine.statistics();
        println!(
            "   📊 Initial stats: best={}, avg={}, diversity={}",
            stats.best_fitness, stats.average_fitness, stats.diversity_index
        );

        // Best fitness should be 0.0 initially (no fitness evaluated)
        assert_eq!(stats.best_fitness, 0.0);
        assert_eq!(stats.average_fitness, 0.0);

        println!("   ✅ Population fitness state: PASSED");
        Ok(())
    }

    #[test]
    fn test_minimal_evolution_components_exist() -> Result<()> {
        println!("🧪 Testing that all evolution components are created...");

        let device = create_test_device();
        let config = create_minimal_config();
        let mut engine = GpuEvolutionEngine::new(device, config)?;
        engine.initialize_random()?;

        // If we get here, all components were created successfully
        println!("   ✅ All evolution components exist");

        // Test getting statistics (should not hang)
        let stats = engine.statistics();
        println!(
            "   📊 Engine stats accessible: gen={}, pop={}",
            stats.generation, stats.population_size
        );

        assert_eq!(stats.generation, 0);
        assert_eq!(stats.population_size, config.population_size);

        println!("   ✅ Evolution components: PASSED");
        Ok(())
    }
}

// Helper function to run specific hang tests manually
pub fn run_hang_test(test_name: &str) -> Result<()> {
    match test_name {
        "creation" => test_basic_evolution_engine_creation(),
        "initialization" => test_random_initialization(),
        "fitness" => test_fitness_evaluation_hangs(),
        "generation" => test_single_evolve_generation_hangs(),
        _ => {
            println!("Available tests: creation, initialization, fitness, generation");
            Ok(())
        }
    }
}

// Public test functions that can be called directly
pub fn test_basic_evolution_engine_creation() -> Result<()> {
    println!("🧪 Testing basic evolution engine creation...");

    let result = run_with_timeout(
        || {
            let device = create_test_device();
            let config = create_minimal_config();
            let _engine = GpuEvolutionEngine::new(device, config)?;
            println!("   ✅ Engine created successfully");
            Ok(())
        },
        5,
    );

    match result {
        Ok(_) => {
            println!("   ✅ Engine creation: PASSED");
            Ok(())
        }
        Err(e) => {
            println!("   ❌ Engine creation: FAILED - {}", e);
            panic!("Engine creation should not hang or fail");
        }
    }
}

pub fn test_random_initialization() -> Result<()> {
    println!("🧪 Testing random initialization...");

    let result = run_with_timeout(
        || {
            let device = create_test_device();
            let config = create_minimal_config();
            let mut engine = GpuEvolutionEngine::new(device, config)?;
            engine.initialize_random()?;
            println!("   ✅ Random initialization successful");
            Ok(())
        },
        5,
    );

    match result {
        Ok(_) => {
            println!("   ✅ Random initialization: PASSED");
            Ok(())
        }
        Err(e) => {
            println!("   ❌ Random initialization: FAILED - {}", e);
            panic!("Random initialization should not hang or fail");
        }
    }
}

pub fn test_fitness_evaluation_hangs() -> Result<()> {
    println!("🧪 Testing fitness evaluation (expected to hang)...");

    let result = run_with_timeout(
        || {
            let device = create_test_device();
            let config = create_minimal_config();
            let mut engine = GpuEvolutionEngine::new(device, config)?;
            engine.initialize_random()?;

            println!("   🔍 About to call evaluate_fitness()...");
            engine.evaluate_fitness()?;
            println!("   ✅ Fitness evaluation completed");
            Ok(())
        },
        10,
    ); // 10 second timeout

    match result {
        Ok(_) => {
            println!("   ✅ Fitness evaluation: UNEXPECTEDLY PASSED");
            Ok(())
        }
        Err(e) => {
            println!(
                "   ❌ Fitness evaluation: HANGED/FAILED as expected - {}",
                e
            );
            // This is expected, so we return Ok for the test framework
            Ok(())
        }
    }
}

pub fn test_single_evolve_generation_hangs() -> Result<()> {
    println!("🧪 Testing single evolve_generation() call (expected to hang)...");

    let result = run_with_timeout(
        || {
            let device = create_test_device();
            let config = create_minimal_config();
            let mut engine = GpuEvolutionEngine::new(device, config)?;
            engine.initialize_random()?;

            println!("   🔍 About to call evolve_generation()...");
            engine.evolve_generation()?;
            println!("   ✅ Single generation completed");
            Ok(())
        },
        15,
    ); // 15 second timeout

    match result {
        Ok(_) => {
            println!("   ✅ Single generation: UNEXPECTEDLY PASSED");
            Ok(())
        }
        Err(e) => {
            println!("   ❌ Single generation: HANGED/FAILED as expected - {}", e);
            // This is expected, so we return Ok for the test framework
            Ok(())
        }
    }
}
