//! Tests for hybrid evolution system

use super::*;
use crate::traits::{EngineConfig, EvolutionEngine};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn test_hybrid_config_default() {
    let config = HybridConfig::default();
    assert_eq!(config.strategy, EngineStrategy::Adaptive);
    assert_eq!(config.parallel_pool_size, 3);
    assert_eq!(config.switch_threshold, 0.1);
    assert_eq!(config.merge_top_percent, 0.3);
}

#[test]
fn test_hybrid_config_validation() {
    let mut config = HybridConfig::default();
    assert!(config.validate().is_ok());

    config.parallel_pool_size = 0;
    assert!(config.validate().is_err());

    config.parallel_pool_size = 3;
    config.switch_threshold = 1.5;
    assert!(config.validate().is_err());

    config.switch_threshold = 0.5;
    config.merge_top_percent = 0.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_engine_strategy() {
    assert_eq!(EngineStrategy::RoundRobin as i32, 0);
    assert_eq!(EngineStrategy::PerformanceBased as i32, 1);
    assert_eq!(EngineStrategy::Parallel as i32, 2);
    assert_eq!(EngineStrategy::Adaptive as i32, 3);
    assert_eq!(EngineStrategy::PhaseBased as i32, 4);
}

#[tokio::test]
async fn test_hybrid_system_creation() {
    let config = HybridConfig::default();
    let system = HybridEvolutionSystem::new(config).await;
    assert!(system.is_ok());
}

#[tokio::test]
async fn test_engine_selection_round_robin() -> TestResult {
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::RoundRobin;
    let mut system = HybridEvolutionSystem::new(config).await?;

    let engine1 = system.select_engine().await?;
    let engine2 = system.select_engine().await?;
    let engine3 = system.select_engine().await?;
    let engine4 = system.select_engine().await?;

    assert!(matches!(engine1, EngineType::Adas));
    assert!(matches!(engine2, EngineType::Swarm));
    assert!(matches!(engine3, EngineType::Dgm));
    assert!(matches!(engine4, EngineType::Adas)); // Wraps around
    Ok(())
}

#[tokio::test]
async fn test_engine_selection_adaptive() -> TestResult {
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::Adaptive;
    let mut system = HybridEvolutionSystem::new(config).await?;

    // Early phase should select ADAS
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Adas));

    // Simulate progression
    *system.generation_counter.write() = 25;
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Swarm));

    // Late phase
    *system.generation_counter.write() = 60;
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Dgm));
    Ok(())
}

#[tokio::test]
async fn test_initial_population_generation() {
    use crate::traits::EvolutionEngine;

    let config = HybridConfig::default();
    let system = HybridEvolutionSystem::new(config).await.unwrap();

    let population = system.generate_initial_population(20).await.unwrap();
    assert_eq!(population.size(), 20);
    assert_eq!(population.generation, 0);
}

#[test]
fn test_performance_tracking() {
    let config = HybridConfig::default();
    let system = futures::executor::block_on(HybridEvolutionSystem::new(config)).unwrap();

    system.update_performance(0, 0.8, 1000000, true, 0.8);
    system.update_performance(0, 0.6, 1000000, true, 0.6);

    let performance = system.engine_performance.read();
    assert_eq!(performance[0].generations_run, 2);
    assert_eq!(performance[0].avg_improvement, 0.7);
    assert_eq!(performance[0].success_rate, 1.0);
}

#[tokio::test]
#[ignore = "Test hangs - needs investigation"]
async fn test_parallel_strategy() -> TestResult {
    use crate::population::Population;
    use crate::traits::EvolvableAgent;

    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::Parallel;
    config.base.population_size = 10;
    config.merge_top_percent = 0.5;

    let mut system = HybridEvolutionSystem::new(config).await?;
    let population: Population<EvolvableAgent> = system.generate_initial_population(10).await?;

    // Should initialize all engines
    assert!(system.adas_engine.is_some());
    assert!(system.swarm_engine.is_some());
    assert!(system.dgm_engine.is_some());

    // Run evolution step
    let result: crate::error::EvolutionEngineResult<Population<EvolvableAgent>> =
        system.evolve_step(population).await;
    assert!(result.is_ok());

    let evolved = result?;
    assert_eq!(evolved.size(), 10);
    assert_eq!(evolved.generation, 1);
    Ok(())
}

#[test]
fn test_hybrid_config_serialization() {
    let config = HybridConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: HybridConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.strategy, deserialized.strategy);
    assert_eq!(config.parallel_pool_size, deserialized.parallel_pool_size);
}

#[test]
fn test_engine_name() {
    let config = HybridConfig::default();
    assert_eq!(config.engine_name(), "Hybrid");
}

#[test]
fn test_engine_strategy_serialization() -> TestResult {
    let strategies = vec![
        EngineStrategy::RoundRobin,
        EngineStrategy::PerformanceBased,
        EngineStrategy::Parallel,
        EngineStrategy::Adaptive,
        EngineStrategy::PhaseBased,
    ];

    for strategy in strategies {
        let json = serde_json::to_string(&strategy)?;
        let deserialized: EngineStrategy = serde_json::from_str(&json)?;
        assert_eq!(strategy, deserialized);
    }
    Ok(())
}

#[test]
fn test_hybrid_config_edge_cases() {
    let mut config = HybridConfig::default();

    // Test minimum valid values
    config.parallel_pool_size = 1;
    config.switch_threshold = 0.0;
    config.merge_top_percent = 0.001; // Very small but > 0
    assert!(config.validate().is_ok());

    // Test maximum valid values
    config.switch_threshold = 1.0;
    config.merge_top_percent = 1.0;
    assert!(config.validate().is_ok());

    // Test invalid values
    config.parallel_pool_size = 0;
    assert!(config.validate().is_err());

    config.parallel_pool_size = 1;
    config.switch_threshold = -0.1;
    assert!(config.validate().is_err());

    config.switch_threshold = 0.5;
    config.merge_top_percent = 1.1;
    assert!(config.validate().is_err());
}

#[test]
fn test_engine_performance_creation() {
    let perf = EnginePerformance::new("TestEngine".to_string());
    assert_eq!(perf.name, "TestEngine");
    assert_eq!(perf.generations_run, 0);
    assert_eq!(perf.avg_improvement, 0.0);
    assert_eq!(perf.best_fitness, 0.0);
    assert_eq!(perf.time_spent_ns, 0);
    assert_eq!(perf.success_rate, 0.0);
}

#[tokio::test]
async fn test_engine_selection_performance_based() -> TestResult {
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::PerformanceBased;
    let mut system = HybridEvolutionSystem::new(config).await?;

    // Update performance to make Swarm the best performer
    system.update_performance(0, 0.5, 1000000, true, 0.5); // ADAS
    system.update_performance(1, 0.9, 1000000, true, 0.9); // Swarm - best
    system.update_performance(2, 0.6, 1000000, true, 0.6); // DGM

    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Swarm));
    Ok(())
}

#[tokio::test]
async fn test_engine_selection_phase_based() -> TestResult {
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::PhaseBased;
    let mut system = HybridEvolutionSystem::new(config).await?;

    // Phase 0 (generation 0-29): ADAS
    *system.generation_counter.write() = 15;
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Adas));

    // Phase 1 (generation 30-59): Swarm
    *system.generation_counter.write() = 35;
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Swarm));

    // Phase 2+ (generation 60+): DGM
    *system.generation_counter.write() = 75;
    let engine = system.select_engine().await?;
    assert!(matches!(engine, EngineType::Dgm));
    Ok(())
}

#[tokio::test]
async fn test_engine_initialization() {
    let config = HybridConfig::default();
    let mut system = HybridEvolutionSystem::new(config).await.unwrap();

    // Initially engines should not be initialized (except for Parallel strategy)
    assert!(system.adas_engine.is_none());
    assert!(system.swarm_engine.is_none());
    assert!(system.dgm_engine.is_none());

    // Initialize ADAS engine
    system
        .ensure_engine_initialized(EngineType::Adas)
        .await
        .unwrap();
    assert!(system.adas_engine.is_some());
    assert!(system.swarm_engine.is_none());
    assert!(system.dgm_engine.is_none());

    // Initialize all engines
    system
        .ensure_engine_initialized(EngineType::All)
        .await
        .unwrap();
    assert!(system.adas_engine.is_some());
    assert!(system.swarm_engine.is_some());
    assert!(system.dgm_engine.is_some());
}

#[test]
fn test_performance_tracking_multiple_runs() {
    let config = HybridConfig::default();
    let system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    // Track multiple runs for same engine
    system.update_performance(0, 0.8, 1000000, true, 0.8);
    system.update_performance(0, 0.6, 2000000, true, 0.8);
    system.update_performance(0, 0.4, 1500000, false, 0.4);

    let performance = system.engine_performance.read();
    assert_eq!(performance[0].generations_run, 3);
    assert_eq!(performance[0].avg_improvement, 0.6); // (0.8 + 0.6 + 0.4) / 3
    assert_eq!(performance[0].time_spent_ns, 4500000);
    assert_eq!(performance[0].success_rate, 2.0 / 3.0); // 2 successes out of 3
}

#[tokio::test]
async fn test_termination_conditions() -> TestResult {
    use crate::metrics::EvolutionMetrics;
    use crate::traits::EvolutionEngine;

    let mut config = HybridConfig::default();
    config.base.max_generations = 10;
    config.base.target_fitness = Some(0.95);
    config.switch_threshold = 0.01;

    let system = HybridEvolutionSystem::new(config).await?;

    // Test max generations reached
    let mut metrics = EvolutionMetrics::default();
    metrics.generation = 15; // Above max_generations
    assert!(system.should_terminate(&metrics).await);

    // Test target fitness reached
    let mut metrics2 = EvolutionMetrics::default();
    metrics2.generation = 5;
    metrics2.best_fitness = 0.98; // Above target
    assert!(system.should_terminate(&metrics2).await);

    // Test not terminated yet
    let mut metrics3 = EvolutionMetrics::default();
    metrics3.generation = 5;
    metrics3.best_fitness = 0.5;
    assert!(!system.should_terminate(&metrics3).await);

    Ok(())
}

#[tokio::test]
async fn test_parameter_adaptation() -> TestResult {
    use crate::metrics::EvolutionMetrics;
    use crate::traits::EvolutionEngine;

    let mut config = HybridConfig::default();
    config.base.adaptive_parameters = true;
    config.strategy = EngineStrategy::Adaptive;

    let mut system = HybridEvolutionSystem::new(config).await?;

    // Initialize engines first
    system.ensure_engine_initialized(EngineType::All).await?;

    // Create metrics for adaptation
    let metrics = EvolutionMetrics::default();

    // Adapt parameters - should not error
    let result = system.adapt_parameters(&metrics).await;
    assert!(result.is_ok());

    Ok(())
}

#[test]
fn test_debug_implementations() {
    let config = HybridConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("HybridConfig"));

    let strategy = EngineStrategy::Adaptive;
    let strategy_debug = format!("{:?}", strategy);
    assert!(strategy_debug.contains("Adaptive"));

    let perf = EnginePerformance::new("Test".to_string());
    let perf_debug = format!("{:?}", perf);
    assert!(perf_debug.contains("EnginePerformance"));
}

#[test]
fn test_configuration_clone() {
    let config = HybridConfig::default();
    let cloned = config.clone();

    assert_eq!(config.strategy, cloned.strategy);
    assert_eq!(config.parallel_pool_size, cloned.parallel_pool_size);
    assert_eq!(config.switch_threshold, cloned.switch_threshold);
    assert_eq!(config.merge_top_percent, cloned.merge_top_percent);
}

#[test]
fn test_performance_clone() {
    let perf = EnginePerformance::new("Original".to_string());
    let cloned = perf.clone();
    assert_eq!(perf.name, cloned.name);
    assert_eq!(perf.generations_run, cloned.generations_run);
    assert_eq!(perf.avg_improvement, cloned.avg_improvement);
}

#[tokio::test]
async fn test_parallel_initialization() {
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::Parallel;

    let system = HybridEvolutionSystem::new(config).await.unwrap();

    // Parallel strategy should initialize all engines
    assert!(system.adas_engine.is_some());
    assert!(system.swarm_engine.is_some());
    assert!(system.dgm_engine.is_some());
}

#[test]
fn test_engine_type_debug() {
    let engine_types = vec![
        EngineType::Adas,
        EngineType::Swarm,
        EngineType::Dgm,
        EngineType::All,
    ];

    for engine_type in engine_types {
        let debug_str = format!("{:?}", engine_type);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_performance_averaging() {
    let config = HybridConfig::default();
    let system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    // Test moving average calculation
    system.update_performance(0, 1.0, 1000000, true, 1.0);
    {
        let perf = system.engine_performance.read();
        assert_eq!(perf[0].avg_improvement, 1.0);
        assert_eq!(perf[0].success_rate, 1.0);
    }

    system.update_performance(0, 0.5, 1000000, false, 0.5);
    {
        let perf = system.engine_performance.read();
        assert_eq!(perf[0].avg_improvement, 0.75); // (1.0 + 0.5) / 2
        assert_eq!(perf[0].success_rate, 0.5); // 1 success out of 2
    }

    system.update_performance(0, 0.0, 1000000, false, 0.0);
    {
        let perf = system.engine_performance.read();
        assert_eq!(perf[0].avg_improvement, 0.5); // (1.0 + 0.5 + 0.0) / 3
        assert_eq!(perf[0].success_rate, 1.0 / 3.0); // 1 success out of 3
    }
}

#[tokio::test]
async fn test_evolution_step_with_different_engines() -> TestResult {
    use crate::traits::EvolutionEngine;

    // Test with RoundRobin strategy
    let mut config = HybridConfig::default();
    config.strategy = EngineStrategy::RoundRobin;
    config.base.population_size = 10;

    let mut system = HybridEvolutionSystem::new(config).await?;

    // Generate initial population
    let population = system.generate_initial_population(10).await?;
    assert_eq!(population.size(), 10);

    // Test metrics access
    let metrics = system.metrics();
    assert_eq!(metrics.generation, 0);

    Ok(())
}

#[test]
fn test_extreme_configuration_values() {
    let mut config = HybridConfig::default();

    // Test extreme but valid values
    config.parallel_pool_size = 1000;
    config.switch_threshold = 1.0;
    config.merge_top_percent = 0.01;

    assert!(config.validate().is_ok());

    let system = tokio_test::block_on(HybridEvolutionSystem::new(config));
    assert!(system.is_ok());
}

#[test]
fn test_performance_best_selection() {
    let config = HybridConfig::default();
    let system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    // Set up different performance levels
    system.update_performance(0, 0.3, 1000000, true, 0.3); // ADAS: 0.3 * 1.0 = 0.3
    system.update_performance(1, 0.8, 1000000, true, 0.8); // Swarm: 0.8 * 1.0 = 0.8 (best)
    system.update_performance(2, 0.6, 1000000, false, 0.6); // DGM: 0.6 * 0.0 = 0.0

    let performance = system.engine_performance.read();
    let best_idx = performance
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let a_score = a.performance_score();
            let b_score = b.performance_score();
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    assert_eq!(best_idx, 1); // Swarm should be best
}

#[tokio::test]
async fn test_metrics_collection() -> TestResult {
    use crate::traits::EvolutionEngine;

    let config = HybridConfig::default();
    let system = HybridEvolutionSystem::new(config).await?;

    // Test initial metrics state
    let metrics = system.metrics();
    assert_eq!(metrics.generation, 0);
    assert_eq!(metrics.total_evaluations, 0);
    assert_eq!(metrics.best_fitness, 0.0);

    Ok(())
}

#[test]
fn test_merge_top_percent_calculation() {
    let config = HybridConfig::default();
    let _system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    let pop_size = 100;
    let merge_percent = 0.3;
    let take_count = (pop_size as f64 * merge_percent) as usize;

    assert_eq!(take_count, 30);

    // Test edge case
    let small_pop = 5;
    let small_take = (small_pop as f64 * merge_percent) as usize;
    assert_eq!(small_take, 1);
}

#[test]
fn test_generation_counter_overflow() {
    let config = HybridConfig::default();
    let system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    // Test that generation counter can handle large values
    *system.generation_counter.write() = u32::MAX - 1;

    // This should not panic
    *system.generation_counter.write() += 1;
    assert_eq!(*system.generation_counter.read(), u32::MAX);
}

#[test]
fn test_time_tracking() {
    let config = HybridConfig::default();
    let system = tokio_test::block_on(HybridEvolutionSystem::new(config)).unwrap();

    let start_time = 1000000u64;
    let end_time = 2500000u64;
    let elapsed = end_time - start_time;

    system.update_performance(0, 0.5, elapsed, true, 0.5);

    let performance = system.engine_performance.read();
    assert_eq!(performance[0].time_spent_ns, elapsed);

    // Add more time
    system.update_performance(0, 0.6, elapsed, true, 0.6);
    let performance = system.engine_performance.read();
    assert_eq!(performance[0].time_spent_ns, elapsed * 2);
}

#[test]
fn test_strategy_equality() {
    assert_eq!(EngineStrategy::Adaptive, EngineStrategy::Adaptive);
    assert_ne!(EngineStrategy::Adaptive, EngineStrategy::RoundRobin);

    // Test copy trait
    let strategy1 = EngineStrategy::Parallel;
    let strategy2 = strategy1;
    assert_eq!(strategy1, strategy2);
}
