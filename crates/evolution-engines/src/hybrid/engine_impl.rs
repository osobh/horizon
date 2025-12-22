//! EvolutionEngine trait implementation for HybridEvolutionSystem

use super::{
    config::{EngineStrategy, HybridConfig},
    system::{EngineType, HybridEvolutionSystem},
};
use crate::{
    adas::AdasEngine,
    error::{EvolutionEngineError, EvolutionEngineResult},
    metrics::EvolutionMetrics,
    population::Population,
    traits::{EvolutionEngine, EvolvableAgent},
};
use async_trait::async_trait;

#[async_trait]
impl EvolutionEngine for HybridEvolutionSystem {
    type Entity = EvolvableAgent;
    type Config = HybridConfig;

    async fn initialize(config: Self::Config) -> EvolutionEngineResult<Self> {
        Self::new(config).await
    }

    async fn evolve_step(
        &mut self,
        population: Population<Self::Entity>,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        self.metrics_collector.start_generation();
        let start_time = std::time::Instant::now();

        // Select strategy
        let engine_type = self.select_engine().await?;

        // Handle parallel execution separately
        if let EngineType::All = engine_type {
            let result = self.evolve_parallel(population).await?;
            *self.generation_counter.write() += 1;
            return Ok(result);
        }

        // Ensure selected engine is initialized
        self.ensure_engine_initialized(engine_type).await?;

        // Run selected engine
        let (result, engine_idx) = match engine_type {
            EngineType::Adas => {
                if let Some(engine) = &mut self.adas_engine {
                    (engine.evolve_step(population).await, 0)
                } else {
                    return Err(EvolutionEngineError::Other(
                        "ADAS engine not initialized".to_string(),
                    ));
                }
            }
            EngineType::Swarm => {
                if let Some(engine) = &mut self.swarm_engine {
                    (engine.evolve_step(population).await, 1)
                } else {
                    return Err(EvolutionEngineError::Other(
                        "Swarm engine not initialized".to_string(),
                    ));
                }
            }
            EngineType::Dgm => {
                if let Some(engine) = &mut self.dgm_engine {
                    (engine.evolve_step(population).await, 2)
                } else {
                    return Err(EvolutionEngineError::Other(
                        "DGM engine not initialized".to_string(),
                    ));
                }
            }
            _ => unreachable!(),
        };

        let elapsed = start_time.elapsed().as_nanos() as u64;

        // Update metrics
        if let Ok(ref pop) = result {
            let best_fitness = pop
                .individuals
                .iter()
                .filter_map(|i| i.fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let avg_fitness = pop
                .individuals
                .iter()
                .filter_map(|i| i.fitness)
                .sum::<f64>()
                / pop.individuals.len() as f64;

            self.update_performance(
                engine_idx,
                best_fitness,
                elapsed,
                best_fitness > 0.5,
                best_fitness,
            );

            self.metrics_collector.end_generation(
                best_fitness,
                avg_fitness,
                0.5, // Simplified diversity
                pop.individuals.len() as u64,
            );
        }

        *self.generation_counter.write() += 1;
        result
    }

    async fn generate_initial_population(
        &self,
        size: usize,
    ) -> EvolutionEngineResult<Population<Self::Entity>> {
        // Use ADAS for initial population as it has good exploration
        let engine = AdasEngine::new(self.config.adas_config.clone())?;
        engine.generate_initial_population(size).await
    }

    async fn should_terminate(&self, metrics: &EvolutionMetrics) -> bool {
        // Check base termination conditions
        if metrics.generation >= self.config.base.max_generations {
            return true;
        }

        if let Some(target) = self.config.base.target_fitness {
            if metrics.best_fitness >= target {
                return true;
            }
        }

        // Check if all engines have stagnated
        let performance = self.engine_performance.read();
        let all_stagnated = performance
            .iter()
            .all(|p| p.generations_run > 10 && p.avg_improvement < self.config.switch_threshold);

        all_stagnated
    }

    fn metrics(&self) -> &EvolutionMetrics {
        self.metrics_collector.metrics()
    }

    async fn adapt_parameters(&mut self, metrics: &EvolutionMetrics) -> EvolutionEngineResult<()> {
        if self.config.base.adaptive_parameters {
            // Adapt individual engine parameters
            if let Some(engine) = &mut self.adas_engine {
                engine.adapt_parameters(metrics).await?;
            }
            if let Some(engine) = &mut self.swarm_engine {
                engine.adapt_parameters(metrics).await?;
            }
            if let Some(engine) = &mut self.dgm_engine {
                engine.adapt_parameters(metrics).await?;
            }

            // Adapt hybrid strategy based on performance
            let performance = self.engine_performance.read();
            let best_performer = performance
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| (p.avg_improvement * 1000.0) as i64)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // If one engine is significantly better, consider switching to performance-based
            if performance[best_performer].avg_improvement
                > performance.iter().map(|p| p.avg_improvement).sum::<f64>() / 3.0 * 1.5
            {
                if self.config.strategy == EngineStrategy::Adaptive {
                    // Could switch to PerformanceBased, but keeping current strategy for now
                }
            }
        }

        Ok(())
    }
}
