//! Main hybrid evolution system implementation

use super::{
    config::{EngineStrategy, HybridConfig},
    performance::EnginePerformance,
};
use crate::{
    adas::{AdasConfig, AdasEngine},
    dgm::{DgmConfig, DgmEngine},
    error::{EvolutionEngineError, EvolutionEngineResult},
    metrics::{EvolutionMetrics, MetricsCollector},
    population::{Individual, Population},
    swarm::{SwarmConfig, SwarmEngine},
    traits::{EngineConfig, EvolutionEngine, EvolvableAgent},
};
use parking_lot::RwLock;
use std::sync::Arc;

/// Engine type enumeration
#[derive(Debug, Clone, Copy)]
pub enum EngineType {
    Adas,
    Swarm,
    Dgm,
    All,
}

/// Hybrid evolution system
pub struct HybridEvolutionSystem {
    /// Configuration
    pub(super) config: HybridConfig,
    /// Metrics collector
    pub(super) metrics_collector: MetricsCollector,
    /// ADAS engine
    pub(super) adas_engine: Option<AdasEngine>,
    /// Swarm engine
    pub(super) swarm_engine: Option<SwarmEngine>,
    /// DGM engine
    pub(super) dgm_engine: Option<DgmEngine>,
    /// Engine performance tracking
    pub(super) engine_performance: Arc<RwLock<Vec<EnginePerformance>>>,
    /// Current engine index (for round-robin)
    pub(super) current_engine_index: Arc<RwLock<usize>>,
    /// Generation counter
    pub(super) generation_counter: Arc<RwLock<u32>>,
}

impl HybridEvolutionSystem {
    /// Create new hybrid system
    pub async fn new(config: HybridConfig) -> EvolutionEngineResult<Self> {
        config.validate()?;

        // Initialize engines based on strategy
        let (adas, swarm, dgm) = match config.strategy {
            EngineStrategy::Parallel => {
                // Initialize all engines for parallel execution
                let adas = Some(AdasEngine::new(config.adas_config.clone())?);
                let swarm = Some(SwarmEngine::new(config.swarm_config.clone())?);
                let dgm = Some(DgmEngine::new(config.dgm_config.clone())?);
                (adas, swarm, dgm)
            }
            _ => {
                // Initialize engines lazily for other strategies
                (None, None, None)
            }
        };

        let engine_performance = vec![
            EnginePerformance::new("ADAS".to_string()),
            EnginePerformance::new("Swarm".to_string()),
            EnginePerformance::new("DGM".to_string()),
        ];

        Ok(Self {
            config,
            metrics_collector: MetricsCollector::new(),
            adas_engine: adas,
            swarm_engine: swarm,
            dgm_engine: dgm,
            engine_performance: Arc::new(RwLock::new(engine_performance)),
            current_engine_index: Arc::new(RwLock::new(0)),
            generation_counter: Arc::new(RwLock::new(0)),
        })
    }

    /// Select engine based on strategy
    pub(super) async fn select_engine(&mut self) -> EvolutionEngineResult<EngineType> {
        match self.config.strategy {
            EngineStrategy::RoundRobin => {
                let mut index = self.current_engine_index.write();
                let current = *index;
                *index = (*index + 1) % 3;
                Ok(match current {
                    0 => EngineType::Adas,
                    1 => EngineType::Swarm,
                    _ => EngineType::Dgm,
                })
            }
            EngineStrategy::PerformanceBased => {
                let performance = self.engine_performance.read();
                let best_engine = performance
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

                Ok(match best_engine {
                    0 => EngineType::Adas,
                    1 => EngineType::Swarm,
                    _ => EngineType::Dgm,
                })
            }
            EngineStrategy::Parallel => Ok(EngineType::All),
            EngineStrategy::Adaptive => {
                let metrics = self.metrics_collector.metrics();
                let generation = *self.generation_counter.read();

                // Early exploration phase - use ADAS for architecture search
                if generation < 20 {
                    Ok(EngineType::Adas)
                }
                // Mid phase - use Swarm for optimization
                else if generation < 50 && metrics.convergence_rate < 0.8 {
                    Ok(EngineType::Swarm)
                }
                // Late phase - use DGM for self-improvement
                else {
                    Ok(EngineType::Dgm)
                }
            }
            EngineStrategy::PhaseBased => {
                let generation = *self.generation_counter.read();
                Ok(match generation / 30 {
                    0 => EngineType::Adas,
                    1 => EngineType::Swarm,
                    _ => EngineType::Dgm,
                })
            }
        }
    }

    /// Initialize engine if not already initialized
    pub(super) async fn ensure_engine_initialized(
        &mut self,
        engine_type: EngineType,
    ) -> EvolutionEngineResult<()> {
        match engine_type {
            EngineType::Adas => {
                if self.adas_engine.is_none() {
                    self.adas_engine = Some(AdasEngine::new(self.config.adas_config.clone())?);
                }
            }
            EngineType::Swarm => {
                if self.swarm_engine.is_none() {
                    self.swarm_engine = Some(SwarmEngine::new(self.config.swarm_config.clone())?);
                }
            }
            EngineType::Dgm => {
                if self.dgm_engine.is_none() {
                    self.dgm_engine = Some(DgmEngine::new(self.config.dgm_config.clone())?);
                }
            }
            EngineType::All => {
                if self.adas_engine.is_none() {
                    self.adas_engine = Some(AdasEngine::new(self.config.adas_config.clone())?);
                }
                if self.swarm_engine.is_none() {
                    self.swarm_engine = Some(SwarmEngine::new(self.config.swarm_config.clone())?);
                }
                if self.dgm_engine.is_none() {
                    self.dgm_engine = Some(DgmEngine::new(self.config.dgm_config.clone())?);
                }
            }
        }
        Ok(())
    }

    /// Update engine performance metrics
    pub(super) fn update_performance(
        &self,
        engine_idx: usize,
        improvement: f64,
        time_ns: u64,
        success: bool,
        best_fitness: f64,
    ) {
        let mut performance = self.engine_performance.write();
        if let Some(perf) = performance.get_mut(engine_idx) {
            perf.update(improvement, time_ns, success, best_fitness);
        }
    }

    /// Run parallel evolution
    pub(super) async fn evolve_parallel(
        &mut self,
        population: Population<EvolvableAgent>,
    ) -> EvolutionEngineResult<Population<EvolvableAgent>> {
        self.ensure_engine_initialized(EngineType::All).await?;

        let pop_size = population.size();
        let start_time = std::time::Instant::now();

        // Clone populations for parallel execution
        let pop1 = population.clone();
        let pop2 = population.clone();
        let pop3 = population;

        // Run engines sequentially (avoiding Send trait issues)
        let mut results = Vec::new();

        if let Some(engine) = &mut self.adas_engine {
            results.push(engine.evolve_step(pop1).await);
        } else {
            results.push(Err(EvolutionEngineError::Other(
                "ADAS engine not initialized".to_string(),
            )));
        }

        if let Some(engine) = &mut self.swarm_engine {
            results.push(engine.evolve_step(pop2).await);
        } else {
            results.push(Err(EvolutionEngineError::Other(
                "Swarm engine not initialized".to_string(),
            )));
        }

        if let Some(engine) = &mut self.dgm_engine {
            results.push(engine.evolve_step(pop3).await);
        } else {
            results.push(Err(EvolutionEngineError::Other(
                "DGM engine not initialized".to_string(),
            )));
        }
        let elapsed = start_time.elapsed().as_nanos() as u64;

        // Collect successful results
        let mut all_individuals = Vec::new();
        for (idx, result) in results.into_iter().enumerate() {
            if let Ok(pop) = result {
                // Update performance metrics
                let improvement = pop
                    .individuals
                    .iter()
                    .filter_map(|i| i.fitness)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                self.update_performance(
                    idx,
                    improvement,
                    elapsed / 3,
                    improvement > 0.5,
                    improvement,
                );

                all_individuals.extend(pop.individuals);
            }
        }

        // Sort by fitness and select top individuals
        all_individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let take_count = (pop_size as f64 * self.config.merge_top_percent) as usize;
        let mut selected = all_individuals
            .into_iter()
            .take(take_count)
            .collect::<Vec<_>>();

        // Fill remaining slots with clones of best individuals
        while selected.len() < pop_size {
            let idx = selected.len() % take_count;
            selected.push(selected[idx].clone());
        }

        let mut new_population = Population::from_individuals(selected);
        new_population.generation = (*self.generation_counter.read() + 1) as u64;

        Ok(new_population)
    }
}
