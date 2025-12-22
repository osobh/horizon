//! Configuration types for hybrid evolution system

use crate::{
    adas::AdasConfig,
    dgm::DgmConfig,
    error::{EvolutionEngineError, EvolutionEngineResult},
    swarm::SwarmConfig,
    traits::EngineConfig,
};
use serde::{Deserialize, Serialize};

/// Engine selection strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineStrategy {
    /// Rotate engines in round-robin fashion
    RoundRobin,
    /// Select based on performance metrics
    PerformanceBased,
    /// Use all engines in parallel and merge results
    Parallel,
    /// Adaptive selection based on convergence
    Adaptive,
    /// Use specific engine based on phase
    PhaseBased,
}

/// Hybrid system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Base configuration
    pub base: crate::config::EvolutionEngineConfig,
    /// Engine selection strategy
    pub strategy: EngineStrategy,
    /// ADAS engine configuration
    pub adas_config: AdasConfig,
    /// Swarm engine configuration
    pub swarm_config: SwarmConfig,
    /// DGM engine configuration
    pub dgm_config: DgmConfig,
    /// Parallel execution pool size
    pub parallel_pool_size: usize,
    /// Engine switch threshold
    pub switch_threshold: f64,
    /// Merge strategy for parallel execution
    pub merge_top_percent: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            base: crate::config::EvolutionEngineConfig::default(),
            strategy: EngineStrategy::Adaptive,
            adas_config: AdasConfig::default(),
            swarm_config: SwarmConfig::default(),
            dgm_config: DgmConfig::default(),
            parallel_pool_size: 3,
            switch_threshold: 0.1,
            merge_top_percent: 0.3,
        }
    }
}

impl EngineConfig for HybridConfig {
    fn validate(&self) -> EvolutionEngineResult<()> {
        self.base.validate()?;
        self.adas_config.validate()?;
        self.swarm_config.validate()?;
        self.dgm_config.validate()?;

        if self.parallel_pool_size == 0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Parallel pool size must be greater than 0".to_string(),
            });
        }

        if self.switch_threshold < 0.0 || self.switch_threshold > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Switch threshold must be between 0 and 1".to_string(),
            });
        }

        if self.merge_top_percent <= 0.0 || self.merge_top_percent > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Merge top percent must be between 0 and 1".to_string(),
            });
        }

        Ok(())
    }

    fn engine_name(&self) -> &str {
        "Hybrid"
    }
}
