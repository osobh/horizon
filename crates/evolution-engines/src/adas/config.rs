//! ADAS configuration types

use super::search_spaces::{ArchitectureSearchSpace, BehaviorSearchSpace};
use crate::{
    error::{EvolutionEngineError, EvolutionEngineResult},
    traits::EngineConfig,
};
use serde::{Deserialize, Serialize};

/// ADAS engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdasConfig {
    /// Base configuration
    pub base: crate::config::EvolutionEngineConfig,
    /// Architecture search space
    pub architecture_space: ArchitectureSearchSpace,
    /// Behavior search space
    pub behavior_space: BehaviorSearchSpace,
    /// Meta-learning rate
    pub meta_learning_rate: f64,
    /// Architecture mutation probability
    pub architecture_mutation_prob: f64,
    /// Behavior mutation probability
    pub behavior_mutation_prob: f64,
}

impl Default for AdasConfig {
    fn default() -> Self {
        Self {
            base: crate::config::EvolutionEngineConfig::default(),
            architecture_space: ArchitectureSearchSpace::default(),
            behavior_space: BehaviorSearchSpace::default(),
            meta_learning_rate: 0.01,
            architecture_mutation_prob: 0.3,
            behavior_mutation_prob: 0.7,
        }
    }
}

impl EngineConfig for AdasConfig {
    fn validate(&self) -> EvolutionEngineResult<()> {
        self.base.validate()?;

        if self.meta_learning_rate <= 0.0 || self.meta_learning_rate > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Meta learning rate must be in (0, 1]".to_string(),
            });
        }

        if self.architecture_mutation_prob < 0.0 || self.architecture_mutation_prob > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Architecture mutation probability must be in [0, 1]".to_string(),
            });
        }

        Ok(())
    }

    fn engine_name(&self) -> &str {
        "ADAS"
    }
}
