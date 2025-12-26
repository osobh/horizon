//! Configuration types for DGM evolution engine

use crate::{
    dgm_self_assessment::SelfAssessmentConfig,
    error::{EvolutionEngineError, EvolutionEngineResult},
    traits::EngineConfig,
};
use serde::{Deserialize, Serialize};

/// DGM engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DgmConfig {
    /// Base configuration
    pub base: crate::config::EvolutionEngineConfig,
    /// Growth patterns
    pub growth_patterns: GrowthPatterns,
    /// Self-improvement parameters
    pub improvement_params: ImprovementParameters,
    /// Discovery rate
    pub discovery_rate: f64,
    /// Growth momentum
    pub growth_momentum: f64,
    /// Pattern retention threshold
    pub pattern_retention_threshold: f64,
    /// Self-assessment configuration
    pub self_assessment: SelfAssessmentConfig,
}

impl Default for DgmConfig {
    fn default() -> Self {
        Self {
            base: crate::config::EvolutionEngineConfig::default(),
            growth_patterns: GrowthPatterns::default(),
            improvement_params: ImprovementParameters::default(),
            discovery_rate: 0.2,
            growth_momentum: 0.7,
            pattern_retention_threshold: 0.6,
            self_assessment: SelfAssessmentConfig::default(),
        }
    }
}

impl EngineConfig for DgmConfig {
    fn validate(&self) -> EvolutionEngineResult<()> {
        self.base.validate()?;

        if self.discovery_rate < 0.0 || self.discovery_rate > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Discovery rate must be between 0 and 1".to_string(),
            });
        }

        if self.growth_momentum < 0.0 || self.growth_momentum > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Growth momentum must be between 0 and 1".to_string(),
            });
        }

        if self.pattern_retention_threshold < 0.0 || self.pattern_retention_threshold > 1.0 {
            return Err(EvolutionEngineError::InvalidConfiguration {
                message: "Pattern retention threshold must be between 0 and 1".to_string(),
            });
        }

        Ok(())
    }

    fn engine_name(&self) -> &str {
        "DGM"
    }
}

/// Growth patterns for DGM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthPatterns {
    /// Maximum pattern history
    pub max_history: usize,
    /// Pattern similarity threshold
    pub similarity_threshold: f64,
    /// Pattern consolidation interval
    pub consolidation_interval: u32,
}

impl Default for GrowthPatterns {
    fn default() -> Self {
        Self {
            max_history: 100,
            similarity_threshold: 0.8,
            consolidation_interval: 10,
        }
    }
}

/// Self-improvement parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementParameters {
    /// Improvement threshold
    pub improvement_threshold: f64,
    /// Learning decay rate
    pub learning_decay: f64,
    /// Exploration bonus
    pub exploration_bonus: f64,
    /// Exploitation penalty
    pub exploitation_penalty: f64,
}

impl Default for ImprovementParameters {
    fn default() -> Self {
        Self {
            improvement_threshold: 0.05,
            learning_decay: 0.95,
            exploration_bonus: 1.2,
            exploitation_penalty: 0.8,
        }
    }
}
