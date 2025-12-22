//! Configuration for evolution engines

use crate::error::EvolutionEngineResult;
use serde::{Deserialize, Serialize};

/// Base configuration for evolution engines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEngineConfig {
    /// Maximum generations to run
    pub max_generations: u32,
    /// Population size
    pub population_size: usize,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Target fitness (optional)
    pub target_fitness: Option<f64>,
    /// Maximum time to run (seconds)
    pub max_runtime_seconds: Option<u64>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable adaptive parameter tuning
    pub adaptive_parameters: bool,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl Default for EvolutionEngineConfig {
    fn default() -> Self {
        Self {
            max_generations: 100,
            population_size: 100,
            mutation_rate: 0.1,
            target_fitness: None,
            max_runtime_seconds: None,
            seed: None,
            adaptive_parameters: true,
            logging: LoggingConfig::default(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log every N generations
    pub log_interval: u32,
    /// Save checkpoints
    pub save_checkpoints: bool,
    /// Checkpoint interval
    pub checkpoint_interval: u32,
    /// Metrics output path
    pub metrics_path: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_interval: 10,
            save_checkpoints: false,
            checkpoint_interval: 50,
            metrics_path: None,
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<usize>,
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,
    /// Maximum GPU memory (MB)
    pub max_gpu_memory_mb: Option<usize>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(8192), // 8GB default
            max_cpu_cores: None,
            max_gpu_memory_mb: Some(4096), // 4GB default
        }
    }
}

impl EvolutionEngineConfig {
    /// Validate configuration
    pub fn validate(&self) -> EvolutionEngineResult<()> {
        if self.population_size == 0 {
            return Err(crate::error::EvolutionEngineError::InvalidConfiguration {
                message: "Population size must be greater than 0".to_string(),
            });
        }

        if self.max_generations == 0 {
            return Err(crate::error::EvolutionEngineError::InvalidConfiguration {
                message: "Max generations must be greater than 0".to_string(),
            });
        }

        if let Some(max_time) = self.max_runtime_seconds {
            if max_time == 0 {
                return Err(crate::error::EvolutionEngineError::InvalidConfiguration {
                    message: "Max runtime must be greater than 0".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Create a builder for the configuration
    pub fn builder() -> EvolutionEngineConfigBuilder {
        EvolutionEngineConfigBuilder::default()
    }
}

/// Builder for EvolutionEngineConfig
#[derive(Default)]
pub struct EvolutionEngineConfigBuilder {
    config: EvolutionEngineConfig,
}

impl EvolutionEngineConfigBuilder {
    /// Set max generations
    pub fn max_generations(mut self, generations: u32) -> Self {
        self.config.max_generations = generations;
        self
    }

    /// Set population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    /// Set target fitness
    pub fn target_fitness(mut self, fitness: f64) -> Self {
        self.config.target_fitness = Some(fitness);
        self
    }

    /// Set max runtime
    pub fn max_runtime_seconds(mut self, seconds: u64) -> Self {
        self.config.max_runtime_seconds = Some(seconds);
        self
    }

    /// Set seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Enable/disable adaptive parameters
    pub fn adaptive_parameters(mut self, enabled: bool) -> Self {
        self.config.adaptive_parameters = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> EvolutionEngineResult<EvolutionEngineConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = EvolutionEngineConfig::default();
        assert_eq!(config.max_generations, 100);
        assert_eq!(config.population_size, 100);
        assert!(config.adaptive_parameters);
    }

    #[test]
    fn test_config_validation() {
        let mut config = EvolutionEngineConfig::default();
        assert!(config.validate().is_ok());

        config.population_size = 0;
        assert!(config.validate().is_err());

        config.population_size = 100;
        config.max_generations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = EvolutionEngineConfig::builder()
            .max_generations(200)
            .population_size(50)
            .target_fitness(0.95)
            .seed(12345)
            .build()
            ?;

        assert_eq!(config.max_generations, 200);
        assert_eq!(config.population_size, 50);
        assert_eq!(config.target_fitness, Some(0.95));
        assert_eq!(config.seed, Some(12345));
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_memory_mb, Some(8192));
        assert_eq!(limits.max_gpu_memory_mb, Some(4096));
        assert_eq!(limits.max_cpu_cores, None);
    }

    #[test]
    fn test_logging_config() {
        let logging = LoggingConfig::default();
        assert_eq!(logging.log_interval, 10);
        assert!(!logging.save_checkpoints);
        assert_eq!(logging.checkpoint_interval, 50);
    }
}
