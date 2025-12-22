//! Bootstrap configuration management

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Bootstrap configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Initial population parameters
    pub population: PopulationConfig,
    /// Evolution parameters
    pub evolution: EvolutionConfig,
    /// Resource limits
    pub resources: ResourceConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Safeguards configuration
    pub safeguards: SafeguardConfig,
}

/// Population configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// Initial population size (10-20 agents recommended)
    pub initial_size: usize,
    /// Maximum population size
    pub max_size: usize,
    /// Minimum population size (triggers intervention)
    pub min_size: usize,
    /// Population growth rate limiter
    pub max_growth_rate: f32,
}

/// Evolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Initial mutation rate (20-30% for diversity)
    pub initial_mutation_rate: f32,
    /// Target mutation rate after stabilization
    pub target_mutation_rate: f32,
    /// Evolution cycle interval
    pub cycle_interval: Duration,
    /// Minimum diversity threshold
    pub min_diversity: f32,
    /// Selection pressure (higher = more selective)
    pub selection_pressure: f32,
}

/// Resource configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// GPU memory per agent (MB)
    pub gpu_memory_per_agent: usize,
    /// CPU compute time slice per agent (ms)
    pub cpu_time_slice: u64,
    /// Maximum kernel execution time (ms)
    pub max_kernel_time: u64,
    /// Storage quota per agent (MB)
    pub storage_quota: usize,
    /// Network bandwidth per agent (KB/s)
    pub network_bandwidth: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
}

/// Safeguard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeguardConfig {
    /// Enable population explosion prevention
    pub prevent_explosion: bool,
    /// Enable diversity crisis detection
    pub diversity_protection: bool,
    /// Enable resource exhaustion prevention
    pub resource_protection: bool,
    /// Enable emergency reset capability
    pub emergency_reset: bool,
    /// Maximum consecutive failures before intervention
    pub max_failures: usize,
}

/// Bootstrap phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapPhase {
    /// Phase 1: Genesis loader initialization
    Genesis,
    /// Phase 2: Template agent creation
    TemplateCreation,
    /// Phase 3: Population growth and specialization
    Specialization,
    /// Phase 4: Ecosystem stabilization
    Stabilization,
    /// Phase 5: Self-sustaining operation
    SelfSustaining,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            population: PopulationConfig {
                initial_size: 15,
                max_size: 1000,
                min_size: 5,
                max_growth_rate: 0.1, // 10% per cycle
            },
            evolution: EvolutionConfig {
                initial_mutation_rate: 0.25,
                target_mutation_rate: 0.05,
                cycle_interval: Duration::from_secs(60),
                min_diversity: 0.3,
                selection_pressure: 0.7,
            },
            resources: ResourceConfig {
                gpu_memory_per_agent: 64, // 64MB
                cpu_time_slice: 100,      // 100ms
                max_kernel_time: 200,     // 200ms
                storage_quota: 10,        // 10MB
                network_bandwidth: 1024,  // 1KB/s
            },
            monitoring: MonitoringConfig {
                metrics_interval: Duration::from_secs(10),
                health_check_interval: Duration::from_secs(30),
                detailed_logging: true,
                checkpoint_interval: Duration::from_secs(300), // 5 minutes
            },
            safeguards: SafeguardConfig {
                prevent_explosion: true,
                diversity_protection: true,
                resource_protection: true,
                emergency_reset: true,
                max_failures: 3,
            },
        }
    }
}

impl BootstrapConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: BootstrapConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.population.initial_size == 0 {
            anyhow::bail!("Initial population size must be greater than 0");
        }

        if self.population.initial_size > self.population.max_size {
            anyhow::bail!("Initial population size cannot exceed maximum size");
        }

        if self.population.min_size >= self.population.max_size {
            anyhow::bail!("Minimum population size must be less than maximum size");
        }

        if !(0.0..=1.0).contains(&self.evolution.initial_mutation_rate) {
            anyhow::bail!("Mutation rate must be between 0.0 and 1.0");
        }

        if self.resources.gpu_memory_per_agent == 0 {
            anyhow::bail!("GPU memory per agent must be greater than 0");
        }

        tracing::info!("Bootstrap configuration validation passed");
        Ok(())
    }

    /// Get current mutation rate based on phase and time
    pub fn current_mutation_rate(&self, phase: BootstrapPhase, progress: f32) -> f32 {
        match phase {
            BootstrapPhase::Genesis => self.evolution.initial_mutation_rate,
            BootstrapPhase::TemplateCreation => self.evolution.initial_mutation_rate,
            BootstrapPhase::Specialization => {
                // Gradually reduce mutation rate during specialization
                let reduction_factor = progress.min(1.0);
                self.evolution.initial_mutation_rate * (1.0 - reduction_factor * 0.5)
            }
            BootstrapPhase::Stabilization => {
                // Continue reducing toward target
                let reduction_factor = progress.min(1.0);
                let intermediate = self.evolution.initial_mutation_rate * 0.5;
                intermediate
                    + (self.evolution.target_mutation_rate - intermediate) * reduction_factor
            }
            BootstrapPhase::SelfSustaining => self.evolution.target_mutation_rate,
        }
    }
}
