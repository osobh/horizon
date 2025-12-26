//! Scenario runner for executing configured agent scenarios

use super::config::{ScenarioConfig, ScenarioType};
use crate::{GpuSwarm, GpuSwarmConfig};
use std::time::{Duration, Instant};

/// Result of running a scenario
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Scenario ID
    pub scenario_id: String,

    /// Total execution time
    pub duration: Duration,

    /// Number of agents simulated
    pub agent_count: usize,

    /// Performance metrics collected
    pub metrics: Vec<PerformanceMetric>,

    /// Whether all objectives were met
    pub objectives_met: bool,

    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Performance metric measurement
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Metric name
    pub name: String,

    /// Measured value
    pub value: f64,

    /// Unit of measurement
    pub unit: String,

    /// Timestamp when measured
    pub timestamp: Duration,
}

/// Scenario runner that executes configured scenarios
pub struct ScenarioRunner {
    device_id: i32,
}

impl ScenarioRunner {
    /// Create a new scenario runner
    pub fn new(device_id: i32) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { device_id })
    }

    /// Run a scenario configuration
    pub fn run_scenario(
        &self,
        config: &ScenarioConfig,
    ) -> Result<ScenarioResult, Box<dyn std::error::Error>> {
        log::info!("Starting scenario: {} ({})", config.name, config.id);

        // Validate configuration
        config.validate()?;

        // Create appropriate swarm configuration
        let mut swarm_config = self.create_swarm_config(config)?;
        swarm_config.device_id = self.device_id;

        // Initialize swarm
        let mut swarm = GpuSwarm::new(swarm_config)?;
        swarm.initialize(config.agent_count)?;

        // Run scenario
        let start_time = Instant::now();
        let mut metrics = Vec::new();
        let errors = Vec::new();

        // Execute scenario based on type
        match &config.scenario_type {
            ScenarioType::Simple { .. } => {
                self.run_simple_scenario(&mut swarm, config, &mut metrics)?;
            }
            ScenarioType::Reasoning { .. } => {
                self.run_reasoning_scenario(&mut swarm, config, &mut metrics)?;
            }
            ScenarioType::Knowledge { .. } => {
                self.run_knowledge_scenario(&mut swarm, config, &mut metrics)?;
            }
        }

        let total_duration = start_time.elapsed();

        // Check objectives
        let objectives_met = self.check_objectives(config, &metrics);

        Ok(ScenarioResult {
            scenario_id: config.id.clone(),
            duration: total_duration,
            agent_count: config.agent_count,
            metrics,
            objectives_met,
            errors,
        })
    }

    /// Create swarm configuration from scenario config
    fn create_swarm_config(
        &self,
        config: &ScenarioConfig,
    ) -> Result<GpuSwarmConfig, Box<dyn std::error::Error>> {
        let base_config = GpuSwarmConfig {
            device_id: 0,
            max_agents: config.agent_count,
            block_size: 256,
            shared_memory_size: 48 * 1024,
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_knowledge_graph: false,
            enable_collective_knowledge: false,
        };

        // Customize based on scenario type
        match &config.scenario_type {
            ScenarioType::Simple { .. } => Ok(base_config),
            ScenarioType::Reasoning { .. } => Ok(GpuSwarmConfig {
                enable_llm: true,
                enable_collective_intelligence: true,
                ..base_config
            }),
            ScenarioType::Knowledge { .. } => Ok(GpuSwarmConfig {
                enable_knowledge_graph: true,
                enable_collective_knowledge: true,
                ..base_config
            }),
        }
    }

    /// Run simple agent scenario
    fn run_simple_scenario(
        &self,
        swarm: &mut GpuSwarm,
        config: &ScenarioConfig,
        metrics: &mut Vec<PerformanceMetric>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let mut step_count = 0;

        // Run simulation for configured duration
        while start_time.elapsed() < config.duration {
            swarm.step()?;
            step_count += 1;

            // Collect metrics periodically
            if step_count % 100 == 0 {
                let elapsed = start_time.elapsed();
                let agents_per_second =
                    (config.agent_count * step_count) as f64 / elapsed.as_secs_f64();

                metrics.push(PerformanceMetric {
                    name: "agents_per_second".to_string(),
                    value: agents_per_second,
                    unit: "agents/s".to_string(),
                    timestamp: elapsed,
                });
            }
        }

        Ok(())
    }

    /// Run reasoning agent scenario
    fn run_reasoning_scenario(
        &self,
        _swarm: &mut GpuSwarm,
        _config: &ScenarioConfig,
        _metrics: &mut Vec<PerformanceMetric>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement reasoning scenario execution
        log::warn!("Reasoning scenario execution not yet implemented");
        Ok(())
    }

    /// Run knowledge agent scenario
    fn run_knowledge_scenario(
        &self,
        _swarm: &mut GpuSwarm,
        _config: &ScenarioConfig,
        _metrics: &mut Vec<PerformanceMetric>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement knowledge scenario execution
        log::warn!("Knowledge scenario execution not yet implemented");
        Ok(())
    }

    /// Check if performance objectives were met
    fn check_objectives(&self, config: &ScenarioConfig, metrics: &[PerformanceMetric]) -> bool {
        for objective in &config.objectives {
            // Find corresponding metric
            let metric_values: Vec<f64> = metrics
                .iter()
                .filter(|m| m.name == objective.metric)
                .map(|m| m.value)
                .collect();

            if metric_values.is_empty() {
                log::warn!("No measurements found for objective: {}", objective.metric);
                return false;
            }

            // Use average value for comparison
            let avg_value = metric_values.iter().sum::<f64>() / metric_values.len() as f64;

            // Check if objective is met
            let met = if objective.maximize {
                avg_value >= objective.target
            } else {
                avg_value <= objective.target
            };

            if !met {
                log::warn!(
                    "Objective not met: {} (target: {}, actual: {})",
                    objective.metric,
                    objective.target,
                    avg_value
                );
                return false;
            }
        }

        true
    }
}
