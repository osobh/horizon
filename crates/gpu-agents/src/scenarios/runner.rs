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
        swarm: &mut GpuSwarm,
        config: &ScenarioConfig,
        metrics: &mut Vec<PerformanceMetric>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use super::reasoning::ReasoningAgentScenario;
        use super::config::ReasoningConfig;

        let start_time = Instant::now();

        // Extract reasoning config from scenario type
        let reasoning_config = match &config.scenario_type {
            ScenarioType::Reasoning { config: cfg } => cfg.clone(),
            _ => return Err("Invalid scenario type for reasoning scenario".into()),
        };

        // Create reasoning scenario
        let mut scenario = ReasoningAgentScenario::new(reasoning_config.clone());
        scenario.initialize_agents(swarm)?;

        let mut step_count = 0;
        let mut total_decisions = 0u64;
        let decision_interval = (1.0 / reasoning_config.decision_frequency * 1000.0) as u64;

        // Run simulation
        while start_time.elapsed() < config.duration {
            // Simulate decision cycle
            let cycle_start = Instant::now();

            // Generate prompts for all agents (simulated - actual LLM calls would go here)
            // Convert agent states to flat f32 slice for prompt generation
            // Format: [x, y, z, vx, vy, goal_x, goal_y] per agent (7 floats each)
            let agent_states_f32: Vec<f32> = (0..swarm.agent_count)
                .flat_map(|i| {
                    vec![
                        (i as f32) * 10.0,  // x position
                        (i as f32) * 5.0,   // y position
                        0.0,                 // z position
                        0.0,                 // vx velocity
                        0.0,                 // vy velocity
                        100.0,               // goal_x
                        100.0,               // goal_y
                    ]
                })
                .collect();
            let prompts = scenario.generate_prompts(&agent_states_f32);

            // Simulate LLM responses (in production, these would come from actual LLM)
            let mock_responses: Vec<String> = prompts.iter().map(|_| {
                // Generate realistic mock response based on agent state
                format!("Move vx=0.{}, vy=0.{}", step_count % 10, (step_count + 1) % 10)
            }).collect();

            // Process responses into velocity commands
            let commands = scenario.process_responses(mock_responses);
            total_decisions += commands.len() as u64;

            // Update agents
            scenario.update(swarm)?;
            step_count += 1;

            // Collect metrics
            if step_count % 100 == 0 {
                let elapsed = start_time.elapsed();

                metrics.push(PerformanceMetric {
                    name: "decisions_per_second".to_string(),
                    value: total_decisions as f64 / elapsed.as_secs_f64(),
                    unit: "decisions/s".to_string(),
                    timestamp: elapsed,
                });

                metrics.push(PerformanceMetric {
                    name: "avg_decision_latency_ms".to_string(),
                    value: cycle_start.elapsed().as_millis() as f64,
                    unit: "ms".to_string(),
                    timestamp: elapsed,
                });
            }

            // Respect decision frequency
            std::thread::sleep(std::time::Duration::from_millis(decision_interval.saturating_sub(
                cycle_start.elapsed().as_millis() as u64
            )));
        }

        log::info!(
            "Reasoning scenario completed: {} steps, {} total decisions",
            step_count,
            total_decisions
        );

        Ok(())
    }

    /// Run knowledge agent scenario
    fn run_knowledge_scenario(
        &self,
        swarm: &mut GpuSwarm,
        config: &ScenarioConfig,
        metrics: &mut Vec<PerformanceMetric>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use super::knowledge::KnowledgeAgentScenario;

        let start_time = Instant::now();

        // Extract knowledge config from scenario type
        let knowledge_config = match &config.scenario_type {
            ScenarioType::Knowledge { config: cfg } => cfg.clone(),
            _ => return Err("Invalid scenario type for knowledge scenario".into()),
        };

        // Create knowledge scenario
        let mut scenario = KnowledgeAgentScenario::new(knowledge_config.clone());
        scenario.initialize_agents(swarm)?;

        let mut step_count = 0;
        let mut total_knowledge_nodes = 0u64;
        let mut total_queries = 0u64;

        // Run simulation
        while start_time.elapsed() < config.duration {
            // Update knowledge graphs
            scenario.update(swarm)?;
            step_count += 1;

            // Simulate knowledge operations
            let agent_count = swarm.agent_count;
            for agent_idx in 0..agent_count.min(10) {
                // Simulate adding knowledge nodes
                if let Some(graph) = scenario.get_graph(agent_idx) {
                    total_knowledge_nodes = graph.len() as u64;
                }

                // Simulate knowledge queries
                total_queries += 1;
            }

            // Collect metrics periodically
            if step_count % 100 == 0 {
                let elapsed = start_time.elapsed();

                metrics.push(PerformanceMetric {
                    name: "total_knowledge_nodes".to_string(),
                    value: scenario.total_nodes() as f64,
                    unit: "nodes".to_string(),
                    timestamp: elapsed,
                });

                metrics.push(PerformanceMetric {
                    name: "knowledge_queries_per_second".to_string(),
                    value: total_queries as f64 / elapsed.as_secs_f64(),
                    unit: "queries/s".to_string(),
                    timestamp: elapsed,
                });

                metrics.push(PerformanceMetric {
                    name: "nodes_per_agent".to_string(),
                    value: scenario.total_nodes() as f64 / agent_count as f64,
                    unit: "nodes/agent".to_string(),
                    timestamp: elapsed,
                });
            }
        }

        log::info!(
            "Knowledge scenario completed: {} steps, {} total nodes, {} queries",
            step_count,
            scenario.total_nodes(),
            total_queries
        );

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
