//! LLM-based reasoning agent scenario implementations

use super::config::{PromptComplexity, ReasoningConfig};
use crate::GpuSwarm;
use std::collections::VecDeque;

/// Velocity command parsed from LLM response
#[derive(Debug, Clone, Copy)]
pub struct VelocityCommand {
    pub vx: f32,
    pub vy: f32,
}

impl Default for VelocityCommand {
    fn default() -> Self {
        Self { vx: 0.0, vy: 0.0 }
    }
}

/// LLM connection state
#[derive(Debug, Clone)]
pub struct LlmConnection {
    pub model: String,
    pub connected: bool,
    pub request_count: usize,
    pub pending_requests: VecDeque<String>,
}

impl LlmConnection {
    /// Create a new LLM connection
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            connected: false,
            request_count: 0,
            pending_requests: VecDeque::new(),
        }
    }

    /// Connect to the LLM service
    pub fn connect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, this would establish a connection to the LLM API
        log::info!("Connecting to LLM model: {}", self.model);
        self.connected = true;
        Ok(())
    }

    /// Queue a request for processing
    pub fn queue_request(&mut self, prompt: String) {
        self.pending_requests.push_back(prompt);
    }

    /// Get pending request count
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
}

/// Reasoning agent scenario executor
pub struct ReasoningAgentScenario {
    config: ReasoningConfig,
    connection: Option<LlmConnection>,
    initialized: bool,
    decision_buffer: Vec<VelocityCommand>,
}

impl ReasoningAgentScenario {
    /// Create a new reasoning agent scenario
    pub fn new(config: ReasoningConfig) -> Self {
        Self {
            config,
            connection: None,
            initialized: false,
            decision_buffer: Vec::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ReasoningConfig {
        &self.config
    }

    /// Initialize reasoning agents
    pub fn initialize_agents(
        &mut self,
        swarm: &mut GpuSwarm,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let agent_count = swarm.agent_count();

        log::info!(
            "Initializing {} reasoning agents with model: {} (complexity: {:?})",
            agent_count,
            self.config.model,
            self.config.prompt_complexity
        );

        // Create LLM connection
        let mut connection = LlmConnection::new(&self.config.model);
        connection.connect()?;
        self.connection = Some(connection);

        // Initialize decision buffer for all agents
        self.decision_buffer = vec![VelocityCommand::default(); agent_count];

        self.initialized = true;

        log::debug!(
            "Reasoning agents initialized with batch_size={}, decision_frequency={}Hz",
            self.config.batch_size,
            self.config.decision_frequency
        );

        Ok(())
    }

    /// Generate prompts based on complexity level
    pub fn generate_prompts(&self, agent_states: &[f32]) -> Vec<String> {
        let prompts = match self.config.prompt_complexity {
            PromptComplexity::Simple => {
                // Simple one-shot prompts
                agent_states
                    .chunks(7)
                    .map(|state| {
                        format!(
                            "Agent at position ({:.2}, {:.2}). Next action?",
                            state.get(0).copied().unwrap_or(0.0),
                            state.get(1).copied().unwrap_or(0.0)
                        )
                    })
                    .collect()
            }
            PromptComplexity::Moderate => {
                // Chain-of-thought reasoning
                agent_states
                    .chunks(7)
                    .map(|state| {
                        format!(
                            "Agent state: position=({:.2}, {:.2}), velocity=({:.2}, {:.2}). \
                             Think step by step about the best next action.",
                            state.get(0).copied().unwrap_or(0.0),
                            state.get(1).copied().unwrap_or(0.0),
                            state.get(3).copied().unwrap_or(0.0),
                            state.get(4).copied().unwrap_or(0.0)
                        )
                    })
                    .collect()
            }
            PromptComplexity::Complex => {
                // Multi-step planning
                agent_states
                    .chunks(7)
                    .map(|state| {
                        format!(
                            "Current state: pos=({:.2}, {:.2}), vel=({:.2}, {:.2}), goal=({:.2}, {:.2}). \
                             Plan a sequence of actions to reach the goal efficiently.",
                            state.get(0).copied().unwrap_or(0.0),
                            state.get(1).copied().unwrap_or(0.0),
                            state.get(3).copied().unwrap_or(0.0),
                            state.get(4).copied().unwrap_or(0.0),
                            state.get(5).copied().unwrap_or(0.0),
                            state.get(6).copied().unwrap_or(0.0)
                        )
                    })
                    .collect()
            }
            PromptComplexity::Advanced => {
                // Full autonomous agent
                agent_states
                    .chunks(7)
                    .map(|state| {
                        format!(
                            "You are an autonomous agent. State: position=({:.2}, {:.2}), \
                             velocity=({:.2}, {:.2}), goal=({:.2}, {:.2}). \
                             Consider your environment, other agents, and long-term objectives. \
                             What is your next action and why?",
                            state.get(0).copied().unwrap_or(0.0),
                            state.get(1).copied().unwrap_or(0.0),
                            state.get(3).copied().unwrap_or(0.0),
                            state.get(4).copied().unwrap_or(0.0),
                            state.get(5).copied().unwrap_or(0.0),
                            state.get(6).copied().unwrap_or(0.0)
                        )
                    })
                    .collect()
            }
        };

        prompts
    }

    /// Process LLM responses into agent actions
    ///
    /// Parses responses looking for velocity/direction commands in common formats:
    /// - "move left/right/up/down"
    /// - "velocity: (x, y)"
    /// - "direction: north/south/east/west"
    /// - Numerical values like "vx=0.5, vy=-0.3"
    pub fn process_responses(&self, responses: Vec<String>) -> Vec<VelocityCommand> {
        responses
            .iter()
            .map(|response| Self::parse_velocity_from_response(response))
            .collect()
    }

    /// Parse a single response into a velocity command
    fn parse_velocity_from_response(response: &str) -> VelocityCommand {
        let response_lower = response.to_lowercase();

        // Try to parse explicit velocity values
        if let Some(cmd) = Self::parse_explicit_velocity(&response_lower) {
            return cmd;
        }

        // Try to parse direction keywords
        if let Some(cmd) = Self::parse_direction_keywords(&response_lower) {
            return cmd;
        }

        // Default: no movement
        VelocityCommand::default()
    }

    /// Parse explicit velocity values like "vx=0.5, vy=-0.3" or "velocity: (0.5, -0.3)"
    fn parse_explicit_velocity(response: &str) -> Option<VelocityCommand> {
        // Try pattern: vx=X, vy=Y
        let vx_pattern = regex::Regex::new(r"vx\s*[=:]\s*([-\d.]+)").ok()?;
        let vy_pattern = regex::Regex::new(r"vy\s*[=:]\s*([-\d.]+)").ok()?;

        if let (Some(vx_cap), Some(vy_cap)) =
            (vx_pattern.captures(response), vy_pattern.captures(response))
        {
            if let (Ok(vx), Ok(vy)) = (
                vx_cap.get(1)?.as_str().parse::<f32>(),
                vy_cap.get(1)?.as_str().parse::<f32>(),
            ) {
                return Some(VelocityCommand {
                    vx: vx.clamp(-1.0, 1.0),
                    vy: vy.clamp(-1.0, 1.0),
                });
            }
        }

        // Try pattern: velocity: (X, Y) or (X, Y)
        let tuple_pattern = regex::Regex::new(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)").ok()?;
        if let Some(cap) = tuple_pattern.captures(response) {
            if let (Ok(vx), Ok(vy)) = (
                cap.get(1)?.as_str().parse::<f32>(),
                cap.get(2)?.as_str().parse::<f32>(),
            ) {
                return Some(VelocityCommand {
                    vx: vx.clamp(-1.0, 1.0),
                    vy: vy.clamp(-1.0, 1.0),
                });
            }
        }

        None
    }

    /// Parse direction keywords like "move left" or "go north"
    fn parse_direction_keywords(response: &str) -> Option<VelocityCommand> {
        let speed = 0.5; // Default movement speed

        // Cardinal directions
        if response.contains("north") || response.contains("up") || response.contains("forward") {
            return Some(VelocityCommand { vx: 0.0, vy: speed });
        }
        if response.contains("south") || response.contains("down") || response.contains("backward")
        {
            return Some(VelocityCommand {
                vx: 0.0,
                vy: -speed,
            });
        }
        if response.contains("east") || response.contains("right") {
            return Some(VelocityCommand { vx: speed, vy: 0.0 });
        }
        if response.contains("west") || response.contains("left") {
            return Some(VelocityCommand {
                vx: -speed,
                vy: 0.0,
            });
        }

        // Diagonal directions
        if response.contains("northeast") {
            return Some(VelocityCommand {
                vx: speed * 0.707,
                vy: speed * 0.707,
            });
        }
        if response.contains("northwest") {
            return Some(VelocityCommand {
                vx: -speed * 0.707,
                vy: speed * 0.707,
            });
        }
        if response.contains("southeast") {
            return Some(VelocityCommand {
                vx: speed * 0.707,
                vy: -speed * 0.707,
            });
        }
        if response.contains("southwest") {
            return Some(VelocityCommand {
                vx: -speed * 0.707,
                vy: -speed * 0.707,
            });
        }

        // Stop keywords
        if response.contains("stop") || response.contains("halt") || response.contains("wait") {
            return Some(VelocityCommand::default());
        }

        None
    }

    /// Update agents with LLM-based decisions
    pub fn update(&mut self, swarm: &mut GpuSwarm) -> Result<(), Box<dyn std::error::Error>> {
        if !self.initialized {
            return Err("Reasoning scenario not initialized".into());
        }

        // GPU kernels handle agent movement with buffered decisions
        swarm.step()?;
        Ok(())
    }

    /// Check if the LLM connection is active
    pub fn is_connected(&self) -> bool {
        self.connection.as_ref().map(|c| c.connected).unwrap_or(false)
    }

    /// Get the decision buffer
    pub fn decisions(&self) -> &[VelocityCommand] {
        &self.decision_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_command_default() {
        let cmd = VelocityCommand::default();
        assert_eq!(cmd.vx, 0.0);
        assert_eq!(cmd.vy, 0.0);
    }

    #[test]
    fn test_parse_direction_keywords() {
        let cmd =
            ReasoningAgentScenario::parse_direction_keywords("I should move north").unwrap();
        assert!(cmd.vy > 0.0);
        assert_eq!(cmd.vx, 0.0);

        let cmd =
            ReasoningAgentScenario::parse_direction_keywords("Going left is best").unwrap();
        assert!(cmd.vx < 0.0);
        assert_eq!(cmd.vy, 0.0);

        let cmd = ReasoningAgentScenario::parse_direction_keywords("Let's stop here").unwrap();
        assert_eq!(cmd.vx, 0.0);
        assert_eq!(cmd.vy, 0.0);
    }

    #[test]
    fn test_parse_explicit_velocity() {
        let cmd =
            ReasoningAgentScenario::parse_explicit_velocity("Set vx=0.5, vy=-0.3").unwrap();
        assert!((cmd.vx - 0.5).abs() < 0.001);
        assert!((cmd.vy - (-0.3)).abs() < 0.001);

        let cmd = ReasoningAgentScenario::parse_explicit_velocity("Move to (0.7, 0.2)").unwrap();
        assert!((cmd.vx - 0.7).abs() < 0.001);
        assert!((cmd.vy - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_process_responses() {
        let config = ReasoningConfig {
            model: "test".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            prompt_complexity: PromptComplexity::Simple,
            batch_size: 10,
            decision_frequency: 10.0,
        };

        let scenario = ReasoningAgentScenario::new(config);

        let responses = vec![
            "I should move north".to_string(),
            "Set vx=0.5, vy=-0.3".to_string(),
            "Random gibberish".to_string(),
        ];

        let commands = scenario.process_responses(responses);
        assert_eq!(commands.len(), 3);
        assert!(commands[0].vy > 0.0); // north
        assert!((commands[1].vx - 0.5).abs() < 0.001); // explicit
        assert_eq!(commands[2].vx, 0.0); // default for unparseable
        assert_eq!(commands[2].vy, 0.0);
    }
}
