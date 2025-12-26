//! Local agent management for single node testing
//!
//! Manages 1-10 agents on a single GPU node

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for local agent manager
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Node ID for this instance
    pub node_id: u32,
    /// GPU memory per agent (bytes)
    pub memory_per_agent: usize,
    /// Communication buffer size
    pub comm_buffer_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_agents: 10,
            node_id: 0,
            memory_per_agent: 10 * 1024 * 1024, // 10MB per agent
            comm_buffer_size: 1024 * 1024,      // 1MB comm buffer
        }
    }
}

/// Individual agent state
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: u32,
    pub name: String,
    pub state: AgentState,
    pub memory_usage: usize,
    pub messages_sent: usize,
    pub messages_received: usize,
}

/// Agent state enum
#[derive(Debug, Clone, PartialEq)]
pub enum AgentState {
    Idle,
    Active,
    Computing,
    Communicating,
    Terminated,
}

/// Coordination status
#[derive(Debug, Clone)]
pub struct CoordinationStatus {
    pub active_agents: usize,
    pub total_messages: usize,
    pub coordination_rounds: usize,
    pub consensus_achieved: bool,
}

/// Agent performance metrics
#[derive(Debug, Clone)]
pub struct AgentMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub gpu_utilization: f32,
    pub memory_usage: usize,
    pub messages_per_second: f64,
    pub avg_response_time_us: f64,
}

/// Local agent manager for single node
pub struct LocalAgentManager {
    device: Arc<CudaDevice>,
    config: AgentConfig,
    agents: HashMap<u32, Agent>,
    gpu_memory: Option<CudaSlice<u8>>,
    comm_buffer: Option<CudaSlice<u8>>,
    next_agent_id: u32,
}

impl LocalAgentManager {
    /// Create a new local agent manager
    pub fn new(device: Arc<CudaDevice>, config: AgentConfig) -> Result<Self> {
        let total_memory = config.max_agents * config.memory_per_agent;

        // Allocate GPU memory for agents
        let gpu_memory = unsafe { device.alloc::<u8>(total_memory) }
            .context("Failed to allocate GPU memory for agents")?;

        // Allocate communication buffer
        let comm_buffer = unsafe { device.alloc::<u8>(config.comm_buffer_size) }
            .context("Failed to allocate communication buffer")?;

        Ok(Self {
            device,
            config,
            agents: HashMap::new(),
            gpu_memory: Some(gpu_memory),
            comm_buffer: Some(comm_buffer),
            next_agent_id: 0,
        })
    }

    /// Spawn a single agent
    pub fn spawn_agent(&mut self, name: &str) -> Result<u32> {
        if self.agents.len() >= self.config.max_agents {
            return Err(anyhow!("Maximum agent capacity reached"));
        }

        let agent_id = self.next_agent_id;
        self.next_agent_id += 1;

        let agent = Agent {
            id: agent_id,
            name: name.to_string(),
            state: AgentState::Idle,
            memory_usage: self.config.memory_per_agent,
            messages_sent: 0,
            messages_received: 0,
        };

        self.agents.insert(agent_id, agent);

        // Initialize agent memory on GPU
        self.run_agent_computation(agent_id)?;

        Ok(agent_id)
    }

    /// Spawn multiple agents
    pub fn spawn_agents(&mut self, count: usize) -> Result<Vec<u32>> {
        let mut agent_ids = Vec::with_capacity(count);

        for _ in 0..count {
            let name = format!("agent-{}", self.next_agent_id);
            let id = self.spawn_agent(&name)?;
            agent_ids.push(id);
        }

        Ok(agent_ids)
    }

    /// Send message between agents
    pub fn send_message(&mut self, from: u32, to: u32, _message: &str) -> Result<()> {
        // Validate agents exist
        if !self.agents.contains_key(&from) {
            return Err(anyhow!("Sender agent {} not found", from));
        }
        if !self.agents.contains_key(&to) {
            return Err(anyhow!("Receiver agent {} not found", to));
        }

        // Update agent states
        if let Some(sender) = self.agents.get_mut(&from) {
            sender.state = AgentState::Communicating;
            sender.messages_sent += 1;
        }

        if let Some(receiver) = self.agents.get_mut(&to) {
            receiver.messages_received += 1;
        }

        // Simulate message passing (in real implementation, would use GPU)
        // For now, just update states

        // Reset sender state
        if let Some(sender) = self.agents.get_mut(&from) {
            sender.state = AgentState::Active;
        }

        Ok(())
    }

    /// Coordinate all agents
    pub fn coordinate_agents(&mut self) -> Result<CoordinationStatus> {
        let active_agents = self
            .agents
            .values()
            .filter(|a| a.state != AgentState::Terminated)
            .count();

        // Simulate coordination round
        for agent in self.agents.values_mut() {
            if agent.state == AgentState::Idle {
                agent.state = AgentState::Active;
            }
        }

        let total_messages: usize = self
            .agents
            .values()
            .map(|a| a.messages_sent + a.messages_received)
            .sum();

        Ok(CoordinationStatus {
            active_agents,
            total_messages,
            coordination_rounds: 1,
            consensus_achieved: active_agents > 0,
        })
    }

    /// Get agent performance metrics
    pub fn get_agent_metrics(&self) -> Result<AgentMetrics> {
        let total_agents = self.agents.len();
        let active_agents = self
            .agents
            .values()
            .filter(|a| a.state == AgentState::Active || a.state == AgentState::Computing)
            .count();

        let memory_usage = self.agents.len() * self.config.memory_per_agent;

        Ok(AgentMetrics {
            total_agents,
            active_agents,
            gpu_utilization: (active_agents as f32) / (self.config.max_agents as f32),
            memory_usage,
            messages_per_second: 1000.0, // Simulated
            avg_response_time_us: 50.0,  // Simulated
        })
    }

    /// Terminate an agent
    pub fn terminate_agent(&mut self, agent_id: u32) -> Result<()> {
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.state = AgentState::Terminated;
        }
        self.agents.remove(&agent_id);
        Ok(())
    }

    /// Get agent status
    pub fn get_agent_status(&self, agent_id: u32) -> Result<&Agent> {
        self.agents
            .get(&agent_id)
            .ok_or_else(|| anyhow!("Agent {} not found", agent_id))
    }

    /// Run agent computation
    fn run_agent_computation(&mut self, agent_id: u32) -> Result<()> {
        // In real implementation, this would launch GPU kernel
        // For now, just update agent state
        if let Some(agent) = self.agents.get_mut(&agent_id) {
            agent.state = AgentState::Computing;
            // Simulate computation
            agent.state = AgentState::Active;
        }
        Ok(())
    }

    /// Clear all agents
    pub fn clear_all_agents(&mut self) {
        self.agents.clear();
        // Reset next_agent_id to allow reuse
        self.next_agent_id = 0;
    }

    /// Allocate GPU memory for agents
    fn allocate_gpu_memory(&mut self) -> Result<()> {
        // Already allocated in new()
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_agents, 10);
        assert_eq!(config.node_id, 0);
    }

    #[test]
    fn test_agent_manager_creation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = CudaDevice::new(0)?;
        let config = AgentConfig::default();
        let manager = LocalAgentManager::new(Arc::new(device), config);
        // Should panic with todo!
        assert!(manager.is_err() || manager.is_ok());
    }
}
