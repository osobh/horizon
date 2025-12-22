//! GPU agent types and structures

use serde::{Deserialize, Serialize};

/// GPU agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSwarmConfig {
    pub device_id: i32,
    pub max_agents: usize,
    pub block_size: u32,
    pub shared_memory_size: usize,
    pub evolution_interval: u32,
    pub enable_llm: bool,
    pub enable_collective_intelligence: bool,
    pub enable_knowledge_graph: bool,
    pub enable_collective_knowledge: bool,
}

impl Default for GpuSwarmConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            max_agents: 1_000_000,
            block_size: 256,
            shared_memory_size: 48 * 1024, // 48KB
            evolution_interval: 100,
            enable_llm: false,
            enable_collective_intelligence: false,
            enable_knowledge_graph: false,
            enable_collective_knowledge: false,
        }
    }
}

/// GPU agent state (mirrors GPU struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgent {
    // Position and movement (24 bytes)
    pub position: [f32; 3],
    pub velocity: [f32; 3],

    // Core state (16 bytes)
    pub fitness: f32,
    pub state: u32,
    pub agent_type: u32,
    pub swarm_id: u32,

    // Memory offsets (16 bytes)
    pub working_memory_offset: u32,
    pub episodic_memory_offset: u32,
    pub semantic_memory_offset: u32,
    pub genome_offset: u32,

    // Communication (72 bytes)
    pub neighbors: [u32; 2],
    pub shared_data: [f32; 16],

    // Padding to 256 bytes
    pub _padding: [u8; 128],
}

impl Default for GpuAgent {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            fitness: 0.0,
            state: 0,
            agent_type: 0,
            swarm_id: 0,
            working_memory_offset: 0,
            episodic_memory_offset: 0,
            semantic_memory_offset: 0,
            genome_offset: 0,
            neighbors: [0; 2],
            shared_data: [0.0; 16],
            _padding: [0; 128],
        }
    }
}

// Verify size is exactly 256 bytes
const _: () = assert!(std::mem::size_of::<GpuAgent>() == 256);
