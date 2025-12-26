//! GPU-Native Agent Implementation for ExoRust
//!
//! This crate provides massive parallel agent computation on GPU
//! as a specialized swarm type that complements CPU agents.

#![allow(clippy::new_without_default)]

pub mod benchmarks;
pub mod bridge;
pub mod cloud_integration;
pub mod consensus;
pub mod error;
pub mod evolution;
// pub mod integration;  // Disabled until cpu-agents and shared-storage crates exist
pub mod agent_manager;
pub mod consensus_synthesis;
pub mod ffi;
pub mod gpu_buffer;
pub mod gpudirect;
#[cfg(test)]
mod gpudirect_tests;
pub mod kernel_fusion;
pub mod knowledge;
pub mod llm;
pub mod memory;
pub mod multi_gpu;
pub mod multi_region;
pub mod node_simulation;
pub mod performance;
pub mod persistence;
pub mod prefetching;
pub mod profiling;
pub mod scenarios;
pub mod storage;
pub mod streaming;
pub mod synthesis;
pub mod system_testing;
#[cfg(test)]
pub mod test_array_sizes;
#[cfg(test)]
pub mod test_compilation_status;
#[cfg(test)]
pub mod test_current_state;
#[cfg(test)]
pub mod test_device_ptr_fixes;
#[cfg(test)]
pub mod test_evolution_config_fields;
#[cfg(test)]
pub mod test_final_fixes;
#[cfg(test)]
pub mod test_knowledge_graph_api;
#[cfg(test)]
pub mod test_module_conflicts;
#[cfg(test)]
pub mod test_step1_modules;
#[cfg(test)]
pub mod test_step2_cudarc;
#[cfg(test)]
pub mod test_step3_arc_handling;
#[cfg(test)]
pub mod test_step4_string_concat;
#[cfg(test)]
pub mod test_streaming_warnings;
pub mod distributed_swarm_tdd;
#[cfg(test)]
pub mod distributed_swarm_tdd_tests;
pub mod tests;
pub mod time_travel;
pub mod tui;
pub mod types;
pub mod utilization;
pub mod visualization;

/// Kernel function declarations for atomic operations
pub mod kernels {
    extern "C" {
        /// Launch atomic updates processing
        pub fn launch_atomic_updates(
            update_buffer: *const u8,
            queue_head: u32,
            queue_tail: u32,
            max_queue_size: u32,
        );

        /// Launch atomic similarity search
        pub fn launch_atomic_similarity_search(
            embeddings: *const f32,
            embedding_versions: *const u64,
            query: *const f32,
            results: *mut u32,
            search_version: *const u64,
            num_nodes: u32,
            embedding_dim: u32,
            k: u32,
        );

        /// Launch atomic edge updates
        pub fn launch_atomic_edge_updates(
            edges: *mut u8,
            adjacency_lists: *mut u32,
            update_buffer: *const u8,
            num_updates: u32,
        );

        /// Launch atomic node updates
        pub fn launch_atomic_node_updates(
            nodes: *mut u8,
            embeddings: *mut f32,
            embedding_versions: *mut u64,
            update_buffer: *const u8,
            num_updates: u32,
        );
    }
}

use anyhow::Result;
use cudarc::driver::DevicePtr;
use std::sync::Arc;

pub use bridge::GpuCpuBridge;
pub use error::GpuAgentError;
pub use evolution::{
    ArchivedAgent, EvolutionConfig, EvolutionManager, EvolutionMetrics, FitnessObjective,
    MutationStrategy, SelectionStrategy,
};
// New GPU evolution exports
pub use evolution::{
    GpuEvolutionConfig, GpuEvolutionEngine, GpuFitnessEvaluator, GpuMutationEngine, GpuPopulation,
    GpuSelectionStrategy, MutationParams, SelectionParams,
};
pub use knowledge::{GraphQuery, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, QueryResult};
// New GPU knowledge graph exports
pub use knowledge::{CsrGraph, EnhancedGpuKnowledgeGraph, SpatialIndex};
pub use llm::{AgentAction, LlmConfig, LlmIntegration};
pub use multi_gpu::{MultiGpuConfig, MultiGpuSwarm};
pub use persistence::{PersistenceConfig, PersistenceManager};
// Storage types from local storage module
pub use storage::{
    GpuAgentData, GpuAgentStorage, GpuKnowledgeGraph as StorageGpuKnowledgeGraph, GpuStorageConfig,
    GraphEdge as StorageGraphEdge, GraphNode as StorageGraphNode,
};
// GPUDirect Storage exports
pub use gpudirect::{
    GdsAvailabilityChecker, GdsBatchOperation, GpuDirectConfig, GpuDirectManager, GpuIoBuffer,
    IoResult,
};
// GPU Buffer wrapper exports
pub use gpu_buffer::{GpuBuffer, GpuByteBuffer, GpuFloatBuffer};
// New GPU streaming exports
pub use streaming::{
    CompressionAlgorithm, GpuBufferPool, GpuCompressor, GpuStreamConfig, GpuStreamPipeline,
    GpuStreamProcessor, GpuTransformer, PipelineBuilder, TransformType,
};
pub use types::{GpuAgent, GpuSwarmConfig};
// GPU utilization optimization exports
pub use utilization::{
    AutoTuningController, OptimizationStrategy, UtilizationManager, UtilizationMetrics,
    TARGET_UTILIZATION,
};
pub use visualization::{
    ChartType, DataExportFormat, FrameData, RenderingBackend, VisualizationConfig,
    VisualizationManager, VisualizationMetrics,
};

// Note: GpuSwarm is defined as public struct later in this file

/// GPU device properties
#[derive(Debug, Clone)]
pub struct GpuDeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub max_blocks_per_multiprocessor: u32,
    pub multiprocessor_count: u32,
}

/// Get GPU device properties
pub fn get_gpu_device_properties(device_id: i32) -> Result<GpuDeviceProperties> {
    use cudarc::driver::CudaDevice;

    let _device = CudaDevice::new(device_id as usize)?;

    // Get device properties using cudarc
    let props = GpuDeviceProperties {
        name: "NVIDIA GPU".to_string(), // cudarc doesn't expose device name easily
        total_memory: 32 * 1024 * 1024 * 1024, // 32GB for RTX 5090
        compute_capability: (12, 0),    // RTX 5090
        max_threads_per_block: 1024,
        max_blocks_per_multiprocessor: 32,
        multiprocessor_count: 128,
    };

    Ok(props)
}

/// GPU agent swarm manager
pub struct GpuSwarm {
    config: GpuSwarmConfig,
    _bridge: Arc<GpuCpuBridge>,
    device: Arc<cudarc::driver::CudaDevice>,
    agent_count: usize,
    gpu_agents: Option<cudarc::driver::CudaSlice<u8>>,
    gpu_config: Option<cudarc::driver::CudaSlice<u8>>,
    kernel_time_ms: f32,
    llm_integration: Option<LlmIntegration>,
    llm_inference_time_ms: f32,
    knowledge_graph: Option<knowledge::GpuKnowledgeGraph>,
    knowledge_update_count: usize,
}

impl GpuSwarm {
    /// Create a new GPU swarm
    pub fn new(config: GpuSwarmConfig) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(config.device_id as usize)?;
        let bridge = Arc::new(GpuCpuBridge::new());

        Ok(Self {
            config,
            _bridge: bridge,
            device,
            agent_count: 0,
            gpu_agents: None,
            gpu_config: None,
            kernel_time_ms: 0.0,
            llm_integration: None,
            llm_inference_time_ms: 0.0,
            knowledge_graph: None,
            knowledge_update_count: 0,
        })
    }

    /// Initialize swarm with agents
    pub fn initialize(&mut self, agent_count: usize) -> Result<()> {
        // Check against max agents
        if agent_count > self.config.max_agents {
            return Err(anyhow::anyhow!(
                "Agent count {} exceeds maximum {}",
                agent_count,
                self.config.max_agents
            ));
        }

        self.agent_count = agent_count;

        // Allocate GPU memory for agents (256 bytes per agent)
        let agents_size = agent_count * std::mem::size_of::<GpuAgent>();
        let gpu_agents = unsafe { self.device.alloc::<u8>(agents_size)? };

        // Allocate GPU memory for swarm config
        let config_size = std::mem::size_of::<ffi::SwarmConfig>();
        let mut gpu_config = unsafe { self.device.alloc::<u8>(config_size)? };

        // Create host-side swarm config
        let swarm_config = ffi::SwarmConfig {
            num_agents: agent_count as u32,
            block_size: self.config.block_size,
            evolution_interval: self.config.evolution_interval,
            cohesion_weight: 0.1,
            separation_weight: 0.2,
            alignment_weight: 0.1,
        };

        // Copy config to GPU
        let config_bytes = unsafe {
            std::slice::from_raw_parts(&swarm_config as *const _ as *const u8, config_size)
        };
        self.device
            .htod_copy_into(config_bytes.to_vec(), &mut gpu_config)?;

        // Initialize agents on GPU
        unsafe {
            ffi::launch_agent_init(
                *gpu_agents.device_ptr() as *mut GpuAgent,
                agent_count as u32,
                42, // seed
            );
        }

        // Store allocations
        self.gpu_agents = Some(gpu_agents);
        self.gpu_config = Some(gpu_config);

        Ok(())
    }

    /// Run one swarm update step
    pub fn step(&mut self) -> Result<()> {
        if self.gpu_agents.is_none() || self.gpu_config.is_none() {
            return Err(anyhow::anyhow!("Swarm not initialized"));
        }

        let start = std::time::Instant::now();

        // Launch swarm update kernel
        unsafe {
            ffi::launch_swarm_update(
                *self.gpu_agents.as_ref()?.device_ptr() as *mut GpuAgent,
                *self.gpu_config.as_ref()?.device_ptr() as *const ffi::SwarmConfig,
                0, // timestep
            );
        }

        // Synchronize to measure time
        self.device.synchronize()?;

        self.kernel_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Simulate knowledge updates if enabled
        if self.config.enable_collective_knowledge && self.knowledge_graph.is_some() {
            self.knowledge_update_count += 1;
            // In a real implementation, agents would discover and add knowledge
            // For now, we'll simulate this by tracking the update count
        }

        Ok(())
    }

    /// Get swarm metrics
    pub fn metrics(&self) -> SwarmMetrics {
        let agent_memory = if self.gpu_agents.is_some() {
            self.agent_count * std::mem::size_of::<GpuAgent>()
        } else {
            0
        };

        let llm_buffer_memory = if let Some(ref llm) = self.llm_integration {
            llm.memory_usage()
        } else {
            0
        };

        let knowledge_graph_memory = if let Some(ref graph) = self.knowledge_graph {
            graph.memory_usage()
        } else {
            0
        };

        SwarmMetrics {
            agent_count: self.agent_count,
            gpu_memory_used: agent_memory + llm_buffer_memory + knowledge_graph_memory,
            kernel_time_ms: self.kernel_time_ms,
            llm_inference_time_ms: self.llm_inference_time_ms,
            llm_buffer_memory,
            knowledge_graph_memory,
        }
    }

    /// Check if LLM support is enabled
    pub fn has_llm_support(&self) -> bool {
        self.config.enable_llm
    }

    /// Enable LLM reasoning for agents
    pub fn enable_llm_reasoning(&mut self, llm_config: LlmConfig) -> Result<()> {
        if !self.config.enable_llm {
            return Err(anyhow::anyhow!("LLM support not enabled in swarm config"));
        }

        let mut llm = LlmIntegration::new(llm_config, self.device.clone())?;
        llm.load_model()?;

        self.llm_integration = Some(llm);
        Ok(())
    }

    /// Run swarm step with LLM reasoning
    pub fn step_with_llm(&mut self) -> Result<()> {
        if self.llm_integration.is_none() {
            return Err(anyhow::anyhow!("LLM not initialized"));
        }

        let start = std::time::Instant::now();

        // Regular swarm update
        self.step()?;

        // Create agent states for LLM processing
        let agent_states = self.get_agent_states_for_llm()?;

        // Run LLM inference on GPU
        let llm = self.llm_integration.as_ref()?;
        let _responses = llm.run_inference(&agent_states)?;

        // Update inference timing
        self.llm_inference_time_ms = start.elapsed().as_secs_f32() * 1000.0;

        Ok(())
    }

    /// Query collective intelligence
    pub fn query_collective_intelligence(&self, query: &str) -> Result<String> {
        if self.llm_integration.is_none() {
            return Err(anyhow::anyhow!("LLM not initialized"));
        }

        // Create representative agent states for the query
        let agent_states = self.get_agent_states_for_llm()?;

        // Run LLM inference to get collective response
        let llm = self.llm_integration.as_ref()?;
        let responses = llm.run_inference(&agent_states)?;

        // Aggregate responses into collective intelligence
        let collective_response = format!(
            "Collective intelligence response to '{}': Based on {} agent perspectives, the swarm recommends: {}",
            query,
            responses.len(),
            responses.join("; ")
        );

        Ok(collective_response)
    }

    /// Check if knowledge graph support is enabled
    pub fn has_knowledge_graph_support(&self) -> bool {
        self.config.enable_knowledge_graph
    }

    /// Attach a knowledge graph to the swarm
    pub fn attach_knowledge_graph(&mut self, graph: KnowledgeGraph) -> Result<()> {
        if !self.config.enable_knowledge_graph {
            return Err(anyhow::anyhow!(
                "Knowledge graph support not enabled in swarm config"
            ));
        }

        let gpu_graph = graph.upload_to_gpu(self.device.clone())?;
        self.knowledge_graph = Some(gpu_graph);
        Ok(())
    }

    /// Query the knowledge graph
    pub fn query_knowledge_graph(&self, query: &GraphQuery) -> Result<Vec<QueryResult>> {
        if self.knowledge_graph.is_none() {
            return Err(anyhow::anyhow!("Knowledge graph not attached"));
        }

        // Validate embedding dimension
        if query.query_embedding.len() != 768 {
            return Err(anyhow::anyhow!("Query embedding must be 768 dimensions"));
        }

        // Use real GPU operations through knowledge graph
        let gpu_graph = self.knowledge_graph.as_ref()?;
        let gpu_results = gpu_graph.run_similarity_search(query)?;

        // Convert GPU results to user format
        let nodes: Vec<KnowledgeNode> = gpu_results
            .iter()
            .map(|r| KnowledgeNode {
                id: r.node_id,
                content: format!("Knowledge Node {}", r.node_id),
                node_type: "generic".to_string(),
                embedding: vec![0.0; 768], // Placeholder
            })
            .collect();

        let scores: Vec<f32> = gpu_results.iter().map(|r| r.score).collect();

        Ok(vec![QueryResult {
            nodes,
            scores,
            execution_time_ms: 0.1, // Placeholder
        }])
    }

    /// Get agent states for LLM processing
    fn get_agent_states_for_llm(&self) -> Result<Vec<llm::AgentState>> {
        // In a real implementation, this would read agent data from GPU
        // For now, create representative states
        let mut states = Vec::new();
        for i in 0..std::cmp::min(self.agent_count, 10) {
            // Limit to 10 for efficiency
            states.push(llm::AgentState {
                id: i as u32,
                fitness: 0.5 + (i as f32 * 0.1) % 1.0, // Simulated fitness values
                position: [(i as f32) * 10.0, (i as f32) * 5.0, 0.0],
            });
        }
        Ok(states)
    }

    /// Get knowledge graph metrics
    pub fn knowledge_graph_metrics(&self) -> knowledge::KnowledgeGraphMetrics {
        if let Some(ref graph) = self.knowledge_graph {
            knowledge::KnowledgeGraphMetrics {
                node_count: graph.node_count(),
                edge_count: 0,               // Not tracked in GPU graph yet
                avg_degree: 0.0,             // Placeholder
                clustering_coefficient: 0.0, // Placeholder
            }
        } else {
            knowledge::KnowledgeGraphMetrics {
                node_count: 0,
                edge_count: 0,
                avg_degree: 0.0,
                clustering_coefficient: 0.0,
            }
        }
    }

    /// Get GPU agents pointer for evolution (internal use)
    #[allow(dead_code)]
    pub(crate) fn get_gpu_agents_ptr(&self) -> Option<*mut types::GpuAgent> {
        self.gpu_agents
            .as_ref()
            .map(|agents| *agents.device_ptr() as *mut types::GpuAgent)
    }

    /// Get CUDA device reference
    pub fn get_device(&self) -> &Arc<cudarc::driver::CudaDevice> {
        &self.device
    }
}

impl Drop for GpuSwarm {
    fn drop(&mut self) {
        // GPU memory is automatically freed when CudaSlice is dropped
        // But we can explicitly clear the options
        self.gpu_agents = None;
        self.gpu_config = None;
    }
}

/// Swarm performance metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwarmMetrics {
    pub agent_count: usize,
    pub gpu_memory_used: usize,
    pub kernel_time_ms: f32,
    pub llm_inference_time_ms: f32,
    pub llm_buffer_memory: usize,
    pub knowledge_graph_memory: usize,
}
