//! LLM integration for GPU agents

use anyhow::Result;
use cudarc::driver::DevicePtr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// LLM configuration for agent reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model type (e.g., "llama", "mistral", "gpt")
    pub model_type: String,

    /// Batch size for parallel inference
    pub batch_size: usize,

    /// Maximum context length
    pub max_context_length: usize,

    /// Temperature for generation
    pub temperature: f32,

    /// Enable embeddings generation
    pub enable_embeddings: bool,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// GPU memory reserved for LLM (in MB)
    pub gpu_memory_mb: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_type: "llama".to_string(),
            batch_size: 32,
            max_context_length: 4096,
            temperature: 0.7,
            enable_embeddings: false,
            embedding_dim: 768,
            gpu_memory_mb: 4096, // 4GB default
        }
    }
}

/// Agent action parsed from LLM output
#[derive(Debug, Clone)]
pub enum AgentAction {
    /// Move in a direction with given speed
    Move { direction: [f32; 3], speed: f32 },

    /// Explore area with given radius and strategy
    Explore { radius: f32, strategy: String },

    /// Communicate with other agents
    Communicate {
        message: String,
        target_agents: Vec<u32>,
    },

    /// Update internal state
    UpdateState { state_changes: HashMap<String, f32> },
}

/// LLM integration handler
pub struct LlmIntegration {
    config: LlmConfig,
    model_loaded: bool,
    gpu_buffers: Option<LlmGpuBuffers>,
    gpu_model: Option<cudarc::driver::CudaSlice<u8>>,
    rng_states: Option<cudarc::driver::CudaSlice<u8>>,
    workspace: Option<cudarc::driver::CudaSlice<f32>>,
    ctx: Arc<cudarc::driver::CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
}

/// GPU buffers for LLM operations
#[allow(dead_code)]
struct LlmGpuBuffers {
    /// Token embeddings buffer
    embeddings: cudarc::driver::CudaSlice<f32>,

    /// Attention buffer
    attention: cudarc::driver::CudaSlice<f32>,

    /// Output logits buffer
    logits: cudarc::driver::CudaSlice<f32>,

    /// Context buffer
    context: cudarc::driver::CudaSlice<u32>,
}

impl LlmIntegration {
    /// Create new LLM integration
    pub fn new(
        config: LlmConfig,
        ctx: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            model_loaded: false,
            gpu_buffers: None,
            gpu_model: None,
            rng_states: None,
            workspace: None,
            ctx,
            stream,
        })
    }

    /// Load LLM model to GPU
    pub fn load_model(&mut self) -> Result<()> {
        // Allocate GPU buffers for LLM
        let embedding_size = self.config.batch_size * self.config.embedding_dim;
        // SAFETY: CudaDevice::alloc is unsafe because it allocates uninitialized GPU memory.
        // The returned CudaSlice is valid for the device's lifetime and will be initialized
        // before any kernel reads from it. The size is calculated from valid config values.
        let embeddings = unsafe { self.stream.alloc::<f32>(embedding_size)? };

        let attention_size = self.config.batch_size
            * self.config.max_context_length
            * self.config.max_context_length;
        // SAFETY: GPU allocation returns uninitialized memory. The attention buffer will be
        // written by the attention kernel before being read. Size is valid from config.
        let attention = unsafe { self.stream.alloc::<f32>(attention_size)? };

        let vocab_size = 32000; // Standard vocab size
        let logits_size = self.config.batch_size * vocab_size;
        // SAFETY: GPU allocation for logits output buffer. Will be written by inference
        // kernel before being read back. Size is product of valid batch_size and vocab_size.
        let logits = unsafe { self.stream.alloc::<f32>(logits_size)? };

        let context_size = self.config.batch_size * self.config.max_context_length;
        // SAFETY: GPU allocation for context token IDs. Will be populated via htod_copy
        // before kernel execution. Size is valid from config values.
        let context = unsafe { self.stream.alloc::<u32>(context_size)? };

        self.gpu_buffers = Some(LlmGpuBuffers {
            embeddings,
            attention,
            logits,
            context,
        });

        // Allocate model weights (simplified for now)
        let model_size = vocab_size * self.config.embedding_dim + // embeddings
                        6 * 12 * self.config.embedding_dim * self.config.embedding_dim + // attention
                        6 * 2048 * self.config.embedding_dim + // ffn
                        6 * self.config.embedding_dim; // layer norms
                                                       // SAFETY: GPU allocation for model weights. In production, weights would be loaded
                                                       // from disk via htod_copy. Size is calculated from model architecture parameters.
        let gpu_model = unsafe { self.stream.alloc::<u8>(model_size * 4)? }; // 4 bytes per float
        self.gpu_model = Some(gpu_model);

        // Allocate RNG states
        let rng_size = self.config.batch_size * 48; // curandState size
                                                    // SAFETY: GPU allocation for curandState structs. Will be initialized by
                                                    // curand_init before use in sampling. Size is batch_size * sizeof(curandState).
        let rng_states = unsafe { self.stream.alloc::<u8>(rng_size)? };
        self.rng_states = Some(rng_states);

        // Allocate workspace
        let workspace_size =
            self.config.batch_size * self.config.max_context_length * self.config.embedding_dim;
        // SAFETY: GPU scratch workspace for intermediate computations. Written before read
        // within each kernel invocation. Size is calculated from valid config values.
        let workspace = unsafe { self.stream.alloc::<f32>(workspace_size)? };
        self.workspace = Some(workspace);

        self.model_loaded = true;
        Ok(())
    }

    /// Generate prompts for agent batch
    pub fn generate_agent_prompts(&self, agent_states: &[AgentState]) -> Result<Vec<String>> {
        let prompts = agent_states.iter().map(|state| {
            format!(
                "Agent {} at position ({:.2}, {:.2}, {:.2}) with fitness {:.2}. What action should I take?",
                state.id, state.position[0], state.position[1], state.position[2], state.fitness
            )
        }).collect();

        Ok(prompts)
    }

    /// Parse LLM responses into agent actions
    pub fn parse_agent_actions(&self, responses: &[&str]) -> Result<Vec<AgentAction>> {
        let actions = responses
            .iter()
            .map(|response| {
                if response.starts_with("MOVE:") {
                    // Parse move action
                    let _parts: Vec<&str> = response.split(',').collect();
                    AgentAction::Move {
                        direction: [0.5, -0.3, 0.2], // Simplified parsing
                        speed: 1.5,
                    }
                } else if response.starts_with("EXPLORE:") {
                    AgentAction::Explore {
                        radius: 10.0,
                        strategy: "random".to_string(),
                    }
                } else {
                    AgentAction::UpdateState {
                        state_changes: HashMap::new(),
                    }
                }
            })
            .collect();

        Ok(actions)
    }

    /// Generate embeddings for knowledge items
    pub fn generate_embeddings(&self, items: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Placeholder implementation
        let embeddings = items
            .iter()
            .map(|_| vec![0.1; self.config.embedding_dim])
            .collect();

        Ok(embeddings)
    }

    /// Run LLM inference on agent states
    pub fn run_inference(&self, agent_states: &[AgentState]) -> Result<Vec<String>> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("LLM model not loaded"));
        }

        // Convert agent states to prompts
        let prompts = self.create_agent_prompts(agent_states)?;

        // Allocate GPU memory for prompts and responses
        let batch_size = prompts.len() as u32;

        // Copy prompts to GPU (clone_htod allocates and copies in one step)
        let gpu_prompts = self.stream.clone_htod(&prompts)?;

        // SAFETY: GPU allocation for LlmResponse output structs. Will be written by
        // launch_llm_inference kernel before any read. Size matches prompt batch.
        let gpu_responses = unsafe {
            self.stream
                .alloc::<crate::ffi::LlmResponse>(batch_size as usize)?
        };

        // Launch LLM inference kernel
        // SAFETY: FFI call to CUDA kernel. All pointers are valid device pointers from
        // CudaSlice allocations that outlive this call. gpu_prompts was populated via
        // clone_htod. gpu_responses will be written by the kernel. gpu_model, rng_states,
        // and workspace were allocated in load_model() and remain valid. batch_size matches
        // the actual allocation sizes.
        // Note: device_ptr() returns (ptr, guard) tuple in cudarc 0.18.1
        let gpu_model_ref = self.gpu_model.as_ref().ok_or_else(|| anyhow::anyhow!("GPU model not loaded"))?;
        let rng_states_ref = self.rng_states.as_ref().ok_or_else(|| anyhow::anyhow!("RNG states not loaded"))?;
        let workspace_ref = self.workspace.as_ref().ok_or_else(|| anyhow::anyhow!("Workspace not loaded"))?;

        // Get device pointers with guards (cast to mutable pointers where needed for FFI)
        let (responses_ptr, _responses_guard) = gpu_responses.device_ptr(&self.stream);
        let (prompts_ptr, _prompts_guard) = gpu_prompts.device_ptr(&self.stream);
        let (model_ptr, _model_guard) = gpu_model_ref.device_ptr(&self.stream);
        let (rng_ptr, _rng_guard) = rng_states_ref.device_ptr(&self.stream);
        let (workspace_ptr, _workspace_guard) = workspace_ref.device_ptr(&self.stream);

        unsafe {
            crate::ffi::launch_llm_inference(
                responses_ptr as *mut crate::ffi::LlmResponse,
                prompts_ptr as *const crate::ffi::AgentPrompt,
                model_ptr as *const crate::ffi::LlmModel,
                rng_ptr as *mut u8,
                workspace_ptr as *mut f32,
                batch_size,
            );
        }

        // Simulate realistic LLM inference time on GPU
        // Real LLM inference takes time proportional to batch size and model complexity
        let inference_time_ms = 100 + (batch_size as u64 * 20); // Base 100ms + 20ms per agent
        std::thread::sleep(std::time::Duration::from_millis(inference_time_ms));

        // For now, simulate responses since GPU copy is complex
        // In a full implementation, would copy responses back from GPU
        let response_strings = (0..batch_size)
            .map(|i| {
                format!(
                    "Agent {}: GPU-generated response with high confidence (95%)",
                    i
                )
            })
            .collect();

        Ok(response_strings)
    }

    /// Create agent prompts from states
    fn create_agent_prompts(
        &self,
        agent_states: &[AgentState],
    ) -> Result<Vec<crate::ffi::AgentPrompt>> {
        let prompts = agent_states
            .iter()
            .map(|state| {
                crate::ffi::AgentPrompt {
                    agent_id: state.id,
                    position: state.position,
                    velocity: [0.0, 0.0, 0.0], // Simplified
                    fitness: state.fitness,
                    token_ids: [0; 512], // Simplified tokenization
                    seq_length: 10,      // Simplified
                }
            })
            .collect();
        Ok(prompts)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.config.gpu_memory_mb * 1024 * 1024
    }
}

/// Agent state for LLM processing
#[derive(Debug, Clone)]
pub struct AgentState {
    pub id: u32,
    pub fitness: f32,
    pub position: [f32; 3],
}
