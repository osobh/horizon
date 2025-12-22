//! FFI bindings to CUDA kernels

use crate::types::GpuAgent;

/// Swarm configuration structure (matches CUDA SwarmConfig)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwarmConfig {
    pub num_agents: u32,
    pub block_size: u32,
    pub evolution_interval: u32,
    pub cohesion_weight: f32,
    pub separation_weight: f32,
    pub alignment_weight: f32,
}

unsafe impl bytemuck::Pod for SwarmConfig {}
unsafe impl bytemuck::Zeroable for SwarmConfig {}

/// LLM model structure on GPU (matches CUDA LLMModel)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlmModel {
    pub token_embeddings: *mut f32,  // [VOCAB_SIZE, EMBEDDING_DIM]
    pub attention_weights: *mut f32, // [NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM, EMBEDDING_DIM]
    pub ffn_weights: *mut f32,       // [NUM_LAYERS, HIDDEN_DIM, EMBEDDING_DIM]
    pub layer_norms: *mut f32,       // [NUM_LAYERS, EMBEDDING_DIM]
    pub output_projection: *mut f32, // [EMBEDDING_DIM, VOCAB_SIZE]
}

/// Agent prompt context (matches CUDA AgentPrompt)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AgentPrompt {
    pub agent_id: u32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub fitness: f32,
    pub token_ids: [u32; 512], // MAX_SEQ_LEN
    pub seq_length: u32,
}

unsafe impl bytemuck::Pod for AgentPrompt {}
unsafe impl bytemuck::Zeroable for AgentPrompt {}

// Implement DeviceRepr for AgentPrompt
unsafe impl cudarc::driver::DeviceRepr for AgentPrompt {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// LLM inference result (matches CUDA LLMResponse)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlmResponse {
    pub agent_id: u32,
    pub response_tokens: [u32; 512], // MAX_SEQ_LEN
    pub response_length: u32,
    pub logits: [f32; 32000], // VOCAB_SIZE
    pub confidence: f32,
}

unsafe impl bytemuck::Pod for LlmResponse {}
unsafe impl bytemuck::Zeroable for LlmResponse {}

// Implement DeviceRepr for LlmResponse
unsafe impl cudarc::driver::DeviceRepr for LlmResponse {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// GPU knowledge node structure (matches CUDA GPUKnowledgeNode)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GPUKnowledgeNode {
    pub id: u32,
    pub node_type_hash: u32,
    pub embedding: [f32; 768], // EMBEDDING_DIM
    pub edge_offset: u32,
    pub edge_count: u32,
}

unsafe impl bytemuck::Pod for GPUKnowledgeNode {}
unsafe impl bytemuck::Zeroable for GPUKnowledgeNode {}

unsafe impl cudarc::driver::DeviceRepr for GPUKnowledgeNode {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// GPU adjacency list entry (matches CUDA GPUAdjacencyEntry)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GPUAdjacencyEntry {
    pub target_id: u32,
    pub weight: f32,
    pub relationship_hash: u32,
}

unsafe impl bytemuck::Pod for GPUAdjacencyEntry {}
unsafe impl bytemuck::Zeroable for GPUAdjacencyEntry {}

unsafe impl cudarc::driver::DeviceRepr for GPUAdjacencyEntry {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// Knowledge graph query structure (matches CUDA GPUGraphQuery)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GPUGraphQuery {
    pub query_embedding: [f32; 768], // EMBEDDING_DIM
    pub max_results: u32,
    pub similarity_threshold: f32,
    pub query_type: u32, // 0=similarity, 1=path_finding, 2=subgraph
}

unsafe impl bytemuck::Pod for GPUGraphQuery {}
unsafe impl bytemuck::Zeroable for GPUGraphQuery {}

unsafe impl cudarc::driver::DeviceRepr for GPUGraphQuery {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// Knowledge graph query result (matches CUDA GPUQueryResult)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GPUQueryResult {
    pub node_id: u32,
    pub score: f32,
    pub path_length: u32,
    pub path: [u32; 20], // MAX_PATH_LENGTH
}

unsafe impl bytemuck::Pod for GPUQueryResult {}
unsafe impl bytemuck::Zeroable for GPUQueryResult {}

unsafe impl cudarc::driver::DeviceRepr for GPUQueryResult {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

extern "C" {
    /// Initialize GPU agents
    pub fn launch_agent_init(agents: *mut GpuAgent, num_agents: u32, seed: u32);

    /// Update GPU agents
    pub fn launch_agent_update(agents: *mut GpuAgent, num_agents: u32, dt: f32, timestep: u32);

    /// Update swarm behavior
    pub fn launch_swarm_update(agents: *mut GpuAgent, config: *const SwarmConfig, timestep: u32);

    /// Initialize working memory
    pub fn launch_memory_init(
        memories: *mut u8, // GPUWorkingMemory
        num_agents: u32,
    );

    /// Launch LLM inference
    pub fn launch_llm_inference(
        responses: *mut LlmResponse,
        prompts: *const AgentPrompt,
        model: *const LlmModel,
        rng_states: *mut u8, // curandState
        workspace: *mut f32,
        batch_size: u32,
    );

    /// Launch LLM embedding lookup
    pub fn launch_llm_embedding_lookup(
        embeddings_out: *mut f32,
        token_ids: *const u32,
        embedding_table: *const f32,
        batch_size: u32,
        seq_len: u32,
        embedding_dim: u32,
    );

    /// Launch LLM attention computation
    pub fn launch_llm_attention(
        attention_out: *mut f32,
        query: *const f32,
        key: *const f32,
        value: *const f32,
        attention_weights: *const f32,
        batch_size: u32,
        seq_len: u32,
        embedding_dim: u32,
        num_heads: u32,
    );

    /// Launch similarity search in knowledge graph
    pub fn launch_similarity_search(
        nodes: *const GPUKnowledgeNode,
        num_nodes: u32,
        query: *const GPUGraphQuery,
        results: *mut GPUQueryResult,
        result_count: *mut u32,
    );

    /// Launch k-nearest neighbors search
    pub fn launch_knn_search(
        nodes: *const GPUKnowledgeNode,
        num_nodes: u32,
        query: *const GPUGraphQuery,
        results: *mut GPUQueryResult,
        k: u32,
    );

    /// Launch BFS pathfinding
    pub fn launch_bfs_pathfinding(
        nodes: *const GPUKnowledgeNode,
        adjacency_list: *const GPUAdjacencyEntry,
        num_nodes: u32,
        start_node_id: u32,
        target_node_id: u32,
        result: *mut GPUQueryResult,
        found: *mut bool,
    );

    /// Launch subgraph extraction
    pub fn launch_subgraph_extraction(
        nodes: *const GPUKnowledgeNode,
        adjacency_list: *const GPUAdjacencyEntry,
        num_nodes: u32,
        seed_nodes: *const u32,
        num_seed_nodes: u32,
        max_depth: u32,
        results: *mut GPUQueryResult,
        result_count: *mut u32,
    );

    /// Launch centrality computation
    pub fn launch_centrality_computation(
        nodes: *const GPUKnowledgeNode,
        adjacency_list: *const GPUAdjacencyEntry,
        num_nodes: u32,
        centrality_scores: *mut f32,
    );

    /// Launch evolution kernel
    pub fn launch_evolution(
        agents: *mut GpuAgent,
        genomes: *mut f32,
        rng_states: *mut u8, // curandState
        num_agents: u32,
        generation: u32,
    );

    /// Initialize RNG states for evolution
    pub fn launch_init_rng(
        states: *mut u8, // curandState
        num_agents: u32,
        seed: u32,
    );

    /// Launch multi-objective fitness evaluation
    pub fn launch_multi_objective_fitness(
        agents: *mut GpuAgent,
        fitness_vectors: *mut f32,
        num_agents: u32,
        num_objectives: u32,
        kernel_time_ms: f32,
        memory_usage_mb: f32,
    );

    /// Launch NSGA-II selection
    pub fn launch_nsga2_selection(
        agents: *mut GpuAgent,
        fitness_vectors: *mut f32,
        selected_indices: *mut u32,
        num_agents: u32,
        num_objectives: u32,
        population_size: u32,
    );

    /// Launch adaptive mutation
    pub fn launch_adaptive_mutation(
        genomes: *mut f32,
        rng_states: *mut u8, // curandState
        num_agents: u32,
        base_mutation_rate: f32,
        population_diversity: f32,
    );

    // =============================================================================
    // String Operations Kernels
    // =============================================================================

    /// Launch string uppercase conversion kernel
    pub fn launch_string_uppercase_kernel(
        input_data: *const u8,
        output_data: *mut u8,
        string_offsets: *const u32,
        string_lengths: *const u32,
        num_strings: u32,
        max_string_length: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    /// Launch string lowercase conversion kernel
    pub fn launch_string_lowercase_kernel(
        input_data: *const u8,
        output_data: *mut u8,
        string_offsets: *const u32,
        string_lengths: *const u32,
        num_strings: u32,
        max_string_length: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    /// Launch string reverse kernel
    pub fn launch_string_reverse_kernel(
        input_data: *const u8,
        output_data: *mut u8,
        string_offsets: *const u32,
        string_lengths: *const u32,
        num_strings: u32,
        max_string_length: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    /// Launch pattern matching kernel
    pub fn launch_string_pattern_match_kernel(
        input_data: *const u8,
        match_results: *mut u8,
        string_offsets: *const u32,
        string_lengths: *const u32,
        pattern: *const u8,
        pattern_length: u32,
        num_strings: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    /// Launch string replacement kernel
    pub fn launch_string_replace_kernel(
        input_data: *const u8,
        output_data: *mut u8,
        string_offsets: *const u32,
        string_lengths: *const u32,
        output_lengths: *mut u32,
        pattern: *const u8,
        pattern_length: u32,
        replacement: *const u8,
        replacement_length: u32,
        num_strings: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    /// Launch bitonic sort kernel for strings
    pub fn launch_string_bitonic_sort_kernel(
        string_data: *mut u8,
        string_offsets: *mut u32,
        string_lengths: *mut u32,
        string_indices: *mut u32,
        num_strings: u32,
        stage: u32,
        step: u32,
        ascending: bool,
        stream: *mut std::ffi::c_void, // cudaStream_t
    );

    // Huffman coding GPU kernels

    /// Launch Huffman encoding kernel
    pub fn launch_huffman_encode(
        d_input: *const u8,
        d_output: *mut u8,
        d_code_table: *const HuffmanCodeGPU,
        input_length: u32,
        d_output_bits: *mut u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    ) -> i32; // cudaError_t

    /// Launch Huffman decoding kernel
    pub fn launch_huffman_decode(
        d_input: *const u8,
        d_output: *mut u8,
        d_tree_nodes: *const HuffmanTreeNodeGPU,
        root_index: u32,
        input_bit_length: u32,
        output_length: u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    ) -> i32; // cudaError_t

    /// Launch frequency counting kernel
    pub fn launch_count_frequencies(
        d_input: *const u8,
        input_length: u32,
        d_frequencies: *mut u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    ) -> i32; // cudaError_t

    /// Launch warp-parallel Huffman encoding
    pub fn launch_warp_parallel_encode(
        d_input: *const u8,
        d_output: *mut u8,
        d_code_table: *const HuffmanCodeGPU,
        input_length: u32,
        d_output_bytes: *mut u32,
        stream: *mut std::ffi::c_void, // cudaStream_t
    ) -> i32; // cudaError_t
}

/// GPU Huffman code structure (matches CUDA HuffmanCode)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HuffmanCodeGPU {
    pub bits: u32,  // Packed bits (up to 32 bits)
    pub length: u8, // Number of bits in the code
}

unsafe impl bytemuck::Pod for HuffmanCodeGPU {}
unsafe impl bytemuck::Zeroable for HuffmanCodeGPU {}

unsafe impl cudarc::driver::DeviceRepr for HuffmanCodeGPU {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// GPU Huffman tree node structure (matches CUDA HuffmanTreeNode)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HuffmanTreeNodeGPU {
    pub symbol: u8,  // Symbol (for leaf nodes)
    pub is_leaf: u8, // 1 if leaf node, 0 if internal
    pub left: u16,   // Index of left child (0 if leaf)
    pub right: u16,  // Index of right child (0 if leaf)
}

unsafe impl bytemuck::Pod for HuffmanTreeNodeGPU {}
unsafe impl bytemuck::Zeroable for HuffmanTreeNodeGPU {}

unsafe impl cudarc::driver::DeviceRepr for HuffmanTreeNodeGPU {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}
