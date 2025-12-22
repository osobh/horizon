//! Tests for LLM integration with GPU agents

#[cfg(test)]
mod tests {
    use gpu_agents::llm::AgentState;
    use gpu_agents::{GpuSwarm, GpuSwarmConfig, LlmConfig, LlmIntegration};

    #[test]
    fn test_llm_config_creation() {
        let config = LlmConfig::default();
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_context_length, 4096);
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_llm_integration_initialization() {
        let llm_config = LlmConfig::default();
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        let integration = LlmIntegration::new(llm_config, device);
        assert!(integration.is_ok());
    }

    #[test]
    fn test_gpu_swarm_with_llm() {
        let mut swarm_config = GpuSwarmConfig::default();
        swarm_config.enable_llm = true;

        let swarm = GpuSwarm::new(swarm_config);
        assert!(swarm.is_ok());

        let swarm = swarm?;
        assert!(swarm.has_llm_support());
    }

    #[test]
    fn test_agent_prompt_generation() {
        let llm_config = LlmConfig::default();
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        let integration = LlmIntegration::new(llm_config, device).unwrap();

        // Test generating prompts for a batch of agents
        let agent_states = vec![
            AgentState {
                id: 0,
                fitness: 0.8,
                position: [1.0, 2.0, 3.0],
            },
            AgentState {
                id: 1,
                fitness: 0.6,
                position: [4.0, 5.0, 6.0],
            },
        ];

        let prompts = integration.generate_agent_prompts(&agent_states);
        assert!(prompts.is_ok());

        let prompts = prompts.unwrap();
        assert_eq!(prompts.len(), 2);
    }

    #[test]
    fn test_agent_action_parsing() {
        let llm_config = LlmConfig::default();
        let device = cudarc::driver::CudaDevice::new(0).unwrap();
        let integration = LlmIntegration::new(llm_config, device).unwrap();

        // Test parsing LLM responses into agent actions
        let responses = vec![
            "MOVE: direction=[0.5, -0.3, 0.2], speed=1.5",
            "EXPLORE: radius=10.0, strategy=random",
        ];

        let actions = integration.parse_agent_actions(&responses);
        assert!(actions.is_ok());

        let actions = actions.unwrap();
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_gpu_llm_batch_inference() {
        let mut swarm_config = GpuSwarmConfig::default();
        swarm_config.enable_llm = true;

        let mut swarm = GpuSwarm::new(swarm_config).unwrap();
        swarm.initialize(64)?; // Small batch for testing

        // Enable LLM for agents
        let result = swarm.enable_llm_reasoning(LlmConfig::default());
        assert!(result.is_ok());

        // Run inference step
        let result = swarm.step_with_llm();
        assert!(result.is_ok());

        // Check that agents were updated based on LLM output
        let metrics = swarm.metrics();
        assert!(metrics.llm_inference_time_ms > 0.0);
    }

    #[test]
    fn test_knowledge_embedding_integration() {
        let llm_config = LlmConfig {
            enable_embeddings: true,
            embedding_dim: 768,
            ..Default::default()
        };

        let device = cudarc::driver::CudaDevice::new(0)?;
        let integration = LlmIntegration::new(llm_config, device)?;

        // Test embedding generation for agent knowledge
        let knowledge_items = vec![
            "The swarm is approaching a resource-rich area",
            "Danger detected at coordinates (100, 200, 50)",
        ];

        let embeddings = integration.generate_embeddings(&knowledge_items);
        assert!(embeddings.is_ok());

        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
    }

    #[test]
    fn test_collective_intelligence_aggregation() {
        let mut swarm_config = GpuSwarmConfig::default();
        swarm_config.enable_llm = true;
        swarm_config.enable_collective_intelligence = true;

        let mut swarm = GpuSwarm::new(swarm_config)?;
        swarm.initialize(128)?;
        swarm.enable_llm_reasoning(LlmConfig::default())?;

        // Run multiple steps to build collective knowledge
        for _ in 0..5 {
            swarm.step_with_llm().unwrap();
        }

        // Query collective intelligence
        let query = "What is the best direction for the swarm to move?";
        let response = swarm.query_collective_intelligence(query);
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(!response.is_empty());
    }

    #[test]
    fn test_gpu_memory_with_llm_buffers() {
        let mut swarm_config = GpuSwarmConfig::default();
        swarm_config.enable_llm = true;

        let mut swarm = GpuSwarm::new(swarm_config).unwrap();
        swarm.initialize(1024)?;
        swarm.enable_llm_reasoning(LlmConfig::default())?;

        let metrics = swarm.metrics();

        // Check that LLM buffers are allocated in addition to agent memory
        let base_agent_memory = 1024 * 256; // 1024 agents * 256 bytes
        assert!(metrics.gpu_memory_used > base_agent_memory);
        assert!(metrics.llm_buffer_memory > 0);
    }
}
