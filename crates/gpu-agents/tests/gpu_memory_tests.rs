//! Tests for GPU memory allocation and management

#[cfg(test)]
mod tests {
    use gpu_agents::{GpuSwarm, GpuSwarmConfig};

    #[test]
    fn test_gpu_swarm_creation() {
        let config = GpuSwarmConfig::default();
        let swarm = GpuSwarm::new(config);
        assert!(swarm.is_ok());
    }

    #[test]
    fn test_agent_initialization() {
        let config = GpuSwarmConfig::default();
        let mut swarm = GpuSwarm::new(config).unwrap();

        let agent_count = 1024;
        let result = swarm.initialize(agent_count);
        assert!(result.is_ok());

        let metrics = swarm.metrics();
        assert_eq!(metrics.agent_count, agent_count);
    }

    #[test]
    fn test_gpu_memory_allocation() {
        let config = GpuSwarmConfig::default();
        let mut swarm = GpuSwarm::new(config).unwrap();

        let agent_count = 1024;
        swarm.initialize(agent_count)?;

        let metrics = swarm.metrics();
        // Each agent is 256 bytes
        let expected_memory = agent_count * 256;
        assert_eq!(metrics.gpu_memory_used, expected_memory);
    }

    #[test]
    fn test_swarm_step_execution() {
        let config = GpuSwarmConfig::default();
        let mut swarm = GpuSwarm::new(config).unwrap();

        swarm.initialize(1024).unwrap();
        let result = swarm.step();
        assert!(result.is_ok());

        let metrics = swarm.metrics();
        assert!(metrics.kernel_time_ms > 0.0);
    }

    #[test]
    fn test_large_swarm_allocation() {
        let config = GpuSwarmConfig::default();
        let mut swarm = GpuSwarm::new(config).unwrap();

        // Test with 1M agents
        let agent_count = 1_000_000;
        let result = swarm.initialize(agent_count);
        assert!(result.is_ok());

        let metrics = swarm.metrics();
        assert_eq!(metrics.agent_count, agent_count);
        assert_eq!(metrics.gpu_memory_used, agent_count * 256);
    }

    #[test]
    fn test_exceeds_max_agents() {
        let mut config = GpuSwarmConfig::default();
        config.max_agents = 1000;

        let mut swarm = GpuSwarm::new(config).unwrap();

        // Try to allocate more than max
        let result = swarm.initialize(2000);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_steps() {
        let config = GpuSwarmConfig::default();
        let mut swarm = GpuSwarm::new(config).unwrap();

        swarm.initialize(1024).unwrap();

        // Run multiple steps
        for _ in 0..10 {
            swarm.step()?;
        }

        let metrics = swarm.metrics();
        assert!(metrics.kernel_time_ms > 0.0);
    }

    #[test]
    fn test_gpu_device_properties() {
        let device_props = gpu_agents::get_gpu_device_properties(0);
        assert!(device_props.is_ok());

        let props = device_props.unwrap();
        assert!(props.total_memory > 0);
        assert!(props.compute_capability.0 >= 7); // At least compute 7.0
    }
}
