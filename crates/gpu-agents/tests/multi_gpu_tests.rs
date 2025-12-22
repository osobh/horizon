//! Tests for multi-GPU support

#[cfg(test)]
mod tests {
    use gpu_agents::{MultiGpuConfig, MultiGpuSwarm};

    #[test]
    fn test_multi_gpu_config_creation() {
        let config = MultiGpuConfig {
            gpu_devices: vec![0, 1],
            agents_per_gpu: 1_000_000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "balanced".to_string(),
        };

        assert_eq!(config.gpu_devices.len(), 2);
        assert_eq!(config.total_agent_capacity(), 2_000_000);
    }

    #[test]
    fn test_multi_gpu_swarm_creation() {
        let config = MultiGpuConfig::default();
        let swarm = MultiGpuSwarm::new(config);

        assert!(swarm.is_ok());
        let swarm = swarm?;
        assert_eq!(swarm.device_count(), 1); // Default single GPU
    }

    #[test]
    fn test_agent_distribution_across_gpus() {
        // Test with single GPU (most systems have at least one)
        let config = MultiGpuConfig {
            gpu_devices: vec![0],
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config)?;
        swarm.initialize(1000).unwrap();

        let distribution = swarm.get_agent_distribution();
        assert_eq!(distribution.len(), 1);
        assert_eq!(distribution[0], 1000);

        // Test the distribution logic with a mock multi-GPU setup
        // In real implementation, would check available GPUs first
    }

    #[test]
    fn test_inter_gpu_communication() {
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 100,
            enable_peer_access: true,
            sync_interval: 10,
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config)?;
        swarm.initialize(100)?;

        // Enable inter-GPU communication (no-op for single GPU)
        let result = swarm.enable_gpu_peer_access();
        assert!(result.is_ok());

        // Test communication bandwidth (simulated for single GPU)
        let bandwidth = swarm.measure_inter_gpu_bandwidth();
        assert!(bandwidth > 0.0); // Should return simulated bandwidth
    }

    #[test]
    fn test_synchronized_step_execution() {
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 1, // Sync every step
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config)?;
        swarm.initialize(1000)?;

        // Execute synchronized step
        swarm.synchronized_step().unwrap();

        let metrics = swarm.metrics();
        assert!(metrics.sync_overhead_ms >= 0.0);
        assert_eq!(metrics.steps_executed, 1);
    }

    #[test]
    fn test_load_balancing_strategies() {
        let mut config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "balanced".to_string(),
        };

        // Test balanced strategy
        let swarm = MultiGpuSwarm::new(config.clone())?;
        assert_eq!(swarm.get_partition_strategy(), "balanced");

        // Test spatial partitioning
        config.partition_strategy = "spatial".to_string();
        let swarm = MultiGpuSwarm::new(config.clone()).unwrap();
        assert_eq!(swarm.get_partition_strategy(), "spatial");

        // Test dynamic load balancing
        config.partition_strategy = "dynamic".to_string();
        let swarm = MultiGpuSwarm::new(config).unwrap();
        assert_eq!(swarm.get_partition_strategy(), "dynamic");
    }

    #[test]
    fn test_gpu_failure_handling() {
        // This test validates the failure handling logic
        // In a real multi-GPU system, it would redistribute agents
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config).unwrap();
        swarm.initialize(1000).unwrap();

        // Test invalid GPU ID
        let result = swarm.handle_gpu_failure(1); // GPU 1 doesn't exist
        assert!(result.is_err());

        // Verify swarm is still functional
        let distribution = swarm.get_agent_distribution();
        assert_eq!(distribution.len(), 1);
        assert_eq!(distribution[0], 1000);
    }

    #[test]
    fn test_multi_gpu_memory_tracking() {
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 10000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config)?;
        swarm.initialize(10000)?;

        let memory_usage = swarm.get_memory_usage_per_gpu();
        assert_eq!(memory_usage.len(), 1);

        // GPU should have memory allocated
        assert!(memory_usage[0] > 0);

        // Total memory should match expected
        let total_memory = memory_usage.iter().sum::<usize>();
        let expected_memory = 10000 * 256; // 256 bytes per agent
        assert_eq!(total_memory, expected_memory);
    }

    #[test]
    fn test_collective_operations_multi_gpu() {
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 10,
            partition_strategy: "balanced".to_string(),
        };

        let mut swarm = MultiGpuSwarm::new(config)?;
        swarm.initialize(1000)?;

        // Test all-reduce operation (single GPU returns its own sum)
        let sum = swarm.all_reduce_fitness().unwrap();
        assert!(sum >= 0.0);

        // Test broadcast operation
        let result = swarm.broadcast_parameters(vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());

        // Test gather operation
        let gathered = swarm.gather_best_agents(10).unwrap();
        assert_eq!(gathered.len(), 10);
    }

    #[test]
    fn test_scaling_to_billion_agents() {
        // Test configuration for 1 billion agents across 16 GPUs
        let config = MultiGpuConfig {
            gpu_devices: (0..16).collect(), // 16 GPUs
            agents_per_gpu: 67_000_000,     // ~67M agents per GPU (within 32GB limit)
            enable_peer_access: true,
            sync_interval: 1000,
            partition_strategy: "spatial".to_string(),
        };

        // Just test config creation and validation
        assert_eq!(config.gpu_devices.len(), 16);
        assert_eq!(config.total_agent_capacity(), 1_072_000_000); // >1 billion

        // In real test, we'd need 16 GPUs to actually create this
        // let swarm = MultiGpuSwarm::new(config);
        // assert!(swarm.is_ok());
    }

    #[test]
    fn test_heterogeneous_gpu_support() {
        // Test GPU capability detection
        let config = MultiGpuConfig {
            gpu_devices: vec![0], // Single GPU test
            agents_per_gpu: 1000,
            enable_peer_access: true,
            sync_interval: 100,
            partition_strategy: "capability_aware".to_string(),
        };

        let swarm = MultiGpuSwarm::new(config)?;

        // Get GPU capabilities
        let capabilities = swarm.get_gpu_capabilities();
        assert_eq!(capabilities.len(), 1);

        // GPU should report its compute capability
        let cap = &capabilities[0];
        assert!(cap.compute_capability.0 >= 7); // Assuming modern GPUs
        assert!(cap.memory_size > 0);
    }
}
