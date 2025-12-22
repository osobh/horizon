#[cfg(test)]
mod test_evolution_config_fields {
    use crate::evolution::{FitnessObjective, GpuEvolutionConfig};

    #[test]
    fn test_gpu_evolution_config_fields() {
        // Test that GpuEvolutionConfig has the expected fields
        let config = GpuEvolutionConfig {
            genome_size: 100,
            fitness_objectives: vec![FitnessObjective::Performance],
            elite_percentage: 0.1,
            block_size: 256,
        };

        assert_eq!(config.genome_size, 100);
        assert_eq!(config.fitness_objectives.len(), 1);
        assert_eq!(config.elite_percentage, 0.1);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_evolution_manager_new_requires_device() {
        use cudarc::driver::CudaDevice;
        use std::sync::Arc;

        // Test that EvolutionManager::new requires a device argument
        let device = Arc::new(CudaDevice::new(0)?);
        let config = GpuEvolutionConfig {
            genome_size: 100,
            fitness_objectives: vec![],
            elite_percentage: 0.1,
            block_size: 256,
        };

        // This should compile - EvolutionManager::new takes (device, config)
        let _manager = crate::evolution::EvolutionManager::new(device, config);
    }
}
