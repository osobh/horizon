//! Shared utilities and test modules for distributed swarm integration tests

use exorust_evolution_engines::{
    config::{EvolutionEngineConfig, LoggingConfig, ResourceLimits},
    swarm::{SwarmConfig},
    swarm_distributed::{
        DistributedSwarmConfig, FaultToleranceConfig, LoadBalanceConfig,
        LoadBalanceStrategy, NetworkConfig, RecoveryStrategy,
    },
    swarm_particle::ParticleParameters,
    swarm_topology::SwarmTopology,
    traits::MockEvolvableAgent,
};

pub mod node_tests;
pub mod distribution_tests;
pub mod fault_tolerance_tests;
pub mod performance_tests;
pub mod network_tests;

/// Helper function to create distributed swarm configuration
pub fn create_distributed_config(node_id: &str, port: u16) -> DistributedSwarmConfig {
    DistributedSwarmConfig {
        base_config: SwarmConfig {
            base: EvolutionEngineConfig {
                population_size: 20,
                max_generations: 100,
                mutation_rate: 0.1,
                target_fitness: None,
                max_runtime_seconds: None,
                seed: None,
                adaptive_parameters: false,
                logging: LoggingConfig::default(),
                resource_limits: ResourceLimits::default(),
            },
            topology: SwarmTopology::Ring,
            particle_params: ParticleParameters::default(),
            social_influence: 1.49445,
            cognitive_influence: 1.49445,
            inertia_weight: 0.9,
            neighborhood_size: 5,
        },
        node_id: node_id.to_string(),
        network_config: NetworkConfig {
            listen_addr: "127.0.0.1".to_string(),
            port,
            bootstrap_peers: vec![],
            max_connections: 10,
            heartbeat_interval_ms: 1000,
        },
        load_balance_config: LoadBalanceConfig {
            target_particles_per_node: 10,
            strategy: LoadBalanceStrategy::EvenDistribution,
            rebalance_threshold: 0.2,
            migration_batch_size: 5,
        },
        fault_tolerance_config: FaultToleranceConfig {
            checkpoint_interval: 10,
            backup_replicas: 2,
            failure_timeout_ms: 5000,
            recovery_strategy: RecoveryStrategy::Hybrid,
        },
    }
}

/// Helper function to create a mock evolvable agent  
pub fn create_mock_agent(id: u32) -> MockEvolvableAgent {
    MockEvolvableAgent::new(format!("agent_{}", id))
}