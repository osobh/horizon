//! Adaptive Consensus Algorithm Test Suite
//!
//! This module tests adaptive consensus algorithms that dynamically select
//! the optimal consensus mechanism based on network conditions, node count,
//! and GPU acceleration capabilities. Tests cover sub-microsecond consensus
//! latency and real-world adaptation scenarios.
//!
//! ALL TESTS IN THIS FILE ARE DESIGNED TO FAIL INITIALLY (RED PHASE)
//! They represent adaptive algorithm functionality that needs to be implemented.

use exorust_consensus::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Adaptive consensus configuration for testing
#[derive(Debug, Clone)]
struct AdaptiveConsensusConfig {
    /// Available consensus algorithms
    available_algorithms: Vec<ConsensusAlgorithmType>,
    /// GPU acceleration enabled
    gpu_acceleration: bool,
    /// Target consensus latency (sub-microsecond)
    target_latency: Duration,
    /// Network condition monitoring enabled
    network_monitoring: bool,
    /// Algorithm switching threshold
    performance_threshold: f32,
    /// Adaptation interval
    adaptation_interval: Duration,
}

/// Consensus algorithm types for adaptive selection
#[derive(Debug, Clone, PartialEq)]
enum ConsensusAlgorithmType {
    /// Traditional PBFT (Byzantine fault tolerant)
    PBFT,
    /// GPU-accelerated PBFT
    GpuPBFT,
    /// Fast consensus for small networks
    FastConsensus,
    /// Streaming consensus for high throughput
    StreamingConsensus,
    /// Hybrid algorithm combining multiple approaches
    HybridConsensus,
    /// Custom GPU-native consensus
    GpuNativeConsensus,
}

/// Network condition metrics for algorithm selection
#[derive(Debug, Clone)]
struct NetworkConditions {
    /// Number of active nodes
    node_count: usize,
    /// Average network latency
    avg_latency: Duration,
    /// Network bandwidth utilization
    bandwidth_utilization: f32,
    /// Partition probability
    partition_risk: f32,
    /// Byzantine node percentage
    byzantine_percentage: f32,
    /// GPU availability across nodes
    gpu_availability: f32,
}

/// Performance metrics for algorithm evaluation
#[derive(Debug, Clone)]
struct AlgorithmPerformance {
    /// Consensus latency
    latency: Duration,
    /// Throughput (consensus per second)
    throughput: f32,
    /// Resource utilization
    cpu_utilization: f32,
    /// GPU utilization
    gpu_utilization: f32,
    /// Memory usage
    memory_usage: usize,
    /// Network messages sent
    network_messages: usize,
    /// Success rate
    success_rate: f32,
}

/// Adaptive consensus engine that selects optimal algorithms
struct AdaptiveConsensusEngine {
    config: AdaptiveConsensusConfig,
    current_algorithm: ConsensusAlgorithmType,
    algorithm_implementations: HashMap<ConsensusAlgorithmType, Box<dyn ConsensusAlgorithm>>,
    performance_history: HashMap<ConsensusAlgorithmType, Vec<AlgorithmPerformance>>,
    network_monitor: NetworkMonitor,
    gpu_accelerator: Option<GpuConsensusAccelerator>,
}

impl AdaptiveConsensusEngine {
    async fn new(config: AdaptiveConsensusConfig) -> Result<Self, ConsensusError> {
        // This will fail - adaptive consensus engine not implemented
        Err(ConsensusError::NotImplemented(
            "AdaptiveConsensusEngine not yet implemented".to_string()
        ))
    }

    async fn select_optimal_algorithm(
        &mut self,
        conditions: &NetworkConditions
    ) -> Result<ConsensusAlgorithmType, ConsensusError> {
        // This will fail - algorithm selection not implemented
        Err(ConsensusError::NotImplemented(
            "Algorithm selection not yet implemented".to_string()
        ))
    }

    async fn run_consensus_round(
        &mut self,
        value: String,
        validators: Vec<ValidatorId>
    ) -> Result<ConsensusResult, ConsensusError> {
        // This will fail - adaptive consensus round not implemented
        Err(ConsensusError::NotImplemented(
            "Adaptive consensus round not yet implemented".to_string()
        ))
    }

    async fn benchmark_algorithm(
        &mut self,
        algorithm: ConsensusAlgorithmType,
        conditions: &NetworkConditions
    ) -> Result<AlgorithmPerformance, ConsensusError> {
        // This will fail - algorithm benchmarking not implemented
        Err(ConsensusError::NotImplemented(
            "Algorithm benchmarking not yet implemented".to_string()
        ))
    }

    async fn adapt_to_conditions(&mut self, conditions: &NetworkConditions) -> Result<(), ConsensusError> {
        // This will fail - adaptive behavior not implemented
        Err(ConsensusError::NotImplemented(
            "Adaptive behavior not yet implemented".to_string()
        ))
    }

    fn get_performance_metrics(&self) -> HashMap<ConsensusAlgorithmType, AlgorithmPerformance> {
        // This will fail - metrics collection not implemented
        HashMap::new()
    }
}

/// Trait for consensus algorithm implementations
#[async_trait::async_trait]
trait ConsensusAlgorithm: Send + Sync {
    async fn run_consensus(
        &mut self,
        value: String,
        validators: Vec<ValidatorId>,
        network_conditions: &NetworkConditions
    ) -> Result<ConsensusResult, ConsensusError>;

    fn expected_performance(&self, conditions: &NetworkConditions) -> AlgorithmPerformance;
    
    fn supports_gpu_acceleration(&self) -> bool;
    
    fn optimal_node_range(&self) -> (usize, usize);
}

/// Network condition monitor
struct NetworkMonitor {
    current_conditions: NetworkConditions,
    condition_history: Vec<(Instant, NetworkConditions)>,
}

impl NetworkMonitor {
    fn new() -> Self {
        Self {
            current_conditions: NetworkConditions {
                node_count: 0,
                avg_latency: Duration::from_millis(100),
                bandwidth_utilization: 0.0,
                partition_risk: 0.0,
                byzantine_percentage: 0.0,
                gpu_availability: 0.0,
            },
            condition_history: Vec::new(),
        }
    }

    async fn update_conditions(&mut self) -> Result<NetworkConditions, ConsensusError> {
        // This will fail - network monitoring not implemented
        Err(ConsensusError::NotImplemented(
            "Network monitoring not yet implemented".to_string()
        ))
    }

    fn predict_conditions(&self, lookahead: Duration) -> NetworkConditions {
        // Return current conditions as placeholder
        self.current_conditions.clone()
    }
}

/// GPU consensus accelerator
struct GpuConsensusAccelerator {
    device_count: u32,
    memory_pool: Vec<u8>,
    acceleration_kernels: HashMap<ConsensusAlgorithmType, String>,
}

impl GpuConsensusAccelerator {
    async fn new() -> Result<Self, ConsensusError> {
        // This will fail - GPU accelerator not implemented
        Err(ConsensusError::NotImplemented(
            "GPU consensus accelerator not yet implemented".to_string()
        ))
    }

    async fn accelerate_consensus(
        &mut self,
        algorithm: ConsensusAlgorithmType,
        messages: Vec<ConsensusMessage>
    ) -> Result<ConsensusResult, ConsensusError> {
        // This will fail - GPU acceleration not implemented
        Err(ConsensusError::NotImplemented(
            "GPU consensus acceleration not yet implemented".to_string()
        ))
    }

    fn get_gpu_utilization(&self) -> f32 {
        0.0 // Placeholder
    }
}

#[derive(Debug, Clone)]
struct ConsensusResult {
    consensus_achieved: bool,
    agreed_value: Option<String>,
    participating_validators: Vec<ValidatorId>,
    consensus_latency: Duration,
    algorithm_used: ConsensusAlgorithmType,
    performance_metrics: AlgorithmPerformance,
}

// ===== FAILING TESTS (RED PHASE) =====

#[tokio::test]
async fn test_algorithm_selection_small_network() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::FastConsensus,
            ConsensusAlgorithmType::PBFT,
            ConsensusAlgorithmType::GpuPBFT,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(500), // Sub-microsecond target
        network_monitoring: true,
        performance_threshold: 0.95,
        adaptation_interval: Duration::from_millis(100),
    };

    // This test WILL FAIL - adaptive engine not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create adaptive engine");

    // Small network conditions (< 100 nodes)
    let small_network = NetworkConditions {
        node_count: 50,
        avg_latency: Duration::from_micros(10),
        bandwidth_utilization: 0.3,
        partition_risk: 0.1,
        byzantine_percentage: 0.05,
        gpu_availability: 0.8,
    };

    let selected_algorithm = engine.select_optimal_algorithm(&small_network).await
        .expect("Algorithm selection should succeed");

    // For small networks, should prefer fast consensus
    assert_eq!(selected_algorithm, ConsensusAlgorithmType::FastConsensus,
              "Small networks should use fast consensus");

    // Run consensus with selected algorithm
    let validators: Vec<ValidatorId> = (0..50).map(|_| ValidatorId::new()).collect();
    let result = engine.run_consensus_round("block_data_small".to_string(), validators).await
        .expect("Small network consensus should succeed");

    assert!(result.consensus_achieved, "Small network should achieve consensus");
    assert!(result.consensus_latency < config.target_latency,
           "Small network consensus should be sub-microsecond: {:?}", result.consensus_latency);
    assert_eq!(result.algorithm_used, ConsensusAlgorithmType::FastConsensus,
              "Should use the selected algorithm");
}

#[tokio::test]
async fn test_algorithm_selection_large_network() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::PBFT,
            ConsensusAlgorithmType::GpuPBFT,
            ConsensusAlgorithmType::GpuNativeConsensus,
            ConsensusAlgorithmType::HybridConsensus,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(800),
        network_monitoring: true,
        performance_threshold: 0.90,
        adaptation_interval: Duration::from_millis(200),
    };

    // This test WILL FAIL - large network handling not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create adaptive engine");

    // Large network conditions (> 10,000 nodes)
    let large_network = NetworkConditions {
        node_count: 100_000,
        avg_latency: Duration::from_micros(50),
        bandwidth_utilization: 0.7,
        partition_risk: 0.2,
        byzantine_percentage: 0.15,
        gpu_availability: 0.9,
    };

    let selected_algorithm = engine.select_optimal_algorithm(&large_network).await
        .expect("Large network algorithm selection should succeed");

    // For large networks with high GPU availability, should prefer GPU-native consensus
    assert_eq!(selected_algorithm, ConsensusAlgorithmType::GpuNativeConsensus,
              "Large networks with high GPU availability should use GPU-native consensus");

    // Run consensus with many validators
    let validators: Vec<ValidatorId> = (0..100_000).map(|_| ValidatorId::new()).collect();
    let result = engine.run_consensus_round("block_data_large".to_string(), validators).await
        .expect("Large network consensus should succeed");

    assert!(result.consensus_achieved, "Large network should achieve consensus");
    assert!(result.consensus_latency < Duration::from_millis(2),
           "Large network consensus should be fast: {:?}", result.consensus_latency);
    assert!(result.performance_metrics.gpu_utilization > 80.0,
           "GPU utilization should be high: {}%", result.performance_metrics.gpu_utilization);
}

#[tokio::test]
async fn test_byzantine_adaptation() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::PBFT,
            ConsensusAlgorithmType::GpuPBFT,
            ConsensusAlgorithmType::HybridConsensus,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(600),
        network_monitoring: true,
        performance_threshold: 0.85,
        adaptation_interval: Duration::from_millis(150),
    };

    // This test WILL FAIL - Byzantine adaptation not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create adaptive engine");

    // High Byzantine threat conditions
    let byzantine_network = NetworkConditions {
        node_count: 10_000,
        avg_latency: Duration::from_micros(30),
        bandwidth_utilization: 0.5,
        partition_risk: 0.3,
        byzantine_percentage: 0.30, // 30% Byzantine nodes
        gpu_availability: 0.7,
    };

    let selected_algorithm = engine.select_optimal_algorithm(&byzantine_network).await
        .expect("Byzantine algorithm selection should succeed");

    // Should select robust PBFT variant for high Byzantine threat
    assert!(matches!(selected_algorithm, 
                    ConsensusAlgorithmType::PBFT | 
                    ConsensusAlgorithmType::GpuPBFT | 
                    ConsensusAlgorithmType::HybridConsensus),
           "High Byzantine threat should use robust algorithm: {:?}", selected_algorithm);

    // Adapt to changing conditions
    engine.adapt_to_conditions(&byzantine_network).await
        .expect("Byzantine adaptation should succeed");

    let validators: Vec<ValidatorId> = (0..10_000).map(|_| ValidatorId::new()).collect();
    let result = engine.run_consensus_round("byzantine_test_block".to_string(), validators).await
        .expect("Byzantine consensus should succeed");

    assert!(result.consensus_achieved, "Should achieve consensus despite Byzantine nodes");
    assert!(result.performance_metrics.success_rate > 0.95,
           "Success rate should be high despite Byzantine nodes: {}", 
           result.performance_metrics.success_rate);
}

#[tokio::test]
async fn test_sub_microsecond_gpu_consensus() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::GpuNativeConsensus,
            ConsensusAlgorithmType::GpuPBFT,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(300), // Aggressive sub-microsecond target
        network_monitoring: true,
        performance_threshold: 0.98,
        adaptation_interval: Duration::from_millis(50),
    };

    // This test WILL FAIL - sub-microsecond GPU consensus not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create GPU consensus engine");

    // Optimal GPU conditions
    let gpu_optimized = NetworkConditions {
        node_count: 1000,
        avg_latency: Duration::from_nanos(50),
        bandwidth_utilization: 0.4,
        partition_risk: 0.05,
        byzantine_percentage: 0.10,
        gpu_availability: 1.0, // Perfect GPU availability
    };

    let selected_algorithm = engine.select_optimal_algorithm(&gpu_optimized).await
        .expect("GPU algorithm selection should succeed");

    assert_eq!(selected_algorithm, ConsensusAlgorithmType::GpuNativeConsensus,
              "Perfect GPU conditions should select GPU-native consensus");

    let validators: Vec<ValidatorId> = (0..1000).map(|_| ValidatorId::new()).collect();
    
    // Run multiple rounds to verify consistent sub-microsecond performance
    let mut latencies = Vec::new();
    for i in 0..100 {
        let result = engine.run_consensus_round(format!("gpu_block_{}", i), validators.clone()).await
            .expect(&format!("GPU consensus round {} should succeed", i));
        
        latencies.push(result.consensus_latency);
        
        assert!(result.consensus_achieved, "GPU round {} should achieve consensus", i);
        assert!(result.consensus_latency <= config.target_latency,
               "Round {} should meet sub-microsecond target: {:?} <= {:?}", 
               i, result.consensus_latency, config.target_latency);
        assert!(result.performance_metrics.gpu_utilization > 95.0,
               "GPU utilization should be optimal: {}%", result.performance_metrics.gpu_utilization);
    }

    let average_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let max_latency = *latencies.iter().max().unwrap();
    
    assert!(average_latency < config.target_latency,
           "Average latency should be sub-microsecond: {:?}", average_latency);
    assert!(max_latency < Duration::from_nanos(500),
           "Maximum latency should be bounded: {:?}", max_latency);
}

#[tokio::test]
async fn test_dynamic_algorithm_switching() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::FastConsensus,
            ConsensusAlgorithmType::PBFT,
            ConsensusAlgorithmType::GpuPBFT,
            ConsensusAlgorithmType::StreamingConsensus,
            ConsensusAlgorithmType::HybridConsensus,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(700),
        network_monitoring: true,
        performance_threshold: 0.90,
        adaptation_interval: Duration::from_millis(100),
    };

    // This test WILL FAIL - dynamic switching not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create dynamic switching engine");

    let validators: Vec<ValidatorId> = (0..5000).map(|_| ValidatorId::new()).collect();

    // Simulate changing network conditions
    let scenarios = vec![
        // Start with small, fast network
        ("small_fast", NetworkConditions {
            node_count: 100,
            avg_latency: Duration::from_micros(5),
            bandwidth_utilization: 0.2,
            partition_risk: 0.05,
            byzantine_percentage: 0.05,
            gpu_availability: 0.5,
        }),
        // Scale up to medium network
        ("medium_stable", NetworkConditions {
            node_count: 1000,
            avg_latency: Duration::from_micros(20),
            bandwidth_utilization: 0.5,
            partition_risk: 0.1,
            byzantine_percentage: 0.1,
            gpu_availability: 0.8,
        }),
        // High-threat Byzantine environment
        ("high_byzantine", NetworkConditions {
            node_count: 2000,
            avg_latency: Duration::from_micros(40),
            bandwidth_utilization: 0.7,
            partition_risk: 0.25,
            byzantine_percentage: 0.25,
            gpu_availability: 0.9,
        }),
        // High-throughput streaming scenario
        ("high_throughput", NetworkConditions {
            node_count: 5000,
            avg_latency: Duration::from_micros(15),
            bandwidth_utilization: 0.9,
            partition_risk: 0.1,
            byzantine_percentage: 0.1,
            gpu_availability: 1.0,
        }),
    ];

    let mut previous_algorithm = None;
    let mut switch_count = 0;

    for (scenario_name, conditions) in scenarios {
        // Update conditions and adapt
        engine.adapt_to_conditions(&conditions).await
            .expect(&format!("Adaptation should succeed for {}", scenario_name));

        let selected_algorithm = engine.select_optimal_algorithm(&conditions).await
            .expect(&format!("Algorithm selection should succeed for {}", scenario_name));

        if let Some(prev_alg) = previous_algorithm {
            if prev_alg != selected_algorithm {
                switch_count += 1;
            }
        }
        previous_algorithm = Some(selected_algorithm.clone());

        // Run consensus with current algorithm
        let result = engine.run_consensus_round(
            format!("adaptive_block_{}", scenario_name), 
            validators[..conditions.node_count.min(validators.len())].to_vec()
        ).await.expect(&format!("Consensus should succeed for {}", scenario_name));

        assert!(result.consensus_achieved, "Scenario {} should achieve consensus", scenario_name);
        assert_eq!(result.algorithm_used, selected_algorithm,
                  "Should use the selected algorithm for {}", scenario_name);

        // Verify algorithm choice makes sense for conditions
        match scenario_name {
            "small_fast" => assert_eq!(selected_algorithm, ConsensusAlgorithmType::FastConsensus,
                                     "Small fast network should use FastConsensus"),
            "high_byzantine" => assert!(matches!(selected_algorithm, 
                                               ConsensusAlgorithmType::PBFT | 
                                               ConsensusAlgorithmType::GpuPBFT),
                                      "High Byzantine threat should use PBFT variant"),
            "high_throughput" => assert!(matches!(selected_algorithm,
                                                ConsensusAlgorithmType::StreamingConsensus |
                                                ConsensusAlgorithmType::GpuNativeConsensus),
                                       "High throughput should use streaming or GPU consensus"),
            _ => {} // Other scenarios can vary
        }
    }

    // Should have switched algorithms at least once
    assert!(switch_count >= 2, "Should adapt by switching algorithms: {} switches", switch_count);
}

#[tokio::test]
async fn test_performance_benchmarking() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::PBFT,
            ConsensusAlgorithmType::GpuPBFT,
            ConsensusAlgorithmType::FastConsensus,
            ConsensusAlgorithmType::GpuNativeConsensus,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(500),
        network_monitoring: true,
        performance_threshold: 0.95,
        adaptation_interval: Duration::from_millis(100),
    };

    // This test WILL FAIL - performance benchmarking not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create benchmarking engine");

    let benchmark_conditions = NetworkConditions {
        node_count: 1000,
        avg_latency: Duration::from_micros(20),
        bandwidth_utilization: 0.5,
        partition_risk: 0.1,
        byzantine_percentage: 0.15,
        gpu_availability: 0.8,
    };

    // Benchmark all available algorithms
    let mut algorithm_performances = HashMap::new();
    
    for algorithm in &config.available_algorithms {
        let performance = engine.benchmark_algorithm(algorithm.clone(), &benchmark_conditions).await
            .expect(&format!("Benchmarking {:?} should succeed", algorithm));
        
        algorithm_performances.insert(algorithm.clone(), performance.clone());
        
        // Basic performance validation
        assert!(performance.success_rate > 0.8,
               "Algorithm {:?} should have high success rate: {}", 
               algorithm, performance.success_rate);
        assert!(performance.latency <= Duration::from_millis(10),
               "Algorithm {:?} should have reasonable latency: {:?}", 
               algorithm, performance.latency);
        
        // GPU algorithms should have higher GPU utilization
        if matches!(algorithm, ConsensusAlgorithmType::GpuPBFT | ConsensusAlgorithmType::GpuNativeConsensus) {
            assert!(performance.gpu_utilization > 50.0,
                   "GPU algorithm {:?} should use GPU: {}%", 
                   algorithm, performance.gpu_utilization);
        }
    }

    // Compare performances
    let pbft_perf = &algorithm_performances[&ConsensusAlgorithmType::PBFT];
    let gpu_pbft_perf = &algorithm_performances[&ConsensusAlgorithmType::GpuPBFT];
    let fast_consensus_perf = &algorithm_performances[&ConsensusAlgorithmType::FastConsensus];

    // GPU PBFT should be faster than regular PBFT
    assert!(gpu_pbft_perf.latency < pbft_perf.latency,
           "GPU PBFT should be faster than regular PBFT: {:?} < {:?}",
           gpu_pbft_perf.latency, pbft_perf.latency);

    // FastConsensus should have lowest latency
    assert!(fast_consensus_perf.latency <= gpu_pbft_perf.latency,
           "FastConsensus should be fastest: {:?} <= {:?}",
           fast_consensus_perf.latency, gpu_pbft_perf.latency);

    // Get overall performance metrics
    let overall_metrics = engine.get_performance_metrics();
    assert!(!overall_metrics.is_empty(), "Should have performance metrics");
}

#[tokio::test]
async fn test_million_node_adaptive_consensus() {
    let config = AdaptiveConsensusConfig {
        available_algorithms: vec![
            ConsensusAlgorithmType::GpuNativeConsensus,
            ConsensusAlgorithmType::HybridConsensus,
            ConsensusAlgorithmType::StreamingConsensus,
        ],
        gpu_acceleration: true,
        target_latency: Duration::from_nanos(800), // Sub-microsecond for million nodes
        network_monitoring: true,
        performance_threshold: 0.85,
        adaptation_interval: Duration::from_millis(500),
    };

    // This test WILL FAIL - million node adaptive consensus not implemented
    let mut engine = AdaptiveConsensusEngine::new(config.clone()).await
        .expect("Failed to create million-node adaptive engine");

    // Million node network conditions
    let million_node_network = NetworkConditions {
        node_count: 1_000_000,
        avg_latency: Duration::from_micros(100),
        bandwidth_utilization: 0.8,
        partition_risk: 0.15,
        byzantine_percentage: 0.20,
        gpu_availability: 0.95,
    };

    let selected_algorithm = engine.select_optimal_algorithm(&million_node_network).await
        .expect("Million node algorithm selection should succeed");

    // Should select GPU-native or hybrid for million nodes
    assert!(matches!(selected_algorithm,
                    ConsensusAlgorithmType::GpuNativeConsensus |
                    ConsensusAlgorithmType::HybridConsensus),
           "Million nodes should use GPU-native or hybrid consensus: {:?}", selected_algorithm);

    // Generate million validators (this itself tests scalability)
    let validators: Vec<ValidatorId> = (0..1_000_000).map(|_| ValidatorId::new()).collect();
    
    let start_time = Instant::now();
    let result = engine.run_consensus_round("million_node_block".to_string(), validators).await
        .expect("Million node consensus should succeed");
    let total_time = start_time.elapsed();

    assert!(result.consensus_achieved, "Million node network should achieve consensus");
    assert!(result.consensus_latency < Duration::from_millis(5),
           "Million node consensus should be fast: {:?}", result.consensus_latency);
    assert!(total_time < Duration::from_secs(60),
           "Million node setup should complete within reasonable time: {:?}", total_time);
    
    assert!(result.performance_metrics.gpu_utilization > 90.0,
           "Million node consensus should maximize GPU utilization: {}%", 
           result.performance_metrics.gpu_utilization);
    assert!(result.performance_metrics.success_rate > 0.95,
           "Million node consensus should have high success rate: {}", 
           result.performance_metrics.success_rate);
}

// Helper implementations and extensions

impl ConsensusError {
    fn NotImplemented(msg: String) -> Self {
        ConsensusError::ValidationFailed(format!("Not implemented: {}", msg))
    }
}

/// Create test validators for benchmarking
fn create_test_validators(count: usize) -> Vec<ValidatorId> {
    (0..count).map(|_| ValidatorId::new()).collect()
}

/// Simulate network conditions for testing
fn simulate_network_conditions(
    node_count: usize,
    byzantine_ratio: f32,
    gpu_availability: f32
) -> NetworkConditions {
    NetworkConditions {
        node_count,
        avg_latency: Duration::from_micros(10 + (node_count / 100) as u64),
        bandwidth_utilization: 0.3 + (node_count as f32 / 10000.0).min(0.6),
        partition_risk: byzantine_ratio * 0.5,
        byzantine_percentage: byzantine_ratio,
        gpu_availability,
    }
}

/// Calculate expected performance for algorithm/condition combination
fn calculate_expected_performance(
    algorithm: &ConsensusAlgorithmType,
    conditions: &NetworkConditions
) -> AlgorithmPerformance {
    // Placeholder implementation - actual logic would be much more sophisticated
    let base_latency = match algorithm {
        ConsensusAlgorithmType::FastConsensus => Duration::from_nanos(200),
        ConsensusAlgorithmType::PBFT => Duration::from_nanos(800),
        ConsensusAlgorithmType::GpuPBFT => Duration::from_nanos(400),
        ConsensusAlgorithmType::GpuNativeConsensus => Duration::from_nanos(300),
        ConsensusAlgorithmType::StreamingConsensus => Duration::from_nanos(500),
        ConsensusAlgorithmType::HybridConsensus => Duration::from_nanos(600),
    };

    let node_scaling_factor = (conditions.node_count as f32).log10() / 3.0; // Log scaling
    let scaled_latency = base_latency.mul_f32(1.0 + node_scaling_factor);

    AlgorithmPerformance {
        latency: scaled_latency,
        throughput: 1000.0 / scaled_latency.as_secs_f32(),
        cpu_utilization: 60.0,
        gpu_utilization: if algorithm.to_string().contains("Gpu") { 85.0 } else { 5.0 },
        memory_usage: conditions.node_count * 1024, // 1KB per node
        network_messages: conditions.node_count * 3, // 3 messages per node
        success_rate: 0.98 - (conditions.byzantine_percentage * 0.1),
    }
}

impl ToString for ConsensusAlgorithmType {
    fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}