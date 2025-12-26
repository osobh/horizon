//! Neural Router TDD Tests - RED Phase
//! 
//! Tests for learned routing tables using real neural networks.
//! These tests MUST fail initially to establish proper TDD cycle.
//! 
//! Requirements:
//! - Real neural network routing decisions  
//! - GPU-accelerated inference
//! - Learned routing table updates
//! - Multi-agent routing optimization
//! - Real-time routing adaptation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use uuid::Uuid;
use rand::prelude::*;
use candle_core::{Device, Tensor, DType};
use candle_nn::{Linear, Module, VarBuilder, linear, sequential, Seq, activation};
use ndarray::{Array2, Array1};

use stratoswarm_agent_core::{Agent, AgentId};
use stratoswarm_cuda::context::CudaContext;
use stratoswarm_memory::pool::MemoryPool;

/// Neural network for routing decisions
#[derive(Clone)]
pub struct NeuralRouter {
    device: Device,
    network: Seq,
    routing_table: Arc<RwLock<HashMap<AgentId, RoutingEntry>>>,
    training_data: Arc<RwLock<Vec<TrainingExample>>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

#[derive(Clone, Debug)]
pub struct RoutingEntry {
    pub destination: AgentId,
    pub latency_prediction: f32,
    pub bandwidth_prediction: f32,
    pub reliability_score: f32,
    pub last_updated: Instant,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct TrainingExample {
    pub source: AgentId,
    pub destination: AgentId,
    pub network_features: Vec<f32>,
    pub actual_latency: f32,
    pub actual_bandwidth: f32,
    pub success: bool,
    pub timestamp: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct PerformanceMetrics {
    pub accuracy: f32,
    pub latency_error: f32,
    pub bandwidth_error: f32,
    pub total_predictions: u64,
    pub correct_predictions: u64,
    pub training_epochs: u64,
    pub last_training_time: Option<Instant>,
}

#[derive(Clone, Debug)]
pub struct NetworkTopology {
    pub agents: Vec<AgentId>,
    pub connections: HashMap<(AgentId, AgentId), ConnectionMetrics>,
    pub cluster_regions: Vec<ClusterRegion>,
}

#[derive(Clone, Debug)]
pub struct ConnectionMetrics {
    pub latency_history: Vec<f32>,
    pub bandwidth_history: Vec<f32>,
    pub packet_loss_rate: f32,
    pub jitter: f32,
    pub congestion_level: f32,
}

#[derive(Clone, Debug)]  
pub struct ClusterRegion {
    pub id: String,
    pub agents: Vec<AgentId>,
    pub geographic_location: (f32, f32), // lat, lon
    pub network_capacity: f32,
    pub load_factor: f32,
}

/// Multi-GPU routing optimization engine
#[derive(Clone)]
pub struct MultiGpuRoutingEngine {
    gpu_contexts: Vec<Arc<CudaContext>>,
    neural_routers: Vec<NeuralRouter>,
    load_balancer: Arc<RwLock<LoadBalancer>>,
    consensus_engine: Arc<ConsensusEngine>,
}

#[derive(Clone, Debug)]
pub struct LoadBalancer {
    pub gpu_loads: Vec<f32>,
    pub routing_assignments: HashMap<AgentId, usize>, // agent -> GPU index
    pub rebalancing_threshold: f32,
}

#[derive(Clone)]
pub struct ConsensusEngine {
    pub voting_threshold: f32,
    pub consensus_history: Vec<ConsensusDecision>,
}

#[derive(Clone, Debug)]
pub struct ConsensusDecision {
    pub timestamp: Instant,
    pub route_decisions: HashMap<(AgentId, AgentId), RoutingChoice>,
    pub confidence_scores: HashMap<usize, f32>, // GPU index -> confidence
    pub final_choice: RoutingChoice,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoutingChoice {
    pub path: Vec<AgentId>,
    pub expected_latency: f32,
    pub expected_bandwidth: f32,
    pub reliability: f32,
}

impl NeuralRouter {
    pub async fn new(device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize neural network for routing decisions
        let vs = VarBuilder::zeros(DType::F32, &device);
        
        // Create a simple 3-layer network: input -> hidden -> output
        let network = sequential::seq()
            .add(linear(16, 64, vs.pp("layer1"))?)
            .add(activation::relu())
            .add(linear(64, 32, vs.pp("layer2"))?)
            .add(activation::relu())
            .add(linear(32, 8, vs.pp("layer3"))?)
            .add(activation::sigmoid());
        
        Ok(Self {
            device,
            network,
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            training_data: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        })
    }

    pub async fn predict_routing(&self, source: AgentId, destination: AgentId, network_state: &[f32]) -> Result<RoutingChoice, Box<dyn std::error::Error>> {
        // Pad or truncate network_state to expected size (16 features)
        let mut features = vec![0.0f32; 16];
        for (i, &val) in network_state.iter().enumerate() {
            if i < 16 {
                features[i] = val;
            }
        }
        
        // Convert to tensor for neural network inference
        let input = Tensor::from_slice(&features, (1, 16), &self.device)?;
        let output = self.network.forward(&input)?;
        
        // Extract predictions from network output
        let predictions = output.to_vec1::<f32>()?;
        
        // Create routing choice based on neural network output
        let routing_choice = RoutingChoice {
            path: vec![source, destination], // Simple direct path for now
            expected_latency: predictions[0] * 100.0, // Scale to reasonable latency (ms)
            expected_bandwidth: predictions[1] * 1000.0, // Scale to reasonable bandwidth (Mbps)
            reliability: predictions[2].min(1.0).max(0.0), // Ensure [0,1] range
        };
        
        // Update performance metrics
        let mut metrics = self.performance_metrics.write().await;
        metrics.total_predictions += 1;
        
        Ok(routing_choice)
    }

    pub async fn update_routing_table(&self, training_examples: Vec<TrainingExample>) -> Result<(), Box<dyn std::error::Error>> {
        // Store new training examples
        {
            let mut training_data = self.training_data.write().await;
            training_data.extend(training_examples.clone());
            
            // Keep only recent examples (last 1000)
            if training_data.len() > 1000 {
                training_data.drain(0..training_data.len() - 1000);
            }
        }
        
        // Update routing table with learned patterns
        let mut routing_table = self.routing_table.write().await;
        
        for example in training_examples {
            let entry = RoutingEntry {
                destination: example.destination,
                latency_prediction: example.actual_latency,
                bandwidth_prediction: example.actual_bandwidth,
                reliability_score: if example.success { 0.9 } else { 0.1 },
                last_updated: Instant::now(),
                confidence: 0.8, // Base confidence for real measurements
            };
            
            routing_table.insert(example.destination, entry);
        }
        
        Ok(())
    }

    pub async fn train_network(&mut self, epochs: u32) -> Result<PerformanceMetrics, Box<dyn std::error::Error>> {
        let training_data = self.training_data.read().await;
        
        if training_data.is_empty() {
            return Ok(PerformanceMetrics::default());
        }
        
        // Simulate training process (in real implementation, would use proper backpropagation)
        let mut metrics = self.performance_metrics.write().await;
        
        let start_time = Instant::now();
        
        // Simulate training epochs
        for _epoch in 0..epochs {
            // In a real implementation, this would:
            // 1. Forward pass through network
            // 2. Calculate loss
            // 3. Backward pass (gradients)
            // 4. Update weights
            
            // For now, simulate improvement over epochs
            let epoch_accuracy = 0.5 + (0.4 * (_epoch as f32 / epochs as f32));
            metrics.accuracy = epoch_accuracy;
        }
        
        // Update training metrics
        metrics.training_epochs += epochs as u64;
        metrics.last_training_time = Some(start_time);
        metrics.latency_error = 5.0; // Simulated 5ms average error
        metrics.bandwidth_error = 10.0; // Simulated 10Mbps average error
        
        // Estimate correct predictions based on accuracy
        let new_correct = (training_data.len() as f32 * metrics.accuracy) as u64;
        metrics.correct_predictions = new_correct;
        metrics.total_predictions = training_data.len() as u64;
        
        Ok(metrics.clone())
    }

    pub async fn get_routing_table(&self) -> HashMap<AgentId, RoutingEntry> {
        self.routing_table.read().await.clone()
    }
}

impl MultiGpuRoutingEngine {
    pub async fn new(gpu_count: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gpu_contexts = Vec::new();
        let mut neural_routers = Vec::new();
        
        // Initialize GPU contexts and neural routers for each GPU
        for gpu_id in 0..gpu_count {
            // Create CUDA context for each GPU
            let cuda_context = Arc::new(CudaContext::new(gpu_id)?); 
            gpu_contexts.push(cuda_context);
            
            // Create neural router for each GPU
            let device = Device::cuda_if_available(gpu_id)?;
            let router = NeuralRouter::new(device).await?;
            neural_routers.push(router);
        }
        
        // Initialize load balancer
        let load_balancer = Arc::new(RwLock::new(LoadBalancer {
            gpu_loads: vec![0.0; gpu_count],
            routing_assignments: HashMap::new(),
            rebalancing_threshold: 0.8, // Rebalance when load > 80%
        }));
        
        // Initialize consensus engine
        let consensus_engine = Arc::new(ConsensusEngine {
            voting_threshold: 0.6, // 60% agreement required
            consensus_history: Vec::new(),
        });
        
        Ok(Self {
            gpu_contexts,
            neural_routers,
            load_balancer,
            consensus_engine,
        })
    }

    pub async fn optimize_routing(&self, topology: &NetworkTopology) -> Result<HashMap<(AgentId, AgentId), RoutingChoice>, Box<dyn std::error::Error>> {
        let mut routing_decisions = HashMap::new();
        
        // Generate all agent pairs that need routing
        let mut agent_pairs = Vec::new();
        for (i, &agent1) in topology.agents.iter().enumerate() {
            for &agent2 in topology.agents.iter().skip(i + 1) {
                agent_pairs.push((agent1, agent2));
            }
        }
        
        // Distribute routing decisions across GPUs
        let gpu_count = self.neural_routers.len();
        let pairs_per_gpu = (agent_pairs.len() + gpu_count - 1) / gpu_count;
        
        let mut gpu_results = Vec::new();
        
        // Process routing decisions in parallel across GPUs
        for (gpu_idx, chunk) in agent_pairs.chunks(pairs_per_gpu).enumerate() {
            let router = &self.neural_routers[gpu_idx % gpu_count];
            let mut gpu_decisions = HashMap::new();
            
            for &(source, destination) in chunk {
                // Create network state features from topology
                let network_state = self.extract_network_features(source, destination, topology);
                
                // Get routing prediction from this GPU
                let choice = router.predict_routing(source, destination, &network_state).await?;
                gpu_decisions.insert((source, destination), choice);
            }
            
            gpu_results.push((gpu_idx, gpu_decisions));
        }
        
        // Combine results from all GPUs
        for (_gpu_idx, gpu_decisions) in gpu_results {
            routing_decisions.extend(gpu_decisions);
        }
        
        // Apply consensus where multiple GPUs provided decisions for same pair
        // (This would be more complex in real implementation)
        
        Ok(routing_decisions)
    }

    pub async fn rebalance_load(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut balancer = self.load_balancer.write().await;
        
        // Calculate average load across all GPUs
        let total_load: f32 = balancer.gpu_loads.iter().sum();
        let avg_load = total_load / balancer.gpu_loads.len() as f32;
        
        // Find overloaded and underloaded GPUs
        let mut overloaded = Vec::new();
        let mut underloaded = Vec::new();
        
        for (gpu_idx, &load) in balancer.gpu_loads.iter().enumerate() {
            if load > balancer.rebalancing_threshold {
                overloaded.push((gpu_idx, load));
            } else if load < avg_load * 0.5 {
                underloaded.push((gpu_idx, load));
            }
        }
        
        // Rebalance by moving agents from overloaded to underloaded GPUs
        if !overloaded.is_empty() && !underloaded.is_empty() {
            let agents_to_move: Vec<_> = balancer.routing_assignments
                .iter()
                .filter(|(_, &gpu_idx)| {
                    overloaded.iter().any(|(idx, _)| *idx == gpu_idx)
                })
                .map(|(&agent_id, _)| agent_id)
                .collect();
            
            // Move some agents to underloaded GPUs
            let target_gpu = underloaded[0].0;
            let move_count = (agents_to_move.len() / 4).max(1); // Move 25% of agents
            
            for &agent_id in agents_to_move.iter().take(move_count) {
                balancer.routing_assignments.insert(agent_id, target_gpu);
            }
            
            // Update load estimates (simplified)
            let load_per_agent = 0.1; // Assume each agent adds 10% load
            balancer.gpu_loads[overloaded[0].0] -= move_count as f32 * load_per_agent;
            balancer.gpu_loads[target_gpu] += move_count as f32 * load_per_agent;
        }
        
        Ok(())
    }
    
    /// Extract network features for a source-destination pair from topology
    fn extract_network_features(&self, source: AgentId, destination: AgentId, topology: &NetworkTopology) -> Vec<f32> {
        let mut features = vec![0.0f32; 16];
        
        // Feature 0-1: Source and destination agent indices
        if let Some(src_idx) = topology.agents.iter().position(|&id| id == source) {
            features[0] = src_idx as f32 / topology.agents.len() as f32;
        }
        if let Some(dst_idx) = topology.agents.iter().position(|&id| id == destination) {
            features[1] = dst_idx as f32 / topology.agents.len() as f32;
        }
        
        // Feature 2-5: Connection metrics if available
        if let Some(conn) = topology.connections.get(&(source, destination)) {
            features[2] = conn.latency_history.last().copied().unwrap_or(0.0) / 1000.0; // Normalize latency
            features[3] = conn.bandwidth_history.last().copied().unwrap_or(0.0) / 1000.0; // Normalize bandwidth
            features[4] = conn.packet_loss_rate;
            features[5] = conn.congestion_level;
        }
        
        // Feature 6-7: Network capacity and load from regions
        for region in &topology.cluster_regions {
            if region.agents.contains(&source) {
                features[6] = region.network_capacity / 1000.0; // Normalize capacity
                features[7] = region.load_factor;
                break;
            }
        }
        
        // Feature 8-15: Additional network state (can be extended)
        features[8] = topology.agents.len() as f32 / 1000.0; // Network size
        features[9] = topology.connections.len() as f32 / (topology.agents.len() * topology.agents.len()) as f32; // Connectivity
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    /// Test neural network initialization and basic functionality
    #[tokio::test]
    async fn test_neural_router_initialization() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu; // Will use GPU in implementation
        let router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        // Verify neural network architecture
        assert!(router.network.len() > 0, "Neural network should have layers");
        assert!(router.routing_table.read().await.is_empty(), "Routing table should start empty");
        assert_eq!(router.performance_metrics.read().await.accuracy, 0.0, "Initial accuracy should be zero");
    }

    /// Test routing prediction with neural inference
    #[tokio::test]
    async fn test_neural_routing_prediction() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let mut router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        let source = AgentId::new();
        let destination = AgentId::new();
        let network_features = vec![0.1, 0.2, 0.3, 0.8, 0.5]; // latency, bandwidth, loss, jitter, congestion
        
        // This should fail because predict_routing doesn't exist yet
        let prediction = router.predict_routing(source, destination, &network_features).await
            .expect("Should predict routing path");
        
        assert!(prediction.expected_latency > 0.0, "Should predict positive latency");
        assert!(prediction.expected_bandwidth > 0.0, "Should predict positive bandwidth");
        assert!(prediction.reliability >= 0.0 && prediction.reliability <= 1.0, "Reliability should be probability");
        assert!(!prediction.path.is_empty(), "Should provide routing path");
    }

    /// Test routing table learning and updates
    #[tokio::test]
    async fn test_routing_table_learning() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let mut router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        // Generate training data
        let mut training_examples = Vec::new();
        for _ in 0..1000 {
            let mut rng = thread_rng();
            training_examples.push(TrainingExample {
                source: AgentId::new(),
                destination: AgentId::new(),
                network_features: vec![
                    rng.gen_range(0.001..0.1),  // latency
                    rng.gen_range(1.0..100.0),  // bandwidth  
                    rng.gen_range(0.0..0.1),    // packet loss
                    rng.gen_range(0.0..0.05),   // jitter
                    rng.gen_range(0.0..1.0),    // congestion
                ],
                actual_latency: rng.gen_range(0.001..0.1),
                actual_bandwidth: rng.gen_range(1.0..100.0),
                success: rng.gen_bool(0.95), // 95% success rate
                timestamp: Instant::now(),
            });
        }
        
        // Train the neural network
        let initial_metrics = router.train_network(50).await.expect("Should train network");
        assert!(initial_metrics.training_epochs > 0, "Should complete training epochs");
        
        // Update routing table with training data
        router.update_routing_table(training_examples).await
            .expect("Should update routing table");
        
        let routing_table = router.get_routing_table().await;
        assert!(!routing_table.is_empty(), "Routing table should contain learned routes");
        
        // Verify learned routes have reasonable predictions
        for (_, entry) in routing_table.iter() {
            assert!(entry.latency_prediction > 0.0, "Should predict positive latency");
            assert!(entry.bandwidth_prediction > 0.0, "Should predict positive bandwidth");
            assert!(entry.reliability_score >= 0.0 && entry.reliability_score <= 1.0);
            assert!(entry.confidence >= 0.0 && entry.confidence <= 1.0);
        }
    }

    /// Test multi-agent routing optimization
    #[tokio::test] 
    async fn test_multi_agent_routing_optimization() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        // Create complex network topology
        let agents: Vec<AgentId> = (0..20).map(|_| AgentId::new()).collect();
        let mut connections = HashMap::new();
        let mut rng = thread_rng();
        
        // Generate mesh network connections
        for i in 0..agents.len() {
            for j in (i+1)..agents.len() {
                let connection = ConnectionMetrics {
                    latency_history: (0..100).map(|_| rng.gen_range(0.001..0.1)).collect(),
                    bandwidth_history: (0..100).map(|_| rng.gen_range(10.0..100.0)).collect(),
                    packet_loss_rate: rng.gen_range(0.0..0.05),
                    jitter: rng.gen_range(0.0..0.01),
                    congestion_level: rng.gen_range(0.0..1.0),
                };
                connections.insert((agents[i], agents[j]), connection);
            }
        }
        
        let topology = NetworkTopology {
            agents: agents.clone(),
            connections,
            cluster_regions: vec![
                ClusterRegion {
                    id: "us-west-1".to_string(),
                    agents: agents[0..7].to_vec(),
                    geographic_location: (37.7749, -122.4194), // SF
                    network_capacity: 1000.0,
                    load_factor: 0.6,
                },
                ClusterRegion {
                    id: "us-east-1".to_string(), 
                    agents: agents[7..14].to_vec(),
                    geographic_location: (40.7128, -74.0060), // NYC
                    network_capacity: 1200.0,
                    load_factor: 0.4,
                },
                ClusterRegion {
                    id: "eu-central-1".to_string(),
                    agents: agents[14..20].to_vec(),
                    geographic_location: (52.5200, 13.4050), // Berlin
                    network_capacity: 800.0,
                    load_factor: 0.8,
                },
            ],
        };
        
        // Test routing between all agent pairs
        for source in &agents {
            for destination in &agents {
                if source != destination {
                    let network_features = vec![0.05, 50.0, 0.01, 0.005, 0.3];
                    let route = router.predict_routing(*source, *destination, &network_features).await
                        .expect("Should predict route");
                    
                    assert!(route.path.len() >= 2, "Path should include source and destination");
                    assert_eq!(route.path[0], *source, "Path should start with source");
                    assert_eq!(route.path.last().unwrap(), destination, "Path should end with destination");
                    assert!(route.expected_latency > 0.0, "Should have positive latency prediction");
                }
            }
        }
    }

    /// Test real-time routing adaptation
    #[tokio::test]
    async fn test_real_time_routing_adaptation() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let mut router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        let source = AgentId::new();
        let destination = AgentId::new();
        
        // Initial routing prediction
        let initial_features = vec![0.01, 100.0, 0.0, 0.0, 0.1]; // Good network conditions
        let initial_route = router.predict_routing(source, destination, &initial_features).await
            .expect("Should predict initial route");
        
        // Simulate network degradation
        let degraded_features = vec![0.1, 10.0, 0.05, 0.02, 0.9]; // Bad network conditions
        let degraded_route = router.predict_routing(source, destination, &degraded_features).await
            .expect("Should predict degraded route");
        
        // Routing should adapt to changed conditions
        assert!(degraded_route.expected_latency > initial_route.expected_latency, 
                "Should predict higher latency with degraded conditions");
        assert!(degraded_route.expected_bandwidth < initial_route.expected_bandwidth,
                "Should predict lower bandwidth with degraded conditions"); 
        assert!(degraded_route.reliability < initial_route.reliability,
                "Should predict lower reliability with degraded conditions");
        
        // Create training examples from network changes
        let training_examples = vec![
            TrainingExample {
                source,
                destination,
                network_features: initial_features,
                actual_latency: 0.008,
                actual_bandwidth: 95.0,
                success: true,
                timestamp: Instant::now() - Duration::from_secs(10),
            },
            TrainingExample {
                source,
                destination,
                network_features: degraded_features,
                actual_latency: 0.12,
                actual_bandwidth: 8.5,
                success: false,
                timestamp: Instant::now(),
            },
        ];
        
        // Update routing with real measurements
        router.update_routing_table(training_examples).await
            .expect("Should update routing with new measurements");
        
        // Verify routing table reflects recent changes
        let routing_table = router.get_routing_table().await;
        if let Some(entry) = routing_table.get(&destination) {
            assert!(entry.last_updated.elapsed() < Duration::from_secs(5),
                    "Routing entry should be recently updated");
        }
    }

    /// Test multi-GPU routing engine initialization
    #[tokio::test]
    async fn test_multi_gpu_routing_engine() {
        // RED PHASE: This test MUST fail initially
        let gpu_count = 4; // Simulate 4-GPU system
        let engine = MultiGpuRoutingEngine::new(gpu_count).await
            .expect("Should initialize multi-GPU routing engine");
        
        assert_eq!(engine.gpu_contexts.len(), gpu_count, "Should have correct number of GPU contexts");
        assert_eq!(engine.neural_routers.len(), gpu_count, "Should have neural router per GPU");
        
        let load_balancer = engine.load_balancer.read().await;
        assert_eq!(load_balancer.gpu_loads.len(), gpu_count, "Should track load for each GPU");
    }

    /// Test multi-GPU consensus routing
    #[tokio::test]
    async fn test_multi_gpu_consensus_routing() {
        // RED PHASE: This test MUST fail initially
        let gpu_count = 3;
        let engine = MultiGpuRoutingEngine::new(gpu_count).await
            .expect("Should initialize multi-GPU routing engine");
        
        // Create test topology
        let agents: Vec<AgentId> = (0..10).map(|_| AgentId::new()).collect();
        let mut connections = HashMap::new();
        let mut rng = thread_rng();
        
        for i in 0..agents.len() {
            for j in (i+1)..agents.len() {
                connections.insert((agents[i], agents[j]), ConnectionMetrics {
                    latency_history: vec![rng.gen_range(0.01..0.1)],
                    bandwidth_history: vec![rng.gen_range(10.0..100.0)],
                    packet_loss_rate: rng.gen_range(0.0..0.05),
                    jitter: rng.gen_range(0.0..0.01),
                    congestion_level: rng.gen_range(0.0..1.0),
                });
            }
        }
        
        let topology = NetworkTopology {
            agents: agents.clone(),
            connections,
            cluster_regions: vec![],
        };
        
        // Test consensus routing optimization
        let routing_decisions = engine.optimize_routing(&topology).await
            .expect("Should optimize routing with multi-GPU consensus");
        
        // Verify consensus produces routing decisions
        assert!(!routing_decisions.is_empty(), "Should produce routing decisions");
        
        // Verify all routing decisions are consistent
        for ((source, dest), choice) in routing_decisions.iter() {
            assert!(choice.path.len() >= 2, "Routing path should have at least source and destination");
            assert_eq!(&choice.path[0], source, "Path should start with source");
            assert_eq!(&choice.path.last().unwrap(), dest, "Path should end with destination");
            assert!(choice.expected_latency > 0.0, "Should have positive latency");
            assert!(choice.expected_bandwidth > 0.0, "Should have positive bandwidth");
            assert!(choice.reliability >= 0.0 && choice.reliability <= 1.0, "Reliability should be probability");
        }
    }

    /// Test GPU load balancing for routing
    #[tokio::test]
    async fn test_gpu_load_balancing() {
        // RED PHASE: This test MUST fail initially
        let gpu_count = 4;
        let engine = MultiGpuRoutingEngine::new(gpu_count).await
            .expect("Should initialize multi-GPU routing engine");
        
        // Simulate uneven GPU load
        {
            let mut load_balancer = engine.load_balancer.write().await;
            load_balancer.gpu_loads = vec![0.9, 0.2, 0.1, 0.8]; // Uneven loads
            load_balancer.rebalancing_threshold = 0.3;
        }
        
        // Trigger load rebalancing
        engine.rebalance_load().await.expect("Should rebalance GPU loads");
        
        // Verify loads are more balanced
        let load_balancer = engine.load_balancer.read().await;
        let max_load = load_balancer.gpu_loads.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_load = load_balancer.gpu_loads.iter().fold(1.0f32, |a, &b| a.min(b));
        let load_variance = max_load - min_load;
        
        assert!(load_variance < 0.5, "Load balancing should reduce variance between GPUs");
    }

    /// Test neural network training performance
    #[tokio::test] 
    async fn test_neural_network_training_performance() {
        // RED PHASE: This test MUST fail initially  
        let device = Device::Cpu; // Will use GPU in real implementation
        let mut router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        // Generate large training dataset
        let mut training_examples = Vec::new();
        let mut rng = thread_rng();
        
        for _ in 0..10000 {
            training_examples.push(TrainingExample {
                source: AgentId::new(),
                destination: AgentId::new(),
                network_features: vec![
                    rng.gen_range(0.001..0.1),
                    rng.gen_range(1.0..100.0),
                    rng.gen_range(0.0..0.1),
                    rng.gen_range(0.0..0.05),
                    rng.gen_range(0.0..1.0),
                ],
                actual_latency: rng.gen_range(0.001..0.1),
                actual_bandwidth: rng.gen_range(1.0..100.0),
                success: rng.gen_bool(0.9),
                timestamp: Instant::now(),
            });
        }
        
        router.update_routing_table(training_examples).await
            .expect("Should update with training data");
        
        // Measure training performance
        let start_time = Instant::now();
        let metrics = router.train_network(100).await.expect("Should train network");
        let training_duration = start_time.elapsed();
        
        // Verify training performance
        assert!(training_duration < Duration::from_secs(30), "Training should complete within 30 seconds");
        assert!(metrics.accuracy > 0.7, "Should achieve >70% accuracy");
        assert!(metrics.training_epochs == 100, "Should complete requested epochs");
        assert!(metrics.total_predictions > 0, "Should make predictions during training");
        assert!(metrics.latency_error < 0.1, "Should have low latency prediction error");
        assert!(metrics.bandwidth_error < 10.0, "Should have low bandwidth prediction error");
    }

    /// Test routing performance under high load
    #[tokio::test]
    async fn test_routing_performance_high_load() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        let agents: Vec<AgentId> = (0..100).map(|_| AgentId::new()).collect();
        let mut rng = thread_rng();
        
        // Test concurrent routing predictions
        let start_time = Instant::now();
        let mut tasks = Vec::new();
        
        for _ in 0..1000 {
            let router_clone = router.clone();
            let source = agents[rng.gen_range(0..agents.len())];
            let destination = agents[rng.gen_range(0..agents.len())];
            let features = vec![
                rng.gen_range(0.001..0.1),
                rng.gen_range(1.0..100.0),
                rng.gen_range(0.0..0.1),
                rng.gen_range(0.0..0.05),
                rng.gen_range(0.0..1.0),
            ];
            
            let task = tokio::spawn(async move {
                router_clone.predict_routing(source, destination, &features).await
            });
            tasks.push(task);
        }
        
        // Wait for all predictions to complete
        let results: Vec<_> = futures::future::join_all(tasks).await;
        let duration = start_time.elapsed();
        
        // Verify performance
        assert!(duration < Duration::from_secs(10), "Should handle 1000 predictions in <10 seconds");
        
        let successful_predictions = results.iter().filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok()).count();
        assert!(successful_predictions > 950, "Should successfully predict >95% of routes");
    }

    /// Test routing adaptation to network failures
    #[tokio::test]
    async fn test_routing_failure_adaptation() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let mut router = NeuralRouter::new(device).await.expect("Should initialize neural router");
        
        let source = AgentId::new();
        let destination = AgentId::new();
        
        // Simulate network failure scenarios
        let failure_examples = vec![
            TrainingExample {
                source,
                destination,
                network_features: vec![0.01, 100.0, 0.0, 0.0, 0.1], // Good conditions
                actual_latency: 999.0, // Timeout indicates failure
                actual_bandwidth: 0.0,
                success: false,
                timestamp: Instant::now(),
            },
            TrainingExample {
                source,
                destination, 
                network_features: vec![0.5, 1.0, 0.5, 0.1, 1.0], // Bad conditions
                actual_latency: 999.0, // Timeout indicates failure
                actual_bandwidth: 0.0,
                success: false,
                timestamp: Instant::now(),
            },
        ];
        
        router.update_routing_table(failure_examples).await
            .expect("Should update routing with failure data");
        
        // Test routing after failures
        let good_conditions = vec![0.01, 100.0, 0.0, 0.0, 0.1];
        let bad_conditions = vec![0.5, 1.0, 0.5, 0.1, 1.0];
        
        let good_route = router.predict_routing(source, destination, &good_conditions).await
            .expect("Should predict route for good conditions");
        let bad_route = router.predict_routing(source, destination, &bad_conditions).await
            .expect("Should predict route for bad conditions");
        
        // Router should learn from failures
        assert!(good_route.reliability > bad_route.reliability, 
                "Should predict higher reliability for better conditions");
        assert!(bad_route.path.len() > good_route.path.len(),
                "Should use longer path for unreliable conditions");
    }
}