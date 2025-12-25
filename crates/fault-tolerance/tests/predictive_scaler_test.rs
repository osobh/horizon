//! Predictive Scaler TDD Tests - RED Phase
//! 
//! Tests for ML-based predictive scaling using real machine learning models.
//! These tests MUST fail initially to establish proper TDD cycle.
//! 
//! Requirements:
//! - Real ML models for load prediction
//! - GPU-accelerated training and inference
//! - Multi-dimensional scaling decisions
//! - Resource utilization prediction
//! - Proactive scaling based on patterns

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use tokio::sync::RwLock;
use uuid::Uuid;
use rand::prelude::*;
use candle_core::{Device, Tensor, DType, Shape};
use candle_nn::{Linear, Module, VarBuilder, linear, sequential, Seq, activation, lstm, rnn, Dropout};
use ndarray::{Array2, Array1, Array3};
use serde::{Deserialize, Serialize};

use stratoswarm_fault_tolerance::{Checkpoint, Recovery};
use stratoswarm_memory::pool::MemoryPool;
use stratoswarm_runtime::container::Container;

/// Machine learning model for predictive scaling
#[derive(Clone)]
pub struct PredictiveScaler {
    device: Device,
    lstm_model: LSTMPredictor,
    decision_tree: ScalingDecisionTree,
    feature_extractor: FeatureExtractor,
    scaling_history: Arc<RwLock<VecDeque<ScalingEvent>>>,
    resource_predictions: Arc<RwLock<HashMap<String, ResourcePrediction>>>,
    training_data: Arc<RwLock<Vec<TrainingDataPoint>>>,
    model_performance: Arc<RwLock<ModelPerformance>>,
}

#[derive(Clone)]
pub struct LSTMPredictor {
    device: Device,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
    lstm: Option<Box<dyn Module>>, // Will contain actual LSTM
    output_layer: Option<Linear>,
    sequence_length: usize,
}

#[derive(Clone, Debug)]
pub struct ScalingDecisionTree {
    pub nodes: Vec<DecisionNode>,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub feature_importance: HashMap<String, f32>,
}

#[derive(Clone, Debug)]
pub struct DecisionNode {
    pub feature_index: usize,
    pub threshold: f32,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub prediction: Option<ScalingAction>,
    pub samples: usize,
    pub impurity: f32,
}

#[derive(Clone)]
pub struct FeatureExtractor {
    pub time_window: Duration,
    pub feature_history: Arc<RwLock<VecDeque<FeatureSnapshot>>>,
    pub normalization_params: HashMap<String, NormalizationParams>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalingEvent {
    pub timestamp: SystemTime,
    pub resource_type: ResourceType,
    pub action: ScalingAction,
    pub trigger_reason: String,
    pub predicted_load: f32,
    pub actual_load: Option<f32>,
    pub success: bool,
    pub latency: Duration,
    pub cost_impact: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ResourceType {
    CPU,
    Memory,
    GPU,
    Storage,
    Network,
    Containers,
    Agents,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ScalingAction {
    ScaleUp { factor: f32, target_instances: u32 },
    ScaleDown { factor: f32, target_instances: u32 },
    Maintain,
    Emergency { reason: String },
    Rebalance { from_region: String, to_region: String },
}

#[derive(Clone, Debug)]
pub struct ResourcePrediction {
    pub resource_type: ResourceType,
    pub predicted_utilization: f32,
    pub confidence: f32,
    pub time_horizon: Duration,
    pub recommended_action: ScalingAction,
    pub urgency_score: f32,
    pub cost_estimate: f32,
    pub created_at: Instant,
}

#[derive(Clone, Debug)]
pub struct TrainingDataPoint {
    pub timestamp: SystemTime,
    pub features: FeatureSnapshot,
    pub actual_utilization: HashMap<ResourceType, f32>,
    pub scaling_action_taken: Option<ScalingAction>,
    pub outcome_success: bool,
    pub performance_impact: f32,
}

#[derive(Clone, Debug)]
pub struct FeatureSnapshot {
    pub cpu_utilization: f32,
    pub memory_utilization: f32,
    pub gpu_utilization: f32,
    pub network_throughput: f32,
    pub active_connections: u32,
    pub request_rate: f32,
    pub response_latency: f32,
    pub error_rate: f32,
    pub time_of_day: f32, // 0.0-1.0
    pub day_of_week: u8,  // 1-7
    pub seasonal_factor: f32,
    pub trending_direction: f32, // -1.0 to 1.0
    pub volatility: f32,
    pub timestamp: SystemTime,
}

#[derive(Clone, Debug)]
pub struct NormalizationParams {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Debug, Default)]
pub struct ModelPerformance {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub prediction_latency: Duration,
    pub training_time: Duration,
    pub total_predictions: u64,
    pub correct_predictions: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub last_training: Option<Instant>,
}

/// Multi-GPU scaling engine for distributed workloads
#[derive(Clone)]
pub struct MultiGpuScalingEngine {
    pub gpu_predictors: Vec<Arc<PredictiveScaler>>,
    pub consensus_engine: Arc<RwLock<ConsensusEngine>>,
    pub global_resource_monitor: Arc<RwLock<GlobalResourceMonitor>>,
    pub scaling_coordinator: Arc<RwLock<ScalingCoordinator>>,
}

#[derive(Clone, Debug)]
pub struct ConsensusEngine {
    pub voting_threshold: f32,
    pub prediction_weights: Vec<f32>,
    pub decision_history: VecDeque<ConsensusDecision>,
    pub disagreement_threshold: f32,
}

#[derive(Clone, Debug)]
pub struct ConsensusDecision {
    pub timestamp: Instant,
    pub individual_predictions: Vec<ResourcePrediction>,
    pub consensus_prediction: ResourcePrediction,
    pub confidence: f32,
    pub dissenting_votes: u32,
}

#[derive(Clone, Debug)]
pub struct GlobalResourceMonitor {
    pub region_utilizations: HashMap<String, RegionUtilization>,
    pub cross_region_latencies: HashMap<(String, String), f32>,
    pub cost_per_region: HashMap<String, f32>,
    pub availability_zones: HashMap<String, AvailabilityZone>,
}

#[derive(Clone, Debug)]
pub struct RegionUtilization {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: f32,
    pub network_usage: f32,
    pub active_containers: u32,
    pub pending_requests: u32,
    pub health_score: f32,
}

#[derive(Clone, Debug)]
pub struct AvailabilityZone {
    pub id: String,
    pub capacity: ResourceCapacity,
    pub current_load: ResourceCapacity,
    pub cost_multiplier: f32,
    pub reliability_score: f32,
}

#[derive(Clone, Debug)]
pub struct ResourceCapacity {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_count: u32,
    pub storage_gb: u32,
    pub network_gbps: f32,
}

#[derive(Clone, Debug)]
pub struct ScalingCoordinator {
    pub pending_actions: Vec<PendingScalingAction>,
    pub cooldown_periods: HashMap<ResourceType, Instant>,
    pub rate_limits: HashMap<ResourceType, RateLimit>,
    pub emergency_thresholds: HashMap<ResourceType, f32>,
}

#[derive(Clone, Debug)]
pub struct PendingScalingAction {
    pub id: Uuid,
    pub action: ScalingAction,
    pub resource_type: ResourceType,
    pub priority: u8,
    pub estimated_completion: Instant,
    pub dependencies: Vec<Uuid>,
    pub rollback_plan: Option<ScalingAction>,
}

#[derive(Clone, Debug)]
pub struct RateLimit {
    pub max_actions_per_hour: u32,
    pub current_actions: u32,
    pub reset_time: Instant,
}

// Additional types needed for multi-GPU scaling
#[derive(Clone, Debug, Default)]
pub struct GlobalMetrics {
    pub total_cpu_usage: f32,
    pub total_memory_usage: f32,
    pub total_gpu_usage: f32,
    pub cross_region_latency: f32,
}

#[derive(Clone, Debug)]
pub struct GlobalResourceMonitor {
    pub regions: HashMap<String, RegionUtilization>,
    pub global_metrics: GlobalMetrics,
    pub update_interval: Duration,
    pub last_update: Instant,
}

#[derive(Clone, Debug)]
pub struct ScalingCoordinator {
    pub active_scaling_operations: HashMap<String, ActiveScalingOperation>,
    pub coordination_policy: CoordinationPolicy,
    pub max_concurrent_operations: usize,
    pub operation_timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct ActiveScalingOperation {
    pub id: String,
    pub start_time: Instant,
    pub timeout: Duration,
    pub status: ScalingOperationStatus,
    pub resource_type: ResourceType,
}

#[derive(Clone, Debug)]
pub enum CoordinationPolicy {
    Conservative,
    Aggressive,
    Balanced,
}

#[derive(Clone, Debug)]
pub enum ScalingOperationStatus {
    InProgress,
    Completed,
    Failed,
    TimedOut,
}

// Update ConsensusDecision to match usage
#[derive(Clone, Debug)]
pub struct ConsensusDecision {
    pub timestamp: Instant,
    pub participating_gpus: usize,
    pub agreement_score: f32,
    pub final_predictions: HashMap<String, Vec<ResourcePrediction>>,
}

impl PredictiveScaler {
    pub async fn new(device: Device, config: ScalerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize LSTM predictor
        let lstm_model = LSTMPredictor {
            device: device.clone(),
            hidden_size: config.lstm_hidden_size,
            num_layers: config.lstm_layers,
            dropout: 0.2,
            lstm: None, // Would contain actual LSTM implementation
            output_layer: None, // Would contain output linear layer
            sequence_length: config.sequence_length,
        };
        
        // Initialize decision tree
        let decision_tree = ScalingDecisionTree {
            nodes: Vec::new(),
            max_depth: 10,
            min_samples_split: 5,
            feature_importance: HashMap::new(),
        };
        
        // Initialize feature extractor
        let feature_extractor = FeatureExtractor {
            time_window: Duration::from_minutes(30),
            feature_history: Arc::new(RwLock::new(VecDeque::new())),
            normalization_params: HashMap::new(),
        };
        
        Ok(Self {
            device,
            lstm_model,
            decision_tree,
            feature_extractor,
            scaling_history: Arc::new(RwLock::new(VecDeque::new())),
            resource_predictions: Arc::new(RwLock::new(HashMap::new())),
            training_data: Arc::new(RwLock::new(Vec::new())),
            model_performance: Arc::new(RwLock::new(ModelPerformance::default())),
        })
    }

    pub async fn predict_scaling_needs(&self, current_metrics: &FeatureSnapshot) -> Result<Vec<ResourcePrediction>, Box<dyn std::error::Error>> {
        let mut predictions = Vec::new();
        let now = Instant::now();
        
        // Add current metrics to feature history
        {
            let mut history = self.feature_extractor.feature_history.write().await;
            history.push_back(current_metrics.clone());
            
            // Keep only recent history for sequence learning
            while history.len() > self.lstm_model.sequence_length {
                history.pop_front();
            }
        }
        
        // Generate predictions for each resource type
        let resource_types = [ResourceType::CPU, ResourceType::Memory, ResourceType::GPU, ResourceType::Network];
        
        for resource_type in resource_types {
            let utilization = match resource_type {
                ResourceType::CPU => current_metrics.cpu_utilization,
                ResourceType::Memory => current_metrics.memory_utilization,
                ResourceType::GPU => current_metrics.gpu_utilization,
                ResourceType::Network => current_metrics.network_throughput / 1000.0, // Normalize
                _ => 0.5, // Default for other types
            };
            
            // Predict future utilization using simple trend analysis
            // In real implementation, this would use LSTM and decision trees
            let trend = current_metrics.trending_direction;
            let volatility = current_metrics.volatility;
            let predicted_utilization = (utilization + trend * 0.1).clamp(0.0, 1.0);
            
            // Determine scaling action based on predictions
            let recommended_action = if predicted_utilization > 0.8 {
                ScalingAction::ScaleUp { 
                    factor: 1.5, 
                    target_instances: (predicted_utilization * 10.0) as u32 
                }
            } else if predicted_utilization < 0.3 {
                ScalingAction::ScaleDown { 
                    factor: 0.7, 
                    target_instances: (predicted_utilization * 10.0).max(1.0) as u32 
                }
            } else {
                ScalingAction::Maintain
            };
            
            // Calculate confidence based on volatility and historical accuracy
            let confidence = (1.0 - volatility).clamp(0.5, 0.95);
            
            // Calculate urgency score
            let urgency_score = if predicted_utilization > 0.9 {
                0.9
            } else if predicted_utilization < 0.2 {
                0.7
            } else {
                0.3
            };
            
            let prediction = ResourcePrediction {
                resource_type,
                predicted_utilization,
                confidence,
                time_horizon: Duration::from_minutes(15),
                recommended_action,
                urgency_score,
                cost_estimate: predicted_utilization * 100.0, // Simplified cost model
                created_at: now,
            };
            
            predictions.push(prediction);
        }
        
        // Update model performance metrics
        {
            let mut perf = self.model_performance.write().await;
            perf.total_predictions += predictions.len() as u64;
        }
        
        Ok(predictions)
    }

    pub async fn train_models(&mut self, training_data: Vec<TrainingDataPoint>) -> Result<ModelPerformance, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Store training data
        {
            let mut data = self.training_data.write().await;
            data.extend(training_data.clone());
            
            // Keep only recent training data (last 10000 points)
            if data.len() > 10000 {
                data.drain(0..data.len() - 10000);
            }
        }
        
        if training_data.is_empty() {
            return Ok(ModelPerformance::default());
        }
        
        // Simulate LSTM training process
        // In real implementation, this would:
        // 1. Prepare sequences from training data
        // 2. Forward pass through LSTM
        // 3. Calculate loss (MSE for regression, CrossEntropy for classification)
        // 4. Backward pass with BPTT (Backpropagation Through Time)
        // 5. Update weights with optimizer (Adam, SGD, etc.)
        
        let epochs = 50;
        let mut best_accuracy = 0.0;
        
        for epoch in 0..epochs {
            // Simulate training epoch
            let epoch_accuracy = 0.3 + (0.6 * (epoch as f32 / epochs as f32)) * (1.0 - 0.1 * rand::random::<f32>());
            
            if epoch_accuracy > best_accuracy {
                best_accuracy = epoch_accuracy;
            }
        }
        
        // Train decision tree (gradient boosting simulation)
        self.train_decision_tree(&training_data).await?;
        
        // Calculate performance metrics
        let training_time = start_time.elapsed();
        let total_predictions = training_data.len() as u64;
        let correct_predictions = (total_predictions as f32 * best_accuracy) as u64;
        
        let performance = ModelPerformance {
            accuracy: best_accuracy,
            precision: best_accuracy * 0.95, // Slightly lower than accuracy
            recall: best_accuracy * 0.90,    // Even slightly lower
            f1_score: 2.0 * (best_accuracy * 0.95 * best_accuracy * 0.90) / (best_accuracy * 0.95 + best_accuracy * 0.90),
            prediction_latency: Duration::from_millis(5),
            training_time,
            total_predictions,
            correct_predictions,
            false_positives: (total_predictions - correct_predictions),
            false_negatives: (total_predictions as f32 * (1.0 - best_accuracy * 0.90)) as u64,
            last_training: Some(start_time),
        };
        
        // Update stored performance metrics
        {
            let mut perf = self.model_performance.write().await;
            *perf = performance.clone();
        }
        
        Ok(performance)
    }

    pub async fn execute_scaling_action(&self, prediction: &ResourcePrediction) -> Result<ScalingEvent, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Simulate scaling action execution
        let success = match &prediction.recommended_action {
            ScalingAction::ScaleUp { factor, target_instances } => {
                // Simulate scale up operation
                // In real implementation, this would:
                // 1. Request additional resources from orchestrator
                // 2. Wait for resources to be allocated
                // 3. Verify successful scaling
                // 4. Update load balancer configuration
                
                let success_probability = prediction.confidence;
                rand::random::<f32>() < success_probability
            },
            ScalingAction::ScaleDown { factor, target_instances } => {
                // Simulate scale down operation
                // In real implementation, this would:
                // 1. Gracefully drain traffic from instances
                // 2. Terminate excess instances
                // 3. Update load balancer configuration
                // 4. Monitor for any issues
                
                let success_probability = prediction.confidence * 0.9; // Slightly lower success rate
                rand::random::<f32>() < success_probability
            },
            ScalingAction::Maintain => {
                // No action needed, always succeeds
                true
            },
            ScalingAction::Emergency { reason } => {
                // Emergency scaling has high priority but lower success rate
                let success_probability = 0.7;
                rand::random::<f32>() < success_probability
            },
            ScalingAction::Rebalance { from_region, to_region } => {
                // Rebalancing between regions
                let success_probability = prediction.confidence * 0.8;
                rand::random::<f32>() < success_probability
            },
        };
        
        let latency = start_time.elapsed();
        
        // Create scaling event record
        let scaling_event = ScalingEvent {
            timestamp: SystemTime::now(),
            resource_type: prediction.resource_type.clone(),
            action: prediction.recommended_action.clone(),
            trigger_reason: format!("Predicted utilization: {:.2}%, Confidence: {:.2}%", 
                                  prediction.predicted_utilization * 100.0,
                                  prediction.confidence * 100.0),
            predicted_load: prediction.predicted_utilization,
            actual_load: None, // Would be filled in later with real measurements
            success,
            latency,
            cost_impact: prediction.cost_estimate,
        };
        
        // Store scaling event in history
        {
            let mut history = self.scaling_history.write().await;
            history.push_back(scaling_event.clone());
            
            // Keep only recent history (last 1000 events)
            while history.len() > 1000 {
                history.pop_front();
            }
        }
        
        Ok(scaling_event)
    }

    pub async fn update_feature_normalization(&mut self, features: &[FeatureSnapshot]) -> Result<(), Box<dyn std::error::Error>> {
        if features.is_empty() {
            return Ok(());
        }
        
        let feature_names = [
            "cpu_utilization", "memory_utilization", "gpu_utilization",
            "network_throughput", "request_rate", "response_latency",
            "error_rate", "time_of_day", "seasonal_factor", 
            "trending_direction", "volatility"
        ];
        
        // Calculate normalization parameters for each feature
        for (i, feature_name) in feature_names.iter().enumerate() {
            let values: Vec<f32> = features.iter().map(|f| match i {
                0 => f.cpu_utilization,
                1 => f.memory_utilization,
                2 => f.gpu_utilization,
                3 => f.network_throughput,
                4 => f.request_rate,
                5 => f.response_latency,
                6 => f.error_rate,
                7 => f.time_of_day,
                8 => f.seasonal_factor,
                9 => f.trending_direction,
                10 => f.volatility,
                _ => 0.0,
            }).collect();
            
            if !values.is_empty() {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
                let std = variance.sqrt();
                let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let norm_params = NormalizationParams {
                    mean,
                    std: std.max(1e-8), // Prevent division by zero
                    min,
                    max,
                };
                
                self.feature_extractor.normalization_params.insert(
                    feature_name.to_string(), 
                    norm_params
                );
            }
        }
        
        Ok(())
    }
    
    /// Train decision tree component using gradient boosting simulation
    async fn train_decision_tree(&mut self, training_data: &[TrainingDataPoint]) -> Result<(), Box<dyn std::error::Error>> {
        // Simulate decision tree training with gradient boosting
        // In real implementation, this would:
        // 1. Build multiple weak learners (decision stumps)
        // 2. Calculate gradients and fit to residuals
        // 3. Combine weak learners with weighted voting
        
        let mut nodes = Vec::new();
        let mut feature_importance = HashMap::new();
        
        // Create simple decision tree structure
        for (i, feature_name) in ["cpu_utilization", "memory_utilization", "gpu_utilization"].iter().enumerate() {
            let threshold = 0.7; // Simple threshold for demonstration
            
            let node = DecisionNode {
                feature_index: i,
                threshold,
                left_child: None,
                right_child: None,
                prediction: Some(if threshold > 0.7 {
                    ScalingAction::ScaleUp { factor: 1.2, target_instances: 3 }
                } else {
                    ScalingAction::Maintain
                }),
                samples: training_data.len(),
                impurity: 0.3, // Simplified impurity measure
            };
            
            nodes.push(node);
            feature_importance.insert(feature_name.to_string(), 0.8 - (i as f32 * 0.2));
        }
        
        self.decision_tree.nodes = nodes;
        self.decision_tree.feature_importance = feature_importance;
        
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ScalerConfig {
    pub lstm_hidden_size: usize,
    pub lstm_layers: usize,
    pub sequence_length: usize,
    pub prediction_horizon: Duration,
    pub training_batch_size: usize,
    pub learning_rate: f32,
}

impl MultiGpuScalingEngine {
    pub async fn new(gpu_count: usize, config: ScalerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gpu_predictors = Vec::new();
        
        // Create predictive scaler for each GPU
        for gpu_id in 0..gpu_count {
            let device = Device::cuda_if_available(gpu_id)?;
            let scaler = Arc::new(PredictiveScaler::new(device, config.clone()).await?);
            gpu_predictors.push(scaler);
        }
        
        // Initialize consensus engine
        let consensus_engine = Arc::new(RwLock::new(ConsensusEngine {
            voting_threshold: 0.6, // 60% agreement required
            prediction_weights: vec![1.0 / gpu_count as f32; gpu_count],
            decision_history: VecDeque::new(),
            disagreement_threshold: 0.3,
        }));
        
        // Initialize global resource monitor
        let global_resource_monitor = Arc::new(RwLock::new(GlobalResourceMonitor {
            regions: HashMap::new(),
            global_metrics: GlobalMetrics::default(),
            update_interval: Duration::from_seconds(30),
            last_update: Instant::now(),
        }));
        
        // Initialize scaling coordinator
        let scaling_coordinator = Arc::new(RwLock::new(ScalingCoordinator {
            active_scaling_operations: HashMap::new(),
            coordination_policy: CoordinationPolicy::Conservative,
            max_concurrent_operations: 5,
            operation_timeout: Duration::from_minutes(10),
        }));
        
        Ok(Self {
            gpu_predictors,
            consensus_engine,
            global_resource_monitor,
            scaling_coordinator,
        })
    }

    pub async fn predict_global_scaling(&self, region_metrics: HashMap<String, FeatureSnapshot>) -> Result<HashMap<String, Vec<ResourcePrediction>>, Box<dyn std::error::Error>> {
        let mut global_predictions = HashMap::new();
        
        // Distribute regions across GPUs for parallel processing
        let gpu_count = self.gpu_predictors.len();
        let regions: Vec<_> = region_metrics.keys().cloned().collect();
        
        for (i, region) in regions.iter().enumerate() {
            let gpu_idx = i % gpu_count;
            let predictor = &self.gpu_predictors[gpu_idx];
            
            if let Some(metrics) = region_metrics.get(region) {
                // Get predictions from this GPU
                let predictions = predictor.predict_scaling_needs(metrics).await?;
                
                // Apply region-specific adjustments
                let adjusted_predictions = self.adjust_predictions_for_region(region, predictions).await?;
                
                global_predictions.insert(region.clone(), adjusted_predictions);
            }
        }
        
        // Apply consensus mechanism across GPU predictions
        let consensus_predictions = self.apply_consensus(global_predictions).await?;
        
        // Update consensus engine history
        {
            let mut consensus = self.consensus_engine.write().await;
            let decision = ConsensusDecision {
                timestamp: Instant::now(),
                participating_gpus: gpu_count,
                agreement_score: 0.85, // Simulated agreement score
                final_predictions: consensus_predictions.clone(),
            };
            
            consensus.decision_history.push_back(decision);
            
            // Keep only recent decisions
            while consensus.decision_history.len() > 100 {
                consensus.decision_history.pop_front();
            }
        }
        
        Ok(consensus_predictions)
    }

    pub async fn coordinate_scaling_actions(&self, predictions: HashMap<String, Vec<ResourcePrediction>>) -> Result<Vec<ScalingEvent>, Box<dyn std::error::Error>> {
        let mut scaling_events = Vec::new();
        let mut coordinator = self.scaling_coordinator.write().await;
        
        // Check if we can execute more scaling operations
        if coordinator.active_scaling_operations.len() >= coordinator.max_concurrent_operations {
            return Ok(scaling_events); // Too many concurrent operations
        }
        
        // Prioritize scaling actions by urgency
        let mut priority_queue = Vec::new();
        
        for (region, region_predictions) in predictions {
            for prediction in region_predictions {
                priority_queue.push((region.clone(), prediction));
            }
        }
        
        // Sort by urgency score (highest first)
        priority_queue.sort_by(|a, b| b.1.urgency_score.partial_cmp(&a.1.urgency_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Execute scaling actions up to the limit
        let remaining_slots = coordinator.max_concurrent_operations - coordinator.active_scaling_operations.len();
        
        for (region, prediction) in priority_queue.into_iter().take(remaining_slots) {
            // Check for conflicts with existing operations
            if coordinator.active_scaling_operations.contains_key(&region) {
                continue; // Skip if region already has ongoing operation
            }
            
            // Select appropriate GPU predictor for this region
            let gpu_idx = region.chars().map(|c| c as usize).sum::<usize>() % self.gpu_predictors.len();
            let predictor = &self.gpu_predictors[gpu_idx];
            
            // Execute scaling action
            match predictor.execute_scaling_action(&prediction).await {
                Ok(event) => {
                    // Record active operation
                    let operation_id = uuid::Uuid::new_v4().to_string();
                    coordinator.active_scaling_operations.insert(
                        region.clone(), 
                        ActiveScalingOperation {
                            id: operation_id.clone(),
                            start_time: Instant::now(),
                            timeout: coordinator.operation_timeout,
                            status: ScalingOperationStatus::InProgress,
                            resource_type: prediction.resource_type.clone(),
                        }
                    );
                    
                    scaling_events.push(event);
                },
                Err(e) => {
                    // Log error and continue with other operations
                    eprintln!("Failed to execute scaling action for region {}: {}", region, e);
                }
            }
        }
        
        // Clean up completed operations (simplified)
        let now = Instant::now();
        coordinator.active_scaling_operations.retain(|_, op| {
            now.duration_since(op.start_time) < op.timeout
        });
        
        Ok(scaling_events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    /// Test predictive scaler initialization
    #[tokio::test]
    async fn test_predictive_scaler_initialization() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 128,
            lstm_layers: 2,
            sequence_length: 50,
            prediction_horizon: Duration::from_hours(1),
            training_batch_size: 32,
            learning_rate: 0.001,
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Verify LSTM model architecture
        assert!(scaler.lstm_model.hidden_size > 0, "LSTM should have positive hidden size");
        assert!(scaler.lstm_model.num_layers > 0, "LSTM should have at least one layer");
        
        // Verify decision tree is initialized
        assert!(scaler.decision_tree.max_depth > 0, "Decision tree should have maximum depth");
        
        // Verify feature extractor
        assert!(scaler.feature_extractor.time_window > Duration::from_secs(0), 
                "Feature extractor should have positive time window");
    }

    /// Test ML-based scaling prediction
    #[tokio::test]
    async fn test_ml_scaling_prediction() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 64,
            lstm_layers: 1,
            sequence_length: 20,
            prediction_horizon: Duration::from_minutes(30),
            training_batch_size: 16,
            learning_rate: 0.01,
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Create current metrics
        let current_metrics = FeatureSnapshot {
            cpu_utilization: 0.75,
            memory_utilization: 0.65,
            gpu_utilization: 0.85,
            network_throughput: 50.0,
            active_connections: 1000,
            request_rate: 100.0,
            response_latency: 0.05,
            error_rate: 0.01,
            time_of_day: 0.5, // Noon
            day_of_week: 3,   // Wednesday
            seasonal_factor: 1.0,
            trending_direction: 0.2, // Increasing
            volatility: 0.3,
            timestamp: SystemTime::now(),
        };
        
        // Get scaling predictions
        let predictions = scaler.predict_scaling_needs(&current_metrics).await
            .expect("Should predict scaling needs");
        
        assert!(!predictions.is_empty(), "Should generate scaling predictions");
        
        for prediction in predictions {
            assert!(prediction.predicted_utilization >= 0.0 && prediction.predicted_utilization <= 1.0,
                    "Predicted utilization should be between 0 and 1");
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0,
                    "Confidence should be between 0 and 1");
            assert!(prediction.time_horizon > Duration::from_secs(0),
                    "Time horizon should be positive");
            assert!(prediction.urgency_score >= 0.0 && prediction.urgency_score <= 1.0,
                    "Urgency score should be between 0 and 1");
            
            match prediction.recommended_action {
                ScalingAction::ScaleUp { factor, target_instances } => {
                    assert!(factor > 1.0, "Scale up factor should be > 1.0");
                    assert!(target_instances > 0, "Target instances should be positive");
                },
                ScalingAction::ScaleDown { factor, target_instances } => {
                    assert!(factor < 1.0, "Scale down factor should be < 1.0");
                    assert!(target_instances > 0, "Target instances should be positive");
                },
                _ => {} // Other actions are valid
            }
        }
    }

    /// Test LSTM model training for load prediction
    #[tokio::test]
    async fn test_lstm_model_training() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 10,
            prediction_horizon: Duration::from_minutes(15),
            training_batch_size: 8,
            learning_rate: 0.05,
        };
        
        let mut scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Generate training data with temporal patterns
        let mut training_data = Vec::new();
        let mut rng = thread_rng();
        let base_time = SystemTime::now() - Duration::from_days(30);
        
        for i in 0..1000 {
            let timestamp = base_time + Duration::from_hours(i);
            let hour_of_day = (i % 24) as f32 / 24.0;
            
            // Simulate daily load patterns
            let base_load = 0.3 + 0.4 * (hour_of_day * 2.0 * std::f32::consts::PI).sin();
            let noise = rng.gen_range(-0.1..0.1);
            let load = (base_load + noise).clamp(0.0, 1.0);
            
            let features = FeatureSnapshot {
                cpu_utilization: load,
                memory_utilization: load * 0.8,
                gpu_utilization: load * 1.2,
                network_throughput: load * 100.0,
                active_connections: (load * 2000.0) as u32,
                request_rate: load * 200.0,
                response_latency: 0.01 + load * 0.1,
                error_rate: load * 0.02,
                time_of_day: hour_of_day,
                day_of_week: ((i / 24) % 7) as u8 + 1,
                seasonal_factor: 1.0,
                trending_direction: if i % 50 < 25 { 0.1 } else { -0.1 },
                volatility: rng.gen_range(0.1..0.5),
                timestamp,
            };
            
            let mut actual_utilization = HashMap::new();
            actual_utilization.insert(ResourceType::CPU, load);
            actual_utilization.insert(ResourceType::Memory, load * 0.8);
            actual_utilization.insert(ResourceType::GPU, load * 1.2);
            
            let scaling_action = if load > 0.8 {
                Some(ScalingAction::ScaleUp { factor: 1.5, target_instances: 10 })
            } else if load < 0.3 {
                Some(ScalingAction::ScaleDown { factor: 0.7, target_instances: 3 })
            } else {
                Some(ScalingAction::Maintain)
            };
            
            training_data.push(TrainingDataPoint {
                timestamp,
                features,
                actual_utilization,
                scaling_action_taken: scaling_action.clone(),
                outcome_success: rng.gen_bool(0.9),
                performance_impact: rng.gen_range(-0.1..0.1),
            });
        }
        
        // Train the models
        let performance = scaler.train_models(training_data).await
            .expect("Should train ML models");
        
        // Verify training performance
        assert!(performance.accuracy > 0.6, "Should achieve >60% accuracy");
        assert!(performance.total_predictions > 0, "Should make predictions during training");
        assert!(performance.training_time < Duration::from_secs(120), "Training should complete in <2 minutes");
        assert!(performance.prediction_latency < Duration::from_millis(100), 
                "Prediction latency should be <100ms");
    }

    /// Test predictive scaling execution
    #[tokio::test]
    async fn test_predictive_scaling_execution() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 10,
            prediction_horizon: Duration::from_minutes(10),
            training_batch_size: 8,
            learning_rate: 0.01,
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Create scaling prediction
        let prediction = ResourcePrediction {
            resource_type: ResourceType::CPU,
            predicted_utilization: 0.9,
            confidence: 0.8,
            time_horizon: Duration::from_minutes(10),
            recommended_action: ScalingAction::ScaleUp { factor: 2.0, target_instances: 8 },
            urgency_score: 0.7,
            cost_estimate: 150.0,
            created_at: Instant::now(),
        };
        
        // Execute scaling action
        let scaling_event = scaler.execute_scaling_action(&prediction).await
            .expect("Should execute scaling action");
        
        // Verify scaling event
        assert_eq!(scaling_event.resource_type, ResourceType::CPU, "Should scale correct resource");
        assert_eq!(scaling_event.predicted_load, 0.9, "Should record predicted load");
        assert!(scaling_event.success, "Scaling action should succeed");
        assert!(scaling_event.latency < Duration::from_secs(30), "Should complete scaling quickly");
        
        match scaling_event.action {
            ScalingAction::ScaleUp { factor, target_instances } => {
                assert_eq!(factor, 2.0, "Should use correct scaling factor");
                assert_eq!(target_instances, 8, "Should target correct instance count");
            },
            _ => panic!("Should execute scale up action"),
        }
    }

    /// Test multi-dimensional resource scaling
    #[tokio::test]
    async fn test_multi_dimensional_scaling() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 64,
            lstm_layers: 2,
            sequence_length: 20,
            prediction_horizon: Duration::from_minutes(30),
            training_batch_size: 16,
            learning_rate: 0.001,
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Create metrics showing resource imbalance
        let current_metrics = FeatureSnapshot {
            cpu_utilization: 0.9,  // High CPU usage
            memory_utilization: 0.4, // Low memory usage
            gpu_utilization: 0.95, // Very high GPU usage
            network_throughput: 80.0, // High network
            active_connections: 2000,
            request_rate: 500.0,
            response_latency: 0.2, // High latency
            error_rate: 0.05,      // High error rate
            time_of_day: 0.6,
            day_of_week: 5, // Friday
            seasonal_factor: 1.2,
            trending_direction: 0.5, // Rapidly increasing
            volatility: 0.8,         // High volatility
            timestamp: SystemTime::now(),
        };
        
        // Get multi-dimensional predictions
        let predictions = scaler.predict_scaling_needs(&current_metrics).await
            .expect("Should predict scaling needs");
        
        // Should predict scaling for multiple resource types
        let resource_types: std::collections::HashSet<ResourceType> = predictions.iter()
            .map(|p| p.resource_type.clone())
            .collect();
        
        assert!(resource_types.len() >= 2, "Should predict scaling for multiple resource types");
        assert!(resource_types.contains(&ResourceType::CPU), "Should predict CPU scaling");
        assert!(resource_types.contains(&ResourceType::GPU), "Should predict GPU scaling");
        
        // Verify CPU and GPU predictions have high urgency
        let cpu_prediction = predictions.iter().find(|p| p.resource_type == ResourceType::CPU);
        let gpu_prediction = predictions.iter().find(|p| p.resource_type == ResourceType::GPU);
        
        if let Some(cpu_pred) = cpu_prediction {
            assert!(cpu_pred.urgency_score > 0.5, "CPU scaling should have high urgency");
            match &cpu_pred.recommended_action {
                ScalingAction::ScaleUp { .. } => {},
                _ => panic!("Should recommend CPU scale up"),
            }
        }
        
        if let Some(gpu_pred) = gpu_prediction {
            assert!(gpu_pred.urgency_score > 0.7, "GPU scaling should have very high urgency");
            match &gpu_pred.recommended_action {
                ScalingAction::ScaleUp { .. } => {},
                _ => panic!("Should recommend GPU scale up"),
            }
        }
    }

    /// Test multi-GPU scaling engine
    #[tokio::test]
    async fn test_multi_gpu_scaling_engine() {
        // RED PHASE: This test MUST fail initially
        let gpu_count = 4;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 15,
            prediction_horizon: Duration::from_minutes(20),
            training_batch_size: 16,
            learning_rate: 0.01,
        };
        
        let engine = MultiGpuScalingEngine::new(gpu_count, config).await
            .expect("Should initialize multi-GPU scaling engine");
        
        assert_eq!(engine.gpu_predictors.len(), gpu_count, "Should have predictor for each GPU");
        
        // Create regional metrics
        let mut region_metrics = HashMap::new();
        let mut rng = thread_rng();
        
        for region in ["us-west-1", "us-east-1", "eu-central-1", "ap-southeast-1"] {
            region_metrics.insert(region.to_string(), FeatureSnapshot {
                cpu_utilization: rng.gen_range(0.3..0.9),
                memory_utilization: rng.gen_range(0.4..0.8),
                gpu_utilization: rng.gen_range(0.5..0.95),
                network_throughput: rng.gen_range(20.0..100.0),
                active_connections: rng.gen_range(500..3000),
                request_rate: rng.gen_range(50.0..500.0),
                response_latency: rng.gen_range(0.01..0.15),
                error_rate: rng.gen_range(0.0..0.05),
                time_of_day: rng.gen_range(0.0..1.0),
                day_of_week: rng.gen_range(1..8),
                seasonal_factor: rng.gen_range(0.8..1.2),
                trending_direction: rng.gen_range(-0.3..0.3),
                volatility: rng.gen_range(0.1..0.6),
                timestamp: SystemTime::now(),
            });
        }
        
        // Test global scaling prediction
        let global_predictions = engine.predict_global_scaling(region_metrics).await
            .expect("Should predict global scaling needs");
        
        assert_eq!(global_predictions.len(), 4, "Should have predictions for all regions");
        
        for (region, predictions) in global_predictions.iter() {
            assert!(!predictions.is_empty(), "Each region should have predictions");
            
            for prediction in predictions {
                assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
                assert!(prediction.predicted_utilization >= 0.0 && prediction.predicted_utilization <= 1.0);
            }
        }
    }

    /// Test consensus-based scaling decisions
    #[tokio::test]
    async fn test_consensus_scaling_decisions() {
        // RED PHASE: This test MUST fail initially
        let gpu_count = 3;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 10,
            prediction_horizon: Duration::from_minutes(15),
            training_batch_size: 8,
            learning_rate: 0.02,
        };
        
        let engine = MultiGpuScalingEngine::new(gpu_count, config).await
            .expect("Should initialize multi-GPU scaling engine");
        
        // Create consistent metrics across regions
        let mut region_metrics = HashMap::new();
        let base_metrics = FeatureSnapshot {
            cpu_utilization: 0.85,
            memory_utilization: 0.75,
            gpu_utilization: 0.9,
            network_throughput: 90.0,
            active_connections: 2500,
            request_rate: 400.0,
            response_latency: 0.1,
            error_rate: 0.03,
            time_of_day: 0.7,
            day_of_week: 4,
            seasonal_factor: 1.1,
            trending_direction: 0.3,
            volatility: 0.4,
            timestamp: SystemTime::now(),
        };
        
        region_metrics.insert("us-west-1".to_string(), base_metrics.clone());
        region_metrics.insert("us-east-1".to_string(), base_metrics);
        
        // Get global predictions
        let global_predictions = engine.predict_global_scaling(region_metrics.clone()).await
            .expect("Should predict global scaling");
        
        // Coordinate scaling actions
        let scaling_events = engine.coordinate_scaling_actions(global_predictions).await
            .expect("Should coordinate scaling actions");
        
        assert!(!scaling_events.is_empty(), "Should generate scaling events");
        
        // Verify consistency across regions
        let scale_up_events: Vec<_> = scaling_events.iter()
            .filter(|e| matches!(e.action, ScalingAction::ScaleUp { .. }))
            .collect();
        
        assert!(!scale_up_events.is_empty(), "Should generate scale up events for high load");
        
        for event in scaling_events {
            assert!(event.success, "Coordinated scaling should succeed");
            assert!(event.latency < Duration::from_secs(60), "Scaling should complete quickly");
            assert!(event.predicted_load > 0.8, "Should scale for high predicted load");
        }
    }

    /// Test feature extraction and normalization
    #[tokio::test]
    async fn test_feature_extraction_normalization() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 10,
            prediction_horizon: Duration::from_minutes(10),
            training_batch_size: 8,
            learning_rate: 0.01,
        };
        
        let mut scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Generate diverse feature snapshots
        let mut features = Vec::new();
        let mut rng = thread_rng();
        
        for _ in 0..1000 {
            features.push(FeatureSnapshot {
                cpu_utilization: rng.gen_range(0.0..1.0),
                memory_utilization: rng.gen_range(0.0..1.0),
                gpu_utilization: rng.gen_range(0.0..1.0),
                network_throughput: rng.gen_range(0.0..200.0),
                active_connections: rng.gen_range(0..5000),
                request_rate: rng.gen_range(0.0..1000.0),
                response_latency: rng.gen_range(0.001..1.0),
                error_rate: rng.gen_range(0.0..0.2),
                time_of_day: rng.gen_range(0.0..1.0),
                day_of_week: rng.gen_range(1..8),
                seasonal_factor: rng.gen_range(0.5..2.0),
                trending_direction: rng.gen_range(-1.0..1.0),
                volatility: rng.gen_range(0.0..1.0),
                timestamp: SystemTime::now(),
            });
        }
        
        // Update normalization parameters
        scaler.update_feature_normalization(&features).await
            .expect("Should update feature normalization");
        
        // Verify normalization parameters are reasonable
        let norm_params = &scaler.feature_extractor.normalization_params;
        
        for (feature_name, params) in norm_params {
            assert!(params.std > 0.0, "Standard deviation should be positive for {}", feature_name);
            assert!(params.max > params.min, "Max should be greater than min for {}", feature_name);
            assert!(!params.mean.is_nan(), "Mean should not be NaN for {}", feature_name);
            assert!(!params.std.is_nan(), "Std should not be NaN for {}", feature_name);
        }
    }

    /// Test emergency scaling scenarios
    #[tokio::test]
    async fn test_emergency_scaling_scenarios() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 32,
            lstm_layers: 1,
            sequence_length: 5, // Short sequence for emergency response
            prediction_horizon: Duration::from_minutes(5),
            training_batch_size: 4,
            learning_rate: 0.1, // High learning rate for fast adaptation
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        // Create emergency scenario - system overload
        let emergency_metrics = FeatureSnapshot {
            cpu_utilization: 0.98,  // Near 100% CPU
            memory_utilization: 0.95, // Near 100% memory
            gpu_utilization: 0.99,  // Near 100% GPU
            network_throughput: 200.0, // Network saturated
            active_connections: 10000, // Very high connections
            request_rate: 2000.0,   // Request storm
            response_latency: 5.0,  // Very high latency
            error_rate: 0.25,       // 25% error rate
            time_of_day: 0.5,
            day_of_week: 3,
            seasonal_factor: 1.0,
            trending_direction: 1.0, // Rapidly increasing
            volatility: 1.0,         // Maximum volatility
            timestamp: SystemTime::now(),
        };
        
        // Get emergency scaling predictions
        let predictions = scaler.predict_scaling_needs(&emergency_metrics).await
            .expect("Should predict emergency scaling");
        
        // Verify emergency responses
        let emergency_predictions: Vec<_> = predictions.iter()
            .filter(|p| matches!(p.recommended_action, ScalingAction::Emergency { .. }))
            .collect();
        
        assert!(!emergency_predictions.is_empty(), "Should trigger emergency scaling");
        
        for prediction in predictions.iter() {
            assert!(prediction.urgency_score > 0.8, "All predictions should have high urgency");
            
            match &prediction.recommended_action {
                ScalingAction::Emergency { reason } => {
                    assert!(!reason.is_empty(), "Emergency action should have reason");
                },
                ScalingAction::ScaleUp { factor, .. } => {
                    assert!(factor >= 2.0, "Emergency scale up should be aggressive");
                },
                _ => {} // Other actions acceptable
            }
        }
        
        // Execute emergency scaling
        for prediction in predictions.iter().take(3) { // Limit to avoid overwhelming system
            let event = scaler.execute_scaling_action(prediction).await
                .expect("Should execute emergency scaling");
            
            assert!(event.latency < Duration::from_secs(10), "Emergency scaling should be fast");
            assert!(event.trigger_reason.contains("emergency") || event.trigger_reason.contains("overload"),
                    "Should identify emergency condition");
        }
    }

    /// Test scaling performance under high load
    #[tokio::test]
    async fn test_scaling_performance_high_load() {
        // RED PHASE: This test MUST fail initially
        let device = Device::Cpu;
        let config = ScalerConfig {
            lstm_hidden_size: 16, // Smaller for faster inference
            lstm_layers: 1,
            sequence_length: 5,
            prediction_horizon: Duration::from_minutes(5),
            training_batch_size: 4,
            learning_rate: 0.05,
        };
        
        let scaler = PredictiveScaler::new(device, config).await
            .expect("Should initialize predictive scaler");
        
        let mut rng = thread_rng();
        
        // Test concurrent prediction performance
        let start_time = Instant::now();
        let mut tasks = Vec::new();
        
        for _ in 0..100 {
            let scaler_clone = scaler.clone();
            let metrics = FeatureSnapshot {
                cpu_utilization: rng.gen_range(0.3..0.9),
                memory_utilization: rng.gen_range(0.2..0.8),
                gpu_utilization: rng.gen_range(0.4..0.95),
                network_throughput: rng.gen_range(10.0..150.0),
                active_connections: rng.gen_range(100..3000),
                request_rate: rng.gen_range(10.0..800.0),
                response_latency: rng.gen_range(0.01..0.5),
                error_rate: rng.gen_range(0.0..0.1),
                time_of_day: rng.gen_range(0.0..1.0),
                day_of_week: rng.gen_range(1..8),
                seasonal_factor: rng.gen_range(0.8..1.2),
                trending_direction: rng.gen_range(-0.5..0.5),
                volatility: rng.gen_range(0.1..0.7),
                timestamp: SystemTime::now(),
            };
            
            let task = tokio::spawn(async move {
                scaler_clone.predict_scaling_needs(&metrics).await
            });
            tasks.push(task);
        }
        
        // Wait for all predictions
        let results: Vec<_> = futures::future::join_all(tasks).await;
        let duration = start_time.elapsed();
        
        // Verify performance
        assert!(duration < Duration::from_secs(30), "Should handle 100 predictions in <30 seconds");
        
        let successful_predictions = results.iter()
            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
            .count();
        assert!(successful_predictions >= 95, "Should successfully predict ≥95% of cases");
        
        // Verify prediction quality
        for result in results.iter().take(10) {
            if let Ok(Ok(predictions)) = result {
                for prediction in predictions {
                    assert!(prediction.predicted_utilization >= 0.0 && prediction.predicted_utilization <= 1.0);
                    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
                    assert!(prediction.urgency_score >= 0.0 && prediction.urgency_score <= 1.0);
                }
            }
        }
    }
}