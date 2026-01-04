//! Main adaptive consensus engine

use super::{
    AdaptationController, AdaptiveConsensusConfig, AlgorithmSelector, ConsensusAlgorithmType,
    ConsensusCoordinator, ConsensusOutcome, GpuConsensusAccelerator, NetworkMonitor,
    OptimizationEngine,
};
use crate::ConsensusError;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main adaptive consensus engine
pub struct AdaptiveConsensusEngine {
    config: AdaptiveConsensusConfig,
    current_algorithm: Arc<RwLock<ConsensusAlgorithmType>>,
    network_monitor: Arc<NetworkMonitor>,
    algorithm_selector: Arc<AlgorithmSelector>,
    optimization_engine: Arc<OptimizationEngine>,
    coordinator: Arc<ConsensusCoordinator>,
    adaptation_controller: Arc<AdaptationController>,
    gpu_accelerator: Option<Arc<GpuConsensusAccelerator>>,
    metrics: Arc<RwLock<EngineMetrics>>,
}

struct EngineMetrics {
    total_rounds: u64,
    successful_rounds: u64,
    failed_rounds: u64,
    algorithm_switches: u64,
    average_latency: f64,
}

impl AdaptiveConsensusEngine {
    /// Create new adaptive consensus engine
    pub async fn new(config: AdaptiveConsensusConfig) -> Result<Self, ConsensusError> {
        let current_algorithm = Arc::new(RwLock::new(config.initial_algorithm));

        let network_monitor = Arc::new(NetworkMonitor::new(config.network_monitoring.clone()));
        let algorithm_selector = Arc::new(AlgorithmSelector::new());
        let optimization_engine = Arc::new(OptimizationEngine::new(config.optimization.clone()));
        let coordinator = Arc::new(ConsensusCoordinator::new());
        let adaptation_controller = Arc::new(AdaptationController::new());

        let gpu_accelerator = if config.gpu_config.enabled {
            Some(Arc::new(GpuConsensusAccelerator::new(
                config.gpu_config.clone(),
            )?))
        } else {
            None
        };

        Ok(Self {
            config,
            current_algorithm,
            network_monitor,
            algorithm_selector,
            optimization_engine,
            coordinator,
            adaptation_controller,
            gpu_accelerator,
            metrics: Arc::new(RwLock::new(EngineMetrics {
                total_rounds: 0,
                successful_rounds: 0,
                failed_rounds: 0,
                algorithm_switches: 0,
                average_latency: 0.0,
            })),
        })
    }

    /// Execute consensus round
    pub async fn execute_consensus(
        &self,
        proposal: &[u8],
        nodes: usize,
    ) -> Result<ConsensusOutcome, ConsensusError> {
        let start = std::time::Instant::now();

        // Get current algorithm
        let algorithm = *self.current_algorithm.read().await;

        // Monitor network conditions
        let conditions = self.network_monitor.get_current_conditions().await;

        // Check if we should adapt
        if self.config.enable_adaptation {
            if let Some(new_algorithm) = self
                .algorithm_selector
                .select_algorithm(&conditions, nodes, self.config.target_latency_us)
                .await
            {
                if new_algorithm != algorithm {
                    *self.current_algorithm.write().await = new_algorithm;
                    let mut metrics = self.metrics.write().await;
                    metrics.algorithm_switches += 1;
                }
            }
        }

        // Execute consensus
        let result = self
            .coordinator
            .execute_round(algorithm, proposal, nodes)
            .await?;

        // Update metrics
        let elapsed = start.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.total_rounds += 1;
        if result.consensus_reached {
            metrics.successful_rounds += 1;
        } else {
            metrics.failed_rounds += 1;
        }

        // Update average latency
        let latency_us = elapsed.as_micros() as f64;
        metrics.average_latency = (metrics.average_latency * (metrics.total_rounds - 1) as f64
            + latency_us)
            / metrics.total_rounds as f64;

        Ok(ConsensusOutcome {
            algorithm,
            time_taken: elapsed.as_micros() as u64,
            consensus_reached: result.consensus_reached,
            nodes_participated: nodes,
            gpu_utilization: self.get_gpu_utilization().await,
        })
    }

    /// Get current GPU utilization
    async fn get_gpu_utilization(&self) -> f32 {
        if let Some(ref gpu) = self.gpu_accelerator {
            gpu.get_utilization().await
        } else {
            0.0
        }
    }

    /// Get engine metrics
    pub async fn get_metrics(&self) -> (u64, u64, u64, f64) {
        let metrics = self.metrics.read().await;
        (
            metrics.total_rounds,
            metrics.successful_rounds,
            metrics.algorithm_switches,
            metrics.average_latency,
        )
    }
}
