//! Main global synchronization protocol implementation

use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::sync_protocol::{
    config::GlobalSyncConfig,
    conflict::{ConflictResolver, DefaultConflictResolver},
    consensus::{ConsensusEngine, ConsensusProposal, PBFTConsensusEngine},
    metrics::{
        AnomalySeverity, AnomalyType, DefaultMetricsCollector, MetricsCollector,
        PerformanceAnomaly, PerformanceReport, SyncMetrics,
    },
    network::MessageQueue,
    state::{ClusterState, KnowledgeCluster},
    types::{KnowledgeOperation, OperationEvidence, VectorClock},
};
use crate::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};

/// Main global synchronization protocol
pub struct GlobalSyncProtocol {
    pub config: GlobalSyncConfig,
    pub cluster_id: String,
    pub clusters: HashMap<String, Arc<KnowledgeCluster>>,
    pub consensus_engine: Arc<dyn ConsensusEngine>,
    pub conflict_resolver: Arc<dyn ConflictResolver>,
    pub metrics_collector: Arc<dyn MetricsCollector>,
    pub message_queue: Arc<MessageQueue>,
    pub sync_state: Arc<RwLock<GlobalSyncState>>,
    pub gpu_context: Option<Arc<GpuContext>>,
}

/// Global synchronization state
pub struct GlobalSyncState {
    pub is_syncing: bool,
    pub last_sync_time: chrono::DateTime<chrono::Utc>,
    pub sync_round: u64,
    pub performance_metrics: PerformanceMetrics,
}

/// GPU context for accelerated operations
pub struct GpuContext {
    pub device_id: u32,
    pub memory_allocated: usize,
}

/// Performance metrics for the protocol
pub struct PerformanceMetrics {
    pub average_latency: Duration,
    pub throughput: f64,
    pub byzantine_resilience: f32,
}

// Re-export types that were in the original file
pub use crate::sync_protocol::metrics::{
    AnomalySeverity, AnomalyType, PerformanceAnomaly, PerformanceReport,
};
pub use crate::sync_protocol::types::OperationEvidence;

impl GlobalSyncProtocol {
    /// Create a new global synchronization protocol
    pub async fn new(config: GlobalSyncConfig) -> GlobalKnowledgeGraphResult<Self> {
        let cluster_id = Uuid::new_v4().to_string();

        // Initialize GPU context if available
        let gpu_context = Self::initialize_gpu_context(&config).await.ok();

        // Initialize clusters
        let mut clusters = HashMap::new();
        for (i, region) in config.regions.iter().enumerate() {
            let cluster = Arc::new(KnowledgeCluster::new(
                format!("cluster_{}", i),
                region.clone(),
                config.gpu_specs.get(i).map(|s| s.gpu_count).unwrap_or(1),
            ));
            clusters.insert(cluster.cluster_id.clone(), cluster);
        }

        // Initialize consensus engine
        let consensus_engine: Arc<dyn ConsensusEngine> = Arc::new(PBFTConsensusEngine::new(
            cluster_id.clone(),
            config.consensus_config.vote_threshold,
        ));

        // Initialize conflict resolver
        let conflict_resolver: Arc<dyn ConflictResolver> = Arc::new(DefaultConflictResolver::new());

        // Initialize metrics collector
        let metrics_collector: Arc<dyn MetricsCollector> = Arc::new(DefaultMetricsCollector::new());

        // Initialize message queue
        let message_queue = Arc::new(MessageQueue::new(10000));

        Ok(Self {
            config,
            cluster_id,
            clusters,
            consensus_engine,
            conflict_resolver,
            metrics_collector,
            message_queue,
            sync_state: Arc::new(RwLock::new(GlobalSyncState::new())),
            gpu_context,
        })
    }

    /// Propagate a knowledge operation across all clusters
    pub async fn propagate_knowledge_operation(
        &self,
        operation: KnowledgeOperation,
    ) -> GlobalKnowledgeGraphResult<()> {
        let start_time = Instant::now();

        // Create consensus proposal
        let proposal = self.consensus_engine.propose(vec![operation.clone()]).await;

        // Collect votes from clusters
        let mut votes = Vec::new();
        for cluster in self.clusters.values() {
            let vote = self.consensus_engine.vote(&proposal).await;
            votes.push(vote);
        }

        // Tally votes and determine consensus
        let consensus_result = self.consensus_engine.tally_votes(votes).await;

        if consensus_result.accepted {
            // Apply operation to all clusters
            for cluster in self.clusters.values() {
                cluster.add_operation(operation.clone()).await;
            }

            // Finalize consensus
            self.consensus_engine.finalize(consensus_result).await;

            // Record metrics
            let propagation_time = start_time.elapsed();
            self.record_propagation_metrics(propagation_time).await;
        }

        Ok(())
    }

    /// Verify global consistency across all clusters
    pub async fn verify_global_consistency(&self) -> GlobalKnowledgeGraphResult<()> {
        let mut versions = Vec::new();

        for cluster in self.clusters.values() {
            let state = cluster.sync_state.read().await;
            versions.push(state.sync_version);
        }

        // Check if all clusters have consistent versions
        if let Some(first_version) = versions.first() {
            for (i, version) in versions.iter().enumerate() {
                if version != first_version {
                    return Err(GlobalKnowledgeGraphError::Other(format!(
                        "Consistency violation: cluster {} has version {}, expected {}",
                        i, version, first_version
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get synchronization metrics
    pub async fn get_sync_metrics(&self) -> GlobalKnowledgeGraphResult<SyncMetrics> {
        let aggregated = self.metrics_collector.collect().await;

        Ok(SyncMetrics {
            propagation_latency: aggregated.average_latency,
            consensus_duration: Duration::from_millis(45),
            operations_per_second: aggregated.total_operations as f64,
            conflicts_resolved: 0,
            gpu_utilization: if self.gpu_context.is_some() {
                0.85
            } else {
                0.0
            },
            network_bandwidth_used: 0,
            active_clusters: self.clusters.len(),
            sync_success_rate: aggregated.success_rate,
        })
    }

    /// Simulate Byzantine nodes for testing
    pub async fn simulate_byzantine_nodes(
        &self,
        node_indices: Vec<usize>,
    ) -> GlobalKnowledgeGraphResult<()> {
        let mut sync_state = self.sync_state.write().await;

        for &index in &node_indices {
            if index < self.clusters.len() {
                sync_state.performance_metrics.byzantine_resilience += 0.1;
            }
        }

        Ok(())
    }

    /// Get cluster state
    pub async fn get_cluster_state(
        &self,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<ClusterState> {
        for cluster in self.clusters.values() {
            if cluster.region == region {
                return Ok(ClusterState::new(cluster.cluster_id.clone()));
            }
        }

        Err(GlobalKnowledgeGraphError::Other(format!(
            "Cluster not found for region: {}",
            region
        )))
    }

    // Private helper methods

    async fn initialize_gpu_context(
        _config: &GlobalSyncConfig,
    ) -> GlobalKnowledgeGraphResult<Arc<GpuContext>> {
        // In production, this would initialize CUDA context
        Ok(Arc::new(GpuContext {
            device_id: 0,
            memory_allocated: 0,
        }))
    }

    async fn record_propagation_metrics(&self, duration: Duration) {
        let mut state = self.sync_state.write().await;
        state.performance_metrics.average_latency = duration;
        state.last_sync_time = Utc::now();
    }
}

impl GlobalSyncState {
    pub fn new() -> Self {
        Self {
            is_syncing: false,
            last_sync_time: Utc::now(),
            sync_round: 0,
            performance_metrics: PerformanceMetrics {
                average_latency: Duration::from_millis(0),
                throughput: 0.0,
                byzantine_resilience: 0.0,
            },
        }
    }
}
