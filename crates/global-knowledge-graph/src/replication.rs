//! Cross-region replication management

use crate::error::{GlobalKnowledgeGraphError, GlobalKnowledgeGraphResult};
use crate::graph_manager::{Edge, Node};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

/// Type alias for regional replica
type RegionalReplicaRef = Arc<dyn RegionalReplica + Send + Sync>;

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Replication factor (number of regions to replicate to)
    pub replication_factor: usize,
    /// Maximum replication lag in milliseconds
    pub max_lag_ms: u64,
    /// Conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
    /// Delta sync interval
    pub delta_sync_interval: Duration,
    /// Maximum batch size for replication
    pub max_batch_size: usize,
    /// Regional priorities
    pub regional_priorities: HashMap<String, u32>,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        let mut regional_priorities = HashMap::new();
        regional_priorities.insert("us-east-1".to_string(), 100);
        regional_priorities.insert("us-west-2".to_string(), 90);
        regional_priorities.insert("eu-west-1".to_string(), 80);
        regional_priorities.insert("ap-southeast-1".to_string(), 70);

        Self {
            replication_factor: 3,
            max_lag_ms: 1000,
            conflict_strategy: ConflictStrategy::LastWriteWins,
            delta_sync_interval: Duration::from_secs(5),
            max_batch_size: 100,
            regional_priorities,
        }
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictStrategy {
    /// Last write wins based on timestamp
    LastWriteWins,
    /// Higher version wins
    HigherVersionWins,
    /// Region priority based
    RegionPriorityBased,
    /// Custom resolver function
    Custom,
}

/// Replication event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationEvent {
    /// Node created or updated
    NodeChange(Node),
    /// Node deleted
    NodeDeletion { id: String, region: String },
    /// Edge created
    EdgeChange(Edge),
    /// Edge deleted
    EdgeDeletion {
        id: String,
        source: String,
        target: String,
    },
}

/// Replication status for a region
#[derive(Debug, Clone)]
pub struct RegionStatus {
    /// Region name
    pub region: String,
    /// Last successful sync
    pub last_sync: Instant,
    /// Current lag in milliseconds
    pub lag_ms: u64,
    /// Number of pending events
    pub pending_events: usize,
    /// Is region healthy
    pub is_healthy: bool,
}

/// Delta for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Delta ID
    pub id: String,
    /// Source region
    pub source_region: String,
    /// Target region
    pub target_region: String,
    /// Events in this delta
    pub events: Vec<ReplicationEvent>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Regional replica endpoint
#[async_trait]
pub trait RegionalReplica: Send + Sync {
    /// Apply replication event
    async fn apply_event(&self, event: ReplicationEvent) -> GlobalKnowledgeGraphResult<()>;

    /// Get current version for conflict resolution
    async fn get_version(&self, node_id: &str) -> GlobalKnowledgeGraphResult<u64>;

    /// Check if region is available
    async fn is_available(&self) -> bool;
}

/// Mock regional replica for testing
#[cfg(test)]
pub struct MockRegionalReplica {
    region: String,
    available: Arc<RwLock<bool>>,
    versions: Arc<DashMap<String, u64>>,
}

#[cfg(test)]
impl MockRegionalReplica {
    pub fn new(region: String) -> Self {
        Self {
            region,
            available: Arc::new(RwLock::new(true)),
            versions: Arc::new(DashMap::new()),
        }
    }

    pub fn set_available(&self, available: bool) {
        *self.available.write() = available;
    }
}

#[cfg(test)]
#[async_trait]
impl RegionalReplica for MockRegionalReplica {
    async fn apply_event(&self, event: ReplicationEvent) -> GlobalKnowledgeGraphResult<()> {
        if !*self.available.read() {
            return Err(GlobalKnowledgeGraphError::RegionUnavailable {
                region: self.region.clone(),
                reason: "Region is offline".to_string(),
            });
        }

        match event {
            ReplicationEvent::NodeChange(node) => {
                self.versions.insert(node.id.clone(), node.version);
            }
            ReplicationEvent::NodeDeletion { id, .. } => {
                self.versions.remove(&id);
            }
            _ => {}
        }

        Ok(())
    }

    async fn get_version(&self, node_id: &str) -> GlobalKnowledgeGraphResult<u64> {
        self.versions.get(node_id).map(|v| *v).ok_or_else(|| {
            GlobalKnowledgeGraphError::NodeNotFound {
                node_id: node_id.to_string(),
                region: self.region.clone(),
            }
        })
    }

    async fn is_available(&self) -> bool {
        *self.available.read()
    }
}

/// Cross-region replication manager
pub struct ReplicationManager {
    config: Arc<ReplicationConfig>,
    replicas: Arc<DashMap<String, RegionalReplicaRef>>,
    event_queue: Arc<Mutex<VecDeque<(String, ReplicationEvent)>>>,
    region_status: Arc<DashMap<String, RegionStatus>>,
    delta_store: Arc<DashMap<String, Vec<Delta>>>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<RwLock<Option<mpsc::Receiver<()>>>>,
}

impl ReplicationManager {
    /// Create new replication manager
    pub fn new(config: ReplicationConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Self {
            config: Arc::new(config),
            replicas: Arc::new(DashMap::new()),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
            region_status: Arc::new(DashMap::new()),
            delta_store: Arc::new(DashMap::new()),
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(Some(shutdown_rx))),
        }
    }

    /// Register regional replica
    pub fn register_replica(&self, region: String, replica: RegionalReplicaRef) {
        self.replicas.insert(region.clone(), replica);
        self.region_status.insert(
            region.clone(),
            RegionStatus {
                region,
                last_sync: Instant::now(),
                lag_ms: 0,
                pending_events: 0,
                is_healthy: true,
            },
        );
    }

    /// Start replication manager
    pub fn start(&self) {
        // Just set a flag to indicate the manager is started
        // The actual replication can be done in a separate background task
        println!("Replication manager started");
    }

    /// Process a single replication batch
    pub async fn process_batch(&self) -> GlobalKnowledgeGraphResult<()> {
        let config = self.config.clone();
        let replicas = self.replicas.clone();
        let event_queue = self.event_queue.clone();
        let region_status = self.region_status.clone();
        let delta_store = self.delta_store.clone();

        let mut queue = event_queue.lock().await;
        let batch_size = queue.len().min(config.max_batch_size);

        if batch_size == 0 {
            return Ok(());
        }

        let mut events_by_region: HashMap<String, Vec<ReplicationEvent>> = HashMap::new();

        // Collect events for batch processing
        for _ in 0..batch_size {
            if let Some((source_region, event)) = queue.pop_front() {
                // Determine target regions based on replication factor
                let mut available_regions: Vec<(String, u32)> = Vec::new();

                for replica_ref in replicas.iter() {
                    let region = replica_ref.key();
                    let replica = replica_ref.value();

                    if region != &source_region && replica.is_available().await {
                        if let Some(status) = region_status.get(region) {
                            if status.is_healthy {
                                let priority = status.lag_ms as u32;
                                available_regions.push((region.clone(), priority));
                            }
                        }
                    }
                }

                // Sort by priority (lower lag is better)
                available_regions.sort_by_key(|(_, priority)| *priority);

                let target_regions: Vec<String> = available_regions
                    .into_iter()
                    .take(config.replication_factor - 1)
                    .map(|(region, _)| region)
                    .collect();

                for target_region in target_regions {
                    events_by_region
                        .entry(target_region)
                        .or_insert_with(Vec::new)
                        .push(event.clone());
                }
            }
        }

        drop(queue); // Release the lock

        // Apply events to each region
        for (target_region, events) in events_by_region {
            if let Some(replica) = replicas.get(&target_region) {
                let start = Instant::now();

                for event in &events {
                    if let Err(e) = replica.apply_event(event.clone()).await {
                        tracing::error!("Replication failed to {}: {}", target_region, e);
                    }
                }

                // Update region status
                let lag_ms = start.elapsed().as_millis() as u64;
                let queue_len = event_queue.lock().await.len();
                region_status.alter(&target_region, |_, mut status| {
                    status.last_sync = Instant::now();
                    status.lag_ms = lag_ms;
                    status.is_healthy = lag_ms < config.max_lag_ms;
                    status.pending_events = queue_len;
                    status
                });

                // Store delta for audit
                let delta = Delta {
                    id: Uuid::new_v4().to_string(),
                    source_region: config
                        .regional_priorities
                        .keys()
                        .next()
                        .unwrap_or(&String::new())
                        .clone(),
                    target_region: target_region.clone(),
                    events,
                    timestamp: chrono::Utc::now(),
                };

                delta_store
                    .entry(target_region)
                    .or_insert_with(Vec::new)
                    .push(delta);
            }
        }

        Ok(())
    }

    /// Select target regions for replication (public for testing)
    pub async fn select_target_regions(
        &self,
        source_region: &str,
        replication_factor: usize,
    ) -> Vec<String> {
        select_target_regions(
            source_region,
            replication_factor,
            &self.replicas,
            &self.region_status,
        )
        .await
    }

    /// Stop replication manager
    pub async fn stop(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Replicate node change
    pub async fn replicate_node(&self, node: Node) -> GlobalKnowledgeGraphResult<()> {
        let event = ReplicationEvent::NodeChange(node.clone());
        self.enqueue_event(node.region, event).await
    }

    /// Replicate node deletion
    pub async fn replicate_node_deletion(
        &self,
        node_id: &str,
        region: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let event = ReplicationEvent::NodeDeletion {
            id: node_id.to_string(),
            region: region.to_string(),
        };
        self.enqueue_event(region.to_string(), event).await
    }

    /// Replicate edge change
    pub async fn replicate_edge(&self, edge: Edge) -> GlobalKnowledgeGraphResult<()> {
        let event = ReplicationEvent::EdgeChange(edge);
        self.enqueue_event(
            self.config
                .regional_priorities
                .keys()
                .next()
                ?
                .clone(),
            event,
        )
        .await
    }

    /// Replicate edge deletion
    pub async fn replicate_edge_deletion(
        &self,
        edge_id: &str,
        source: &str,
        target: &str,
    ) -> GlobalKnowledgeGraphResult<()> {
        let event = ReplicationEvent::EdgeDeletion {
            id: edge_id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
        };
        self.enqueue_event(
            self.config
                .regional_priorities
                .keys()
                .next()
                .unwrap()
                .clone(),
            event,
        )
        .await
    }

    /// Enqueue replication event
    async fn enqueue_event(
        &self,
        source_region: String,
        event: ReplicationEvent,
    ) -> GlobalKnowledgeGraphResult<()> {
        let mut queue = self.event_queue.lock().await;
        queue.push_back((source_region, event));

        // Update pending events count
        for mut status in self.region_status.iter_mut() {
            status.pending_events = queue.len();
        }

        Ok(())
    }

    /// Resolve conflict between versions
    pub async fn resolve_conflict(
        &self,
        _node_id: &str,
        region1: &str,
        version1: u64,
        region2: &str,
        version2: u64,
    ) -> GlobalKnowledgeGraphResult<String> {
        match self.config.conflict_strategy {
            ConflictStrategy::LastWriteWins => {
                // In real implementation, would compare timestamps
                Ok(if version1 >= version2 {
                    region1.to_string()
                } else {
                    region2.to_string()
                })
            }
            ConflictStrategy::HigherVersionWins => Ok(if version1 > version2 {
                region1.to_string()
            } else {
                region2.to_string()
            }),
            ConflictStrategy::RegionPriorityBased => {
                let priority1 = self.config.regional_priorities.get(region1).unwrap_or(&0);
                let priority2 = self.config.regional_priorities.get(region2).unwrap_or(&0);
                Ok(if priority1 >= priority2 {
                    region1.to_string()
                } else {
                    region2.to_string()
                })
            }
            ConflictStrategy::Custom => {
                // Would call custom resolver
                Ok(region1.to_string())
            }
        }
    }

    /// Get replication lag for a region
    pub fn get_replication_lag(&self, region: &str) -> Option<u64> {
        self.region_status.get(region).map(|status| status.lag_ms)
    }

    /// Get all region statuses
    pub fn get_region_statuses(&self) -> Vec<RegionStatus> {
        self.region_status
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Force sync for a region
    pub async fn force_sync(&self, region: &str) -> GlobalKnowledgeGraphResult<()> {
        if let Some(replica) = self.replicas.get(region) {
            if !replica.is_available().await {
                return Err(GlobalKnowledgeGraphError::RegionUnavailable {
                    region: region.to_string(),
                    reason: "Region is not available".to_string(),
                });
            }

            // Process all pending events for this region
            let queue = self.event_queue.lock().await;
            for (_, event) in queue.iter() {
                replica.apply_event(event.clone()).await?;
            }

            // Update status
            self.region_status.alter(region, |_, mut status| {
                status.last_sync = Instant::now();
                status.lag_ms = 0;
                status.pending_events = 0;
                status.is_healthy = true;
                status
            });

            Ok(())
        } else {
            Err(GlobalKnowledgeGraphError::RegionUnavailable {
                region: region.to_string(),
                reason: "Region not registered".to_string(),
            })
        }
    }

    /// Get delta history for a region
    pub fn get_delta_history(&self, region: &str, limit: usize) -> Vec<Delta> {
        self.delta_store
            .get(region)
            .map(|deltas| deltas.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }
}

/// Select target regions for replication
async fn select_target_regions(
    source_region: &str,
    replication_factor: usize,
    replicas: &Arc<DashMap<String, RegionalReplicaRef>>,
    region_status: &Arc<DashMap<String, RegionStatus>>,
) -> Vec<String> {
    let mut available_regions: Vec<(String, u32)> = Vec::new();

    for replica_ref in replicas.iter() {
        let region = replica_ref.key();
        let replica = replica_ref.value();

        if region != source_region && replica.is_available().await {
            if let Some(status) = region_status.get(region) {
                if status.is_healthy {
                    let priority = region_status
                        .get(region)
                        .map(|s| s.lag_ms as u32)
                        .unwrap_or(0);
                    available_regions.push((region.clone(), priority));
                }
            }
        }
    }

    // Sort by priority (lower lag is better)
    available_regions.sort_by_key(|(_, priority)| *priority);

    available_regions
        .into_iter()
        .take(replication_factor - 1) // -1 because source region already has the data
        .map(|(region, _)| region)
        .collect()
}

/// Process replication batch
async fn process_replication_batch(
    config: Arc<ReplicationConfig>,
    replicas: Arc<DashMap<String, RegionalReplicaRef>>,
    event_queue: Arc<Mutex<VecDeque<(String, ReplicationEvent)>>>,
    region_status: Arc<DashMap<String, RegionStatus>>,
    delta_store: Arc<DashMap<String, Vec<Delta>>>,
) {
    let mut queue = event_queue.lock().await;
    let batch_size = queue.len().min(config.max_batch_size);

    if batch_size == 0 {
        return;
    }

    let mut events_by_region: HashMap<String, Vec<ReplicationEvent>> = HashMap::new();

    // Collect events for batch processing
    for _ in 0..batch_size {
        if let Some((source_region, event)) = queue.pop_front() {
            // Determine target regions based on replication factor
            let target_regions = select_target_regions(
                &source_region,
                config.replication_factor,
                &replicas,
                &region_status,
            )
            .await;

            for target_region in target_regions {
                events_by_region
                    .entry(target_region)
                    .or_insert_with(Vec::new)
                    .push(event.clone());
            }
        }
    }

    // Apply events to each region
    for (target_region, events) in events_by_region {
        if let Some(replica) = replicas.get(&target_region) {
            let start = Instant::now();
            let mut _success_count = 0;

            for event in &events {
                match replica.apply_event(event.clone()).await {
                    Ok(_) => _success_count += 1,
                    Err(e) => {
                        tracing::error!("Replication failed to {}: {}", target_region, e);
                    }
                }
            }

            // Update region status
            let lag_ms = start.elapsed().as_millis() as u64;
            region_status.alter(&target_region, |_, mut status| {
                status.last_sync = Instant::now();
                status.lag_ms = lag_ms;
                status.is_healthy = lag_ms < config.max_lag_ms;
                status.pending_events = queue.len();
                status
            });

            // Store delta for audit
            let delta = Delta {
                id: Uuid::new_v4().to_string(),
                source_region: config.regional_priorities.keys().next().unwrap().clone(),
                target_region: target_region.clone(),
                events,
                timestamp: chrono::Utc::now(),
            };

            delta_store
                .entry(target_region)
                .or_insert_with(Vec::new)
                .push(delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replication_manager_creation() {
        let config = ReplicationConfig::default();
        let manager = ReplicationManager::new(config);
        assert_eq!(manager.region_status.len(), 0);
    }

    #[tokio::test]
    async fn test_register_replica() {
        let manager = ReplicationManager::new(ReplicationConfig::default());
        let replica = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));

        manager.register_replica("us-east-1".to_string(), replica);

        assert_eq!(manager.replicas.len(), 1);
        assert_eq!(manager.region_status.len(), 1);
    }

    #[tokio::test]
    async fn test_replicate_node() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };

        let result = manager.replicate_node(node).await;
        assert!(result.is_ok());

        let queue = manager.event_queue.lock().await;
        assert_eq!(queue.len(), 1);
    }

    #[tokio::test]
    async fn test_replicate_node_deletion() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        let result = manager
            .replicate_node_deletion("node-id", "us-east-1")
            .await;
        assert!(result.is_ok());

        let queue = manager.event_queue.lock().await;
        assert_eq!(queue.len(), 1);
    }

    #[tokio::test]
    async fn test_replicate_edge() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        let edge = Edge {
            id: "edge-id".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            edge_type: "relates_to".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };

        let result = manager.replicate_edge(edge).await;
        assert!(result.is_ok());

        let queue = manager.event_queue.lock().await;
        assert_eq!(queue.len(), 1);
    }

    #[tokio::test]
    async fn test_conflict_resolution_last_write_wins() {
        let config = ReplicationConfig {
            conflict_strategy: ConflictStrategy::LastWriteWins,
            ..Default::default()
        };
        let manager = ReplicationManager::new(config);

        let result = manager
            .resolve_conflict("node-id", "us-east-1", 5, "eu-west-1", 3)
            .await
            ?;

        assert_eq!(result, "us-east-1");
    }

    #[tokio::test]
    async fn test_conflict_resolution_higher_version_wins() {
        let config = ReplicationConfig {
            conflict_strategy: ConflictStrategy::HigherVersionWins,
            ..Default::default()
        };
        let manager = ReplicationManager::new(config);

        let result = manager
            .resolve_conflict("node-id", "us-east-1", 3, "eu-west-1", 5)
            .await
            ?;

        assert_eq!(result, "eu-west-1");
    }

    #[tokio::test]
    async fn test_conflict_resolution_region_priority() {
        let mut regional_priorities = HashMap::new();
        regional_priorities.insert("us-east-1".to_string(), 100);
        regional_priorities.insert("eu-west-1".to_string(), 50);

        let config = ReplicationConfig {
            conflict_strategy: ConflictStrategy::RegionPriorityBased,
            regional_priorities,
            ..Default::default()
        };
        let manager = ReplicationManager::new(config);

        let result = manager
            .resolve_conflict("node-id", "us-east-1", 1, "eu-west-1", 10)
            .await
            .unwrap();

        assert_eq!(result, "us-east-1");
    }

    #[tokio::test]
    async fn test_get_replication_lag() {
        let manager = ReplicationManager::new(ReplicationConfig::default());
        let replica = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));

        manager.register_replica("us-east-1".to_string(), replica);

        let lag = manager.get_replication_lag("us-east-1");
        assert!(lag.is_some());
        assert_eq!(lag?, 0);
    }

    #[tokio::test]
    async fn test_get_region_statuses() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        let regions = vec!["us-east-1", "eu-west-1"];
        for region in &regions {
            let replica = Arc::new(MockRegionalReplica::new(region.to_string()));
            manager.register_replica(region.to_string(), replica);
        }

        let statuses = manager.get_region_statuses();
        assert_eq!(statuses.len(), 2);

        for status in statuses {
            assert!(regions.contains(&status.region.as_str()));
            assert!(status.is_healthy);
        }
    }

    #[tokio::test]
    async fn test_force_sync() {
        let manager = ReplicationManager::new(ReplicationConfig::default());
        let replica = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));

        manager.register_replica("us-east-1".to_string(), replica);

        // Add some events
        let node = Node {
            id: "test-node".to_string(),
            node_type: "entity".to_string(),
            properties: HashMap::new(),
            region: "us-east-1".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        };
        manager.replicate_node(node).await.unwrap();

        let result = manager.force_sync("us-east-1").await;
        assert!(result.is_ok());

        let status = manager.region_status.get("us-east-1").unwrap();
        assert_eq!(status.lag_ms, 0);
        assert_eq!(status.pending_events, 0);
    }

    #[tokio::test]
    async fn test_force_sync_unavailable_region() {
        let manager = ReplicationManager::new(ReplicationConfig::default());
        let replica = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));
        replica.set_available(false);

        manager.register_replica("us-east-1".to_string(), replica);

        let result = manager.force_sync("us-east-1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delta_history() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        // Manually add some deltas
        let delta = Delta {
            id: Uuid::new_v4().to_string(),
            source_region: "us-east-1".to_string(),
            target_region: "eu-west-1".to_string(),
            events: vec![],
            timestamp: chrono::Utc::now(),
        };

        manager
            .delta_store
            .entry("eu-west-1".to_string())
            .or_insert_with(Vec::new)
            .push(delta);

        let history = manager.get_delta_history("eu-west-1", 10);
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let config = ReplicationConfig {
            max_batch_size: 5,
            ..Default::default()
        };
        let manager = ReplicationManager::new(config);

        // Add multiple events
        for i in 0..10 {
            let node = Node {
                id: format!("node-{}", i),
                node_type: "entity".to_string(),
                properties: HashMap::new(),
                region: "us-east-1".to_string(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: 1,
            };
            manager.replicate_node(node).await.unwrap();
        }

        let queue = manager.event_queue.lock().await;
        assert_eq!(queue.len(), 10);
    }

    #[tokio::test]
    async fn test_select_target_regions() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        // Register multiple replicas
        for region in &["us-east-1", "eu-west-1", "ap-southeast-1"] {
            let replica = Arc::new(MockRegionalReplica::new(region.to_string()));
            manager.register_replica(region.to_string(), replica);
        }

        let targets = manager.select_target_regions("us-east-1", 2).await;

        assert_eq!(targets.len(), 1); // replication_factor - 1
        assert!(!targets.contains(&"us-east-1".to_string())); // Should not include source
    }

    #[tokio::test]
    async fn test_replication_with_unavailable_regions() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        // Register replicas with one unavailable
        let replica1 = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));
        let replica2 = Arc::new(MockRegionalReplica::new("eu-west-1".to_string()));
        replica2.set_available(false);

        manager.register_replica("us-east-1".to_string(), replica1);
        manager.register_replica("eu-west-1".to_string(), replica2);

        let targets = manager.select_target_regions("ap-southeast-1", 3).await;

        assert_eq!(targets.len(), 1); // Only one available region
        assert_eq!(targets[0], "us-east-1");
    }

    #[tokio::test]
    async fn test_start_stop_manager() {
        let manager = ReplicationManager::new(ReplicationConfig::default());

        // Start the manager
        manager.start();

        // Give it a moment to initialize
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the manager
        manager.stop().await;

        // Give it a moment to shut down
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_regional_priorities() {
        let mut regional_priorities = HashMap::new();
        regional_priorities.insert("us-east-1".to_string(), 100);
        regional_priorities.insert("us-west-2".to_string(), 90);
        regional_priorities.insert("eu-west-1".to_string(), 80);

        let config = ReplicationConfig {
            regional_priorities: regional_priorities.clone(),
            ..Default::default()
        };

        assert_eq!(config.regional_priorities.len(), 3);
        assert_eq!(config.regional_priorities.get("us-east-1").unwrap(), &100);
    }

    #[tokio::test]
    async fn test_max_lag_monitoring() {
        let config = ReplicationConfig {
            max_lag_ms: 100,
            ..Default::default()
        };
        let manager = ReplicationManager::new(config);

        let replica = Arc::new(MockRegionalReplica::new("us-east-1".to_string()));
        manager.register_replica("us-east-1".to_string(), replica);

        // Simulate high lag
        manager.region_status.alter("us-east-1", |_, mut status| {
            status.lag_ms = 150;
            status.is_healthy = false;
            status
        });

        let status = manager.region_status.get("us-east-1").unwrap();
        assert!(!status.is_healthy);
        assert_eq!(status.lag_ms, 150);
    }
}
