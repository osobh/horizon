//! State management for synchronization protocol

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::sync_protocol::types::{KnowledgeOperation, VectorClock};

/// Represents a knowledge cluster in the distributed system
pub struct KnowledgeCluster {
    pub cluster_id: String,
    pub region: String,
    pub gpu_count: usize,
    pub operations: Arc<RwLock<Vec<KnowledgeOperation>>>,
    pub vector_clock: Arc<RwLock<VectorClock>>,
    pub peers: Arc<RwLock<HashSet<String>>>,
    pub sync_state: Arc<RwLock<ClusterSyncState>>,
}

/// Synchronization state for a cluster
#[derive(Debug, Clone)]
pub struct ClusterSyncState {
    pub last_sync: DateTime<Utc>,
    pub pending_operations: Vec<Uuid>,
    pub acknowledged_operations: HashSet<Uuid>,
    pub sync_in_progress: bool,
    pub sync_version: u64,
}

/// State of a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    pub cluster_id: String,
    pub is_primary: bool,
    pub is_healthy: bool,
    pub last_heartbeat: DateTime<Utc>,
    pub operation_count: u64,
}

impl KnowledgeCluster {
    pub fn new(cluster_id: String, region: String, gpu_count: usize) -> Self {
        Self {
            cluster_id: cluster_id.clone(),
            region,
            gpu_count,
            operations: Arc::new(RwLock::new(Vec::new())),
            vector_clock: Arc::new(RwLock::new(VectorClock::new())),
            peers: Arc::new(RwLock::new(HashSet::new())),
            sync_state: Arc::new(RwLock::new(ClusterSyncState::new())),
        }
    }

    pub async fn add_operation(&self, operation: KnowledgeOperation) {
        let mut operations = self.operations.write().await;
        operations.push(operation);

        // Update vector clock
        let mut clock = self.vector_clock.write().await;
        clock.increment(&self.cluster_id);
    }

    pub async fn get_operations_since(&self, version: u64) -> Vec<KnowledgeOperation> {
        let operations = self.operations.read().await;
        operations.iter().skip(version as usize).cloned().collect()
    }

    pub async fn add_peer(&self, peer_id: String) {
        let mut peers = self.peers.write().await;
        peers.insert(peer_id);
    }

    pub async fn remove_peer(&self, peer_id: &str) {
        let mut peers = self.peers.write().await;
        peers.remove(peer_id);
    }

    pub async fn get_peers(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.iter().cloned().collect()
    }

    pub async fn mark_sync_complete(&self) {
        let mut state = self.sync_state.write().await;
        state.sync_in_progress = false;
        state.last_sync = Utc::now();
        state.sync_version += 1;
        state.pending_operations.clear();
    }
}

impl ClusterSyncState {
    pub fn new() -> Self {
        Self {
            last_sync: Utc::now(),
            pending_operations: Vec::new(),
            acknowledged_operations: HashSet::new(),
            sync_in_progress: false,
            sync_version: 0,
        }
    }

    pub fn mark_operation_acknowledged(&mut self, operation_id: Uuid) {
        self.acknowledged_operations.insert(operation_id);
        self.pending_operations.retain(|&id| id != operation_id);
    }

    pub fn add_pending_operation(&mut self, operation_id: Uuid) {
        if !self.acknowledged_operations.contains(&operation_id) {
            self.pending_operations.push(operation_id);
        }
    }
}

impl ClusterState {
    pub fn new(cluster_id: String) -> Self {
        Self {
            cluster_id,
            is_primary: false,
            is_healthy: true,
            last_heartbeat: Utc::now(),
            operation_count: 0,
        }
    }

    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Utc::now();
    }

    pub fn is_responsive(&self) -> bool {
        let elapsed = Utc::now().signed_duration_since(self.last_heartbeat);
        elapsed.num_seconds() < 30 // Consider unresponsive after 30 seconds
    }
}
