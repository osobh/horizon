//! Knowledge transfer coordination with bandwidth management

use std::sync::Arc;
use std::collections::VecDeque;
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;
use dashmap::DashMap;

/// Knowledge transfer coordinator with bandwidth management
pub struct KnowledgeTransferCoordinator {
    transfer_queue: Arc<RwLock<VecDeque<TransferRequest>>>,
    bandwidth_limiter: Arc<Semaphore>,
    transfer_stats: Arc<DashMap<String, TransferStats>>,
    compression_enabled: bool,
    encryption_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub package_id: Uuid,
    pub source_cluster: String,
    pub target_cluster: String,
    pub priority: TransferPriority,
    pub estimated_size_mb: f64,
    pub compression_ratio: f64,
    pub retry_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    pub total_transfers: u64,
    pub successful_transfers: u64,
    pub failed_transfers: u64,
    pub total_bytes_transferred: u64,
    pub average_speed_mbps: f64,
}

impl KnowledgeTransferCoordinator {
    pub fn new(max_concurrent_transfers: usize) -> Self {
        Self {
            transfer_queue: Arc::new(RwLock::new(VecDeque::new())),
            bandwidth_limiter: Arc::new(Semaphore::new(max_concurrent_transfers)),
            transfer_stats: Arc::new(DashMap::new()),
            compression_enabled: true,
            encryption_enabled: true,
        }
    }
    
    pub async fn enqueue_transfer(&self, request: TransferRequest) {
        let mut queue = self.transfer_queue.write().await;
        
        // Insert based on priority
        let position = queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());
        
        queue.insert(position, request);
    }
    
    pub async fn get_next_transfer(&self) -> Option<TransferRequest> {
        let mut queue = self.transfer_queue.write().await;
        queue.pop_front()
    }
    
    pub async fn update_stats(&self, cluster_id: &str, success: bool, bytes: u64, speed_mbps: f64) {
        let mut entry = self.transfer_stats
            .entry(cluster_id.to_string())
            .or_insert_with(TransferStats::default);

        entry.total_transfers += 1;
        if success {
            entry.successful_transfers += 1;
        } else {
            entry.failed_transfers += 1;
        }
        entry.total_bytes_transferred += bytes;

        // Update average speed with exponential moving average
        let alpha = 0.1;
        entry.average_speed_mbps = entry.average_speed_mbps * (1.0 - alpha) + speed_mbps * alpha;
    }

    pub async fn get_stats(&self, cluster_id: &str) -> Option<TransferStats> {
        self.transfer_stats.get(cluster_id).map(|r| r.clone())
    }
}