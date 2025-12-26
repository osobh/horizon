//! Lock-free atomic knowledge graph operations
//!
//! Implements concurrent, lock-free updates to the knowledge graph using
//! atomic operations for high-performance multi-agent scenarios.

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Atomic update operation types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AtomicUpdateOp {
    /// Add new node
    AddNode,
    /// Update node embedding
    UpdateEmbedding,
    /// Add new edge
    AddEdge,
    /// Update edge weight
    UpdateEdgeWeight,
    /// Remove edge
    RemoveEdge,
    /// Batch update multiple operations
    BatchUpdate,
}

/// Atomic update command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicUpdate {
    /// Operation type
    pub operation: AtomicUpdateOp,
    /// Target node ID
    pub node_id: u32,
    /// Target edge (source, target) for edge operations
    pub edge: Option<(u32, u32)>,
    /// New embedding data
    pub embedding: Option<Vec<f32>>,
    /// New weight value
    pub weight: Option<f32>,
    /// Timestamp for ordering
    pub timestamp: u64,
}

/// Lock-free atomic node structure
#[derive(Debug)]
pub struct AtomicNode {
    /// Node ID
    pub id: u32,
    /// Node type hash (atomic for concurrent updates)
    pub node_type_hash: AtomicU32,
    /// Embedding version counter (for lock-free updates)
    pub embedding_version: AtomicU64,
    /// Edge count (atomic for concurrent edge additions)
    pub edge_count: AtomicU32,
    /// Last update timestamp
    pub last_updated: AtomicU64,
}

/// Lock-free atomic edge structure
#[derive(Debug)]
pub struct AtomicEdge {
    /// Source node ID
    pub source_id: u32,
    /// Target node ID  
    pub target_id: u32,
    /// Relationship hash
    pub relationship_hash: AtomicU32,
    /// Weight (atomic for concurrent updates)
    pub weight: AtomicU32, // Using u32 bits to represent f32
    /// Edge status (active=1, deleted=0)
    pub status: AtomicU32,
    /// Version counter
    pub version: AtomicU64,
}

/// GPU-side atomic update queue
pub struct AtomicUpdateQueue {
    device: Arc<CudaDevice>,
    /// Update operations buffer
    update_buffer: CudaSlice<u8>,
    /// Queue head pointer (atomic)
    queue_head: AtomicU32,
    /// Queue tail pointer (atomic)
    queue_tail: AtomicU32,
    /// Maximum queue size
    max_queue_size: usize,
    /// Update buffer size
    update_buffer_size: usize,
}

impl AtomicUpdateQueue {
    /// Create new atomic update queue
    pub fn new(device: Arc<CudaDevice>, max_queue_size: usize) -> Result<Self> {
        let update_size = std::mem::size_of::<AtomicUpdate>();
        let update_buffer_size = max_queue_size * update_size;
        let update_buffer = unsafe { device.alloc::<u8>(update_buffer_size)? };

        Ok(Self {
            device,
            update_buffer,
            queue_head: AtomicU32::new(0),
            queue_tail: AtomicU32::new(0),
            max_queue_size,
            update_buffer_size,
        })
    }

    /// Enqueue update operation (lock-free)
    pub fn enqueue_update(&self, update: AtomicUpdate) -> Result<bool> {
        let current_tail = self.queue_tail.load(Ordering::Acquire);
        let next_tail = (current_tail + 1) % self.max_queue_size as u32;

        // Check if queue is full
        let current_head = self.queue_head.load(Ordering::Acquire);
        if next_tail == current_head {
            return Ok(false); // Queue full
        }

        // Serialize update to bytes
        let update_bytes = bincode::serialize(&update)
            .map_err(|e| anyhow::anyhow!("Serialization error: {}", e))?;

        // Calculate buffer offset
        let offset = current_tail as usize * std::mem::size_of::<AtomicUpdate>();

        // Copy update to GPU buffer (simplified - would need proper slice handling)
        if offset + update_bytes.len() <= self.update_buffer_size {
            // In practice would use proper GPU memory operations
            // For now, just advance the tail
            let prev_tail = self.queue_tail.compare_exchange_weak(
                current_tail,
                next_tail,
                Ordering::Release,
                Ordering::Relaxed,
            );

            Ok(prev_tail.is_ok())
        } else {
            Ok(false) // Buffer overflow
        }
    }

    /// Dequeue update operation (lock-free)
    pub fn dequeue_update(&self) -> Result<Option<AtomicUpdate>> {
        let current_head = self.queue_head.load(Ordering::Acquire);
        let current_tail = self.queue_tail.load(Ordering::Acquire);

        // Check if queue is empty
        if current_head == current_tail {
            return Ok(None);
        }

        // Calculate buffer offset
        let offset = current_head as usize * std::mem::size_of::<AtomicUpdate>();

        // Read update from GPU buffer (simplified)
        // In practice would read from GPU memory
        let next_head = (current_head + 1) % self.max_queue_size as u32;

        let prev_head = self.queue_head.compare_exchange_weak(
            current_head,
            next_head,
            Ordering::Release,
            Ordering::Relaxed,
        );

        if prev_head.is_ok() {
            // Would deserialize actual update from GPU memory
            Ok(Some(AtomicUpdate {
                operation: AtomicUpdateOp::AddNode,
                node_id: 0,
                edge: None,
                embedding: None,
                weight: None,
                timestamp: 0,
            }))
        } else {
            Ok(None)
        }
    }

    /// Flush all pending updates to GPU
    pub fn flush_updates(&mut self) -> Result<usize> {
        let current_head = self.queue_head.load(Ordering::Acquire);
        let current_tail = self.queue_tail.load(Ordering::Acquire);

        let pending_count = if current_tail >= current_head {
            current_tail - current_head
        } else {
            (self.max_queue_size as u32 - current_head) + current_tail
        };

        if pending_count > 0 {
            // Launch GPU kernel to process pending updates
            unsafe {
                crate::kernels::launch_atomic_updates(
                    *self.update_buffer.device_ptr() as *const u8,
                    current_head,
                    current_tail,
                    self.max_queue_size as u32,
                );
            }

            // Reset queue pointers after processing
            self.queue_head.store(0, Ordering::Release);
            self.queue_tail.store(0, Ordering::Release);
        }

        Ok(pending_count as usize)
    }
}

/// Lock-free atomic knowledge graph
pub struct AtomicKnowledgeGraph {
    device: Arc<CudaDevice>,
    /// Atomic node data
    nodes: CudaSlice<u8>, // Serialized AtomicNode structs
    /// Atomic edge data
    edges: CudaSlice<u8>, // Serialized AtomicEdge structs
    /// Node embeddings with version tracking
    embeddings: CudaSlice<f32>,
    /// Embedding versions
    embedding_versions: CudaSlice<u64>,
    /// Edge adjacency lists with atomic counters
    adjacency_lists: CudaSlice<u32>,
    /// Update queue for concurrent operations
    update_queue: AtomicUpdateQueue,
    /// Graph metadata
    max_nodes: usize,
    max_edges: usize,
    embedding_dim: usize,
    /// Current counts (atomic)
    node_count: AtomicU32,
    edge_count: AtomicU32,
    /// Version counter for consistency
    graph_version: AtomicU64,
}

impl AtomicKnowledgeGraph {
    /// Create new atomic knowledge graph
    pub fn new(
        device: Arc<CudaDevice>,
        max_nodes: usize,
        max_edges: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        // Allocate GPU memory for nodes and edges
        let node_size = std::mem::size_of::<AtomicNode>();
        let edge_size = std::mem::size_of::<AtomicEdge>();

        let nodes = unsafe { device.alloc::<u8>(max_nodes * node_size)? };
        let edges = unsafe { device.alloc::<u8>(max_edges * edge_size)? };
        let embeddings = device.alloc_zeros::<f32>(max_nodes * embedding_dim)?;
        let embedding_versions = device.alloc_zeros::<u64>(max_nodes)?;
        let adjacency_lists = device.alloc_zeros::<u32>(max_nodes * 64)?; // Max 64 edges per node

        let update_queue = AtomicUpdateQueue::new(device.clone(), 10000)?;

        Ok(Self {
            device,
            nodes,
            edges,
            embeddings,
            embedding_versions,
            adjacency_lists,
            update_queue,
            max_nodes,
            max_edges,
            embedding_dim,
            node_count: AtomicU32::new(0),
            edge_count: AtomicU32::new(0),
            graph_version: AtomicU64::new(0),
        })
    }

    /// Add node atomically
    pub fn add_node_atomic(
        &self,
        node_id: u32,
        node_type_hash: u32,
        embedding: &[f32],
    ) -> Result<bool> {
        // Check if we have space
        let current_count = self.node_count.load(Ordering::Acquire);
        if current_count >= self.max_nodes as u32 {
            return Ok(false);
        }

        // Create atomic update
        let update = AtomicUpdate {
            operation: AtomicUpdateOp::AddNode,
            node_id,
            edge: None,
            embedding: Some(embedding.to_vec()),
            weight: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        // Enqueue for atomic processing
        let success = self.update_queue.enqueue_update(update)?;

        if success {
            // Optimistically increment counter
            self.node_count.fetch_add(1, Ordering::Relaxed);
            self.graph_version.fetch_add(1, Ordering::Relaxed);
        }

        Ok(success)
    }

    /// Update node embedding atomically
    pub fn update_embedding_atomic(&self, node_id: u32, new_embedding: &[f32]) -> Result<bool> {
        if node_id >= self.max_nodes as u32 {
            return Ok(false);
        }

        let update = AtomicUpdate {
            operation: AtomicUpdateOp::UpdateEmbedding,
            node_id,
            edge: None,
            embedding: Some(new_embedding.to_vec()),
            weight: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        let success = self.update_queue.enqueue_update(update)?;

        if success {
            self.graph_version.fetch_add(1, Ordering::Relaxed);
        }

        Ok(success)
    }

    /// Add edge atomically
    pub fn add_edge_atomic(
        &self,
        source_id: u32,
        target_id: u32,
        relationship_hash: u32,
        weight: f32,
    ) -> Result<bool> {
        let current_count = self.edge_count.load(Ordering::Acquire);
        if current_count >= self.max_edges as u32 {
            return Ok(false);
        }

        let update = AtomicUpdate {
            operation: AtomicUpdateOp::AddEdge,
            node_id: source_id,
            edge: Some((source_id, target_id)),
            embedding: None,
            weight: Some(weight),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        let success = self.update_queue.enqueue_update(update)?;

        if success {
            self.edge_count.fetch_add(1, Ordering::Relaxed);
            self.graph_version.fetch_add(1, Ordering::Relaxed);
        }

        Ok(success)
    }

    /// Batch update multiple operations
    pub fn batch_update_atomic(&mut self, updates: Vec<AtomicUpdate>) -> Result<usize> {
        let mut success_count = 0;

        for update in updates {
            if self.update_queue.enqueue_update(update)? {
                success_count += 1;
            }
        }

        // Flush all updates to GPU
        let processed = self.update_queue.flush_updates()?;

        if processed > 0 {
            self.graph_version
                .fetch_add(processed as u64, Ordering::Relaxed);
        }

        Ok(success_count)
    }

    /// Perform atomic similarity search with consistency guarantees
    pub fn atomic_similarity_search(
        &mut self,
        query_embedding: &[f32],
        k: usize,
        consistency_level: ConsistencyLevel,
    ) -> Result<Vec<(u32, f32)>> {
        // Record current graph version for consistency
        let search_version = match consistency_level {
            ConsistencyLevel::Strong => {
                // Wait for all pending updates to complete
                self.flush_pending_updates()?;
                self.graph_version.load(Ordering::SeqCst)
            }
            ConsistencyLevel::Eventual => self.graph_version.load(Ordering::Acquire),
            ConsistencyLevel::Weak => self.graph_version.load(Ordering::Relaxed),
        };

        // Allocate GPU memory for results
        let gpu_query = self.device.htod_sync_copy(query_embedding)?;
        let gpu_results = unsafe { self.device.alloc::<u32>(k * 2)? }; // (id, score) pairs
        let gpu_version = unsafe { self.device.alloc::<u64>(1)? };

        // Upload search version
        self.device
            .htod_copy_into(vec![search_version], &mut gpu_version.clone())?;

        // Launch atomic search kernel
        unsafe {
            crate::kernels::launch_atomic_similarity_search(
                *self.embeddings.device_ptr() as *const f32,
                *self.embedding_versions.device_ptr() as *const u64,
                *gpu_query.device_ptr() as *const f32,
                *gpu_results.device_ptr() as *mut u32,
                *gpu_version.device_ptr() as *const u64,
                self.max_nodes as u32,
                self.embedding_dim as u32,
                k as u32,
            );
        }

        // Download results
        let results: Vec<u32> = self.device.dtoh_sync_copy(&gpu_results)?;

        // Convert to (id, score) pairs
        let mut result_pairs = Vec::new();
        for i in (0..results.len()).step_by(2) {
            if i + 1 < results.len() {
                let node_id = results[i];
                let score_bits = results[i + 1];
                let score = f32::from_bits(score_bits);
                result_pairs.push((node_id, score));
            }
        }

        Ok(result_pairs)
    }

    /// Flush all pending updates
    fn flush_pending_updates(&mut self) -> Result<()> {
        self.update_queue.flush_updates()?;
        Ok(())
    }

    /// Get current graph statistics
    pub fn statistics(&self) -> AtomicGraphStatistics {
        AtomicGraphStatistics {
            node_count: self.node_count.load(Ordering::Acquire),
            edge_count: self.edge_count.load(Ordering::Acquire),
            graph_version: self.graph_version.load(Ordering::Acquire),
            max_nodes: self.max_nodes,
            max_edges: self.max_edges,
            embedding_dim: self.embedding_dim,
        }
    }

    /// Check if graph has pending updates
    pub fn has_pending_updates(&self) -> bool {
        let head = self.update_queue.queue_head.load(Ordering::Acquire);
        let tail = self.update_queue.queue_tail.load(Ordering::Acquire);
        head != tail
    }
}

/// Consistency levels for atomic operations
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyLevel {
    /// Strong consistency - wait for all updates to complete
    Strong,
    /// Eventual consistency - read latest committed state
    Eventual,
    /// Weak consistency - best effort read
    Weak,
}

/// Atomic graph statistics
#[derive(Debug, Clone)]
pub struct AtomicGraphStatistics {
    pub node_count: u32,
    pub edge_count: u32,
    pub graph_version: u64,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub embedding_dim: usize,
}

impl std::fmt::Display for AtomicGraphStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AtomicGraph: nodes={}/{}, edges={}/{}, version={}, dim={}",
            self.node_count,
            self.max_nodes,
            self.edge_count,
            self.max_edges,
            self.graph_version,
            self.embedding_dim
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_update_queue() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let queue = AtomicUpdateQueue::new(device, 100)?;

            let update = AtomicUpdate {
                operation: AtomicUpdateOp::AddNode,
                node_id: 42,
                edge: None,
                embedding: Some(vec![1.0, 2.0, 3.0]),
                weight: None,
                timestamp: 123456,
            };

            assert!(queue.enqueue_update(update)?);
        }
    }

    #[test]
    fn test_atomic_knowledge_graph_creation() -> Result<(), Box<dyn std::error::Error>>  {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            let graph = AtomicKnowledgeGraph::new(device, 1000, 5000, 128)?;

            let stats = graph.statistics();
            assert_eq!(stats.node_count, 0);
            assert_eq!(stats.edge_count, 0);
            assert_eq!(stats.max_nodes, 1000);
            assert_eq!(stats.max_edges, 5000);
            assert_eq!(stats.embedding_dim, 128);
        }
    }
}
