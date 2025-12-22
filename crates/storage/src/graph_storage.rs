//! Main graph storage implementation

use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;

use crate::error::StorageError;
use crate::graph_csr::GraphCSR;
use crate::graph_format::NodeRecord;
use crate::graph_wal::{GraphWAL, NodeUpdates, WALEntry};

/// Main graph storage manager
pub struct GraphStorage {
    #[allow(dead_code)]
    base_path: PathBuf,
    node_file: Arc<Mutex<File>>,
    edge_csr: Arc<Mutex<GraphCSR>>,
    wal: Arc<GraphWAL>,
    node_count: Arc<Mutex<usize>>,
    max_nodes: usize,
}

impl GraphStorage {
    /// Create a new graph storage instance
    pub async fn create(base_path: PathBuf, initial_nodes: usize) -> Result<Self, StorageError> {
        // Create directory structure
        tokio::fs::create_dir_all(&base_path).await?;
        tokio::fs::create_dir_all(base_path.join("nodes")).await?;
        tokio::fs::create_dir_all(base_path.join("edges")).await?;
        tokio::fs::create_dir_all(base_path.join("wal")).await?;
        tokio::fs::create_dir_all(base_path.join("snapshots")).await?;

        // Initialize node storage file
        let node_file_path = base_path.join("nodes/records.bin");
        let node_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&node_file_path)
            .await?;

        // Pre-allocate space for initial nodes
        let file_size = initial_nodes * NodeRecord::SIZE;
        node_file.set_len(file_size as u64).await?;

        // Initialize CSR edge storage
        let edge_csr = GraphCSR::new(initial_nodes);

        // Initialize WAL
        let wal = GraphWAL::new(base_path.join("wal")).await?;

        Ok(Self {
            base_path,
            node_file: Arc::new(Mutex::new(node_file)),
            edge_csr: Arc::new(Mutex::new(edge_csr)),
            wal: Arc::new(wal),
            node_count: Arc::new(Mutex::new(0)),
            max_nodes: initial_nodes,
        })
    }

    /// Open existing graph storage
    pub async fn open(base_path: PathBuf) -> Result<Self, StorageError> {
        // Check if storage exists
        if !base_path.exists() {
            return Err(StorageError::StorageNotInitialized);
        }

        // Open node file
        let node_file_path = base_path.join("nodes/records.bin");
        let node_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&node_file_path)
            .await?;

        // Get file size to determine max nodes
        let metadata = node_file.metadata().await?;
        let max_nodes = (metadata.len() as usize) / NodeRecord::SIZE;

        // TODO: Load CSR from disk
        let edge_csr = GraphCSR::new(max_nodes);

        // Open WAL
        let wal = GraphWAL::new(base_path.join("wal")).await?;

        Ok(Self {
            base_path,
            node_file: Arc::new(Mutex::new(node_file)),
            edge_csr: Arc::new(Mutex::new(edge_csr)),
            wal: Arc::new(wal),
            node_count: Arc::new(Mutex::new(0)), // TODO: Load actual count
            max_nodes,
        })
    }

    /// Write a node to storage
    pub async fn write_node(&self, node: &NodeRecord) -> Result<(), StorageError> {
        if node.id as usize >= self.max_nodes {
            return Err(StorageError::InvalidNode {
                id: node.id,
                reason: "Node ID exceeds maximum nodes".to_string(),
            });
        }

        // Prepare data outside of locks for better performance
        let offset = node.id * NodeRecord::SIZE as u64;
        let node_bytes = node.to_bytes();
        let new_count = node.id as usize + 1;

        // Log to WAL first (durability before data write)
        self.wal
            .append(WALEntry::NodeWrite {
                id: node.id,
                data: *node,
            })
            .await?;

        // Perform file I/O with minimal lock scope
        {
            let mut file = self.node_file.lock().await;
            file.seek(tokio::io::SeekFrom::Start(offset)).await?;
            file.write_all(&node_bytes).await?;
            file.sync_all().await?;
        }

        // Update node count with minimal lock scope
        {
            let mut count = self.node_count.lock().await;
            *count = (*count).max(new_count);
        }

        Ok(())
    }

    /// Read a node from storage
    pub async fn read_node(&self, id: u64) -> Result<NodeRecord, StorageError> {
        if id as usize >= self.max_nodes {
            return Err(StorageError::InvalidNode {
                id,
                reason: "Node ID exceeds maximum nodes".to_string(),
            });
        }

        // Prepare offset and buffer outside of lock
        let offset = id * NodeRecord::SIZE as u64;
        let mut buffer = [0u8; NodeRecord::SIZE]; // Use array for stack allocation

        // Minimize lock scope for file I/O
        {
            let mut file = self.node_file.lock().await;
            file.seek(tokio::io::SeekFrom::Start(offset)).await?;
            file.read_exact(&mut buffer).await?;
        }

        NodeRecord::from_bytes(&buffer).ok_or_else(|| StorageError::InvalidDataFormat {
            reason: "Failed to parse node record".to_string(),
        })
    }

    /// Update node properties
    pub async fn update_node(&self, id: u64, updates: NodeUpdates) -> Result<(), StorageError> {
        // Read current node
        let mut node = self.read_node(id).await?;

        // Apply updates
        if let Some(flags) = updates.flags {
            node.flags = flags;
        }
        if let Some(embedding_offset) = updates.embedding_offset {
            node.embedding_offset = embedding_offset;
        }
        if let Some(property_offset) = updates.property_offset {
            node.property_offset = property_offset;
        }
        if let Some(edge_count) = updates.edge_count {
            node.edge_count = edge_count;
        }
        if let Some(importance_score) = updates.importance_score {
            node.importance_score = importance_score;
        }

        node.mark_accessed();

        // Log update to WAL
        self.wal
            .append(WALEntry::NodeUpdate { id, updates })
            .await?;

        // Write updated node
        self.write_node(&node).await
    }

    /// Add an edge
    pub async fn add_edge(
        &self,
        from: u64,
        to: u64,
        edge_type: u32,
        weight: f32,
    ) -> Result<(), StorageError> {
        // Log to WAL
        self.wal
            .append(WALEntry::EdgeAdd {
                from,
                to,
                edge_type,
                weight,
            })
            .await?;

        // Add to CSR
        let mut csr = self.edge_csr.lock().await;

        csr.add_edge(from as usize, to, edge_type, weight)?;

        // Update source node's edge count
        self.update_node(
            from,
            NodeUpdates {
                flags: None,
                embedding_offset: None,
                property_offset: None,
                edge_count: Some(csr.degree(from as usize)? as u32),
                importance_score: None,
            },
        )
        .await?;

        Ok(())
    }

    /// Get all edges from a node
    pub async fn get_edges(&self, from: u64) -> Result<Vec<crate::graph_csr::Edge>, StorageError> {
        let csr = self.edge_csr.lock().await;

        Ok(csr.get_edges(from as usize)?.collect())
    }

    /// Get node count
    pub async fn node_count(&self) -> Result<usize, StorageError> {
        let count = self.node_count.lock().await;

        Ok(*count)
    }

    /// Write multiple nodes efficiently in a batch
    pub async fn write_nodes_batch(&self, nodes: &[NodeRecord]) -> Result<(), StorageError> {
        if nodes.is_empty() {
            return Ok(());
        }

        // Validate all nodes first
        for node in nodes {
            if node.id as usize >= self.max_nodes {
                return Err(StorageError::InvalidNode {
                    id: node.id,
                    reason: "Node ID exceeds maximum nodes".to_string(),
                });
            }
        }

        // Log all operations to WAL first
        for node in nodes {
            self.wal
                .append(WALEntry::NodeWrite {
                    id: node.id,
                    data: *node,
                })
                .await?;
        }

        // Prepare write operations outside of lock
        let mut writes = Vec::with_capacity(nodes.len());
        for node in nodes {
            let offset = node.id * NodeRecord::SIZE as u64;
            let bytes = node.to_bytes();
            writes.push((offset, bytes));
        }

        // Perform all file writes in a single lock acquisition
        {
            let mut file = self.node_file.lock().await;
            for (offset, bytes) in writes {
                file.seek(tokio::io::SeekFrom::Start(offset)).await?;
                file.write_all(&bytes).await?;
            }
            file.sync_all().await?;
        }

        // Update node count
        if let Some(max_node) = nodes.iter().map(|n| n.id).max() {
            let new_count = max_node as usize + 1;
            let mut count = self.node_count.lock().await;
            *count = (*count).max(new_count);
        }

        Ok(())
    }

    /// Force a checkpoint
    pub async fn checkpoint(&self) -> Result<(), StorageError> {
        self.wal.checkpoint().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_format::node_flags;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_graph_storage_create() {
        let dir = tempdir()?;
        let _storage = GraphStorage::create(dir.path().to_path_buf(), 1000)
            .await
            .unwrap();

        // Verify directory structure
        assert!(dir.path().join("nodes").exists());
        assert!(dir.path().join("edges").exists());
        assert!(dir.path().join("wal").exists());
        assert!(dir.path().join("snapshots").exists());
        assert!(dir.path().join("nodes/records.bin").exists());
    }

    #[tokio::test]
    async fn test_node_write_read() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        let mut node = NodeRecord::new(42, 7);
        node.flags = node_flags::ACTIVE | node_flags::VERIFIED;
        node.importance_score = 0.85;

        storage.write_node(&node).await?;

        let read_node = storage.read_node(42).await?;
        assert_eq!(read_node.id, 42);
        assert_eq!(read_node.type_id, 7);
        assert_eq!(read_node.flags, node_flags::ACTIVE | node_flags::VERIFIED);
        assert_eq!(read_node.importance_score, 0.85);
    }

    #[tokio::test]
    async fn test_node_update() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        let node = NodeRecord::new(10, 5);
        storage.write_node(&node).await?;

        storage
            .update_node(
                10,
                NodeUpdates {
                    flags: Some(node_flags::SYNTHETIC),
                    embedding_offset: Some(1234),
                    property_offset: None,
                    edge_count: None,
                    importance_score: Some(0.95),
                },
            )
            .await
            .unwrap();

        let updated = storage.read_node(10).await?;
        assert_eq!(updated.flags, node_flags::SYNTHETIC);
        assert_eq!(updated.embedding_offset, 1234);
        assert_eq!(updated.importance_score, 0.95);
        assert!(updated.access_count > 0);
    }

    #[tokio::test]
    async fn test_edge_operations() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        // Create nodes
        storage.write_node(&NodeRecord::new(1, 1)).await?;
        storage.write_node(&NodeRecord::new(2, 1)).await?;
        storage.write_node(&NodeRecord::new(3, 1)).await?;

        // Add edges
        storage.add_edge(1, 2, 10, 0.5).await?;
        storage.add_edge(1, 3, 11, 0.7).await?;

        // Get edges
        let edges = storage.get_edges(1).await?;
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].target, 2);
        assert_eq!(edges[0].edge_type, 10);
        assert_eq!(edges[0].weight, 0.5);
        assert_eq!(edges[1].target, 3);
        assert_eq!(edges[1].edge_type, 11);
        assert_eq!(edges[1].weight, 0.7);

        // Check that node's edge count was updated
        let node = storage.read_node(1).await?;
        assert_eq!(node.edge_count, 2);
    }

    #[tokio::test]
    async fn test_invalid_node_id() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 10)
            .await
            .unwrap();

        let node = NodeRecord::new(100, 1); // ID exceeds max_nodes
        let result = storage.write_node(&node).await;

        assert!(result.is_err());
        match result {
            Err(StorageError::InvalidNode { id, .. }) => assert_eq!(id, 100),
            _ => panic!("Expected InvalidNode error"),
        }
    }

    #[tokio::test]
    async fn test_node_count() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        assert_eq!(storage.node_count().await?, 0);

        storage.write_node(&NodeRecord::new(5, 1)).await?;
        assert_eq!(storage.node_count().await?, 6); // 0-5

        storage.write_node(&NodeRecord::new(10, 1)).await?;
        assert_eq!(storage.node_count().await?, 11); // 0-10
    }

    #[tokio::test]
    async fn test_checkpoint() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        storage.write_node(&NodeRecord::new(1, 1)).await?;
        storage.checkpoint().await?;

        // Checkpoint should succeed without error
    }

    #[tokio::test]
    async fn test_open_existing_storage() {
        let dir = tempdir()?;

        // Create storage first
        let storage1 = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        // Write some data
        let node = NodeRecord::new(5, 2);
        storage1.write_node(&node).await?;
        storage1.add_edge(5, 10, 1, 0.5).await?;

        // Drop the first storage
        drop(storage1);

        // Open existing storage
        let storage2 = GraphStorage::open(dir.path().to_path_buf()).await?;

        // Verify we can read the data
        let read_node = storage2.read_node(5).await?;
        assert_eq!(read_node.id, 5);
        assert_eq!(read_node.type_id, 2);

        // Verify max_nodes is calculated correctly
        assert_eq!(storage2.max_nodes, 100);
    }

    #[tokio::test]
    async fn test_open_nonexistent_storage() {
        let dir = tempdir()?;
        let nonexistent_path = dir.path().join("nonexistent");

        let result = GraphStorage::open(nonexistent_path).await;
        assert!(matches!(result, Err(StorageError::StorageNotInitialized)));
    }

    #[tokio::test]
    async fn test_batch_write_nodes() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        // Create multiple nodes for batch write
        let nodes = vec![
            NodeRecord::new(1, 10),
            NodeRecord::new(2, 20),
            NodeRecord::new(3, 30),
            NodeRecord::new(5, 50), // Non-sequential ID
        ];

        // Batch write should succeed
        storage.write_nodes_batch(&nodes).await?;

        // Verify all nodes were written correctly
        for node in &nodes {
            let read_node = storage.read_node(node.id).await?;
            assert_eq!(read_node.id, node.id);
            assert_eq!(read_node.type_id, node.type_id);
        }

        // Verify node count is updated correctly (should be 6: 0-5)
        assert_eq!(storage.node_count().await?, 6);
    }

    #[tokio::test]
    async fn test_empty_batch_write() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
            .await
            .unwrap();

        // Empty batch should succeed without error
        storage.write_nodes_batch(&[]).await?;
        assert_eq!(storage.node_count().await?, 0);
    }

    #[tokio::test]
    async fn test_batch_write_invalid_node() {
        let dir = tempdir()?;
        let storage = GraphStorage::create(dir.path().to_path_buf(), 10)
            .await
            .unwrap();

        // Create batch with one invalid node ID
        let nodes = vec![
            NodeRecord::new(1, 10),
            NodeRecord::new(100, 20), // Invalid: exceeds max_nodes
            NodeRecord::new(2, 30),
        ];

        // Batch write should fail without writing any nodes
        let result = storage.write_nodes_batch(&nodes).await;
        assert!(result.is_err());

        // Verify no nodes were written
        assert_eq!(storage.node_count().await?, 0);
    }

    //    #[tokio::test]
    //    async fn test_mutex_poisoning_write_node_file() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Poison the node_file mutex
    //        let poisoned_file = Arc::clone(&storage.node_file);
    //        thread::spawn(move || {
    //            let _guard = poisoned_file.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to write a node
    //        let node = NodeRecord::new(1, 1);
    //        let result = storage.write_node(&node).await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "node file")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_write_node_count() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Poison the node_count mutex
    //        let poisoned_count = Arc::clone(&storage.node_count);
    //        thread::spawn(move || {
    //            let _guard = poisoned_count.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to write a node
    //        let node = NodeRecord::new(1, 1);
    //        let result = storage.write_node(&node).await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "node count")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_read_node() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Write a node first
    //        storage.write_node(&NodeRecord::new(1, 1)).await?;
    //
    //        // Poison the node_file mutex
    //        let poisoned_file = Arc::clone(&storage.node_file);
    //        thread::spawn(move || {
    //            let _guard = poisoned_file.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to read the node
    //        let result = storage.read_node(1).await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "node file")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_add_edge() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Create nodes first
    //        storage.write_node(&NodeRecord::new(1, 1)).await?;
    //        storage.write_node(&NodeRecord::new(2, 1)).await?;
    //
    //        // Poison the edge_csr mutex
    //        let poisoned_csr = Arc::clone(&storage.edge_csr);
    //        thread::spawn(move || {
    //            let _guard = poisoned_csr.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to add an edge
    //        let result = storage.add_edge(1, 2, 10, 0.5).await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "edge CSR")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_get_edges() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Poison the edge_csr mutex
    //        let poisoned_csr = Arc::clone(&storage.edge_csr);
    //        thread::spawn(move || {
    //            let _guard = poisoned_csr.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to get edges
    //        let result = storage.get_edges(1).await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "edge CSR")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_node_count() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Poison the node_count mutex
    //        let poisoned_count = Arc::clone(&storage.node_count);
    //        thread::spawn(move || {
    //            let _guard = poisoned_count.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to get node count
    //        let result = storage.node_count().await;
    //
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "node count")
    //        );
    //    }
    //
    //    #[tokio::test]
    //    async fn test_update_node_with_flags_only() {
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Write initial node
    //        let node = NodeRecord::new(42, 0);
    //        storage.write_node(&node).await?;
    //
    //        // Update only flags
    //        let updates = NodeUpdates {
    //            flags: Some(0xFF),
    //            embedding_offset: None,
    //            property_offset: None,
    //            edge_count: None,
    //            importance_score: None,
    //        };
    //
    //        storage.update_node(42, updates).await?;
    //
    //        // Read back and verify
    //        let updated = storage.read_node(42).await?;
    //        assert_eq!(updated.flags, 0xFF);
    //        assert_eq!(updated.embedding_offset, 0);
    //    }
    //
    //    #[tokio::test]
    //    async fn test_mutex_poisoning_read_node_lock() {
    //        use std::thread;
    //
    //        let dir = tempdir()?;
    //        let storage = GraphStorage::create(dir.path().to_path_buf(), 100)
    //            .await
    //            .unwrap();
    //
    //        // Write a node first
    //        let node = NodeRecord::new(5, 0);
    //        storage.write_node(&node).await?;
    //
    //        // Poison the node_file mutex
    //        let poisoned_file = Arc::clone(&storage.node_file);
    //        thread::spawn(move || {
    //            let _guard = poisoned_file.lock()?;
    //            panic!("Poisoning mutex");
    //        })
    //        .join()
    //        .unwrap_err();
    //
    //        // Try to read the node
    //        let result = storage.read_node(5).await;
    //        assert!(
    //            matches!(result, Err(StorageError::LockPoisoned { resource }) if resource == "node file")
    //        );
    //    }
}
