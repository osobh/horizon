//! Inventory storage trait and implementations
//!
//! Provides an abstract storage interface for node inventory with
//! pluggable backends.

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{InventoryError, Result};
use crate::node::{InventorySummary, NodeInfo, NodeStatus};

/// Inventory data that gets persisted
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct InventoryData {
    /// Version for migration purposes
    pub version: u32,
    /// All registered nodes
    pub nodes: HashMap<String, NodeInfo>,
}

impl InventoryData {
    /// Current data format version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create new empty inventory data
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            nodes: HashMap::new(),
        }
    }
}

/// Abstract inventory storage interface
#[async_trait]
pub trait InventoryStore: Send + Sync {
    /// Add a new node to inventory
    async fn add_node(&mut self, node: NodeInfo) -> Result<()>;

    /// Get a node by ID
    async fn get_node(&self, id: &str) -> Result<Option<NodeInfo>>;

    /// Update an existing node
    async fn update_node(&mut self, node: NodeInfo) -> Result<()>;

    /// Remove a node by ID
    async fn remove_node(&mut self, id: &str) -> Result<NodeInfo>;

    /// List all nodes
    async fn list_nodes(&self) -> Result<Vec<NodeInfo>>;

    /// Update node status
    async fn update_status(&mut self, id: &str, status: NodeStatus) -> Result<()>;

    /// Find node by name or ID
    async fn find_node(&self, name_or_id: &str) -> Result<Option<NodeInfo>>;

    /// Find node by address
    async fn find_by_address(&self, address: &str) -> Result<Option<NodeInfo>>;

    /// List nodes by status
    async fn list_by_status(&self, status: NodeStatus) -> Result<Vec<NodeInfo>>;

    /// List nodes by tag
    async fn list_by_tag(&self, tag: &str) -> Result<Vec<NodeInfo>>;

    /// Get inventory summary
    async fn summary(&self) -> Result<InventorySummary>;

    /// Get node count
    async fn count(&self) -> Result<usize>;

    /// Check if inventory is empty
    async fn is_empty(&self) -> Result<bool>;
}

/// In-memory inventory store with JSON file persistence
#[derive(Debug)]
pub struct InMemoryStore {
    /// Path to inventory file
    inventory_path: PathBuf,
    /// In-memory inventory data
    data: InventoryData,
}

impl InMemoryStore {
    /// Create a new in-memory store with default path (~/.hpc/inventory/nodes.json)
    ///
    /// # Errors
    /// Returns error if home directory cannot be found or inventory directory cannot be created.
    pub fn new() -> Result<Self> {
        let inventory_dir = Self::inventory_dir()?;
        std::fs::create_dir_all(&inventory_dir).map_err(|e| {
            InventoryError::StorageError(format!(
                "Failed to create inventory directory {:?}: {e}",
                inventory_dir
            ))
        })?;

        let inventory_path = inventory_dir.join("nodes.json");
        let data = Self::load_from_path(&inventory_path)?;

        Ok(Self {
            inventory_path,
            data,
        })
    }

    /// Create store with custom path (useful for testing)
    #[must_use]
    pub fn with_path(inventory_path: PathBuf) -> Self {
        let data = Self::load_from_path(&inventory_path).unwrap_or_default();
        Self {
            inventory_path,
            data,
        }
    }

    /// Get the default inventory directory path
    ///
    /// # Errors
    /// Returns error if home directory cannot be found.
    pub fn inventory_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().ok_or(InventoryError::StorageError(
            "Cannot find home directory".to_string(),
        ))?;
        Ok(home.join(".hpc").join("inventory"))
    }

    /// Get the keys directory path
    ///
    /// # Errors
    /// Returns error if inventory directory cannot be determined.
    pub fn keys_dir() -> Result<PathBuf> {
        Ok(Self::inventory_dir()?.join("keys"))
    }

    /// Load inventory data from a file path
    fn load_from_path(path: &PathBuf) -> Result<InventoryData> {
        if path.exists() {
            let content = std::fs::read_to_string(path).map_err(|e| {
                InventoryError::StorageError(format!("Failed to read inventory file: {e}"))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                InventoryError::StorageError(format!("Failed to parse inventory file: {e}"))
            })
        } else {
            Ok(InventoryData::new())
        }
    }

    /// Save inventory to disk
    fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.data)
            .map_err(|e| InventoryError::StorageError(format!("Failed to serialize inventory: {e}")))?;
        std::fs::write(&self.inventory_path, content).map_err(|e| {
            InventoryError::StorageError(format!(
                "Failed to write inventory file {:?}: {e}",
                self.inventory_path
            ))
        })?;
        Ok(())
    }

    /// Generate a unique node name based on address
    #[must_use]
    pub fn generate_node_name(&self, address: &str) -> String {
        let base_name = address.replace('.', "-").replace(':', "-");

        let mut name = format!("node-{base_name}");
        let mut counter = 1;

        while self.data.nodes.values().any(|n| n.name == name) {
            name = format!("node-{base_name}-{counter}");
            counter += 1;
        }

        name
    }

    /// Get mutable reference to a node (for sync operations)
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut NodeInfo> {
        self.data.nodes.get_mut(id)
    }

    /// Find node by name or ID (mutable, for sync operations)
    pub fn find_node_mut(&mut self, name_or_id: &str) -> Option<&mut NodeInfo> {
        if self.data.nodes.contains_key(name_or_id) {
            return self.data.nodes.get_mut(name_or_id);
        }
        self.data.nodes.values_mut().find(|n| n.name == name_or_id)
    }

    /// Remove by name or ID (sync version)
    ///
    /// # Errors
    /// Returns error if node is not found.
    pub fn remove_by_name_or_id(&mut self, name_or_id: &str) -> Result<NodeInfo> {
        let id = self
            .data
            .nodes
            .values()
            .find(|n| n.id == name_or_id || n.name == name_or_id)
            .map(|n| n.id.clone())
            .ok_or_else(|| InventoryError::NodeNotFound(name_or_id.to_string()))?;

        let node = self.data.nodes.remove(&id).ok_or_else(|| {
            InventoryError::NodeNotFound(id.clone())
        })?;
        self.save()?;
        Ok(node)
    }

    /// Persist changes (call after batch operations)
    ///
    /// # Errors
    /// Returns error if save fails.
    pub fn persist(&self) -> Result<()> {
        self.save()
    }
}

#[async_trait]
impl InventoryStore for InMemoryStore {
    async fn add_node(&mut self, node: NodeInfo) -> Result<()> {
        if self.data.nodes.contains_key(&node.id) {
            return Err(InventoryError::DuplicateNode(node.id));
        }

        // Check for duplicate address
        if self.data.nodes.values().any(|n| n.address == node.address) {
            return Err(InventoryError::DuplicateAddress(node.address));
        }

        self.data.nodes.insert(node.id.clone(), node);
        self.save()?;
        Ok(())
    }

    async fn get_node(&self, id: &str) -> Result<Option<NodeInfo>> {
        Ok(self.data.nodes.get(id).cloned())
    }

    async fn update_node(&mut self, node: NodeInfo) -> Result<()> {
        if !self.data.nodes.contains_key(&node.id) {
            return Err(InventoryError::NodeNotFound(node.id));
        }
        self.data.nodes.insert(node.id.clone(), node);
        self.save()?;
        Ok(())
    }

    async fn remove_node(&mut self, id: &str) -> Result<NodeInfo> {
        let node = self
            .data
            .nodes
            .remove(id)
            .ok_or_else(|| InventoryError::NodeNotFound(id.to_string()))?;
        self.save()?;
        Ok(node)
    }

    async fn list_nodes(&self) -> Result<Vec<NodeInfo>> {
        let mut nodes: Vec<_> = self.data.nodes.values().cloned().collect();
        nodes.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(nodes)
    }

    async fn update_status(&mut self, id: &str, status: NodeStatus) -> Result<()> {
        let node = self
            .data
            .nodes
            .get_mut(id)
            .ok_or_else(|| InventoryError::NodeNotFound(id.to_string()))?;
        node.set_status(status);
        self.save()?;
        Ok(())
    }

    async fn find_node(&self, name_or_id: &str) -> Result<Option<NodeInfo>> {
        // First try exact ID match
        if let Some(node) = self.data.nodes.get(name_or_id) {
            return Ok(Some(node.clone()));
        }

        // Then try name match
        Ok(self
            .data
            .nodes
            .values()
            .find(|n| n.name == name_or_id)
            .cloned())
    }

    async fn find_by_address(&self, address: &str) -> Result<Option<NodeInfo>> {
        Ok(self
            .data
            .nodes
            .values()
            .find(|n| n.address == address)
            .cloned())
    }

    async fn list_by_status(&self, status: NodeStatus) -> Result<Vec<NodeInfo>> {
        Ok(self
            .data
            .nodes
            .values()
            .filter(|n| n.status == status)
            .cloned()
            .collect())
    }

    async fn list_by_tag(&self, tag: &str) -> Result<Vec<NodeInfo>> {
        Ok(self
            .data
            .nodes
            .values()
            .filter(|n| n.tags.contains(&tag.to_string()))
            .cloned()
            .collect())
    }

    async fn summary(&self) -> Result<InventorySummary> {
        let nodes: Vec<_> = self.data.nodes.values().cloned().collect();
        Ok(InventorySummary::from_nodes(&nodes))
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.data.nodes.len())
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(self.data.nodes.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{CredentialRef, NodeMode};
    use tempfile::TempDir;

    fn create_test_store(temp_dir: &TempDir) -> InMemoryStore {
        let inventory_path = temp_dir.path().join("nodes.json");
        InMemoryStore::with_path(inventory_path)
    }

    fn create_test_node(name: &str, address: &str) -> NodeInfo {
        NodeInfo::new(
            name.to_string(),
            address.to_string(),
            22,
            "user".to_string(),
            CredentialRef::SshAgent,
            NodeMode::Docker,
        )
    }

    #[tokio::test]
    async fn test_add_and_get_node() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("test-node", "192.168.1.100");
        let node_id = node.id.clone();

        store.add_node(node).await.unwrap();

        let retrieved = store.get_node(&node_id).await.unwrap().unwrap();
        assert_eq!(retrieved.name, "test-node");
        assert_eq!(retrieved.address, "192.168.1.100");
    }

    #[tokio::test]
    async fn test_find_node_by_name() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("my-gpu-node", "10.0.0.50");
        store.add_node(node).await.unwrap();

        let found = store.find_node("my-gpu-node").await.unwrap().unwrap();
        assert_eq!(found.address, "10.0.0.50");
    }

    #[tokio::test]
    async fn test_remove_node() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("to-remove", "1.2.3.4");
        let node_id = node.id.clone();
        store.add_node(node).await.unwrap();

        assert_eq!(store.count().await.unwrap(), 1);

        let removed = store.remove_node(&node_id).await.unwrap();
        assert_eq!(removed.name, "to-remove");
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_duplicate_address_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node1 = create_test_node("node1", "192.168.1.1");
        let node2 = create_test_node("node2", "192.168.1.1");

        store.add_node(node1).await.unwrap();
        let result = store.add_node(node2).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InventoryError::DuplicateAddress(_)));
    }

    #[test]
    fn test_generate_node_name() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let name1 = store.generate_node_name("192.168.1.100");
        assert_eq!(name1, "node-192-168-1-100");

        // Add a node with that name
        let mut node = create_test_node(&name1, "192.168.1.100");
        node.name = name1;
        store.data.nodes.insert(node.id.clone(), node);

        // Next name should have a counter
        let name2 = store.generate_node_name("192.168.1.100");
        assert_eq!(name2, "node-192-168-1-100-1");
    }

    #[tokio::test]
    async fn test_list_by_status() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let mut node1 = create_test_node("connected", "1.1.1.1");
        node1.status = NodeStatus::Connected;

        let mut node2 = create_test_node("pending", "2.2.2.2");
        node2.status = NodeStatus::Pending;

        let mut node3 = create_test_node("also-connected", "3.3.3.3");
        node3.status = NodeStatus::Connected;

        store.add_node(node1).await.unwrap();
        store.add_node(node2).await.unwrap();
        store.add_node(node3).await.unwrap();

        let connected = store.list_by_status(NodeStatus::Connected).await.unwrap();
        assert_eq!(connected.len(), 2);

        let pending = store.list_by_status(NodeStatus::Pending).await.unwrap();
        assert_eq!(pending.len(), 1);
    }
}
