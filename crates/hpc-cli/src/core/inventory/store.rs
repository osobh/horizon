//! Inventory store
//!
//! Persistence layer for node inventory data.
//! This wraps the shared hpc-inventory crate's InMemoryStore.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use hpc_inventory::{InventorySummary, NodeInfo, NodeStatus};

/// Inventory data that gets persisted
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InventoryData {
    /// Version for migration purposes
    pub version: u32,
    /// All registered nodes
    pub nodes: HashMap<String, NodeInfo>,
}

impl InventoryData {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            nodes: HashMap::new(),
        }
    }
}

/// Inventory store for managing node persistence
pub struct InventoryStore {
    /// Path to inventory file
    inventory_path: PathBuf,
    /// In-memory inventory data
    data: InventoryData,
}

impl InventoryStore {
    /// Create a new inventory store
    pub fn new() -> Result<Self> {
        let inventory_dir = Self::inventory_dir()?;
        std::fs::create_dir_all(&inventory_dir)
            .with_context(|| format!("Failed to create inventory directory: {:?}", inventory_dir))?;

        let inventory_path = inventory_dir.join("nodes.json");

        let data = if inventory_path.exists() {
            let content = std::fs::read_to_string(&inventory_path)
                .with_context(|| format!("Failed to read inventory file: {:?}", inventory_path))?;
            serde_json::from_str(&content).unwrap_or_else(|_| InventoryData::new())
        } else {
            InventoryData::new()
        };

        Ok(Self {
            inventory_path,
            data,
        })
    }

    /// Get the inventory directory path
    pub fn inventory_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Cannot find home directory")?;
        Ok(home.join(".hpc").join("inventory"))
    }

    /// Get the keys directory path
    pub fn keys_dir() -> Result<PathBuf> {
        Ok(Self::inventory_dir()?.join("keys"))
    }

    /// Save inventory to disk
    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.data)
            .context("Failed to serialize inventory")?;
        std::fs::write(&self.inventory_path, content)
            .with_context(|| format!("Failed to write inventory file: {:?}", self.inventory_path))?;
        Ok(())
    }

    /// Add a new node to inventory
    pub fn add_node(&mut self, node: NodeInfo) -> Result<()> {
        if self.data.nodes.contains_key(&node.id) {
            anyhow::bail!("Node with ID {} already exists", node.id);
        }

        // Check for duplicate address
        if self.find_by_address(&node.address).is_some() {
            anyhow::bail!("Node with address {} already exists", node.address);
        }

        self.data.nodes.insert(node.id.clone(), node);
        self.save()?;
        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Option<&NodeInfo> {
        self.data.nodes.get(id)
    }

    /// Get a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut NodeInfo> {
        self.data.nodes.get_mut(id)
    }

    /// Find a node by name or ID
    pub fn find_node(&self, name_or_id: &str) -> Option<&NodeInfo> {
        // First try exact ID match
        if let Some(node) = self.data.nodes.get(name_or_id) {
            return Some(node);
        }

        // Then try name match
        self.data.nodes.values().find(|n| n.name == name_or_id)
    }

    /// Find a node by name or ID (mutable)
    pub fn find_node_mut(&mut self, name_or_id: &str) -> Option<&mut NodeInfo> {
        // First try exact ID match
        if self.data.nodes.contains_key(name_or_id) {
            return self.data.nodes.get_mut(name_or_id);
        }

        // Then try name match
        self.data.nodes.values_mut().find(|n| n.name == name_or_id)
    }

    /// Find a node by address
    pub fn find_by_address(&self, address: &str) -> Option<&NodeInfo> {
        self.data.nodes.values().find(|n| n.address == address)
    }

    /// Update a node
    pub fn update_node(&mut self, node: NodeInfo) -> Result<()> {
        if !self.data.nodes.contains_key(&node.id) {
            anyhow::bail!("Node with ID {} not found", node.id);
        }
        self.data.nodes.insert(node.id.clone(), node);
        self.save()?;
        Ok(())
    }

    /// Remove a node by ID
    pub fn remove_node(&mut self, id: &str) -> Result<NodeInfo> {
        let node = self
            .data
            .nodes
            .remove(id)
            .with_context(|| format!("Node with ID {} not found", id))?;
        self.save()?;
        Ok(node)
    }

    /// Remove a node by name or ID
    pub fn remove_by_name_or_id(&mut self, name_or_id: &str) -> Result<NodeInfo> {
        let id = self
            .find_node(name_or_id)
            .map(|n| n.id.clone())
            .with_context(|| format!("Node '{}' not found", name_or_id))?;
        self.remove_node(&id)
    }

    /// Get all nodes
    pub fn list_nodes(&self) -> Vec<&NodeInfo> {
        let mut nodes: Vec<_> = self.data.nodes.values().collect();
        nodes.sort_by(|a, b| a.name.cmp(&b.name));
        nodes
    }

    /// Get nodes filtered by status
    pub fn list_by_status(&self, status: NodeStatus) -> Vec<&NodeInfo> {
        self.data
            .nodes
            .values()
            .filter(|n| n.status == status)
            .collect()
    }

    /// Get nodes filtered by tag
    pub fn list_by_tag(&self, tag: &str) -> Vec<&NodeInfo> {
        self.data
            .nodes
            .values()
            .filter(|n| n.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Get inventory summary
    pub fn summary(&self) -> InventorySummary {
        let nodes: Vec<_> = self.data.nodes.values().cloned().collect();
        InventorySummary::from_nodes(&nodes)
    }

    /// Get node count
    pub fn count(&self) -> usize {
        self.data.nodes.len()
    }

    /// Check if inventory is empty
    pub fn is_empty(&self) -> bool {
        self.data.nodes.is_empty()
    }

    /// Generate a unique node name based on address
    pub fn generate_node_name(&self, address: &str) -> String {
        let base_name = address
            .replace('.', "-")
            .replace(':', "-");

        let mut name = format!("node-{}", base_name);
        let mut counter = 1;

        while self.data.nodes.values().any(|n| n.name == name) {
            name = format!("node-{}-{}", base_name, counter);
            counter += 1;
        }

        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hpc_inventory::{CredentialRef, NodeMode};
    use tempfile::TempDir;

    fn create_test_store(temp_dir: &TempDir) -> InventoryStore {
        let inventory_path = temp_dir.path().join("nodes.json");
        InventoryStore {
            inventory_path,
            data: InventoryData::new(),
        }
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

    #[test]
    fn test_add_and_get_node() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("test-node", "192.168.1.100");
        let node_id = node.id.clone();

        store.add_node(node).unwrap();

        let retrieved = store.get_node(&node_id).unwrap();
        assert_eq!(retrieved.name, "test-node");
        assert_eq!(retrieved.address, "192.168.1.100");
    }

    #[test]
    fn test_find_node_by_name() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("my-gpu-node", "10.0.0.50");
        store.add_node(node).unwrap();

        let found = store.find_node("my-gpu-node").unwrap();
        assert_eq!(found.address, "10.0.0.50");
    }

    #[test]
    fn test_remove_node() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node = create_test_node("to-remove", "1.2.3.4");
        let node_id = node.id.clone();
        store.add_node(node).unwrap();

        assert_eq!(store.count(), 1);

        let removed = store.remove_node(&node_id).unwrap();
        assert_eq!(removed.name, "to-remove");
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_duplicate_address_rejected() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let node1 = create_test_node("node1", "192.168.1.1");
        let node2 = create_test_node("node2", "192.168.1.1");

        store.add_node(node1).unwrap();
        let result = store.add_node(node2);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
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
        store.add_node(node).unwrap();

        // Next name should have a counter
        let name2 = store.generate_node_name("192.168.1.100");
        assert_eq!(name2, "node-192-168-1-100-1");
    }

    #[test]
    fn test_list_by_status() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = create_test_store(&temp_dir);

        let mut node1 = create_test_node("connected", "1.1.1.1");
        node1.status = NodeStatus::Connected;

        let mut node2 = create_test_node("pending", "2.2.2.2");
        node2.status = NodeStatus::Pending;

        let mut node3 = create_test_node("also-connected", "3.3.3.3");
        node3.status = NodeStatus::Connected;

        store.add_node(node1).unwrap();
        store.add_node(node2).unwrap();
        store.add_node(node3).unwrap();

        let connected = store.list_by_status(NodeStatus::Connected);
        assert_eq!(connected.len(), 2);

        let pending = store.list_by_status(NodeStatus::Pending);
        assert_eq!(pending.len(), 1);
    }
}
