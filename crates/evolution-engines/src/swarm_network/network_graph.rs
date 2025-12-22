//! Network graph representation for distributed swarm topology

use super::types::{EdgeWeight, NetworkPartition};
use crate::error::EvolutionEngineResult;
use crate::swarm_distributed::SwarmNode;
use std::collections::HashMap;

/// Network graph representing inter-node connectivity
pub struct NetworkGraph {
    /// Nodes in the network
    pub(crate) nodes: HashMap<String, SwarmNode>,
    /// Adjacency list: node_id -> [connected_node_ids]
    pub(crate) adjacency: HashMap<String, Vec<String>>,
    /// Edge weights (latency, bandwidth, etc.)
    pub(crate) edge_weights: HashMap<(String, String), EdgeWeight>,
}

impl NetworkGraph {
    /// Create new network graph
    pub async fn new() -> EvolutionEngineResult<Self> {
        Ok(Self {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            edge_weights: HashMap::new(),
        })
    }

    /// Add node to graph
    pub async fn add_node(&mut self, node: SwarmNode) -> EvolutionEngineResult<()> {
        self.nodes.insert(node.node_id.clone(), node.clone());
        self.adjacency.insert(node.node_id, Vec::new());
        Ok(())
    }

    /// Remove node from graph
    pub async fn remove_node(&mut self, node_id: &str) -> EvolutionEngineResult<()> {
        self.nodes.remove(node_id);
        self.adjacency.remove(node_id);

        // Remove from all adjacency lists
        for (_, neighbors) in &mut self.adjacency {
            neighbors.retain(|id| id != node_id);
        }

        // Remove all edges involving this node
        self.edge_weights
            .retain(|(from, to), _| from != node_id && to != node_id);

        Ok(())
    }

    /// Update edge weight between nodes
    pub async fn update_edge(
        &mut self,
        from_node: &str,
        to_node: &str,
        weight: EdgeWeight,
    ) -> EvolutionEngineResult<()> {
        // Add to adjacency if not exists
        if let Some(neighbors) = self.adjacency.get_mut(from_node) {
            if !neighbors.contains(&to_node.to_string()) {
                neighbors.push(to_node.to_string());
            }
        }

        // Store edge weight
        self.edge_weights
            .insert((from_node.to_string(), to_node.to_string()), weight);

        Ok(())
    }

    /// Get network partitions using connected components
    pub async fn get_partitions(&self) -> EvolutionEngineResult<Vec<NetworkPartition>> {
        let mut partitions = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id) {
                let mut component = Vec::new();
                self.dfs(node_id, &mut visited, &mut component);

                partitions.push(NetworkPartition {
                    id: format!("partition_{}", partitions.len()),
                    nodes: component,
                    particles: Vec::new(),   // Stub implementation
                    load: 0.0,               // Stub implementation
                    communication_cost: 0.0, // Stub implementation
                });
            }
        }

        Ok(partitions)
    }

    /// Get all nodes in the graph
    pub fn get_nodes(&self) -> &HashMap<String, SwarmNode> {
        &self.nodes
    }

    /// Get adjacency list
    pub fn get_adjacency(&self) -> &HashMap<String, Vec<String>> {
        &self.adjacency
    }

    /// Get edge weight between two nodes
    pub fn get_edge_weight(&self, from_node: &str, to_node: &str) -> Option<&EdgeWeight> {
        self.edge_weights
            .get(&(from_node.to_string(), to_node.to_string()))
    }

    /// Check if nodes are connected
    pub fn are_connected(&self, node1: &str, node2: &str) -> bool {
        if let Some(neighbors) = self.adjacency.get(node1) {
            neighbors.contains(&node2.to_string())
        } else {
            false
        }
    }

    /// Get shortest path between two nodes (simple BFS)
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        let mut parent = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        parent.insert(neighbor.clone(), current.clone());
                        queue.push_back(neighbor.clone());

                        if neighbor == to {
                            // Reconstruct path
                            let mut path = vec![neighbor.clone()];
                            let mut curr = neighbor.clone();
                            while let Some(prev) = parent.get(&curr) {
                                path.push(prev.clone());
                                curr = prev.clone();
                            }
                            path.reverse();
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Depth-first search for connected components
    fn dfs(
        &self,
        node_id: &str,
        visited: &mut std::collections::HashSet<String>,
        component: &mut Vec<String>,
    ) {
        visited.insert(node_id.to_string());
        component.push(node_id.to_string());

        if let Some(neighbors) = self.adjacency.get(node_id) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.dfs(neighbor, visited, component);
                }
            }
        }
    }
}
