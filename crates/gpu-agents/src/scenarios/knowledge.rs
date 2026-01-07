//! Knowledge graph agent scenario implementations

use super::config::{KnowledgeConfig, MemoryPattern};
use crate::GpuSwarm;
use std::collections::HashMap;

/// Knowledge graph node
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: u64,
    pub content: String,
    pub timestamp: f32,
    pub connections: Vec<u64>,
    pub weight: f32,
}

impl KnowledgeNode {
    /// Create a new knowledge node
    pub fn new(id: u64, content: impl Into<String>) -> Self {
        Self {
            id,
            content: content.into(),
            timestamp: 0.0,
            connections: Vec::new(),
            weight: 1.0,
        }
    }

    /// Add a connection to another node
    pub fn connect(&mut self, other_id: u64) {
        if !self.connections.contains(&other_id) {
            self.connections.push(other_id);
        }
    }
}

/// Knowledge graph for agent memory
#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    nodes: HashMap<u64, KnowledgeNode>,
    next_id: u64,
    max_nodes: usize,
    memory_pattern: MemoryPattern,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new(max_nodes: usize, memory_pattern: MemoryPattern) -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            max_nodes,
            memory_pattern,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, content: impl Into<String>) -> u64 {
        // If at capacity, remove nodes based on memory pattern
        if self.nodes.len() >= self.max_nodes {
            self.evict_node();
        }

        let id = self.next_id;
        self.next_id += 1;

        let node = KnowledgeNode::new(id, content);
        self.nodes.insert(id, node);
        id
    }

    /// Connect two nodes
    pub fn connect(&mut self, from_id: u64, to_id: u64) {
        if let Some(node) = self.nodes.get_mut(&from_id) {
            node.connect(to_id);
        }
    }

    /// Get a node by ID
    pub fn get(&self, id: u64) -> Option<&KnowledgeNode> {
        self.nodes.get(&id)
    }

    /// Get node count
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Evict a node based on memory pattern
    fn evict_node(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        let id_to_remove = match self.memory_pattern {
            MemoryPattern::Random => {
                // Remove a random node
                *self.nodes.keys().next().unwrap()
            }
            MemoryPattern::Temporal => {
                // Remove oldest node (lowest timestamp)
                *self
                    .nodes
                    .iter()
                    .min_by(|a, b| a.1.timestamp.partial_cmp(&b.1.timestamp).unwrap())
                    .map(|(id, _)| id)
                    .unwrap()
            }
            MemoryPattern::Associative => {
                // Remove least connected node
                *self
                    .nodes
                    .iter()
                    .min_by_key(|(_, node)| node.connections.len())
                    .map(|(id, _)| id)
                    .unwrap()
            }
            MemoryPattern::Hierarchical => {
                // Remove lowest weight node
                *self
                    .nodes
                    .iter()
                    .min_by(|a, b| a.1.weight.partial_cmp(&b.1.weight).unwrap())
                    .map(|(id, _)| id)
                    .unwrap()
            }
        };

        self.nodes.remove(&id_to_remove);
    }

    /// Query nodes by content (simple substring match)
    pub fn query(&self, query: &str) -> Vec<&KnowledgeNode> {
        self.nodes
            .values()
            .filter(|node| node.content.contains(query))
            .collect()
    }
}

/// Knowledge agent scenario executor
pub struct KnowledgeAgentScenario {
    config: KnowledgeConfig,
    graphs: Vec<KnowledgeGraph>,
    initialized: bool,
}

impl KnowledgeAgentScenario {
    /// Create a new knowledge agent scenario
    pub fn new(config: KnowledgeConfig) -> Self {
        Self {
            config,
            graphs: Vec::new(),
            initialized: false,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &KnowledgeConfig {
        &self.config
    }

    /// Initialize knowledge graphs for agents
    pub fn initialize_agents(
        &mut self,
        swarm: &mut GpuSwarm,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let agent_count = swarm.agent_count();

        log::info!(
            "Initializing {} knowledge graphs with {} initial nodes each",
            agent_count,
            self.config.initial_nodes
        );

        // Create a knowledge graph for each agent
        self.graphs.clear();
        for i in 0..agent_count {
            let mut graph = KnowledgeGraph::new(self.config.max_nodes, self.config.memory_pattern);

            // Add initial nodes
            for j in 0..self.config.initial_nodes {
                let content = format!("Agent {} knowledge node {}", i, j);
                let id = graph.add_node(content);

                // Create some connections for associative patterns
                if j > 0 {
                    graph.connect(id, id - 1);
                }
            }

            self.graphs.push(graph);
        }

        self.initialized = true;

        log::debug!(
            "Knowledge graphs initialized with memory pattern: {:?}",
            self.config.memory_pattern
        );

        Ok(())
    }

    /// Update agents with knowledge operations
    pub fn update(&mut self, swarm: &mut GpuSwarm) -> Result<(), Box<dyn std::error::Error>> {
        if !self.initialized {
            return Err("Knowledge scenario not initialized".into());
        }

        // Perform knowledge sharing between agents
        if self.config.sharing_ratio > 0.0 {
            self.share_knowledge();
        }

        // GPU kernels handle agent movement
        swarm.step()?;
        Ok(())
    }

    /// Share knowledge between agents based on sharing ratio
    fn share_knowledge(&mut self) {
        if self.graphs.len() < 2 {
            return;
        }

        // Simple sharing: each agent shares with neighbors
        let num_shares = (self.graphs.len() as f32 * self.config.sharing_ratio) as usize;

        for i in 0..num_shares.min(self.graphs.len() - 1) {
            let src_idx = i;
            let dst_idx = (i + 1) % self.graphs.len();

            // Get a node from source to share
            if let Some(node) = self.graphs[src_idx].nodes.values().next() {
                let content = node.content.clone();
                self.graphs[dst_idx].add_node(format!("Shared: {}", content));
            }
        }
    }

    /// Get a specific agent's knowledge graph
    pub fn get_graph(&self, agent_idx: usize) -> Option<&KnowledgeGraph> {
        self.graphs.get(agent_idx)
    }

    /// Get total knowledge nodes across all agents
    pub fn total_nodes(&self) -> usize {
        self.graphs.iter().map(|g| g.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_node() {
        let mut node = KnowledgeNode::new(1, "test content");
        assert_eq!(node.id, 1);
        assert_eq!(node.content, "test content");
        assert!(node.connections.is_empty());

        node.connect(2);
        node.connect(3);
        assert_eq!(node.connections.len(), 2);

        // Connecting again should not duplicate
        node.connect(2);
        assert_eq!(node.connections.len(), 2);
    }

    #[test]
    fn test_knowledge_graph() {
        let mut graph = KnowledgeGraph::new(10, MemoryPattern::Random);

        let id1 = graph.add_node("first");
        let id2 = graph.add_node("second");
        graph.connect(id1, id2);

        assert_eq!(graph.len(), 2);
        assert!(graph.get(id1).unwrap().connections.contains(&id2));
    }
}
