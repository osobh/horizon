//! Causal graph structures and operations

use chrono::Duration as ChronoDuration;
use std::collections::{HashMap, HashSet};

use crate::{KnowledgeGraph, KnowledgeGraphResult};

/// Causal graph structure optimized for inference
pub struct CausalGraph {
    /// Adjacency matrix for causal relationships
    pub adjacency_matrix: Vec<Vec<f64>>,
    /// Node ID to index mapping
    pub node_to_index: HashMap<String, usize>,
    /// Index to node ID mapping
    pub index_to_node: Vec<String>,
    /// Temporal ordering constraints
    pub temporal_constraints: Vec<TemporalConstraint>,
    /// Confounding variables
    pub confounders: HashSet<String>,
}

/// Temporal constraint between nodes
#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    pub before_node: String,
    pub after_node: String,
    pub min_delay: ChronoDuration,
    pub max_delay: ChronoDuration,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            adjacency_matrix: Vec::new(),
            node_to_index: HashMap::new(),
            index_to_node: Vec::new(),
            temporal_constraints: Vec::new(),
            confounders: HashSet::new(),
        }
    }

    pub fn rebuild_from_knowledge_graph(&mut self, kg: &KnowledgeGraph) -> KnowledgeGraphResult<()> {
        // Extract nodes and build index mapping
        self.node_to_index.clear();
        self.index_to_node.clear();
        
        for (idx, node) in kg.get_all_nodes()?.iter().enumerate() {
            self.node_to_index.insert(node.id.clone(), idx);
            self.index_to_node.push(node.id.clone());
        }

        let num_nodes = self.index_to_node.len();
        self.adjacency_matrix = vec![vec![0.0; num_nodes]; num_nodes];

        // Initialize adjacency matrix based on knowledge graph edges
        for edge in kg.get_all_edges()? {
            if let (Some(&from_idx), Some(&to_idx)) = (
                self.node_to_index.get(&edge.source_id),
                self.node_to_index.get(&edge.target_id),
            ) {
                self.adjacency_matrix[from_idx][to_idx] = 1.0;
            }
        }

        Ok(())
    }

    pub fn add_temporal_constraint(&mut self, constraint: TemporalConstraint) {
        self.temporal_constraints.push(constraint);
    }

    pub fn add_confounder(&mut self, node_id: String) {
        self.confounders.insert(node_id);
    }

    pub fn get_neighbors(&self, node_id: &str) -> Vec<String> {
        if let Some(&node_idx) = self.node_to_index.get(node_id) {
            self.adjacency_matrix[node_idx]
                .iter()
                .enumerate()
                .filter_map(|(idx, &weight)| {
                    if weight > 0.0 {
                        Some(self.index_to_node[idx].clone())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}
