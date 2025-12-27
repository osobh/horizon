//! Compressed Sparse Row format for efficient GPU graph traversal

use crate::error::StorageError;
use std::time::{SystemTime, UNIX_EPOCH};

/// Compressed Sparse Row format optimized for GPU traversal
#[derive(Debug, Clone)]
pub struct GraphCSR {
    /// Offsets into edge arrays for each node
    pub row_offsets: Vec<u64>,
    /// Target node IDs
    pub col_indices: Vec<u64>,
    /// Edge type identifiers
    pub edge_types: Vec<u32>,
    /// Edge weights or strengths
    pub weights: Vec<f32>,
    /// Creation timestamps
    pub timestamps: Vec<u64>,
}

impl GraphCSR {
    /// Create a new CSR graph structure
    pub fn new(num_nodes: usize) -> Self {
        Self {
            row_offsets: vec![0; num_nodes + 1],
            col_indices: Vec::new(),
            edge_types: Vec::new(),
            weights: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Get the current timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Add an edge to the graph
    pub fn add_edge(
        &mut self,
        from: usize,
        to: u64,
        edge_type: u32,
        weight: f32,
    ) -> Result<(), StorageError> {
        if from >= self.row_offsets.len() - 1 {
            return Err(StorageError::InvalidNode {
                id: from as u64,
                reason: "Source node ID out of bounds".to_string(),
            });
        }

        let insert_pos = self.row_offsets[from + 1] as usize;

        // Insert edge data
        self.col_indices.insert(insert_pos, to);
        self.edge_types.insert(insert_pos, edge_type);
        self.weights.insert(insert_pos, weight);
        self.timestamps
            .insert(insert_pos, Self::current_timestamp());

        // Update offsets for subsequent nodes
        for i in (from + 1)..self.row_offsets.len() {
            self.row_offsets[i] += 1;
        }

        Ok(())
    }

    /// Get all edges from a specific node
    pub fn get_edges(&self, from: usize) -> Result<EdgeIterator<'_>, StorageError> {
        if from >= self.row_offsets.len() - 1 {
            return Err(StorageError::InvalidNode {
                id: from as u64,
                reason: "Node ID out of bounds".to_string(),
            });
        }

        let start = self.row_offsets[from] as usize;
        let end = self.row_offsets[from + 1] as usize;

        Ok(EdgeIterator {
            csr: self,
            current: start,
            end,
        })
    }

    /// Get the degree (number of edges) for a node
    pub fn degree(&self, node: usize) -> Result<usize, StorageError> {
        if node >= self.row_offsets.len() - 1 {
            return Err(StorageError::InvalidNode {
                id: node as u64,
                reason: "Node ID out of bounds".to_string(),
            });
        }

        let start = self.row_offsets[node];
        let end = self.row_offsets[node + 1];

        Ok((end - start) as usize)
    }

    /// Get total number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.col_indices.len()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    /// Check if an edge exists
    pub fn has_edge(&self, from: usize, to: u64) -> Result<bool, StorageError> {
        let edges = self.get_edges(from)?;

        for edge in edges {
            if edge.target == to {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Iterator over edges from a specific node
pub struct EdgeIterator<'a> {
    csr: &'a GraphCSR,
    current: usize,
    end: usize,
}

/// Edge information returned by iterator
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub target: u64,
    pub edge_type: u32,
    pub weight: f32,
    pub timestamp: u64,
}

impl<'a> Iterator for EdgeIterator<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }

        let edge = Edge {
            target: self.csr.col_indices[self.current],
            edge_type: self.csr.edge_types[self.current],
            weight: self.csr.weights[self.current],
            timestamp: self.csr.timestamps[self.current],
        };

        self.current += 1;
        Some(edge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        let csr = GraphCSR::new(10);

        assert_eq!(csr.row_offsets.len(), 11);
        assert_eq!(csr.col_indices.len(), 0);
        assert_eq!(csr.edge_types.len(), 0);
        assert_eq!(csr.weights.len(), 0);
        assert_eq!(csr.timestamps.len(), 0);
        assert_eq!(csr.node_count(), 10);
        assert_eq!(csr.edge_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut csr = GraphCSR::new(5);

        csr.add_edge(0, 1, 10, 0.5)?;
        csr.add_edge(0, 2, 11, 0.7)?;
        csr.add_edge(1, 3, 12, 0.3)?;

        assert_eq!(csr.edge_count(), 3);
        assert_eq!(csr.degree(0)?, 2);
        assert_eq!(csr.degree(1)?, 1);
        assert_eq!(csr.degree(2)?, 0);
    }

    #[test]
    fn test_add_edge_out_of_bounds() {
        let mut csr = GraphCSR::new(5);

        let result = csr.add_edge(10, 1, 10, 0.5);
        assert!(result.is_err());

        match result {
            Err(StorageError::InvalidNode { id, .. }) => assert_eq!(id, 10),
            _ => panic!("Expected InvalidNode error"),
        }
    }

    #[test]
    fn test_get_edges() {
        let mut csr = GraphCSR::new(5);

        csr.add_edge(0, 1, 10, 0.5)?;
        csr.add_edge(0, 2, 11, 0.7)?;
        csr.add_edge(0, 4, 12, 0.9)?;

        let edges: Vec<Edge> = csr.get_edges(0).unwrap().collect();

        assert_eq!(edges.len(), 3);
        assert_eq!(edges[0].target, 1);
        assert_eq!(edges[0].edge_type, 10);
        assert_eq!(edges[0].weight, 0.5);

        assert_eq!(edges[1].target, 2);
        assert_eq!(edges[1].edge_type, 11);
        assert_eq!(edges[1].weight, 0.7);

        assert_eq!(edges[2].target, 4);
        assert_eq!(edges[2].edge_type, 12);
        assert_eq!(edges[2].weight, 0.9);
    }

    #[test]
    fn test_has_edge() {
        let mut csr = GraphCSR::new(5);

        csr.add_edge(0, 1, 10, 0.5)?;
        csr.add_edge(0, 2, 11, 0.7)?;

        assert!(csr.has_edge(0, 1).unwrap());
        assert!(csr.has_edge(0, 2).unwrap());
        assert!(!csr.has_edge(0, 3).unwrap());
        assert!(!csr.has_edge(1, 0).unwrap());
    }

    #[test]
    fn test_edge_timestamps() {
        let mut csr = GraphCSR::new(3);

        let before = GraphCSR::current_timestamp();
        csr.add_edge(0, 1, 10, 0.5)?;
        let after = GraphCSR::current_timestamp();

        let edges: Vec<Edge> = csr.get_edges(0).unwrap().collect();
        assert!(!edges.is_empty());
        assert!(edges[0].timestamp >= before);
        assert!(edges[0].timestamp <= after);
    }

    #[test]
    fn test_multiple_sources() {
        let mut csr = GraphCSR::new(4);

        // Add edges from different source nodes
        csr.add_edge(0, 1, 10, 0.5)?;
        csr.add_edge(0, 2, 11, 0.6)?;
        csr.add_edge(1, 2, 12, 0.7)?;
        csr.add_edge(1, 3, 13, 0.8)?;
        csr.add_edge(2, 3, 14, 0.9)?;

        // Check degree of each node
        assert_eq!(csr.degree(0)?, 2);
        assert_eq!(csr.degree(1)?, 2);
        assert_eq!(csr.degree(2)?, 1);
        assert_eq!(csr.degree(3)?, 0);

        // Check specific edges
        assert!(csr.has_edge(0, 1).unwrap());
        assert!(csr.has_edge(0, 2).unwrap());
        assert!(csr.has_edge(1, 2).unwrap());
        assert!(csr.has_edge(1, 3).unwrap());
        assert!(csr.has_edge(2, 3).unwrap());

        // Check non-existent edges
        assert!(!csr.has_edge(2, 1).unwrap());
        assert!(!csr.has_edge(3, 0).unwrap());
    }

    #[test]
    fn test_empty_node_edges() {
        let csr = GraphCSR::new(5);

        let edges: Vec<Edge> = csr.get_edges(0).unwrap().collect();
        assert_eq!(edges.len(), 0);

        assert_eq!(csr.degree(0)?, 0);
        assert_eq!(csr.degree(4)?, 0);
    }

    #[test]
    fn test_get_edges_out_of_bounds() {
        let csr = GraphCSR::new(5);

        // Test accessing edges for a node beyond the graph size
        let result = csr.get_edges(5);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 5));

        let result = csr.get_edges(10);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 10));
    }

    #[test]
    fn test_degree_out_of_bounds() {
        let csr = GraphCSR::new(3);

        // Test getting degree for a node beyond the graph size
        let result = csr.degree(3);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 3));

        let result = csr.degree(100);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 100));
    }

    #[test]
    fn test_has_edge_out_of_bounds() {
        let csr = GraphCSR::new(3);

        // Test has_edge with out of bounds source node
        let result = csr.has_edge(3, 0);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 3));

        let result = csr.has_edge(10, 0);
        assert!(matches!(result, Err(StorageError::InvalidNode { id, .. }) if id == 10));

        // Test has_edge with valid source but out of bounds target (should return Ok(false))
        let result = csr.has_edge(0, 100);
        assert_eq!(result?, false);
    }
}
