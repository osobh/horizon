use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
// These will be used for future validation enhancements
#[allow(unused_imports)]
use validator::{Validate, ValidationError};

use crate::error::{Result, VisualEditorError};

/// Result of topology validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationIssue>,
    pub warnings: Vec<ValidationIssue>,
}

/// A validation issue (error or warning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub message: String,
    pub element_id: Option<String>,
}

/// Type of validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    MissingNode,
    DuplicateId,
    InvalidEdge,
    DisconnectedNode,
    CyclicDependency,
    InvalidProperty,
    PerformanceWarning,
}

/// Topology validator
pub struct TopologyValidator {
    /// Maximum number of nodes allowed
    max_nodes: usize,
    /// Maximum number of edges allowed
    max_edges: usize,
    /// Minimum bandwidth for edges
    min_bandwidth: i32,
}

impl TopologyValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self {
            max_nodes: 1000,
            max_edges: 5000,
            min_bandwidth: 100, // 100 Mbps minimum
        }
    }

    /// Create a validator with custom settings
    pub fn with_limits(max_nodes: usize, max_edges: usize, min_bandwidth: i32) -> Self {
        Self {
            max_nodes,
            max_edges,
            min_bandwidth,
        }
    }

    /// Validate a topology
    pub fn validate(&self, topology: &Value) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Extract topology data
        let nodes = topology
            .get("nodes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| VisualEditorError::InvalidTopology("Missing nodes array".to_string()))?;

        let edges = topology
            .get("edges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| VisualEditorError::InvalidTopology("Missing edges array".to_string()))?;

        // Check node count
        if nodes.len() > self.max_nodes {
            warnings.push(ValidationIssue {
                issue_type: IssueType::PerformanceWarning,
                message: format!(
                    "Topology has {} nodes, which exceeds recommended limit of {}",
                    nodes.len(),
                    self.max_nodes
                ),
                element_id: None,
            });
        }

        // Check edge count
        if edges.len() > self.max_edges {
            warnings.push(ValidationIssue {
                issue_type: IssueType::PerformanceWarning,
                message: format!(
                    "Topology has {} edges, which exceeds recommended limit of {}",
                    edges.len(),
                    self.max_edges
                ),
                element_id: None,
            });
        }

        // Build node ID set
        let mut node_ids = HashSet::new();
        let mut duplicate_ids = Vec::new();

        for node in nodes {
            if let Some(id) = node.get("id").and_then(|v| v.as_str()) {
                if !node_ids.insert(id.to_string()) {
                    duplicate_ids.push(id.to_string());
                }

                // Validate node properties
                if let Err(e) = self.validate_node(node) {
                    errors.push(ValidationIssue {
                        issue_type: IssueType::InvalidProperty,
                        message: e.to_string(),
                        element_id: Some(id.to_string()),
                    });
                }
            } else {
                errors.push(ValidationIssue {
                    issue_type: IssueType::InvalidProperty,
                    message: "Node missing ID".to_string(),
                    element_id: None,
                });
            }
        }

        // Report duplicate IDs
        for id in duplicate_ids {
            errors.push(ValidationIssue {
                issue_type: IssueType::DuplicateId,
                message: format!("Duplicate node ID: {}", id),
                element_id: Some(id),
            });
        }

        // Validate edges
        let mut node_connections: HashMap<String, usize> = HashMap::new();

        for edge in edges {
            if let Err(e) = self.validate_edge(edge, &node_ids) {
                let edge_id = edge
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                errors.push(ValidationIssue {
                    issue_type: IssueType::InvalidEdge,
                    message: e.to_string(),
                    element_id: edge_id,
                });
            } else {
                // Track connections
                if let (Some(source), Some(target)) = (
                    edge.get("source").and_then(|v| v.as_str()),
                    edge.get("target").and_then(|v| v.as_str()),
                ) {
                    *node_connections.entry(source.to_string()).or_insert(0) += 1;
                    *node_connections.entry(target.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Check for disconnected nodes
        for id in &node_ids {
            if !node_connections.contains_key(id) {
                warnings.push(ValidationIssue {
                    issue_type: IssueType::DisconnectedNode,
                    message: format!("Node {} has no connections", id),
                    element_id: Some(id.clone()),
                });
            }
        }

        // Check for cycles (simplified check)
        if edges.len() >= nodes.len() && !nodes.is_empty() {
            warnings.push(ValidationIssue {
                issue_type: IssueType::CyclicDependency,
                message: "Topology may contain cycles".to_string(),
                element_id: None,
            });
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Validate a single node
    fn validate_node(&self, node: &Value) -> Result<()> {
        // Check required fields
        let _ = node
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VisualEditorError::ValidationError("Node missing type".to_string()))?;

        let _ = node
            .get("position")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                VisualEditorError::ValidationError("Node missing position".to_string())
            })?;

        // Validate position
        if let Some(pos) = node.get("position") {
            let _ = pos.get("x").and_then(|v| v.as_f64()).ok_or_else(|| {
                VisualEditorError::ValidationError("Invalid x position".to_string())
            })?;

            let _ = pos.get("y").and_then(|v| v.as_f64()).ok_or_else(|| {
                VisualEditorError::ValidationError("Invalid y position".to_string())
            })?;
        }

        Ok(())
    }

    /// Validate a single edge
    fn validate_edge(&self, edge: &Value, valid_node_ids: &HashSet<String>) -> Result<()> {
        // Check source and target
        let source = edge
            .get("source")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VisualEditorError::ValidationError("Edge missing source".to_string()))?;

        let target = edge
            .get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VisualEditorError::ValidationError("Edge missing target".to_string()))?;

        // Validate nodes exist
        if !valid_node_ids.contains(source) {
            return Err(VisualEditorError::ValidationError(format!(
                "Edge references non-existent source node: {}",
                source
            )));
        }

        if !valid_node_ids.contains(target) {
            return Err(VisualEditorError::ValidationError(format!(
                "Edge references non-existent target node: {}",
                target
            )));
        }

        // Validate bandwidth
        if let Some(bandwidth) = edge.get("bandwidth").and_then(|v| v.as_i64()) {
            if bandwidth < self.min_bandwidth as i64 {
                return Err(VisualEditorError::ValidationError(format!(
                    "Edge bandwidth {} is below minimum {}",
                    bandwidth, self.min_bandwidth
                )));
            }
        } else {
            return Err(VisualEditorError::ValidationError(
                "Edge missing bandwidth".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for TopologyValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_topology() {
        let validator = TopologyValidator::new();
        let topology = json!({
            "nodes": [
                {"id": "n1", "type": "gpu", "position": {"x": 0, "y": 0}},
                {"id": "n2", "type": "cpu", "position": {"x": 100, "y": 0}}
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2", "bandwidth": 1000}
            ]
        });

        let result = validator.validate(&topology).unwrap();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_invalid_edge_reference() {
        let validator = TopologyValidator::new();
        let topology = json!({
            "nodes": [
                {"id": "n1", "type": "gpu", "position": {"x": 0, "y": 0}}
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n99", "bandwidth": 1000}
            ]
        });

        let result = validator.validate(&topology).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
}
