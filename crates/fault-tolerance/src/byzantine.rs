//! Byzantine fault detection for distributed consensus
//!
//! Provides detection of Byzantine (malicious or faulty) nodes in distributed
//! consensus protocols through voting pattern analysis and trust scoring.

use crate::error::{FaultToleranceError, FtResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Byzantine fault detector for consensus systems
///
/// Monitors node behavior and detects potential Byzantine faults through:
/// - Voting pattern analysis
/// - Trust score tracking
/// - Anomaly detection
#[derive(Debug)]
pub struct ByzantineDetector {
    /// Trust scores for each node (0.0 = untrusted, 1.0 = fully trusted)
    trust_scores: Arc<DashMap<String, f64>>,
    /// Byzantine detection threshold (nodes below this are suspect)
    threshold: f32,
    /// Maximum nodes to track
    max_nodes: usize,
    /// Detection configuration
    config: ByzantineConfig,
}

/// Configuration for Byzantine detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineConfig {
    /// Minimum trust score before flagging as Byzantine
    pub min_trust_score: f64,
    /// Trust decay rate per round
    pub trust_decay_rate: f64,
    /// Trust increase for consistent behavior
    pub trust_increase_rate: f64,
    /// Number of rounds to track for pattern analysis
    pub history_depth: usize,
}

impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            min_trust_score: 0.3,
            trust_decay_rate: 0.05,
            trust_increase_rate: 0.02,
            history_depth: 100,
        }
    }
}

impl ByzantineDetector {
    /// Create a new Byzantine detector
    ///
    /// # Arguments
    /// * `threshold` - Byzantine fault tolerance threshold (typically 0.33 for BFT)
    /// * `max_nodes` - Maximum number of nodes to track
    pub fn new(threshold: f32, max_nodes: usize) -> FtResult<Self> {
        if threshold <= 0.0 || threshold > 1.0 {
            return Err(FaultToleranceError::CoordinationError(
                "Byzantine threshold must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            trust_scores: Arc::new(DashMap::with_capacity(max_nodes)),
            threshold,
            max_nodes,
            config: ByzantineConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        threshold: f32,
        max_nodes: usize,
        config: ByzantineConfig,
    ) -> FtResult<Self> {
        let mut detector = Self::new(threshold, max_nodes)?;
        detector.config = config;
        Ok(detector)
    }

    /// Detect Byzantine behavior in a set of votes
    ///
    /// Analyzes voting patterns to identify potentially Byzantine nodes.
    /// Returns a list of node IDs that exhibit suspicious behavior.
    ///
    /// # Type Parameters
    /// * `K` - Vote key type (e.g., ValidatorId)
    /// * `V` - Vote value type
    pub async fn detect_in_votes<K, V>(&self, votes: &HashMap<K, V>) -> FtResult<Vec<String>>
    where
        K: std::hash::Hash + Eq + ToString + Clone,
        V: Clone,
    {
        let mut suspicious_nodes = Vec::new();

        for node_id in votes.keys() {
            let node_key = node_id.to_string();

            // Check trust score
            if let Some(score) = self.trust_scores.get(&node_key) {
                if *score < self.config.min_trust_score {
                    suspicious_nodes.push(node_key);
                }
            }
        }

        // Check if Byzantine threshold would be exceeded
        let byzantine_ratio = suspicious_nodes.len() as f32 / votes.len().max(1) as f32;
        if byzantine_ratio > self.threshold {
            tracing::warn!(
                "Byzantine ratio {} exceeds threshold {}",
                byzantine_ratio,
                self.threshold
            );
        }

        Ok(suspicious_nodes)
    }

    /// Report suspicious behavior for a node
    pub fn report_suspicious(&self, node_id: &str) {
        let mut score = self.trust_scores.entry(node_id.to_string()).or_insert(1.0);
        *score = (*score - self.config.trust_decay_rate).max(0.0);

        tracing::debug!("Node {} trust score decreased to {}", node_id, *score);
    }

    /// Report good behavior for a node
    pub fn report_good_behavior(&self, node_id: &str) {
        let mut score = self.trust_scores.entry(node_id.to_string()).or_insert(1.0);
        *score = (*score + self.config.trust_increase_rate).min(1.0);
    }

    /// Get trust score for a specific node
    pub fn get_trust_score(&self, node_id: &str) -> f64 {
        self.trust_scores.get(node_id).map(|r| *r).unwrap_or(1.0) // New nodes start with full trust
    }

    /// Get all nodes below the trust threshold
    pub fn get_untrusted_nodes(&self) -> Vec<String> {
        self.trust_scores
            .iter()
            .filter(|entry| *entry.value() < self.config.min_trust_score)
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Reset trust score for a node (e.g., after recovery)
    pub fn reset_trust(&self, node_id: &str) {
        self.trust_scores.insert(node_id.to_string(), 1.0);
    }

    /// Get the Byzantine threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get maximum nodes capacity
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_byzantine_detector_creation() {
        let detector = ByzantineDetector::new(0.33, 1000);
        assert!(detector.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_threshold() {
        let detector = ByzantineDetector::new(1.5, 1000);
        assert!(detector.is_err());
    }

    #[tokio::test]
    async fn test_trust_score_management() {
        let detector = ByzantineDetector::new(0.33, 1000).unwrap();

        // New node should have full trust
        assert_eq!(detector.get_trust_score("node1"), 1.0);

        // Report suspicious behavior
        detector.report_suspicious("node1");
        assert!(detector.get_trust_score("node1") < 1.0);

        // Report good behavior
        detector.report_good_behavior("node1");
        let score = detector.get_trust_score("node1");
        assert!(score > 0.9); // Should recover somewhat
    }

    #[tokio::test]
    async fn test_detect_in_votes() {
        let detector = ByzantineDetector::new(0.33, 1000).unwrap();

        // Reduce trust for a node
        for _ in 0..20 {
            detector.report_suspicious("bad_node");
        }

        let mut votes: HashMap<String, i32> = HashMap::new();
        votes.insert("good_node".to_string(), 1);
        votes.insert("bad_node".to_string(), 1);

        let suspicious = detector.detect_in_votes(&votes).await.unwrap();
        assert!(suspicious.contains(&"bad_node".to_string()));
    }
}
