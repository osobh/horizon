//! Temporal knowledge graph functionality
//!
//! This module provides time-based graph operations including:
//! - Temporal nodes and edges with timestamps
//! - Time-based queries and filtering
//! - Temporal evolution tracking
//! - Causality analysis

use super::{KnowledgeEdge, KnowledgeNode};
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Temporal node with timestamp and validity period
#[derive(Debug, Clone)]
pub struct TemporalNode {
    pub base_node: KnowledgeNode,
    pub timestamp: i64,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub confidence: f32,
}

/// Temporal edge with timestamp and duration
#[derive(Debug, Clone)]
pub struct TemporalEdge {
    pub base_edge: KnowledgeEdge,
    pub timestamp: i64,
    pub duration: Option<i64>,
    pub causality_score: f32,
}

/// Time window query parameters
#[derive(Debug, Clone)]
pub struct TimeWindowQuery {
    pub start_time: i64,
    pub end_time: i64,
    pub include_edges: bool,
}

/// Temporal path query parameters
#[derive(Debug, Clone)]
pub struct TemporalPathQuery {
    pub source_id: u32,
    pub target_id: u32,
    pub max_time_delta: i64,
    pub min_causality_score: f32,
}

/// Temporal path result
#[derive(Debug, Clone)]
pub struct TemporalPath {
    pub nodes: Vec<u32>,
    pub timestamps: Vec<i64>,
    pub edges: Vec<u32>,
    pub total_causality_score: f32,
}

/// Aggregation type for temporal data
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Max,
    Min,
}

/// Temporal aggregation config
#[derive(Debug, Clone)]
pub struct TemporalAggregation {
    pub bucket_size_ms: i64,
    pub aggregation_type: AggregationType,
}

/// Temporal aggregation result
#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub buckets: Vec<f32>,
    pub timestamps: Vec<i64>,
}

/// Node evolution tracking
#[derive(Debug, Clone)]
pub struct NodeEvolution {
    pub node_id: u32,
    pub versions: Vec<TemporalNode>,
}

/// Causality analysis result
#[derive(Debug, Clone)]
pub struct CausalityAnalysis {
    pub is_causal: bool,
    pub chain_length: usize,
    pub total_delay: i64,
    pub confidence: f32,
}

/// Temporal anomaly
#[derive(Debug, Clone)]
pub struct TemporalAnomaly {
    pub node_id: u32,
    pub timestamp: i64,
    pub anomaly_score: f32,
    pub reason: String,
}

/// Time window result
#[derive(Debug, Clone)]
pub struct TimeWindowResult {
    pub nodes: Vec<TemporalNode>,
    pub edges: Vec<TemporalEdge>,
}

/// GPU temporal data structure
struct GpuTemporalData {
    node_timestamps: CudaSlice<i64>,
    node_valid_from: CudaSlice<i64>,
    node_valid_to: CudaSlice<i64>,
    edge_timestamps: CudaSlice<i64>,
    causality_scores: CudaSlice<f32>,
}

/// Temporal knowledge graph with time-based operations
pub struct TemporalKnowledgeGraph {
    device: Arc<CudaDevice>,
    nodes: Arc<Mutex<BTreeMap<i64, Vec<TemporalNode>>>>, // Indexed by timestamp
    edges: Arc<Mutex<Vec<TemporalEdge>>>,
    node_versions: Arc<DashMap<u32, Vec<TemporalNode>>>, // Track node evolution
    gpu_data: Option<GpuTemporalData>,
    max_nodes: usize,
}

impl TemporalKnowledgeGraph {
    /// Create new temporal knowledge graph
    pub fn new(device: Arc<CudaDevice>, max_nodes: usize) -> Result<Self> {
        Ok(Self {
            device,
            nodes: Arc::new(Mutex::new(BTreeMap::new())),
            edges: Arc::new(Mutex::new(Vec::new())),
            node_versions: Arc::new(DashMap::new()),
            gpu_data: None,
            max_nodes,
        })
    }

    /// Add temporal node
    pub fn add_temporal_node(&mut self, node: TemporalNode) -> Result<()> {
        let mut nodes = self.nodes.lock()?;
        nodes
            .entry(node.timestamp)
            .or_insert_with(Vec::new)
            .push(node.clone());

        // Track node evolution
        self.node_versions
            .entry(node.base_node.id)
            .or_insert_with(Vec::new)
            .push(node);

        Ok(())
    }

    /// Add temporal edge
    pub fn add_temporal_edge(&mut self, edge: TemporalEdge) -> Result<()> {
        let mut edges = self.edges.lock()?;
        edges.push(edge);
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        let nodes = self.nodes.lock()?;
        nodes.values().map(|v| v.len()).sum()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.lock()?.len()
    }

    /// Get time range of graph
    pub fn get_time_range(&self) -> (i64, i64) {
        let nodes = self.nodes.lock()?;
        if nodes.is_empty() {
            return (i64::MAX, i64::MIN);
        }

        let min_time = *nodes.keys().next()?;
        let max_time = *nodes.keys().last()?;
        (min_time, max_time)
    }

    /// Query nodes within time window
    pub fn query_time_window(&self, query: TimeWindowQuery) -> Result<TimeWindowResult> {
        let nodes = self.nodes.lock()?;
        let mut result_nodes = Vec::new();

        // Get nodes in time range
        for (&timestamp, node_list) in nodes.range(query.start_time..=query.end_time) {
            result_nodes.extend(node_list.clone());
        }

        let mut result_edges = Vec::new();
        if query.include_edges {
            let edges = self.edges.lock()?;
            for edge in edges.iter() {
                if edge.timestamp >= query.start_time && edge.timestamp <= query.end_time {
                    result_edges.push(edge.clone());
                }
            }
        }

        Ok(TimeWindowResult {
            nodes: result_nodes,
            edges: result_edges,
        })
    }

    /// Find temporal paths between nodes
    pub fn find_temporal_paths(&self, query: TemporalPathQuery) -> Result<Vec<TemporalPath>> {
        let edges = self.edges.lock()?;
        let mut paths = Vec::new();

        // Build adjacency list with temporal constraints
        let mut adj_list: HashMap<u32, Vec<(u32, i64, f32)>> = HashMap::new();
        for edge in edges.iter() {
            if edge.causality_score >= query.min_causality_score {
                adj_list
                    .entry(edge.base_edge.source_id)
                    .or_insert_with(Vec::new)
                    .push((
                        edge.base_edge.target_id,
                        edge.timestamp,
                        edge.causality_score,
                    ));
            }
        }

        // BFS with temporal constraints
        let mut queue = VecDeque::new();
        queue.push_back((vec![query.source_id], vec![0i64], 1.0f32));

        while let Some((path, timestamps, causality)) = queue.pop_front() {
            let current = *path.last()?;
            let current_time = *timestamps.last()?;

            if current == query.target_id {
                paths.push(TemporalPath {
                    nodes: path,
                    timestamps,
                    edges: vec![], // Simplified for now
                    total_causality_score: causality,
                });
                continue;
            }

            if let Some(neighbors) = adj_list.get(&current) {
                for &(next, edge_time, edge_causality) in neighbors {
                    if !path.contains(&next)
                        && edge_time > current_time
                        && edge_time - current_time <= query.max_time_delta
                    {
                        let mut new_path = path.clone();
                        new_path.push(next);

                        let mut new_timestamps = timestamps.clone();
                        new_timestamps.push(edge_time);

                        queue.push_back((new_path, new_timestamps, causality * edge_causality));
                    }
                }
            }
        }

        Ok(paths)
    }

    /// Aggregate temporal data
    pub fn aggregate_temporal_data(
        &self,
        config: TemporalAggregation,
    ) -> Result<AggregationResult> {
        let nodes = self.nodes.lock()?;
        let (min_time, max_time) = self.get_time_range();

        if min_time > max_time {
            return Ok(AggregationResult {
                buckets: vec![],
                timestamps: vec![],
            });
        }

        let num_buckets = ((max_time - min_time) / config.bucket_size_ms + 1) as usize;
        let mut buckets = vec![0.0f32; num_buckets];
        let mut counts = vec![0u32; num_buckets];

        // Aggregate nodes into buckets
        for (&timestamp, node_list) in nodes.iter() {
            let bucket_idx = ((timestamp - min_time) / config.bucket_size_ms) as usize;
            if bucket_idx < num_buckets {
                match config.aggregation_type {
                    AggregationType::Count => buckets[bucket_idx] += node_list.len() as f32,
                    AggregationType::Sum => {
                        for node in node_list {
                            buckets[bucket_idx] += node.confidence;
                        }
                    }
                    AggregationType::Average => {
                        for node in node_list {
                            buckets[bucket_idx] += node.confidence;
                            counts[bucket_idx] += 1;
                        }
                    }
                    AggregationType::Max => {
                        for node in node_list {
                            buckets[bucket_idx] = buckets[bucket_idx].max(node.confidence);
                        }
                    }
                    AggregationType::Min => {
                        for node in node_list {
                            if counts[bucket_idx] == 0 {
                                buckets[bucket_idx] = node.confidence;
                            } else {
                                buckets[bucket_idx] = buckets[bucket_idx].min(node.confidence);
                            }
                            counts[bucket_idx] += 1;
                        }
                    }
                }
            }
        }

        // Finalize averages
        if matches!(config.aggregation_type, AggregationType::Average) {
            for i in 0..num_buckets {
                if counts[i] > 0 {
                    buckets[i] /= counts[i] as f32;
                }
            }
        }

        // Generate timestamps
        let timestamps: Vec<i64> = (0..num_buckets)
            .map(|i| min_time + i as i64 * config.bucket_size_ms)
            .collect();

        Ok(AggregationResult {
            buckets,
            timestamps,
        })
    }

    /// Get evolution of a specific node
    pub fn get_node_evolution(&self, node_id: u32) -> Result<NodeEvolution> {
        let node_versions = self.node_versions
            .get(&node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?
            .value()
            .clone();

        Ok(NodeEvolution {
            node_id,
            versions: node_versions,
        })
    }

    /// Check if node is valid at specific time
    pub fn is_node_valid_at(&self, node_id: u32, timestamp: i64) -> Result<bool> {
        if let Some(node_versions) = self.node_versions.get(&node_id) {
            for version in node_versions.value() {
                if timestamp >= version.valid_from {
                    if let Some(valid_to) = version.valid_to {
                        if timestamp <= valid_to {
                            return Ok(true);
                        }
                    } else {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Create snapshot at specific time
    pub fn snapshot_at_time(&self, timestamp: i64) -> Result<TemporalKnowledgeGraph> {
        let snapshot = TemporalKnowledgeGraph::new(self.device.clone(), self.max_nodes)?;

        // Copy nodes valid at timestamp
        for entry in self.node_versions.iter() {
            for version in entry.value() {
                if timestamp >= version.valid_from {
                    if let Some(valid_to) = version.valid_to {
                        if timestamp <= valid_to {
                            // Node is valid at this time
                            let mut snapshot_inner = snapshot;
                            snapshot_inner.add_temporal_node(version.clone())?;
                            return Ok(snapshot_inner);
                        }
                    } else {
                        // No end time, still valid
                        let mut snapshot_inner = snapshot;
                        snapshot_inner.add_temporal_node(version.clone())?;
                        return Ok(snapshot_inner);
                    }
                }
            }
        }

        Ok(snapshot)
    }

    /// Analyze causality chain between nodes
    pub fn analyze_causality_chain(
        &self,
        source_id: u32,
        target_id: u32,
    ) -> Result<CausalityAnalysis> {
        let paths = self.find_temporal_paths(TemporalPathQuery {
            source_id,
            target_id,
            max_time_delta: i64::MAX,
            min_causality_score: 0.0,
        })?;

        if paths.is_empty() {
            return Ok(CausalityAnalysis {
                is_causal: false,
                chain_length: 0,
                total_delay: 0,
                confidence: 0.0,
            });
        }

        // Find best path (highest causality)
        let best_path = paths
            .iter()
            .max_by(|a, b| {
                a.total_causality_score
                    .partial_cmp(&b.total_causality_score)
                    .unwrap()
            })
            .unwrap();

        let total_delay =
            best_path.timestamps.last()? - best_path.timestamps.first()?;

        Ok(CausalityAnalysis {
            is_causal: true,
            chain_length: best_path.nodes.len() - 1,
            total_delay,
            confidence: best_path.total_causality_score,
        })
    }

    /// Detect temporal anomalies
    pub fn detect_temporal_anomalies(&self) -> Result<Vec<TemporalAnomaly>> {
        let mut anomalies = Vec::new();
        let nodes = self.nodes.lock()?;

        // Collect all timestamps
        let mut all_timestamps: Vec<i64> = nodes.keys().cloned().collect();
        if all_timestamps.len() < 3 {
            return Ok(anomalies);
        }

        // Calculate expected intervals
        all_timestamps.sort();
        let mut intervals = Vec::new();
        for i in 1..all_timestamps.len() {
            intervals.push(all_timestamps[i] - all_timestamps[i - 1]);
        }

        // Calculate median interval
        intervals.sort();
        let median_interval = intervals[intervals.len() / 2];
        let threshold = median_interval as f32 * 0.5; // 50% deviation threshold

        // Check each timestamp for anomalies
        for i in 1..all_timestamps.len() {
            let interval = all_timestamps[i] - all_timestamps[i - 1];
            let deviation = (interval - median_interval).abs() as f32;

            if deviation > threshold {
                // Found anomaly
                if let Some(node_list) = nodes.get(&all_timestamps[i]) {
                    for node in node_list {
                        anomalies.push(TemporalAnomaly {
                            node_id: node.base_node.id,
                            timestamp: node.timestamp,
                            anomaly_score: deviation / median_interval as f32,
                            reason: format!(
                                "Irregular timing: expected ~{}ms interval",
                                median_interval
                            ),
                        });
                    }
                }
            }
        }

        Ok(anomalies)
    }

    /// Sync data to GPU
    pub fn sync_to_gpu(&mut self) -> Result<()> {
        let nodes = self.nodes.lock()?;
        let edges = self.edges.lock()?;

        // Flatten temporal data
        let mut node_timestamps = Vec::new();
        let mut node_valid_from = Vec::new();
        let mut node_valid_to = Vec::new();

        for node_list in nodes.values() {
            for node in node_list {
                node_timestamps.push(node.timestamp);
                node_valid_from.push(node.valid_from);
                node_valid_to.push(node.valid_to.unwrap_or(i64::MAX));
            }
        }

        let mut edge_timestamps = Vec::new();
        let mut causality_scores = Vec::new();

        for edge in edges.iter() {
            edge_timestamps.push(edge.timestamp);
            causality_scores.push(edge.causality_score);
        }

        // Upload to GPU
        let gpu_data = GpuTemporalData {
            node_timestamps: self.device.htod_sync_copy(&node_timestamps)?,
            node_valid_from: self.device.htod_sync_copy(&node_valid_from)?,
            node_valid_to: self.device.htod_sync_copy(&node_valid_to)?,
            edge_timestamps: self.device.htod_sync_copy(&edge_timestamps)?,
            causality_scores: self.device.htod_sync_copy(&causality_scores)?,
        };

        self.gpu_data = Some(gpu_data);
        Ok(())
    }

    /// GPU time window query
    pub fn gpu_time_window_query(&self, start_time: i64, end_time: i64) -> Result<Vec<u32>> {
        if self.gpu_data.is_none() {
            return Err(anyhow!("GPU data not synchronized"));
        }

        // In real implementation, would launch GPU kernel
        // For now, return placeholder
        Ok(vec![0, 1, 2])
    }

    /// GPU temporal aggregation
    pub fn gpu_temporal_aggregate(&self, bucket_size: i64) -> Result<Vec<f32>> {
        if self.gpu_data.is_none() {
            return Err(anyhow!("GPU data not synchronized"));
        }

        // In real implementation, would launch GPU kernel
        // For now, return placeholder
        Ok(vec![10.0, 15.0, 20.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_node_creation() {
        let node = TemporalNode {
            base_node: KnowledgeNode {
                id: 1,
                content: "Test".to_string(),
                node_type: "test".to_string(),
                embedding: vec![0.1; 128],
            },
            timestamp: 1000,
            valid_from: 1000,
            valid_to: None,
            confidence: 0.9,
        };

        assert_eq!(node.timestamp, 1000);
        assert_eq!(node.confidence, 0.9);
    }
}
