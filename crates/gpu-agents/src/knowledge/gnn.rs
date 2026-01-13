//! Graph Neural Network functionality for knowledge graphs
//!
//! This module provides GNN operations including:
//! - Node embedding computation
//! - Message passing and aggregation
//! - Graph convolution layers
//! - Attention mechanisms

use super::{KnowledgeEdge, KnowledgeNode};
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceSlice};
use std::collections::HashMap;
use std::sync::Arc;

/// Activation function for GNN layers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU(f32),
}

/// Aggregation function for message passing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationFunction {
    Sum,
    Mean,
    Max,
    Attention,
}

/// GNN configuration
#[derive(Debug, Clone)]
pub struct GnnConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            input_dim: 128,
            hidden_dim: 64,
            output_dim: 32,
            num_layers: 2,
            dropout: 0.0,
            activation: ActivationFunction::ReLU,
            aggregation: AggregationFunction::Mean,
        }
    }
}

/// GNN layer
struct GnnLayer {
    weight_matrix: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    input_dim: usize,
    output_dim: usize,
}

/// Subgraph structure
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub node_ids: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub features: Vec<Vec<f32>>,
}

/// GPU data for GNN
struct GpuGnnData {
    node_features: CudaSlice<f32>,
    edge_indices: CudaSlice<u32>,
    edge_weights: CudaSlice<f32>,
    adjacency_matrix: CudaSlice<f32>,
}

/// Graph Neural Network
pub struct GraphNeuralNetwork {
    device: Arc<CudaContext>,
    config: GnnConfig,
    layers: Vec<GnnLayer>,
    nodes: Vec<KnowledgeNode>,
    edges: Vec<KnowledgeEdge>,
    adjacency_list: HashMap<u32, Vec<(u32, f32)>>,
    gpu_data: Option<GpuGnnData>,
    is_trained: bool,
}

impl GraphNeuralNetwork {
    /// Create new GNN
    pub fn new(device: Arc<CudaContext>, config: GnnConfig) -> Result<Self> {
        let mut layers = Vec::new();

        // Create layers
        let layer_dims = Self::compute_layer_dimensions(&config);

        for i in 0..config.num_layers {
            let input_dim = layer_dims[i];
            let output_dim = layer_dims[i + 1];

            // Initialize weights (Xavier initialization)
            let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
            let weight_data: Vec<f32> = (0..input_dim * output_dim)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();

            let bias_data = vec![0.0f32; output_dim];

            let stream = device.default_stream();
            let layer = GnnLayer {
                weight_matrix: stream.clone_htod(&weight_data)?,
                bias: stream.clone_htod(&bias_data)?,
                input_dim,
                output_dim,
            };

            layers.push(layer);
        }

        Ok(Self {
            device,
            config,
            layers,
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency_list: HashMap::new(),
            gpu_data: None,
            is_trained: false,
        })
    }

    /// Compute layer dimensions
    fn compute_layer_dimensions(config: &GnnConfig) -> Vec<usize> {
        let mut dims = vec![config.input_dim];

        for i in 0..config.num_layers {
            if i == config.num_layers - 1 {
                dims.push(config.output_dim);
            } else {
                dims.push(config.hidden_dim);
            }
        }

        dims
    }

    /// Get layer count
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Load graph data
    pub fn load_graph(&mut self, nodes: &[KnowledgeNode], edges: &[KnowledgeEdge]) -> Result<()> {
        self.nodes = nodes.to_vec();
        self.edges = edges.to_vec();

        // Build adjacency list
        self.adjacency_list.clear();
        for edge in &self.edges {
            self.adjacency_list
                .entry(edge.source_id)
                .or_insert_with(Vec::new)
                .push((edge.target_id, edge.weight));
        }

        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Forward pass through GNN
    pub fn forward(&mut self) -> Result<Vec<Vec<f32>>> {
        if self.nodes.is_empty() {
            return Err(anyhow!("No graph loaded"));
        }

        // Start with input features
        let mut features: Vec<Vec<f32>> = self.nodes.iter().map(|n| n.embedding.clone()).collect();

        // Pass through each layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Message passing
            let messages = self.compute_messages_cpu(&features, layer_idx)?;

            // Aggregation
            let aggregated = self.aggregate_messages_cpu(&messages)?;

            // Linear transformation
            features = self.linear_transform_cpu(&aggregated, layer)?;

            // Apply activation
            features = self.apply_activation_cpu(&features)?;

            // Apply dropout (during training)
            if self.config.dropout > 0.0 && self.is_trained {
                features = self.apply_dropout_cpu(&features)?;
            }
        }

        Ok(features)
    }

    /// Compute messages (CPU version)
    fn compute_messages_cpu(
        &self,
        features: &[Vec<f32>],
        _layer_idx: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut messages = vec![Vec::new(); self.nodes.len()];

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if let Some(neighbors) = self.adjacency_list.get(&node.id) {
                for &(neighbor_id, weight) in neighbors {
                    // Find neighbor index
                    if let Some(neighbor_idx) = self.nodes.iter().position(|n| n.id == neighbor_id)
                    {
                        let mut message = features[neighbor_idx].clone();
                        // Scale by edge weight
                        for val in &mut message {
                            *val *= weight;
                        }
                        messages[node_idx].push(message);
                    }
                }
            }
        }

        Ok(messages)
    }

    /// Aggregate messages (CPU version)
    fn aggregate_messages_cpu(&self, messages: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>> {
        let feature_dim = if !messages.is_empty() && !messages[0].is_empty() {
            messages[0][0].len()
        } else {
            self.config.input_dim
        };

        let mut aggregated = Vec::new();

        for node_messages in messages {
            if node_messages.is_empty() {
                // No messages, use zero vector
                aggregated.push(vec![0.0; feature_dim]);
            } else {
                match self.config.aggregation {
                    AggregationFunction::Sum => {
                        let mut sum = vec![0.0; feature_dim];
                        for msg in node_messages {
                            for (i, val) in msg.iter().enumerate() {
                                sum[i] += val;
                            }
                        }
                        aggregated.push(sum);
                    }
                    AggregationFunction::Mean => {
                        let mut sum = vec![0.0; feature_dim];
                        for msg in node_messages {
                            for (i, val) in msg.iter().enumerate() {
                                sum[i] += val;
                            }
                        }
                        let count = node_messages.len() as f32;
                        for val in &mut sum {
                            *val /= count;
                        }
                        aggregated.push(sum);
                    }
                    AggregationFunction::Max => {
                        let mut max_vals = vec![f32::NEG_INFINITY; feature_dim];
                        for msg in node_messages {
                            for (i, val) in msg.iter().enumerate() {
                                max_vals[i] = max_vals[i].max(*val);
                            }
                        }
                        aggregated.push(max_vals);
                    }
                    AggregationFunction::Attention => {
                        // Simplified attention - uniform weights
                        let weight = 1.0 / node_messages.len() as f32;
                        let mut weighted_sum = vec![0.0; feature_dim];
                        for msg in node_messages {
                            for (i, val) in msg.iter().enumerate() {
                                weighted_sum[i] += val * weight;
                            }
                        }
                        aggregated.push(weighted_sum);
                    }
                }
            }
        }

        Ok(aggregated)
    }

    /// Linear transformation (CPU version)
    fn linear_transform_cpu(
        &self,
        features: &[Vec<f32>],
        layer: &GnnLayer,
    ) -> Result<Vec<Vec<f32>>> {
        // Download weights from GPU
        let stream = self.device.default_stream();
        let weights: Vec<f32> = stream.clone_dtoh(&layer.weight_matrix)?;
        let bias: Vec<f32> = stream.clone_dtoh(&layer.bias)?;

        let mut output = Vec::new();

        for feature in features {
            let mut out_feature = vec![0.0; layer.output_dim];

            // Matrix multiplication: out = feature * weight + bias
            for i in 0..layer.output_dim {
                for j in 0..layer.input_dim {
                    out_feature[i] += feature[j] * weights[j * layer.output_dim + i];
                }
                out_feature[i] += bias[i];
            }

            output.push(out_feature);
        }

        Ok(output)
    }

    /// Apply activation function (CPU version)
    fn apply_activation_cpu(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut activated = features.to_vec();

        for feature in &mut activated {
            for val in feature {
                *val = match self.config.activation {
                    ActivationFunction::ReLU => val.max(0.0),
                    ActivationFunction::Tanh => val.tanh(),
                    ActivationFunction::Sigmoid => 1.0 / (1.0 + (-*val).exp()),
                    ActivationFunction::LeakyReLU(alpha) => {
                        if *val < 0.0 {
                            *val * alpha
                        } else {
                            *val
                        }
                    }
                };
            }
        }

        Ok(activated)
    }

    /// Apply dropout (CPU version)
    fn apply_dropout_cpu(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut dropped = features.to_vec();
        let keep_prob = 1.0 - self.config.dropout;

        for feature in &mut dropped {
            for val in feature {
                if rand::random::<f32>() > keep_prob {
                    *val = 0.0;
                } else {
                    *val /= keep_prob; // Scale to maintain expected value
                }
            }
        }

        Ok(dropped)
    }

    /// Compute messages for specific layer
    pub fn compute_messages(&mut self, layer_idx: usize) -> Result<Vec<Vec<f32>>> {
        let features = self.forward()?;
        let messages = self.compute_messages_cpu(&features, layer_idx)?;

        // Flatten messages for each node
        let mut flattened = Vec::new();
        for node_messages in messages {
            if node_messages.is_empty() {
                flattened.push(vec![0.0; self.config.hidden_dim]);
            } else {
                flattened.push(node_messages[0].clone());
            }
        }

        Ok(flattened)
    }

    /// Compute attention weights
    pub fn compute_attention_weights(&mut self) -> Result<Vec<Vec<f32>>> {
        let num_nodes = self.nodes.len();
        let mut attention_weights = vec![vec![0.0; num_nodes]; num_nodes];

        // Simple uniform attention for now
        for i in 0..num_nodes {
            let node = &self.nodes[i];
            if let Some(neighbors) = self.adjacency_list.get(&node.id) {
                let num_neighbors = neighbors.len() as f32;
                for &(neighbor_id, _) in neighbors {
                    if let Some(j) = self.nodes.iter().position(|n| n.id == neighbor_id) {
                        attention_weights[i][j] = 1.0 / num_neighbors;
                    }
                }
            }
        }

        Ok(attention_weights)
    }

    /// Classify nodes
    pub fn classify_nodes(&mut self) -> Result<Vec<u32>> {
        let embeddings = self.forward()?;

        // Simple argmax classification
        let predictions: Vec<u32> = embeddings
            .iter()
            .map(|emb| {
                emb.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect();

        Ok(predictions)
    }

    /// Predict link between nodes
    pub fn predict_link(&mut self, source_id: u32, target_id: u32) -> Result<f32> {
        let embeddings = self.forward()?;

        // Find node indices
        let source_idx = self
            .nodes
            .iter()
            .position(|n| n.id == source_id)
            .ok_or_else(|| anyhow!("Source node not found"))?;
        let target_idx = self
            .nodes
            .iter()
            .position(|n| n.id == target_id)
            .ok_or_else(|| anyhow!("Target node not found"))?;

        // Compute similarity (dot product)
        let source_emb = &embeddings[source_idx];
        let target_emb = &embeddings[target_idx];

        let dot_product: f32 = source_emb
            .iter()
            .zip(target_emb.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Apply sigmoid to get probability
        let score = 1.0 / (1.0 + (-dot_product).exp());

        Ok(score)
    }

    /// Get graph-level embedding
    pub fn get_graph_embedding(&mut self) -> Result<Vec<f32>> {
        let node_embeddings = self.forward()?;

        // Simple mean pooling
        let mut graph_embedding = vec![0.0; self.config.output_dim];
        let num_nodes = node_embeddings.len() as f32;

        for node_emb in &node_embeddings {
            for (i, val) in node_emb.iter().enumerate() {
                graph_embedding[i] += val / num_nodes;
            }
        }

        Ok(graph_embedding)
    }

    /// Extract k-hop subgraph
    pub fn extract_subgraph(&self, center_node: u32, k: usize) -> Result<Subgraph> {
        let mut visited = HashMap::new();
        let mut queue = vec![(center_node, 0)];
        let mut subgraph_nodes = Vec::new();
        let mut subgraph_edges = Vec::new();

        visited.insert(center_node, 0);

        while let Some((node_id, depth)) = queue.pop() {
            if depth > k {
                continue;
            }

            subgraph_nodes.push(node_id);

            if let Some(neighbors) = self.adjacency_list.get(&node_id) {
                for &(neighbor_id, _) in neighbors {
                    subgraph_edges.push((node_id, neighbor_id));

                    if !visited.contains_key(&neighbor_id) && depth < k {
                        visited.insert(neighbor_id, depth + 1);
                        queue.push((neighbor_id, depth + 1));
                    }
                }
            }
        }

        // Extract features
        let features: Vec<Vec<f32>> = subgraph_nodes
            .iter()
            .filter_map(|&node_id| {
                self.nodes
                    .iter()
                    .find(|n| n.id == node_id)
                    .map(|n| n.embedding.clone())
            })
            .collect();

        Ok(Subgraph {
            node_ids: subgraph_nodes,
            edges: subgraph_edges,
            features,
        })
    }

    /// Train step
    pub fn train_step(&mut self, labels: &[u32], learning_rate: f32) -> Result<f32> {
        // Forward pass
        let predictions = self.forward()?;

        // Compute loss (cross-entropy)
        let mut loss = 0.0;
        for (i, (pred, &label)) in predictions.iter().zip(labels.iter()).enumerate() {
            let label_idx = label as usize;
            if label_idx < pred.len() {
                // Softmax + cross-entropy
                let exp_sum: f32 = pred.iter().map(|x| x.exp()).sum();
                let prob = pred[label_idx].exp() / exp_sum;
                loss -= prob.ln();
            }
        }
        loss /= labels.len() as f32;

        // Backward pass (simplified - just update weights)
        let stream = self.device.default_stream();
        for layer in &mut self.layers {
            // Download weights
            let mut weights: Vec<f32> = stream.clone_dtoh(&layer.weight_matrix)?;

            // Simple gradient descent update
            for w in &mut weights {
                *w -= learning_rate * 0.01; // Small fixed gradient
            }

            // Upload updated weights
            layer.weight_matrix = stream.clone_htod(&weights)?;
        }

        self.is_trained = true;
        Ok(loss)
    }

    /// Sync to GPU
    pub fn sync_to_gpu(&mut self) -> Result<()> {
        // Flatten node features
        let mut features = Vec::new();
        for node in &self.nodes {
            features.extend(&node.embedding);
        }

        // Flatten edge data
        let mut edge_indices = Vec::new();
        let mut edge_weights = Vec::new();
        for edge in &self.edges {
            edge_indices.push(edge.source_id);
            edge_indices.push(edge.target_id);
            edge_weights.push(edge.weight);
        }

        // Create adjacency matrix (simplified)
        let num_nodes = self.nodes.len();
        let adjacency = vec![0.0f32; num_nodes * num_nodes];

        // Upload to GPU
        let stream = self.device.default_stream();
        let gpu_data = GpuGnnData {
            node_features: stream.clone_htod(&features)?,
            edge_indices: stream.clone_htod(&edge_indices)?,
            edge_weights: stream.clone_htod(&edge_weights)?,
            adjacency_matrix: stream.clone_htod(&adjacency)?,
        };

        self.gpu_data = Some(gpu_data);
        Ok(())
    }

    /// GPU message passing
    pub fn gpu_message_passing(&self, layer_idx: usize) -> Result<Vec<f32>> {
        if self.gpu_data.is_none() {
            return Err(anyhow!("GPU data not synchronized"));
        }

        // In real implementation, would launch GPU kernel
        // For now, return placeholder
        Ok(vec![0.1; self.config.hidden_dim * self.nodes.len()])
    }

    /// GPU aggregation
    pub fn gpu_aggregate_messages(&self) -> Result<Vec<Vec<f32>>> {
        if self.gpu_data.is_none() {
            return Err(anyhow!("GPU data not synchronized"));
        }

        // In real implementation, would launch GPU kernel
        // For now, return placeholder
        let mut result = Vec::new();
        for _ in 0..self.nodes.len() {
            result.push(vec![0.1; self.config.hidden_dim]);
        }
        Ok(result)
    }
}

// Simple random number generation
mod rand {
    pub fn random<T>() -> T
    where
        T: Default + From<f32>,
    {
        // Simple pseudo-random
        let val = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_nanos() % 1000) as f32 / 1000.0)
            .unwrap_or(0.5);
        T::from(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_config_default() {
        let config = GnnConfig::default();
        assert_eq!(config.input_dim, 128);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.output_dim, 32);
        assert_eq!(config.num_layers, 2);
    }
}
