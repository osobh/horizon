//! Knowledge graph shaders with embedded ML.
//!
//! These shaders implement knowledge graph operations with neural network
//! similarity computation and graph neural network operations embedded
//! directly in the compute kernels.
//!
//! # Key Innovation: Embedded Similarity Networks
//!
//! Instead of computing embeddings on CPU or making round-trips,
//! we embed small neural networks for:
//! - Similarity computation between node embeddings
//! - Graph neural network message passing
//! - Attention-based neighbor aggregation
//!
//! # Operations Implemented
//!
//! - **Similarity Search**: K-nearest neighbors with embedded distance
//! - **Graph Traversal**: BFS/DFS with early termination
//! - **GNN Operations**: Message passing, aggregation, update
//! - **Atomic Graph Updates**: Lock-free node/edge modifications

use crate::{BufferBinding, ShaderInfo};

/// Knowledge graph operations with embedded similarity networks.
///
/// This shader implements efficient similarity search using neural network
/// distance functions embedded directly in the kernel.
pub const GRAPH: &str = r#"
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Structures
// =============================================================================

// Graph node representation
struct GraphNode {
    uint node_id;
    uint node_type;           // Type of node (entity, concept, etc.)
    uint embedding_offset;    // Offset into embeddings buffer
    uint embedding_dim;       // Dimension of embedding
    uint edge_start;          // Start index in edge list
    uint edge_count;          // Number of edges
    float importance;         // Node importance score
    uint flags;               // Node flags (active, visited, etc.)
};

// Graph edge representation
struct GraphEdge {
    uint source;
    uint target;
    uint edge_type;
    float weight;
};

// Similarity search parameters
struct SimilarityParams {
    uint num_nodes;
    uint embedding_dim;
    uint k;                   // Number of nearest neighbors
    float threshold;          // Minimum similarity threshold
    uint use_neural_distance; // 1 = use neural network, 0 = cosine
};

// Embedded similarity network weights
// Learns a distance metric: input: [query, node] -> similarity score
struct SimilarityNetworkWeights {
    // Layer 1: concat(query, node) -> hidden (32)
    float layer1_weights[256 * 32];  // 128 + 128 max input
    float layer1_bias[32];

    // Layer 2: hidden -> hidden (16)
    float layer2_weights[32 * 16];
    float layer2_bias[16];

    // Layer 3: hidden -> similarity (1)
    float layer3_weights[16];
    float layer3_bias;
};

// K-NN result
struct KnnResult {
    uint node_id;
    float similarity;
};

// =============================================================================
// Embedded Similarity Network
// =============================================================================

// Compute similarity using embedded neural network
inline float compute_neural_similarity(
    device const float* query,
    device const float* node_embedding,
    uint dim,
    constant SimilarityNetworkWeights& weights
) {
    // Concatenate query and node embedding as input
    float input[256];
    uint input_dim = min(dim * 2, 256u);

    for (uint i = 0; i < min(dim, 128u); i++) {
        input[i] = query[i];
        input[i + 128] = node_embedding[i];
    }

    // Layer 1
    float hidden1[32];
    for (uint h = 0; h < 32; h++) {
        float sum = weights.layer1_bias[h];
        for (uint i = 0; i < input_dim; i++) {
            sum += input[i] * weights.layer1_weights[h * 256 + i];
        }
        hidden1[h] = tanh(sum);
    }

    // Layer 2
    float hidden2[16];
    for (uint h = 0; h < 16; h++) {
        float sum = weights.layer2_bias[h];
        for (uint i = 0; i < 32; i++) {
            sum += hidden1[i] * weights.layer2_weights[h * 32 + i];
        }
        hidden2[h] = tanh(sum);
    }

    // Output
    float similarity = weights.layer3_bias;
    for (uint i = 0; i < 16; i++) {
        similarity += hidden2[i] * weights.layer3_weights[i];
    }

    // Sigmoid to [0, 1]
    return 1.0f / (1.0f + exp(-similarity));
}

// Compute cosine similarity (fallback)
inline float compute_cosine_similarity(
    device const float* query,
    device const float* node_embedding,
    uint dim
) {
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_n = 0.0f;

    for (uint i = 0; i < dim; i++) {
        dot += query[i] * node_embedding[i];
        norm_q += query[i] * query[i];
        norm_n += node_embedding[i] * node_embedding[i];
    }

    float denom = sqrt(norm_q) * sqrt(norm_n);
    return denom > 0.0f ? dot / denom : 0.0f;
}

// =============================================================================
// Similarity Search Kernels
// =============================================================================

// Compute similarity for all nodes (first pass)
kernel void compute_similarities(
    device const GraphNode* nodes [[buffer(0)]],
    device const float* embeddings [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device float* similarities [[buffer(3)]],
    constant SimilarityParams& params [[buffer(4)]],
    constant SimilarityNetworkWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_nodes) return;

    GraphNode node = nodes[tid];
    device const float* node_embedding = embeddings + node.embedding_offset;

    float sim;
    if (params.use_neural_distance) {
        sim = compute_neural_similarity(query, node_embedding, params.embedding_dim, weights);
    } else {
        sim = compute_cosine_similarity(query, node_embedding, params.embedding_dim);
    }

    similarities[tid] = sim;
}

// Find top-k using parallel reduction (simplified version)
// Each threadgroup finds its local top-k, then merge
kernel void find_top_k(
    device const float* similarities [[buffer(0)]],
    device KnnResult* results [[buffer(1)]],
    constant SimilarityParams& params [[buffer(2)]],
    threadgroup float* shared_sims [[threadgroup(0)]],
    threadgroup uint* shared_ids [[threadgroup(1)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Load local data
    float my_sim = (tid < params.num_nodes) ? similarities[tid] : -INFINITY;
    uint my_id = tid;

    shared_sims[lid] = my_sim;
    shared_ids[lid] = my_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort in shared memory (simplified for top-k)
    for (uint step = 2; step <= group_size; step *= 2) {
        for (uint substep = step / 2; substep > 0; substep /= 2) {
            uint partner = lid ^ substep;
            if (partner < group_size) {
                bool ascending = ((lid / step) % 2) == 0;
                bool swap_needed = ascending ?
                    (shared_sims[lid] < shared_sims[partner]) :
                    (shared_sims[lid] > shared_sims[partner]);

                if (lid < partner && swap_needed) {
                    float temp_sim = shared_sims[lid];
                    uint temp_id = shared_ids[lid];
                    shared_sims[lid] = shared_sims[partner];
                    shared_ids[lid] = shared_ids[partner];
                    shared_sims[partner] = temp_sim;
                    shared_ids[partner] = temp_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write top-k from this group
    if (lid < params.k) {
        uint result_idx = group_id * params.k + lid;
        results[result_idx].node_id = shared_ids[group_size - 1 - lid];
        results[result_idx].similarity = shared_sims[group_size - 1 - lid];
    }
}

// =============================================================================
// Graph Traversal Kernels
// =============================================================================

// BFS frontier expansion
kernel void bfs_expand(
    device const GraphNode* nodes [[buffer(0)]],
    device const GraphEdge* edges [[buffer(1)]],
    device uint* frontier [[buffer(2)]],
    device uint* next_frontier [[buffer(3)]],
    device atomic_uint* next_frontier_size [[buffer(4)]],
    device uint* visited [[buffer(5)]],
    device uint* distances [[buffer(6)]],
    constant uint& frontier_size [[buffer(7)]],
    constant uint& current_distance [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= frontier_size) return;

    uint node_id = frontier[tid];
    GraphNode node = nodes[node_id];

    // Expand neighbors
    for (uint e = 0; e < node.edge_count; e++) {
        GraphEdge edge = edges[node.edge_start + e];
        uint neighbor = edge.target;

        // Atomic check-and-set for visited
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(
            (device atomic_uint*)&visited[neighbor], &expected, 1u,
            memory_order_relaxed, memory_order_relaxed)) {
            // First to visit this node
            distances[neighbor] = current_distance + 1;

            // Add to next frontier
            uint idx = atomic_fetch_add_explicit(next_frontier_size, 1u, memory_order_relaxed);
            next_frontier[idx] = neighbor;
        }
    }
}

// Weighted shortest path (Bellman-Ford step)
kernel void bellman_ford_step(
    device const GraphNode* nodes [[buffer(0)]],
    device const GraphEdge* edges [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device atomic_uint* changed [[buffer(3)]],
    constant uint& num_nodes [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_nodes) return;

    GraphNode node = nodes[tid];
    float my_dist = distances[tid];

    for (uint e = 0; e < node.edge_count; e++) {
        GraphEdge edge = edges[node.edge_start + e];
        float new_dist = distances[edge.source] + edge.weight;

        if (new_dist < my_dist) {
            my_dist = new_dist;
            atomic_store_explicit(changed, 1u, memory_order_relaxed);
        }
    }

    distances[tid] = my_dist;
}
"#;

/// Atomic graph operations for lock-free updates.
pub const ATOMIC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Atomic node update
struct AtomicNodeUpdate {
    uint node_id;
    uint update_type;  // 0 = importance, 1 = flags
    float value;
    uint mask;
};

// Apply atomic updates to nodes
kernel void apply_atomic_updates(
    device GraphNode* nodes [[buffer(0)]],
    device const AtomicNodeUpdate* updates [[buffer(1)]],
    constant uint& num_updates [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_updates) return;

    AtomicNodeUpdate update = updates[tid];
    device GraphNode& node = nodes[update.node_id];

    if (update.update_type == 0) {
        // Atomic importance update
        atomic_add_float((device atomic_uint*)&node.importance, update.value);
    } else if (update.update_type == 1) {
        // Atomic flags update
        atomic_fetch_or_explicit(
            (device atomic_uint*)&node.flags,
            update.mask,
            memory_order_relaxed
        );
    }
}

// Atomic edge weight update
kernel void update_edge_weights(
    device GraphEdge* edges [[buffer(0)]],
    device const uint* edge_indices [[buffer(1)]],
    device const float* weight_deltas [[buffer(2)]],
    constant uint& num_updates [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_updates) return;

    uint edge_idx = edge_indices[tid];
    float delta = weight_deltas[tid];

    atomic_add_float((device atomic_uint*)&edges[edge_idx].weight, delta);
}
"#;

/// Graph Neural Network (GNN) operations.
///
/// Implements message passing neural networks with embedded aggregation.
pub const GNN: &str = r#"
#include <metal_stdlib>
using namespace metal;

// GNN parameters
struct GnnParams {
    uint num_nodes;
    uint embedding_dim;
    uint hidden_dim;
    uint num_layers;
    float dropout_rate;
};

// GNN layer weights (one layer)
struct GnnLayerWeights {
    // Message network: concat(source, target, edge) -> message
    float message_weights[384 * 64];  // 128 * 3 -> 64
    float message_bias[64];

    // Update network: concat(node, aggregated) -> new_node
    float update_weights[192 * 64];   // 64 + 128 -> 64
    float update_bias[64];

    // Attention weights (optional)
    float attention_weights[64];
};

// Compute message from source to target
inline void compute_message(
    device const float* source_embedding,
    device const float* target_embedding,
    float edge_weight,
    thread float* message,
    uint dim,
    constant GnnLayerWeights& weights
) {
    // Concatenate inputs
    float input[384];
    for (uint i = 0; i < min(dim, 128u); i++) {
        input[i] = source_embedding[i];
        input[i + 128] = target_embedding[i];
        input[i + 256] = edge_weight;  // Broadcast edge weight
    }

    // Message MLP
    for (uint h = 0; h < 64; h++) {
        float sum = weights.message_bias[h];
        for (uint i = 0; i < min(dim * 3, 384u); i++) {
            sum += input[i] * weights.message_weights[h * 384 + i];
        }
        message[h] = max(0.0f, sum);  // ReLU
    }
}

// Aggregate messages for a node
inline void aggregate_messages(
    device const float* messages,
    uint num_messages,
    thread float* aggregated,
    uint message_dim
) {
    // Mean aggregation
    for (uint d = 0; d < message_dim; d++) {
        float sum = 0.0f;
        for (uint m = 0; m < num_messages; m++) {
            sum += messages[m * message_dim + d];
        }
        aggregated[d] = num_messages > 0 ? sum / float(num_messages) : 0.0f;
    }
}

// GNN message passing kernel
kernel void gnn_message_passing(
    device const GraphNode* nodes [[buffer(0)]],
    device const GraphEdge* edges [[buffer(1)]],
    device const float* node_embeddings [[buffer(2)]],
    device float* messages_out [[buffer(3)]],
    constant GnnParams& params [[buffer(4)]],
    constant GnnLayerWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_nodes) return;

    GraphNode node = nodes[tid];
    device const float* target_emb = node_embeddings + node.embedding_offset;

    // Compute messages from all neighbors
    for (uint e = 0; e < node.edge_count; e++) {
        GraphEdge edge = edges[node.edge_start + e];
        device const float* source_emb = node_embeddings + nodes[edge.source].embedding_offset;

        float message[64];
        compute_message(source_emb, target_emb, edge.weight, message,
                        params.embedding_dim, weights);

        // Store message
        device float* msg_out = messages_out + (node.edge_start + e) * 64;
        for (uint d = 0; d < 64; d++) {
            msg_out[d] = message[d];
        }
    }
}

// GNN aggregation and update kernel
kernel void gnn_aggregate_update(
    device const GraphNode* nodes [[buffer(0)]],
    device const float* messages [[buffer(1)]],
    device const float* node_embeddings [[buffer(2)]],
    device float* new_embeddings [[buffer(3)]],
    constant GnnParams& params [[buffer(4)]],
    constant GnnLayerWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_nodes) return;

    GraphNode node = nodes[tid];
    device const float* my_emb = node_embeddings + node.embedding_offset;

    // Aggregate incoming messages
    float aggregated[64];
    for (uint d = 0; d < 64; d++) {
        float sum = 0.0f;
        for (uint e = 0; e < node.edge_count; e++) {
            device const float* msg = messages + (node.edge_start + e) * 64;
            sum += msg[d];
        }
        aggregated[d] = node.edge_count > 0 ? sum / float(node.edge_count) : 0.0f;
    }

    // Update: combine old embedding with aggregated messages
    float input[192];
    for (uint i = 0; i < 64; i++) {
        input[i] = aggregated[i];
    }
    for (uint i = 0; i < min(params.embedding_dim, 128u); i++) {
        input[i + 64] = my_emb[i];
    }

    // Update MLP
    device float* new_emb = new_embeddings + node.embedding_offset;
    for (uint h = 0; h < min(params.embedding_dim, 64u); h++) {
        float sum = weights.update_bias[h];
        for (uint i = 0; i < 192; i++) {
            sum += input[i] * weights.update_weights[h * 192 + i];
        }
        new_emb[h] = tanh(sum);  // Tanh activation
    }
}

// Attention-based GNN aggregation
kernel void gnn_attention_aggregate(
    device const GraphNode* nodes [[buffer(0)]],
    device const GraphEdge* edges [[buffer(1)]],
    device const float* node_embeddings [[buffer(2)]],
    device float* new_embeddings [[buffer(3)]],
    constant GnnParams& params [[buffer(4)]],
    constant GnnLayerWeights& weights [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_nodes) return;

    GraphNode node = nodes[tid];
    device const float* my_emb = node_embeddings + node.embedding_offset;

    // Compute attention scores for neighbors
    float attention_scores[32];  // Max 32 neighbors for attention
    float max_score = -INFINITY;

    uint num_neighbors = min(node.edge_count, 32u);
    for (uint e = 0; e < num_neighbors; e++) {
        GraphEdge edge = edges[node.edge_start + e];
        device const float* neighbor_emb = node_embeddings + nodes[edge.source].embedding_offset;

        // Attention score = dot(weights, concat(my, neighbor))
        float score = 0.0f;
        for (uint d = 0; d < min(params.embedding_dim, 32u); d++) {
            score += weights.attention_weights[d] * my_emb[d];
            score += weights.attention_weights[d + 32] * neighbor_emb[d];
        }
        attention_scores[e] = score;
        max_score = max(max_score, score);
    }

    // Softmax normalization
    float sum_exp = 0.0f;
    for (uint e = 0; e < num_neighbors; e++) {
        attention_scores[e] = exp(attention_scores[e] - max_score);
        sum_exp += attention_scores[e];
    }
    for (uint e = 0; e < num_neighbors; e++) {
        attention_scores[e] /= sum_exp;
    }

    // Weighted aggregation
    device float* new_emb = new_embeddings + node.embedding_offset;
    for (uint d = 0; d < min(params.embedding_dim, 64u); d++) {
        float weighted_sum = 0.0f;
        for (uint e = 0; e < num_neighbors; e++) {
            GraphEdge edge = edges[node.edge_start + e];
            device const float* neighbor_emb = node_embeddings + nodes[edge.source].embedding_offset;
            weighted_sum += attention_scores[e] * neighbor_emb[d];
        }
        new_emb[d] = tanh(weighted_sum);
    }
}
"#;

/// Shader info for the graph shader.
pub const GRAPH_INFO: ShaderInfo = ShaderInfo {
    name: "graph",
    description: "Knowledge graph operations with embedded similarity networks",
    kernel_functions: &[
        "compute_similarities",
        "find_top_k",
        "bfs_expand",
        "bellman_ford_step",
    ],
    buffer_bindings: &[
        BufferBinding {
            index: 0,
            name: "nodes",
            description: "Array of GraphNode structs",
            read_only: true,
        },
        BufferBinding {
            index: 1,
            name: "embeddings",
            description: "Node embeddings buffer",
            read_only: true,
        },
        BufferBinding {
            index: 2,
            name: "query",
            description: "Query embedding for similarity search",
            read_only: true,
        },
        BufferBinding {
            index: 3,
            name: "similarities",
            description: "Output similarity scores",
            read_only: false,
        },
    ],
};

/// Shader info for atomic operations.
pub const ATOMIC_INFO: ShaderInfo = ShaderInfo {
    name: "atomic",
    description: "Atomic lock-free graph updates",
    kernel_functions: &["apply_atomic_updates", "update_edge_weights"],
    buffer_bindings: &[],
};

/// Shader info for GNN.
pub const GNN_INFO: ShaderInfo = ShaderInfo {
    name: "gnn",
    description: "Graph Neural Network message passing and aggregation",
    kernel_functions: &[
        "gnn_message_passing",
        "gnn_aggregate_update",
        "gnn_attention_aggregate",
    ],
    buffer_bindings: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_shader_content() {
        assert!(GRAPH.contains("GraphNode"));
        assert!(GRAPH.contains("SimilarityNetworkWeights"));
        assert!(GRAPH.contains("compute_neural_similarity"));
        assert!(GRAPH.contains("compute_similarities"));
        assert!(GRAPH.contains("bfs_expand"));
    }

    #[test]
    fn test_atomic_shader_content() {
        assert!(ATOMIC.contains("AtomicNodeUpdate"));
        assert!(ATOMIC.contains("apply_atomic_updates"));
        assert!(ATOMIC.contains("update_edge_weights"));
    }

    #[test]
    fn test_gnn_shader_content() {
        assert!(GNN.contains("GnnLayerWeights"));
        assert!(GNN.contains("compute_message"));
        assert!(GNN.contains("gnn_message_passing"));
        assert!(GNN.contains("gnn_aggregate_update"));
        assert!(GNN.contains("gnn_attention_aggregate"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_graph_shader_compilation() {
        use stratoswarm_metal_core::metal3::{is_available, Metal3Backend};
        use stratoswarm_metal_core::backend::MetalBackend;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = Metal3Backend::new().expect("Failed to create Metal backend");

        // Combine common + graph shaders
        let source = format!(
            "{}\n{}\n{}",
            crate::common::RNG,
            crate::common::ATOMICS,
            GRAPH
        );

        let result = backend.create_compute_pipeline(&source, "compute_similarities");
        assert!(result.is_ok(), "Failed to compile graph shader: {:?}", result.err());

        let result = backend.create_compute_pipeline(&source, "bfs_expand");
        assert!(result.is_ok(), "Failed to compile bfs_expand: {:?}", result.err());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_gnn_shader_compilation() {
        use stratoswarm_metal_core::metal3::{is_available, Metal3Backend};
        use stratoswarm_metal_core::backend::MetalBackend;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = Metal3Backend::new().expect("Failed to create Metal backend");

        // GNN shader needs graph structures
        let source = format!(
            "{}\n{}\n{}\n{}",
            crate::common::RNG,
            crate::common::ATOMICS,
            GRAPH,
            GNN
        );

        let result = backend.create_compute_pipeline(&source, "gnn_message_passing");
        assert!(result.is_ok(), "Failed to compile gnn shader: {:?}", result.err());
    }
}
