#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>

// Maximum values
#define MAX_NODES 1000000
#define MAX_EDGES 5000000
#define MAX_EMBEDDING_DIM 1024
#define MAX_QUEUE_SIZE 10000

// Atomic update operation types (matching Rust enum)
enum AtomicUpdateOp {
    ADD_NODE = 0,
    UPDATE_EMBEDDING = 1,
    ADD_EDGE = 2,
    UPDATE_EDGE_WEIGHT = 3,
    REMOVE_EDGE = 4,
    BATCH_UPDATE = 5
};

// Atomic update structure (matching Rust struct)
struct AtomicUpdate {
    uint32_t operation;     // AtomicUpdateOp
    uint32_t node_id;
    uint32_t edge_source;   // For edge operations
    uint32_t edge_target;   // For edge operations
    float weight;           // For edge weight updates
    uint64_t timestamp;
    uint32_t embedding_offset; // Offset in embedding buffer
    uint32_t embedding_size;   // Size of embedding data
};

// Atomic node structure
struct AtomicNode {
    uint32_t id;
    uint32_t node_type_hash;
    uint64_t embedding_version;
    uint32_t edge_count;
    uint64_t last_updated;
};

// Atomic edge structure
struct AtomicEdge {
    uint32_t source_id;
    uint32_t target_id;
    uint32_t relationship_hash;
    uint32_t weight_bits;    // f32 as u32 bits for atomic operations
    uint32_t status;         // 1=active, 0=deleted
    uint64_t version;
};

// Device functions for atomic operations

__device__ float atomic_exchange_float(float* address, float val) {
    uint32_t* address_as_uint = (uint32_t*)address;
    uint32_t old_val = atomicExch(address_as_uint, __float_as_uint(val));
    return __uint_as_float(old_val);
}

__device__ float atomic_add_float(float* address, float val) {
    uint32_t* address_as_uint = (uint32_t*)address;
    uint32_t old_val, new_val;
    do {
        old_val = *address_as_uint;
        new_val = __float_as_uint(__uint_as_float(old_val) + val);
    } while (atomicCAS(address_as_uint, old_val, new_val) != old_val);
    return __uint_as_float(old_val);
}

__device__ bool atomic_cas_version(uint64_t* address, uint64_t expected, uint64_t desired) {
    unsigned long long* ull_address = (unsigned long long*)address;
    return atomicCAS(ull_address, expected, desired) == expected;
}

// Kernel for processing atomic updates to nodes
__global__ void atomic_node_updates_kernel(
    AtomicNode* nodes,
    float* embeddings,
    uint64_t* embedding_versions,
    const AtomicUpdate* updates,
    const float* embedding_data,
    uint32_t num_updates,
    uint32_t max_nodes,
    uint32_t embedding_dim
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_updates) return;
    
    const AtomicUpdate* update = &updates[tid];
    
    if (update->node_id >= max_nodes) return;
    
    AtomicNode* node = &nodes[update->node_id];
    
    if (update->operation == ADD_NODE) {
        // Initialize new node atomically
        atomicExch(&node->id, update->node_id);
        atomicExch(&node->edge_count, 0);
        atomicExch((unsigned long long*)&node->last_updated, update->timestamp);
        
        // Update embedding if provided
        if (update->embedding_size > 0 && update->embedding_size <= embedding_dim) {
            uint32_t embedding_start = update->node_id * embedding_dim;
            
            // Atomically increment embedding version
            uint64_t new_version = atomicAdd((unsigned long long*)&embedding_versions[update->node_id], 1ULL);
            
            // Copy embedding data
            for (uint32_t i = 0; i < update->embedding_size; i++) {
                if (embedding_start + i < max_nodes * embedding_dim) {
                    float new_val = embedding_data[update->embedding_offset + i];
                    atomic_exchange_float(&embeddings[embedding_start + i], new_val);
                }
            }
            
            // Update node's embedding version
            atomicExch((unsigned long long*)&node->embedding_version, new_version);
        }
        
    } else if (update->operation == UPDATE_EMBEDDING) {
        // Update embedding atomically
        if (update->embedding_size > 0 && update->embedding_size <= embedding_dim) {
            uint32_t embedding_start = update->node_id * embedding_dim;
            
            // Atomically increment version
            uint64_t old_version = atomicAdd((unsigned long long*)&embedding_versions[update->node_id], 1ULL);
            uint64_t new_version = old_version + 1;
            
            // Update embedding data
            for (uint32_t i = 0; i < update->embedding_size; i++) {
                if (embedding_start + i < max_nodes * embedding_dim) {
                    float new_val = embedding_data[update->embedding_offset + i];
                    atomic_exchange_float(&embeddings[embedding_start + i], new_val);
                }
            }
            
            // Update node version
            atomicExch((unsigned long long*)&node->embedding_version, new_version);
            atomicExch((unsigned long long*)&node->last_updated, update->timestamp);
        }
    }
}

// Kernel for processing atomic updates to edges
__global__ void atomic_edge_updates_kernel(
    AtomicEdge* edges,
    uint32_t* adjacency_lists,
    const AtomicUpdate* updates,
    uint32_t num_updates,
    uint32_t max_edges,
    uint32_t max_nodes_adj
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_updates) return;
    
    const AtomicUpdate* update = &updates[tid];
    
    if (update->operation == ADD_EDGE) {
        // Find available edge slot
        uint32_t edge_idx = UINT32_MAX;
        
        // Linear search for empty slot (simplified - could use better allocation)
        for (uint32_t i = 0; i < max_edges; i++) {
            uint32_t expected = 0;
            if (atomicCAS(&edges[i].status, expected, 1) == 0) {
                edge_idx = i;
                break;
            }
        }
        
        if (edge_idx != UINT32_MAX) {
            AtomicEdge* edge = &edges[edge_idx];
            
            // Initialize edge atomically
            atomicExch(&edge->source_id, update->edge_source);
            atomicExch(&edge->target_id, update->edge_target);
            atomicExch(&edge->weight_bits, __float_as_uint(update->weight));
            atomicExch((unsigned long long*)&edge->version, update->timestamp);
            
            // Update adjacency list (simplified)
            uint32_t adj_offset = update->edge_source * 64; // Max 64 edges per node
            for (uint32_t i = 0; i < 64; i++) {
                if (adj_offset + i < max_nodes_adj * 64) {
                    if (atomicCAS(&adjacency_lists[adj_offset + i], 0, update->edge_target) == 0) {
                        break;
                    }
                }
            }
        }
        
    } else if (update->operation == UPDATE_EDGE_WEIGHT) {
        // Find edge and update weight
        for (uint32_t i = 0; i < max_edges; i++) {
            AtomicEdge* edge = &edges[i];
            
            if (edge->source_id == update->edge_source && 
                edge->target_id == update->edge_target &&
                edge->status == 1) {
                
                // Atomically update weight
                atomicExch(&edge->weight_bits, __float_as_uint(update->weight));
                atomicExch((unsigned long long*)&edge->version, update->timestamp);
                break;
            }
        }
        
    } else if (update->operation == REMOVE_EDGE) {
        // Find edge and mark as deleted
        for (uint32_t i = 0; i < max_edges; i++) {
            AtomicEdge* edge = &edges[i];
            
            if (edge->source_id == update->edge_source && 
                edge->target_id == update->edge_target &&
                edge->status == 1) {
                
                // Atomically mark as deleted
                atomicExch(&edge->status, 0);
                atomicExch((unsigned long long*)&edge->version, update->timestamp);
                break;
            }
        }
    }
}

// Kernel for atomic similarity search with version consistency
__global__ void atomic_similarity_search_kernel(
    const float* embeddings,
    const uint64_t* embedding_versions,
    const float* query_embedding,
    uint32_t* results,
    const uint64_t* search_version,
    uint32_t num_nodes,
    uint32_t embedding_dim,
    uint32_t k
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    // Check if this node's embedding is consistent with search version
    uint64_t node_version = embedding_versions[tid];
    uint64_t required_version = *search_version;
    
    // Skip nodes that were updated after the search version
    if (node_version > required_version) return;
    
    // Compute cosine similarity
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    uint32_t embedding_offset = tid * embedding_dim;
    
    for (uint32_t i = 0; i < embedding_dim; i++) {
        if (embedding_offset + i < num_nodes * embedding_dim) {
            float emb_val = embeddings[embedding_offset + i];
            float query_val = query_embedding[i];
            
            dot_product += emb_val * query_val;
            norm1 += emb_val * emb_val;
            norm2 += query_val * query_val;
        }
    }
    
    float norm_product = sqrtf(norm1) * sqrtf(norm2);
    float similarity = (norm_product > 1e-8f) ? (dot_product / norm_product) : 0.0f;
    
    // Store result if similarity is good enough
    if (similarity > 0.5f) { // Threshold
        // Find position in results array (simplified - would use better top-k algorithm)
        for (uint32_t i = 0; i < k; i++) {
            uint32_t current_score_bits = results[i * 2 + 1];
            float current_score = __uint_as_float(current_score_bits);
            
            if (similarity > current_score) {
                // Atomic update
                if (atomicCAS(&results[i * 2], 0, tid) == 0) {
                    atomicExch(&results[i * 2 + 1], __float_as_uint(similarity));
                    break;
                }
            }
        }
    }
}

// Kernel for atomic graph traversal with consistency
__global__ void atomic_graph_traversal_kernel(
    const AtomicNode* nodes,
    const AtomicEdge* edges,
    const uint32_t* adjacency_lists,
    uint32_t start_node,
    uint32_t* visited,
    uint32_t* path,
    uint32_t* path_length,
    uint64_t search_version,
    uint32_t max_nodes,
    uint32_t max_depth
) {
    uint32_t tid = threadIdx.x;
    
    // Shared memory for BFS queue
    __shared__ uint32_t queue[256];
    __shared__ uint32_t queue_head, queue_tail;
    __shared__ uint32_t current_depth;
    
    if (tid == 0) {
        queue_head = 0;
        queue_tail = 1;
        queue[0] = start_node;
        current_depth = 0;
        *path_length = 0;
    }
    
    // Initialize visited array
    for (uint32_t i = tid; i < max_nodes; i += blockDim.x) {
        visited[i] = 0;
    }
    
    __syncthreads();
    
    if (start_node < max_nodes) {
        atomicExch(&visited[start_node], 1);
    }
    
    __syncthreads();
    
    // BFS traversal with atomic operations
    while (queue_head < queue_tail && current_depth < max_depth) {
        uint32_t level_size = queue_tail - queue_head;
        
        // Process current level
        for (uint32_t i = tid; i < level_size; i += blockDim.x) {
            if (queue_head + i >= queue_tail) break;
            
            uint32_t current_node = queue[queue_head + i];
            
            if (current_node >= max_nodes) continue;
            
            // Check node version consistency
            const AtomicNode* node = &nodes[current_node];
            if (node->last_updated > search_version) continue;
            
            // Add to path
            uint32_t path_pos = atomicAdd(path_length, 1);
            if (path_pos < max_depth) {
                path[path_pos] = current_node;
            }
            
            // Explore neighbors
            uint32_t adj_offset = current_node * 64;
            for (uint32_t j = 0; j < 64; j++) {
                if (adj_offset + j >= max_nodes * 64) break;
                
                uint32_t neighbor = adjacency_lists[adj_offset + j];
                if (neighbor == 0 || neighbor >= max_nodes) continue;
                
                // Try to visit neighbor
                if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                    // Add to queue
                    uint32_t queue_pos = atomicAdd(&queue_tail, 1);
                    if (queue_pos < 256) {
                        queue[queue_pos] = neighbor;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Move to next level
        if (tid == 0) {
            queue_head = queue_tail;
            current_depth++;
        }
        
        __syncthreads();
    }
}

// Launch functions
extern "C" {
    
void launch_atomic_updates(
    const uint8_t* update_buffer,
    uint32_t queue_head,
    uint32_t queue_tail,
    uint32_t max_queue_size
) {
    // Process updates in queue range
    uint32_t num_updates = (queue_tail >= queue_head) ? 
        (queue_tail - queue_head) : 
        ((max_queue_size - queue_head) + queue_tail);
    
    if (num_updates == 0) return;
    
    // This is a simplified implementation
    // In practice, would deserialize updates and call appropriate kernels
    printf("Processing %u atomic updates\n", num_updates);
}

void launch_atomic_similarity_search(
    const float* embeddings,
    const uint64_t* embedding_versions,
    const float* query,
    uint32_t* results,
    const uint64_t* search_version,
    uint32_t num_nodes,
    uint32_t embedding_dim,
    uint32_t k
) {
    dim3 block_size(256);
    dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
    
    atomic_similarity_search_kernel<<<grid_size, block_size>>>(
        embeddings, embedding_versions, query, results, search_version,
        num_nodes, embedding_dim, k
    );
    
    cudaDeviceSynchronize();
}

void launch_atomic_edge_updates(
    uint8_t* edges,
    uint32_t* adjacency_lists,
    const uint8_t* update_buffer,
    uint32_t num_updates
) {
    if (num_updates == 0) return;
    
    dim3 block_size(256);
    dim3 grid_size((num_updates + block_size.x - 1) / block_size.x);
    
    // Cast to appropriate types
    AtomicEdge* typed_edges = (AtomicEdge*)edges;
    const AtomicUpdate* typed_updates = (const AtomicUpdate*)update_buffer;
    
    atomic_edge_updates_kernel<<<grid_size, block_size>>>(
        typed_edges, adjacency_lists, typed_updates, num_updates,
        MAX_EDGES, MAX_NODES
    );
    
    cudaDeviceSynchronize();
}

void launch_atomic_node_updates(
    uint8_t* nodes,
    float* embeddings,
    uint64_t* embedding_versions,
    const uint8_t* update_buffer,
    uint32_t num_updates
) {
    if (num_updates == 0) return;
    
    dim3 block_size(256);
    dim3 grid_size((num_updates + block_size.x - 1) / block_size.x);
    
    // Cast to appropriate types
    AtomicNode* typed_nodes = (AtomicNode*)nodes;
    const AtomicUpdate* typed_updates = (const AtomicUpdate*)update_buffer;
    
    atomic_node_updates_kernel<<<grid_size, block_size>>>(
        typed_nodes, embeddings, embedding_versions, typed_updates,
        nullptr, // embedding_data - would need proper handling
        num_updates, MAX_NODES, MAX_EMBEDDING_DIM
    );
    
    cudaDeviceSynchronize();
}

void launch_atomic_graph_traversal(
    const uint8_t* nodes,
    const uint8_t* edges,
    const uint32_t* adjacency_lists,
    uint32_t start_node,
    uint32_t* visited,
    uint32_t* path,
    uint32_t* path_length,
    uint64_t search_version,
    uint32_t max_nodes,
    uint32_t max_depth
) {
    dim3 block_size(256);
    dim3 grid_size(1); // Single block for coordinated traversal
    
    const AtomicNode* typed_nodes = (const AtomicNode*)nodes;
    const AtomicEdge* typed_edges = (const AtomicEdge*)edges;
    
    atomic_graph_traversal_kernel<<<grid_size, block_size>>>(
        typed_nodes, typed_edges, adjacency_lists, start_node,
        visited, path, path_length, search_version,
        max_nodes, max_depth
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"