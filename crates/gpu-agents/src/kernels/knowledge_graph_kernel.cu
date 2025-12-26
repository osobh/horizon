#include "types.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// Knowledge graph configuration
#define MAX_NODES 1000000
#define MAX_EDGES 5000000
#define EMBEDDING_DIM 768
#define MAX_QUERY_RESULTS 100
#define MAX_PATH_LENGTH 20

// GPU knowledge node structure
struct GPUKnowledgeNode {
    uint32_t id;
    uint32_t node_type_hash;    // Hash of node type string
    float embedding[EMBEDDING_DIM];
    uint32_t edge_offset;       // Offset in adjacency list
    uint32_t edge_count;        // Number of outgoing edges
};

// GPU knowledge edge structure  
struct GPUKnowledgeEdge {
    uint32_t source_id;
    uint32_t target_id;
    uint32_t relationship_hash; // Hash of relationship string
    float weight;
};

// GPU adjacency list entry
struct GPUAdjacencyEntry {
    uint32_t target_id;
    float weight;
    uint32_t relationship_hash;
};

// Knowledge graph query structure
struct GPUGraphQuery {
    float query_embedding[EMBEDDING_DIM];
    uint32_t max_results;
    float similarity_threshold;
    uint32_t query_type;        // 0=similarity, 1=path_finding, 2=subgraph
};

// Query result structure
struct GPUQueryResult {
    uint32_t node_id;
    float score;
    uint32_t path_length;       // For path queries
    uint32_t path[MAX_PATH_LENGTH]; // Path nodes
};

// Device function to compute cosine similarity
__device__ float compute_cosine_similarity(
    const float* embedding1,
    const float* embedding2,
    int dim)
{
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot_product += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
    }
    
    float norm_product = sqrtf(norm1) * sqrtf(norm2);
    return (norm_product > 1e-8f) ? (dot_product / norm_product) : 0.0f;
}

// Device function to compute Euclidean distance
__device__ float compute_euclidean_distance(
    const float* embedding1,
    const float* embedding2,
    int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = embedding1[i] - embedding2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Kernel for similarity search in knowledge graph
__global__ void similarity_search_kernel(
    const GPUKnowledgeNode* nodes,
    uint32_t num_nodes,
    const GPUGraphQuery* query,
    GPUQueryResult* results,
    uint32_t* result_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    const GPUKnowledgeNode* node = &nodes[tid];
    
    // Compute similarity to query embedding
    float similarity = compute_cosine_similarity(
        node->embedding,
        query->query_embedding,
        EMBEDDING_DIM
    );
    
    // Check if similarity meets threshold
    if (similarity >= query->similarity_threshold) {
        // Atomic increment to get result index
        uint32_t result_idx = atomicInc(result_count, query->max_results);
        
        if (result_idx < query->max_results) {
            GPUQueryResult* result = &results[result_idx];
            result->node_id = node->id;
            result->score = similarity;
            result->path_length = 0; // No path for similarity search
        }
    }
}

// Kernel for breadth-first search pathfinding
__global__ void bfs_pathfinding_kernel(
    const GPUKnowledgeNode* nodes,
    const GPUAdjacencyEntry* adjacency_list,
    uint32_t num_nodes,
    uint32_t start_node_id,
    uint32_t target_node_id,
    GPUQueryResult* result,
    bool* found)
{
    // Shared memory for BFS queue and visited array
    __shared__ uint32_t queue[1024];
    __shared__ uint32_t parent[1024];
    __shared__ bool visited[1024];
    __shared__ uint32_t queue_start, queue_end;
    
    uint32_t tid = threadIdx.x;
    uint32_t local_nodes = min(num_nodes, (uint32_t)1024);
    
    // Initialize shared memory
    if (tid == 0) {
        queue_start = 0;
        queue_end = 1;
        queue[0] = start_node_id;
        *found = false;
    }
    
    if (tid < local_nodes) {
        visited[tid] = false;
        parent[tid] = UINT32_MAX;
    }
    
    __syncthreads();
    
    // Mark start node as visited
    if (tid == 0 && start_node_id < local_nodes) {
        visited[start_node_id] = true;
    }
    
    __syncthreads();
    
    // BFS loop
    while (queue_start < queue_end && !(*found)) {
        uint32_t current_level_size = queue_end - queue_start;
        
        // Process current level in parallel
        for (uint32_t i = tid; i < current_level_size; i += blockDim.x) {
            if (queue_start + i >= queue_end) break;
            
            uint32_t current_node = queue[queue_start + i];
            
            if (current_node == target_node_id) {
                *found = true;
                
                // Reconstruct path
                uint32_t path_node = target_node_id;
                uint32_t path_len = 0;
                
                while (path_node != UINT32_MAX && path_len < MAX_PATH_LENGTH) {
                    result->path[path_len++] = path_node;
                    path_node = (path_node < local_nodes) ? parent[path_node] : UINT32_MAX;
                }
                
                result->path_length = path_len;
                result->node_id = target_node_id;
                result->score = 1.0f / path_len; // Inverse path length as score
                return;
            }
            
            // Explore neighbors (simplified - would need actual adjacency list access)
            if (current_node < num_nodes) {
                const GPUKnowledgeNode* node = &nodes[current_node];
                
                // Add neighbors to queue (simplified implementation)
                for (uint32_t j = 0; j < node->edge_count && j < 10; j++) {
                    uint32_t neighbor = (current_node + j + 1) % num_nodes;
                    
                    if (neighbor < local_nodes && !visited[neighbor]) {
                        visited[neighbor] = true;
                        parent[neighbor] = current_node;
                        
                        uint32_t new_queue_pos = atomicInc((unsigned int*)&queue_end, 1024);
                        if (new_queue_pos < 1024) {
                            queue[new_queue_pos] = neighbor;
                        }
                    }
                }
            }
        }
        
        queue_start = queue_end;
        __syncthreads();
    }
}

// Kernel for k-nearest neighbors search
__global__ void knn_search_kernel(
    const GPUKnowledgeNode* nodes,
    uint32_t num_nodes,
    const GPUGraphQuery* query,
    GPUQueryResult* results,
    uint32_t k)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for local results
    __shared__ float local_scores[256];
    __shared__ uint32_t local_ids[256];
    
    if (tid < blockDim.x) {
        local_scores[tid] = -1.0f;
        local_ids[tid] = UINT32_MAX;
    }
    
    __syncthreads();
    
    // Compute similarity for this thread's node
    if (tid < num_nodes) {
        const GPUKnowledgeNode* node = &nodes[tid];
        
        float similarity = compute_cosine_similarity(
            node->embedding,
            query->query_embedding,
            EMBEDDING_DIM
        );
        
        local_scores[threadIdx.x] = similarity;
        local_ids[threadIdx.x] = node->id;
    }
    
    __syncthreads();
    
    // Parallel reduction to find top-k in this block
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (local_scores[threadIdx.x + stride] > local_scores[threadIdx.x]) {
                local_scores[threadIdx.x] = local_scores[threadIdx.x + stride];
                local_ids[threadIdx.x] = local_ids[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result
    if (threadIdx.x == 0 && local_scores[0] > 0.0f) {
        uint32_t result_idx = atomicInc((unsigned int*)&results[0].path_length, k);
        if (result_idx < k) {
            results[result_idx].node_id = local_ids[0];
            results[result_idx].score = local_scores[0];
            results[result_idx].path_length = 0;
        }
    }
}

// Kernel for subgraph extraction around query nodes
__global__ void subgraph_extraction_kernel(
    const GPUKnowledgeNode* nodes,
    const GPUAdjacencyEntry* adjacency_list,
    uint32_t num_nodes,
    const uint32_t* seed_nodes,
    uint32_t num_seed_nodes,
    uint32_t max_depth,
    GPUQueryResult* results,
    uint32_t* result_count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_seed_nodes) return;
    
    uint32_t seed_node = seed_nodes[tid];
    
    // BFS from seed node up to max_depth
    for (uint32_t depth = 0; depth < max_depth; depth++) {
        if (seed_node < num_nodes) {
            const GPUKnowledgeNode* node = &nodes[seed_node];
            
            // Add current node to results
            uint32_t result_idx = atomicInc(result_count, MAX_QUERY_RESULTS);
            if (result_idx < MAX_QUERY_RESULTS) {
                results[result_idx].node_id = seed_node;
                results[result_idx].score = 1.0f / (depth + 1); // Distance-based score
                results[result_idx].path_length = depth;
            }
            
            // Explore one level deeper (simplified)
            if (depth < max_depth - 1 && node->edge_count > 0) {
                seed_node = (seed_node + 1) % num_nodes; // Simplified neighbor selection
            }
        }
    }
}

// Kernel for graph analytics - centrality computation
__global__ void centrality_computation_kernel(
    const GPUKnowledgeNode* nodes,
    const GPUAdjacencyEntry* adjacency_list,
    uint32_t num_nodes,
    float* centrality_scores)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_nodes) return;
    
    const GPUKnowledgeNode* node = &nodes[tid];
    
    // Compute degree centrality (simplified)
    float degree_centrality = (float)node->edge_count / (num_nodes - 1);
    
    // Compute closeness centrality approximation
    float closeness_sum = 0.0f;
    for (uint32_t i = 0; i < node->edge_count && i < 100; i++) {
        // Simplified distance computation
        closeness_sum += 1.0f; // Each edge has distance 1
    }
    
    float closeness_centrality = (node->edge_count > 0) ? 
        (float)node->edge_count / closeness_sum : 0.0f;
    
    // Combined centrality score
    centrality_scores[tid] = degree_centrality * 0.5f + closeness_centrality * 0.5f;
}

extern "C"
{
    void launch_similarity_search(
        const GPUKnowledgeNode* nodes,
        uint32_t num_nodes,
        const GPUGraphQuery* query,
        GPUQueryResult* results,
        uint32_t* result_count)
    {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        similarity_search_kernel<<<grid_size, block_size>>>(
            nodes, num_nodes, query, results, result_count);
        
        cudaDeviceSynchronize();
    }

    void launch_knn_search(
        const GPUKnowledgeNode* nodes,
        uint32_t num_nodes,
        const GPUGraphQuery* query,
        GPUQueryResult* results,
        uint32_t k)
    {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        knn_search_kernel<<<grid_size, block_size>>>(
            nodes, num_nodes, query, results, k);
        
        cudaDeviceSynchronize();
    }

    void launch_bfs_pathfinding(
        const GPUKnowledgeNode* nodes,
        const GPUAdjacencyEntry* adjacency_list,
        uint32_t num_nodes,
        uint32_t start_node_id,
        uint32_t target_node_id,
        GPUQueryResult* result,
        bool* found)
    {
        dim3 block_size(256);
        dim3 grid_size(1); // Single block for BFS
        
        bfs_pathfinding_kernel<<<grid_size, block_size>>>(
            nodes, adjacency_list, num_nodes, start_node_id, 
            target_node_id, result, found);
        
        cudaDeviceSynchronize();
    }

    void launch_subgraph_extraction(
        const GPUKnowledgeNode* nodes,
        const GPUAdjacencyEntry* adjacency_list,
        uint32_t num_nodes,
        const uint32_t* seed_nodes,
        uint32_t num_seed_nodes,
        uint32_t max_depth,
        GPUQueryResult* results,
        uint32_t* result_count)
    {
        dim3 block_size(256);
        dim3 grid_size((num_seed_nodes + block_size.x - 1) / block_size.x);
        
        subgraph_extraction_kernel<<<grid_size, block_size>>>(
            nodes, adjacency_list, num_nodes, seed_nodes, 
            num_seed_nodes, max_depth, results, result_count);
        
        cudaDeviceSynchronize();
    }

    void launch_centrality_computation(
        const GPUKnowledgeNode* nodes,
        const GPUAdjacencyEntry* adjacency_list,
        uint32_t num_nodes,
        float* centrality_scores)
    {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        centrality_computation_kernel<<<grid_size, block_size>>>(
            nodes, adjacency_list, num_nodes, centrality_scores);
        
        cudaDeviceSynchronize();
    }

    // ========================================================================
    // Temporal Knowledge Graph Kernels
    // ========================================================================

    __global__ void temporal_window_query_kernel(
        const GPUKnowledgeNode* nodes,
        const int64_t* timestamps,
        const int64_t* valid_from,
        const int64_t* valid_to,
        uint32_t num_nodes,
        int64_t query_start,
        int64_t query_end,
        uint32_t* result_indices,
        uint32_t* result_count)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_nodes) return;
        
        int64_t node_timestamp = timestamps[tid];
        int64_t node_valid_from = valid_from[tid];
        int64_t node_valid_to = valid_to[tid];
        
        // Check if node is within time window and valid
        if (node_timestamp >= query_start && node_timestamp <= query_end &&
            query_start >= node_valid_from && query_end <= node_valid_to) {
            
            uint32_t idx = atomicAdd(result_count, 1);
            if (idx < MAX_QUERY_RESULTS) {
                result_indices[idx] = tid;
            }
        }
    }

    __global__ void temporal_aggregation_kernel(
        const int64_t* timestamps,
        const float* values,
        uint32_t num_values,
        int64_t bucket_size,
        int64_t min_time,
        float* buckets,
        uint32_t num_buckets,
        uint32_t aggregation_type) // 0=count, 1=sum, 2=avg, 3=max
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_values) return;
        
        int64_t timestamp = timestamps[tid];
        uint32_t bucket_idx = (timestamp - min_time) / bucket_size;
        
        if (bucket_idx < num_buckets) {
            float value = values[tid];
            
            switch (aggregation_type) {
                case 0: // Count
                    atomicAdd(&buckets[bucket_idx], 1.0f);
                    break;
                case 1: // Sum
                    atomicAdd(&buckets[bucket_idx], value);
                    break;
                case 2: // Average (sum first, divide later)
                    atomicAdd(&buckets[bucket_idx], value);
                    break;
                case 3: // Max
                    atomicMax((int*)&buckets[bucket_idx], __float_as_int(value));
                    break;
            }
        }
    }

    void launch_temporal_window_query(
        const GPUKnowledgeNode* nodes,
        const int64_t* timestamps,
        const int64_t* valid_from,
        const int64_t* valid_to,
        uint32_t num_nodes,
        int64_t query_start,
        int64_t query_end,
        uint32_t* result_indices,
        uint32_t* result_count)
    {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        temporal_window_query_kernel<<<grid_size, block_size>>>(
            nodes, timestamps, valid_from, valid_to, num_nodes,
            query_start, query_end, result_indices, result_count);
        
        cudaDeviceSynchronize();
    }

    // ========================================================================
    // Reasoning Engine Kernels
    // ========================================================================

    __global__ void multi_hop_reasoning_kernel(
        const uint32_t* fact_subjects,
        const uint32_t* fact_predicates,
        const uint32_t* fact_objects,
        const float* fact_confidences,
        uint32_t num_facts,
        uint32_t start_entity,
        uint32_t target_entity,
        uint32_t max_hops,
        uint32_t* path_nodes,
        float* path_confidence)
    {
        __shared__ uint32_t s_queue[1024];
        __shared__ float s_confidences[1024];
        __shared__ uint32_t s_queue_size;
        
        if (threadIdx.x == 0) {
            s_queue[0] = start_entity;
            s_confidences[0] = 1.0f;
            s_queue_size = 1;
            *path_confidence = 0.0f;
        }
        __syncthreads();
        
        // BFS-style reasoning with confidence propagation
        for (uint32_t hop = 0; hop < max_hops && s_queue_size > 0; hop++) {
            uint32_t current_size = s_queue_size;
            if (threadIdx.x == 0) s_queue_size = 0;
            __syncthreads();
            
            for (uint32_t i = threadIdx.x; i < current_size; i += blockDim.x) {
                uint32_t current_entity = s_queue[i];
                float current_confidence = s_confidences[i];
                
                // Search for facts with current entity as subject
                for (uint32_t f = 0; f < num_facts; f++) {
                    if (fact_subjects[f] == current_entity) {
                        uint32_t next_entity = fact_objects[f];
                        float next_confidence = current_confidence * fact_confidences[f];
                        
                        if (next_entity == target_entity) {
                            // Custom atomic max for float
                            float old_val = *path_confidence;
                            while (next_confidence > old_val) {
                                float assumed = old_val;
                                old_val = atomicCAS((unsigned int*)path_confidence, 
                                                  __float_as_uint(assumed), 
                                                  __float_as_uint(next_confidence));
                                if (old_val == assumed) break;
                                old_val = __uint_as_float(old_val);
                            }
                        }
                        
                        // Add to queue if confidence is high enough
                        if (next_confidence > 0.1f) {
                            uint32_t idx = atomicAdd(&s_queue_size, 1);
                            if (idx < 1024) {
                                s_queue[idx] = next_entity;
                                s_confidences[idx] = next_confidence;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    __global__ void rule_application_kernel(
        const uint32_t* fact_subjects,
        const uint32_t* fact_predicates,
        const uint32_t* fact_objects,
        const float* fact_confidences,
        uint32_t num_facts,
        const uint32_t* rule_patterns,
        const float* rule_confidences,
        uint32_t num_rules,
        uint32_t* inferred_facts,
        float* inferred_confidences,
        uint32_t* num_inferred)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_facts) return;
        
        // Apply each rule to each fact
        for (uint32_t r = 0; r < num_rules; r++) {
            uint32_t rule_pattern = rule_patterns[r];
            
            // Transitive rule pattern
            if (rule_pattern == 1) { // TRANSITIVE
                uint32_t subject = fact_subjects[tid];
                uint32_t predicate = fact_predicates[tid];
                uint32_t object = fact_objects[tid];
                
                // Find facts where this object is the subject
                for (uint32_t f2 = 0; f2 < num_facts; f2++) {
                    if (fact_subjects[f2] == object && 
                        fact_predicates[f2] == predicate) {
                        
                        // Infer transitive relation
                        float confidence = fact_confidences[tid] * 
                                         fact_confidences[f2] * 
                                         rule_confidences[r];
                        
                        if (confidence > 0.5f) {
                            uint32_t idx = atomicAdd(num_inferred, 1);
                            if (idx < MAX_QUERY_RESULTS) {
                                inferred_facts[idx * 3] = subject;
                                inferred_facts[idx * 3 + 1] = predicate;
                                inferred_facts[idx * 3 + 2] = fact_objects[f2];
                                inferred_confidences[idx] = confidence;
                            }
                        }
                    }
                }
            }
        }
    }

    void launch_multi_hop_reasoning(
        const uint32_t* fact_subjects,
        const uint32_t* fact_predicates,
        const uint32_t* fact_objects,
        const float* fact_confidences,
        uint32_t num_facts,
        uint32_t start_entity,
        uint32_t target_entity,
        uint32_t max_hops,
        uint32_t* path_nodes,
        float* path_confidence)
    {
        dim3 block_size(256);
        dim3 grid_size(1); // Single block for BFS
        
        multi_hop_reasoning_kernel<<<grid_size, block_size>>>(
            fact_subjects, fact_predicates, fact_objects, fact_confidences,
            num_facts, start_entity, target_entity, max_hops,
            path_nodes, path_confidence);
        
        cudaDeviceSynchronize();
    }

    // ========================================================================
    // Graph Neural Network Kernels
    // ========================================================================

    __global__ void gnn_message_passing_kernel(
        const float* node_features,
        const uint32_t* edge_indices,
        const float* edge_weights,
        uint32_t num_nodes,
        uint32_t num_edges,
        uint32_t feature_dim,
        float* messages)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_edges) return;
        
        uint32_t source = edge_indices[tid * 2];
        uint32_t target = edge_indices[tid * 2 + 1];
        float weight = edge_weights[tid];
        
        // Compute message from source to target
        for (uint32_t d = 0; d < feature_dim; d++) {
            float feature_val = node_features[source * feature_dim + d];
            atomicAdd(&messages[target * feature_dim + d], feature_val * weight);
        }
    }

    __global__ void gnn_aggregation_kernel(
        const float* messages,
        const float* node_features,
        const float* weights,
        const float* bias,
        uint32_t num_nodes,
        uint32_t input_dim,
        uint32_t output_dim,
        uint32_t aggregation_type, // 0=sum, 1=mean, 2=max
        float* output_features)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_nodes) return;
        
        // Apply linear transformation: out = (messages + features) * W + b
        for (uint32_t o = 0; o < output_dim; o++) {
            float sum = bias[o];
            
            for (uint32_t i = 0; i < input_dim; i++) {
                float combined = messages[tid * input_dim + i] + 
                                node_features[tid * input_dim + i];
                sum += combined * weights[i * output_dim + o];
            }
            
            // Apply activation (ReLU)
            output_features[tid * output_dim + o] = fmaxf(0.0f, sum);
        }
    }

    __global__ void gnn_attention_kernel(
        const float* query_features,
        const float* key_features,
        const uint32_t* edge_indices,
        uint32_t num_edges,
        uint32_t feature_dim,
        float* attention_weights)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_edges) return;
        
        uint32_t source = edge_indices[tid * 2];
        uint32_t target = edge_indices[tid * 2 + 1];
        
        // Compute attention score (dot product)
        float score = 0.0f;
        for (uint32_t d = 0; d < feature_dim; d++) {
            score += query_features[target * feature_dim + d] * 
                    key_features[source * feature_dim + d];
        }
        
        // Apply softmax later
        attention_weights[tid] = score / sqrtf((float)feature_dim);
    }

    void launch_gnn_message_passing(
        const float* node_features,
        const uint32_t* edge_indices,
        const float* edge_weights,
        uint32_t num_nodes,
        uint32_t num_edges,
        uint32_t feature_dim,
        float* messages)
    {
        dim3 block_size(256);
        dim3 grid_size((num_edges + block_size.x - 1) / block_size.x);
        
        gnn_message_passing_kernel<<<grid_size, block_size>>>(
            node_features, edge_indices, edge_weights,
            num_nodes, num_edges, feature_dim, messages);
        
        cudaDeviceSynchronize();
    }

    void launch_gnn_aggregation(
        const float* messages,
        const float* node_features,
        const float* weights,
        const float* bias,
        uint32_t num_nodes,
        uint32_t input_dim,
        uint32_t output_dim,
        uint32_t aggregation_type,
        float* output_features)
    {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        gnn_aggregation_kernel<<<grid_size, block_size>>>(
            messages, node_features, weights, bias,
            num_nodes, input_dim, output_dim, aggregation_type,
            output_features);
        
        cudaDeviceSynchronize();
    }

    // Helper function for atomic max on floats
    __device__ float atomicMaxFloat(float* addr, float value) {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int;
        int expected;
        
        do {
            expected = old;
            old = atomicCAS(addr_as_int, expected,
                          __float_as_int(fmaxf(value, __int_as_float(expected))));
        } while (expected != old);
        
        return __int_as_float(old);
    }
}