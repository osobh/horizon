// Fast synthesis kernel for high-throughput pattern matching
#include <cuda_runtime.h>
#include <cstdint>

// Aligned node structure (64 bytes)
struct __align__(64) GPUNodeAligned {
    uint32_t node_type;
    uint32_t value_hash;
    uint32_t child_count;
    uint32_t children[10];
    uint32_t padding[3]; // Pad to 64 bytes
};

// Fast pattern matching kernel with shared memory
extern "C" __global__ void match_patterns_fast(
    const uint8_t* __restrict__ patterns,
    const uint8_t* __restrict__ ast_nodes,
    uint32_t* __restrict__ matches,
    uint32_t num_patterns,
    uint32_t num_nodes
) {
    // Shared memory for pattern caching
    __shared__ GPUNodeAligned shared_patterns[32];
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_id = threadIdx.x;
    
    // Load patterns into shared memory cooperatively
    if (thread_id < 32 && thread_id < num_patterns) {
        shared_patterns[thread_id] = *((const GPUNodeAligned*)&patterns[thread_id * 64]);
    }
    __syncthreads();
    
    // Process multiple nodes per thread for better throughput
    const uint32_t nodes_per_thread = 4;
    const uint32_t start_node = tid * nodes_per_thread;
    const uint32_t end_node = min(start_node + nodes_per_thread, num_nodes);
    
    // Process assigned nodes
    for (uint32_t node_idx = start_node; node_idx < end_node; node_idx++) {
        // Load AST node with aligned access
        const GPUNodeAligned* node = (const GPUNodeAligned*)&ast_nodes[node_idx * 64];
        
        uint32_t match_flags = 0;
        
        // Check against cached patterns with loop unrolling
        #pragma unroll 8
        for (uint32_t p = 0; p < min(num_patterns, 32u); p++) {
            const GPUNodeAligned& pattern = shared_patterns[p];
            
            // Fast comparison using bitwise operations
            bool type_match = (pattern.node_type == node->node_type);
            bool value_match = (pattern.value_hash == 0) || (pattern.value_hash == node->value_hash);
            bool child_match = (pattern.child_count == node->child_count);
            
            // Set match flag
            if (type_match && value_match && child_match) {
                match_flags |= (1u << p);
            }
        }
        
        // Write results only if there's a match
        if (match_flags != 0) {
            matches[node_idx * 2] = node_idx;
            matches[node_idx * 2 + 1] = match_flags;
        }
    }
}

// Launch function for fast kernel
extern "C" void launch_match_patterns_fast(
    const uint8_t* patterns,
    const uint8_t* ast_nodes,
    uint32_t* matches,
    uint32_t num_patterns,
    uint32_t num_nodes
) {
    const uint32_t threads_per_block = 256;
    const uint32_t nodes_per_thread = 4;
    const uint32_t nodes_per_block = threads_per_block * nodes_per_thread;
    const uint32_t num_blocks = (num_nodes + nodes_per_block - 1) / nodes_per_block;
    
    dim3 block(threads_per_block);
    dim3 grid(num_blocks);
    
    match_patterns_fast<<<grid, block>>>(
        patterns, ast_nodes, matches, num_patterns, num_nodes
    );
    
    cudaDeviceSynchronize();
}