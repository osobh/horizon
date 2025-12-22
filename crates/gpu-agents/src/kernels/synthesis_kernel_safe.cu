// Safe version of synthesis kernels with error checking
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Simple kernel that just sets output to indicate it ran
extern "C" __global__ void safe_match_patterns_kernel(
    const uint8_t* patterns,
    const uint8_t* ast_nodes,
    uint32_t* matches,
    uint32_t num_patterns,
    uint32_t num_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // Simple operation - just mark as no match
    if (tid < num_nodes) {
        matches[tid * 2] = tid;
        matches[tid * 2 + 1] = 0; // No match
    }
}

extern "C" void launch_match_patterns_safe(
    const uint8_t* patterns,
    const uint8_t* ast_nodes,
    uint32_t* matches,
    uint32_t num_patterns,
    uint32_t num_nodes
) {
    printf("[CUDA] launch_match_patterns_safe called with %u patterns, %u nodes\n", 
           num_patterns, num_nodes);
    
    if (num_nodes == 0) {
        printf("[CUDA] No nodes to process, returning\n");
        return;
    }
    
    // Check for null pointers
    if (!patterns || !ast_nodes || !matches) {
        printf("[CUDA] ERROR: Null pointer passed to kernel\n");
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_nodes + block.x - 1) / block.x);
    
    printf("[CUDA] Launching kernel with grid(%u) block(%u)\n", grid.x, block.x);
    
    safe_match_patterns_kernel<<<grid, block>>>(
        patterns, ast_nodes, matches, num_patterns, num_nodes
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA] ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[CUDA] ERROR: Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("[CUDA] Kernel completed successfully\n");
}