#include "types.cuh"
#include <cuda_runtime.h>

// Device function to store in working memory
__device__ void gpu_memory_store(
    GPUWorkingMemory* mem,
    uint32_t          key,
    float*            value,
    uint32_t          timestamp)
{
    uint32_t idx = atomicAdd(&mem->head, 1) % MAX_MEMORY_ENTRIES;
    mem->keys[idx] = key;
    
    // Copy embedding values
    for (int i = 0; i < EMBEDDING_SIZE; i++)
    {
        mem->values[idx][i] = value[i];
    }
    
    mem->timestamps[idx] = timestamp;
    atomicAdd(&mem->access_count[idx], 1);
}

// Device function to recall from working memory
__device__ bool gpu_memory_recall(
    GPUWorkingMemory* mem,
    uint32_t          key,
    float*            out_value)
{
    // Parallel search across memory entries
    for (int i = threadIdx.x; i < MAX_MEMORY_ENTRIES; i += blockDim.x)
    {
        if (mem->keys[i] == key)
        {
            // Copy embedding values
            for (int j = 0; j < EMBEDDING_SIZE; j++)
            {
                out_value[j] = mem->values[i][j];
            }
            atomicAdd(&mem->access_count[i], 1);
            return true;
        }
    }
    return false;
}

// Initialize memory kernel
__global__ void memory_init_kernel(
    GPUWorkingMemory* memories,
    uint32_t          num_agents)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    GPUWorkingMemory* mem = &memories[tid];
    
    // Clear all entries
    for (int i = 0; i < MAX_MEMORY_ENTRIES; i++)
    {
        mem->keys[i]         = 0;
        mem->timestamps[i]   = 0;
        mem->access_count[i] = 0;
        
        for (int j = 0; j < EMBEDDING_SIZE; j++)
        {
            mem->values[i][j] = 0.0f;
        }
    }
    
    mem->head = 0;
}

// Export C interface
extern "C"
{
    void launch_memory_init(GPUWorkingMemory* memories, uint32_t num_agents)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        memory_init_kernel<<<grid_size, block_size>>>(memories, num_agents);
        cudaDeviceSynchronize();
    }
}