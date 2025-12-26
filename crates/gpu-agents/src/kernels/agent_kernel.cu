#include "types.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Device function to update agent position
__device__ void update_agent_position(GPUAgent* agent, float dt)
{
    agent->position.x += agent->velocity.x * dt;
    agent->position.y += agent->velocity.y * dt;
    agent->position.z += agent->velocity.z * dt;
}

// Main agent update kernel
__global__ void agent_update_kernel(
    GPUAgent* agents,
    uint32_t  num_agents,
    float     dt,
    uint32_t  timestep)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    GPUAgent* agent = &agents[tid];
    
    // Update position
    update_agent_position(agent, dt);
    
    // Update fitness based on position (simple example)
    float dist_from_origin = sqrtf(
        agent->position.x * agent->position.x +
        agent->position.y * agent->position.y +
        agent->position.z * agent->position.z
    );
    
    agent->fitness = 1.0f / (1.0f + dist_from_origin * 0.01f);
}

// Initialize agents kernel
__global__ void agent_init_kernel(
    GPUAgent* agents,
    uint32_t  num_agents,
    uint32_t  seed)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    GPUAgent* agent = &agents[tid];
    
    // Initialize with deterministic values for testing
    agent->position.x     = (tid % 100) * 10.0f;
    agent->position.y     = ((tid / 100) % 100) * 10.0f;
    agent->position.z     = 0.0f;
    
    agent->velocity.x     = 1.0f;
    agent->velocity.y     = 0.0f;
    agent->velocity.z     = 0.0f;
    
    agent->fitness        = 0.5f;
    agent->state          = 0;
    agent->agent_type     = tid % 4; // 4 agent types
    agent->swarm_id       = tid / 1000; // Group into swarms of 1000
    
    // Initialize memory offsets
    agent->working_memory_offset  = tid * sizeof(GPUWorkingMemory);
    agent->episodic_memory_offset = 0;
    agent->semantic_memory_offset = 0;
    agent->genome_offset          = tid * GENOME_SIZE * sizeof(float);
    
    // Clear shared data
    for (int i = 0; i < 16; i++)
    {
        agent->shared_data[i] = 0.0f;
    }
    
    // Set neighbors
    agent->neighbors[0] = (tid > 0) ? tid - 1 : num_agents - 1;
    agent->neighbors[1] = (tid < num_agents - 1) ? tid + 1 : 0;
}

// Export C interface for testing
extern "C"
{
    void launch_agent_init(GPUAgent* agents, uint32_t num_agents, uint32_t seed)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        agent_init_kernel<<<grid_size, block_size>>>(agents, num_agents, seed);
        cudaDeviceSynchronize();
    }
    
    void launch_agent_update(GPUAgent* agents, uint32_t num_agents, float dt, uint32_t timestep)
    {
        dim3 block_size(256);
        dim3 grid_size((num_agents + block_size.x - 1) / block_size.x);
        
        agent_update_kernel<<<grid_size, block_size>>>(agents, num_agents, dt, timestep);
        cudaDeviceSynchronize();
    }
}