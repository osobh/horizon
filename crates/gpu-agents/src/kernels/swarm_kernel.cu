#include "types.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Device function to calculate swarm forces
__device__ float3 calculate_swarm_forces(
    GPUAgent*    agent,
    SwarmConfig* config,
    uint32_t     local_id,
    float3*      local_positions,
    float3*      local_velocities)
{
    float3 cohesion   = make_float3(0.0f, 0.0f, 0.0f);
    float3 separation = make_float3(0.0f, 0.0f, 0.0f);
    float3 alignment  = make_float3(0.0f, 0.0f, 0.0f);
    
    int neighbor_count = 0;
    
    // Check neighbors in shared memory
    for (int i = 0; i < blockDim.x; i++)
    {
        if (i == local_id) continue;
        
        float3 diff = make_float3(
            local_positions[i].x - agent->position.x,
            local_positions[i].y - agent->position.y,
            local_positions[i].z - agent->position.z
        );
        
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        
        if (dist < 100.0f) // Neighbor radius
        {
            // Cohesion: move toward center
            cohesion.x += local_positions[i].x;
            cohesion.y += local_positions[i].y;
            cohesion.z += local_positions[i].z;
            
            // Separation: avoid crowding
            if (dist < 10.0f && dist > 0.0f)
            {
                separation.x -= diff.x / (dist * dist);
                separation.y -= diff.y / (dist * dist);
                separation.z -= diff.z / (dist * dist);
            }
            
            // Alignment: match velocities
            alignment.x += local_velocities[i].x;
            alignment.y += local_velocities[i].y;
            alignment.z += local_velocities[i].z;
            
            neighbor_count++;
        }
    }
    
    if (neighbor_count > 0)
    {
        cohesion.x = cohesion.x / neighbor_count - agent->position.x;
        cohesion.y = cohesion.y / neighbor_count - agent->position.y;
        cohesion.z = cohesion.z / neighbor_count - agent->position.z;
        
        alignment.x = alignment.x / neighbor_count;
        alignment.y = alignment.y / neighbor_count;
        alignment.z = alignment.z / neighbor_count;
    }
    
    // Apply weights and combine forces
    float3 force;
    force.x = cohesion.x * config->cohesion_weight +
              separation.x * config->separation_weight +
              alignment.x * config->alignment_weight;
    force.y = cohesion.y * config->cohesion_weight +
              separation.y * config->separation_weight +
              alignment.y * config->alignment_weight;
    force.z = cohesion.z * config->cohesion_weight +
              separation.z * config->separation_weight +
              alignment.z * config->alignment_weight;
    
    return force;
}

// Main swarm behavior kernel
__global__ void swarm_update_kernel(
    GPUAgent*    agents,
    SwarmConfig* config,
    uint32_t     timestep)
{
    // Declare shared memory inside kernel
    __shared__ float3 local_positions[256];
    __shared__ float3 local_velocities[256];
    __shared__ float  local_fitness[256];
    (void)local_fitness; // Suppress unused variable warning
    
    uint32_t tid      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_id = threadIdx.x;
    
    if (tid >= config->num_agents) return;
    
    GPUAgent* agent = &agents[tid];
    
    // Load agent data to shared memory (only if within block size)
    if (local_id < 256 && local_id < config->num_agents) {
        local_positions[local_id]  = agent->position;
        local_velocities[local_id] = agent->velocity;
        local_fitness[local_id]    = agent->fitness;
    }
    
    __syncthreads();
    
    // Calculate swarm forces
    float3 force = calculate_swarm_forces(agent, config, local_id, local_positions, local_velocities);
    
    // Update velocity
    agent->velocity.x += force.x;
    agent->velocity.y += force.y;
    agent->velocity.z += force.z;
    
    // Limit velocity
    float speed = sqrtf(
        agent->velocity.x * agent->velocity.x +
        agent->velocity.y * agent->velocity.y +
        agent->velocity.z * agent->velocity.z
    );
    
    if (speed > 10.0f)
    {
        agent->velocity.x = (agent->velocity.x / speed) * 10.0f;
        agent->velocity.y = (agent->velocity.y / speed) * 10.0f;
        agent->velocity.z = (agent->velocity.z / speed) * 10.0f;
    }
}

// Export C interface
extern "C"
{
    void launch_swarm_update(GPUAgent* agents, SwarmConfig* config, uint32_t timestep)
    {
        // Read config to get num_agents
        SwarmConfig host_config;
        cudaMemcpy(&host_config, config, sizeof(SwarmConfig), cudaMemcpyDeviceToHost);
        
        dim3 block_size(256);
        dim3 grid_size((host_config.num_agents + block_size.x - 1) / block_size.x);
        
        swarm_update_kernel<<<grid_size, block_size>>>(agents, config, timestep);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        }
    }
}