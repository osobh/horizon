#include <cuda_runtime.h>
#include <iostream>
#include "../src/kernels/types.cuh"

extern "C" void launch_swarm_update(GPUAgent* agents, SwarmConfig* config, uint32_t timestep);
extern "C" void launch_agent_init(GPUAgent* agents, uint32_t num_agents, uint32_t seed);

int main()
{
    std::cout << "Testing swarm update directly..." << std::endl;
    
    // Small test with 256 agents
    const int num_agents = 256;
    
    // Allocate GPU memory
    GPUAgent* gpu_agents;
    SwarmConfig* gpu_config;
    
    cudaError_t err = cudaMalloc(&gpu_agents, num_agents * sizeof(GPUAgent));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate agents: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    err = cudaMalloc(&gpu_config, sizeof(SwarmConfig));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate config: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Initialize config on host
    SwarmConfig host_config;
    host_config.num_agents = num_agents;
    host_config.block_size = 256;
    host_config.evolution_interval = 100;
    host_config.cohesion_weight = 0.1f;
    host_config.separation_weight = 0.2f;
    host_config.alignment_weight = 0.1f;
    
    // Copy to device
    err = cudaMemcpy(gpu_config, &host_config, sizeof(SwarmConfig), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy config: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Initialize agents first
    std::cout << "Initializing agents..." << std::endl;
    launch_agent_init(gpu_agents, num_agents, 42);
    
    std::cout << "Launching swarm update kernel..." << std::endl;
    launch_swarm_update(gpu_agents, gpu_config, 1);
    std::cout << "Kernel completed successfully!" << std::endl;
    
    // Cleanup
    cudaFree(gpu_agents);
    cudaFree(gpu_config);
    
    return 0;
}