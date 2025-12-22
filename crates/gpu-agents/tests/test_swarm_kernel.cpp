#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../src/kernels/types.cuh"

// Declare kernel launch functions
extern "C" void launch_agent_init(GPUAgent* agents, uint32_t num_agents, uint32_t seed);
extern "C" void launch_swarm_update(GPUAgent* agents, SwarmConfig* config, uint32_t timestep);

class SwarmKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        num_agents_ = 1024;
        
        // Allocate GPU memory
        cudaMalloc(&gpu_agents_, num_agents_ * sizeof(GPUAgent));
        cudaMalloc(&gpu_config_, sizeof(SwarmConfig));
        
        // Setup swarm config
        SwarmConfig host_config;
        host_config.num_agents        = num_agents_;
        host_config.block_size        = 256;
        host_config.evolution_interval = 100;
        host_config.cohesion_weight   = 0.1f;
        host_config.separation_weight = 0.2f;
        host_config.alignment_weight  = 0.1f;
        
        cudaMemcpy(gpu_config_, &host_config, sizeof(SwarmConfig), cudaMemcpyHostToDevice);
        
        // Initialize agents
        launch_agent_init(gpu_agents_, num_agents_, 42);
    }
    
    void TearDown() override
    {
        cudaFree(gpu_agents_);
        cudaFree(gpu_config_);
    }
    
    GPUAgent*    gpu_agents_;
    SwarmConfig* gpu_config_;
    size_t       num_agents_;
};

TEST_F(SwarmKernelTest, SwarmInitialization)
{
    // Download agents and verify initialization
    std::vector<GPUAgent> host_agents(num_agents_);
    cudaMemcpy(host_agents.data(), gpu_agents_,
               num_agents_ * sizeof(GPUAgent),
               cudaMemcpyDeviceToHost);
    
    // Check agent initialization
    for (size_t i = 0; i < num_agents_; ++i)
    {
        EXPECT_FLOAT_EQ(host_agents[i].position.x, (i % 100) * 10.0f);
        EXPECT_FLOAT_EQ(host_agents[i].position.y, ((i / 100) % 100) * 10.0f);
        EXPECT_FLOAT_EQ(host_agents[i].position.z, 0.0f);
        
        EXPECT_FLOAT_EQ(host_agents[i].velocity.x, 1.0f);
        EXPECT_FLOAT_EQ(host_agents[i].velocity.y, 0.0f);
        EXPECT_FLOAT_EQ(host_agents[i].velocity.z, 0.0f);
        
        EXPECT_FLOAT_EQ(host_agents[i].fitness, 0.5f);
        EXPECT_EQ(host_agents[i].agent_type, i % 4);
        EXPECT_EQ(host_agents[i].swarm_id, i / 1000);
    }
}

TEST_F(SwarmKernelTest, SwarmUpdate)
{
    // Run swarm update
    launch_swarm_update(gpu_agents_, gpu_config_, 1);
    
    // Download and verify
    std::vector<GPUAgent> host_agents(num_agents_);
    cudaMemcpy(host_agents.data(), gpu_agents_,
               num_agents_ * sizeof(GPUAgent),
               cudaMemcpyDeviceToHost);
    
    // Velocities should be modified by swarm forces
    int velocity_changed_count = 0;
    for (size_t i = 0; i < num_agents_; ++i)
    {
        if (host_agents[i].velocity.x != 1.0f ||
            host_agents[i].velocity.y != 0.0f ||
            host_agents[i].velocity.z != 0.0f)
        {
            velocity_changed_count++;
        }
        
        // Velocity should be limited
        float speed = std::sqrt(
            host_agents[i].velocity.x * host_agents[i].velocity.x +
            host_agents[i].velocity.y * host_agents[i].velocity.y +
            host_agents[i].velocity.z * host_agents[i].velocity.z
        );
        EXPECT_LE(speed, 10.1f); // Small tolerance for floating point
    }
    
    // Some agents should have changed velocity due to swarm forces
    EXPECT_GT(velocity_changed_count, 0);
}

TEST_F(SwarmKernelTest, NeighborConnectivity)
{
    // Download and check neighbor setup
    std::vector<GPUAgent> host_agents(num_agents_);
    cudaMemcpy(host_agents.data(), gpu_agents_,
               num_agents_ * sizeof(GPUAgent),
               cudaMemcpyDeviceToHost);
    
    // Verify neighbor indices
    for (size_t i = 0; i < num_agents_; ++i)
    {
        uint32_t expected_prev = (i > 0) ? i - 1 : num_agents_ - 1;
        uint32_t expected_next = (i < num_agents_ - 1) ? i + 1 : 0;
        
        EXPECT_EQ(host_agents[i].neighbors[0], expected_prev);
        EXPECT_EQ(host_agents[i].neighbors[1], expected_next);
    }
}