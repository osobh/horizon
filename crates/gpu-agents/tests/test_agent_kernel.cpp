#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../src/kernels/types.cuh"

// Test utilities
namespace test_utils
{
    template <typename T>
    class CudaBuffer
    {
    public:
        explicit CudaBuffer(size_t count) : count_(count)
        {
            cudaMalloc(&device_ptr_, count * sizeof(T));
        }
        
        ~CudaBuffer()
        {
            cudaFree(device_ptr_);
        }
        
        void upload(const T* host_data)
        {
            cudaMemcpy(device_ptr_, host_data, count_ * sizeof(T), cudaMemcpyHostToDevice);
        }
        
        void download(T* host_data)
        {
            cudaMemcpy(host_data, device_ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
        }
        
        T* get() { return device_ptr_; }
        size_t size() const { return count_; }
        
    private:
        T*     device_ptr_;
        size_t count_;
    };
}

class AgentKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test agents
        test_agents_.resize(num_agents_);
        for (size_t i = 0; i < num_agents_; ++i)
        {
            GPUAgent& agent = test_agents_[i];
            agent.position       = make_float3(i * 10.0f, 0.0f, 0.0f);
            agent.velocity       = make_float3(1.0f, 0.0f, 0.0f);
            agent.fitness        = 0.5f;
            agent.state          = 0;
            agent.agent_type     = 0;
            agent.swarm_id       = 0;
        }
    }
    
    std::vector<GPUAgent> test_agents_;
    const size_t          num_agents_ = 1024;
};

TEST_F(AgentKernelTest, AgentMemoryAllocation)
{
    test_utils::CudaBuffer<GPUAgent> gpu_agents(num_agents_);
    
    // Upload test data
    gpu_agents.upload(test_agents_.data());
    
    // Download and verify
    std::vector<GPUAgent> result(num_agents_);
    gpu_agents.download(result.data());
    
    for (size_t i = 0; i < num_agents_; ++i)
    {
        EXPECT_FLOAT_EQ(result[i].position.x, i * 10.0f);
        EXPECT_FLOAT_EQ(result[i].velocity.x, 1.0f);
        EXPECT_FLOAT_EQ(result[i].fitness, 0.5f);
    }
}

TEST_F(AgentKernelTest, AgentStructSize)
{
    EXPECT_EQ(sizeof(GPUAgent), 256);
    EXPECT_EQ(alignof(GPUAgent), 4);
}

TEST_F(AgentKernelTest, WorkingMemorySize)
{
    size_t expected_size = sizeof(uint32_t) * MAX_MEMORY_ENTRIES +           // keys
                          sizeof(float) * MAX_MEMORY_ENTRIES * EMBEDDING_SIZE + // values
                          sizeof(uint32_t) * MAX_MEMORY_ENTRIES +           // timestamps
                          sizeof(uint32_t) * MAX_MEMORY_ENTRIES +           // access_count
                          sizeof(uint32_t);                                  // head
    
    EXPECT_EQ(sizeof(GPUWorkingMemory), expected_size);
}