#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../src/kernels/types.cuh"

// Declare kernel launch functions
extern "C" void launch_memory_init(GPUWorkingMemory* memories, uint32_t num_agents);

class MemoryKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        num_agents_ = 256;
        cudaMalloc(&gpu_memories_, num_agents_ * sizeof(GPUWorkingMemory));
    }
    
    void TearDown() override
    {
        cudaFree(gpu_memories_);
    }
    
    GPUWorkingMemory* gpu_memories_;
    size_t            num_agents_;
};

TEST_F(MemoryKernelTest, MemoryInitialization)
{
    // Launch initialization kernel
    launch_memory_init(gpu_memories_, num_agents_);
    
    // Download and verify
    std::vector<GPUWorkingMemory> host_memories(num_agents_);
    cudaMemcpy(host_memories.data(), gpu_memories_,
               num_agents_ * sizeof(GPUWorkingMemory),
               cudaMemcpyDeviceToHost);
    
    // Check that all memories are properly initialized
    for (size_t i = 0; i < num_agents_; ++i)
    {
        EXPECT_EQ(host_memories[i].head, 0);
        
        // Check first few entries
        for (int j = 0; j < 10; ++j)
        {
            EXPECT_EQ(host_memories[i].keys[j], 0);
            EXPECT_EQ(host_memories[i].timestamps[j], 0);
            EXPECT_EQ(host_memories[i].access_count[j], 0);
            
            // Check first few embedding values
            for (int k = 0; k < 10; ++k)
            {
                EXPECT_FLOAT_EQ(host_memories[i].values[j][k], 0.0f);
            }
        }
    }
}

TEST_F(MemoryKernelTest, MemoryStructureSize)
{
    // Verify structure sizes match expectations
    size_t expected_size = sizeof(uint32_t) * MAX_MEMORY_ENTRIES * 3 + // keys, timestamps, access_count
                          sizeof(float) * MAX_MEMORY_ENTRIES * EMBEDDING_SIZE + // values
                          sizeof(uint32_t); // head
    
    EXPECT_EQ(sizeof(GPUWorkingMemory), expected_size);
    
    // Verify memory allocation
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    EXPECT_GT(free_mem, 0);
}