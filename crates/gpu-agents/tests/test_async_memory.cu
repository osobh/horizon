// CUDA 13.0 Async Memory Tests for StratoSwarm
// Tests async allocation, memory pools, and RTX 5090 features

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <random>

// Include the async memory header
extern "C" {
    cudaError_t init_async_memory_system();
    cudaError_t cleanup_async_memory_system();
    void* async_allocate(size_t size, cudaStream_t stream);
    cudaError_t async_free(void* ptr, cudaStream_t stream);
    cudaError_t create_memory_pool(size_t initial_size, size_t max_size, cudaStream_t stream);
    cudaError_t batch_async_operations(void** ptrs, size_t* sizes, uint8_t* operations, 
                                       int num_operations, cudaStream_t stream);
    void get_memory_stats(size_t* current_allocated, size_t* peak_allocated);
    float test_async_memory_performance(size_t allocation_size, int num_iterations);
}

class AsyncMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize async memory system
        ASSERT_EQ(cudaSuccess, init_async_memory_system());
        
        // Create test stream
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
    }
    
    void TearDown() override {
        // Cleanup stream
        cudaStreamDestroy(stream_);
        
        // Cleanup async memory system
        cleanup_async_memory_system();
    }
    
    cudaStream_t stream_;
};

// Test 1: Basic async allocation and deallocation
TEST_F(AsyncMemoryTest, BasicAsyncAllocation) {
    const size_t size = 1024 * 1024;  // 1MB
    
    // Allocate memory asynchronously
    void* ptr = async_allocate(size, stream_);
    ASSERT_NE(nullptr, ptr);
    
    // Use the memory (simple memset)
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(ptr, 42, size, stream_));
    
    // Free memory asynchronously
    ASSERT_EQ(cudaSuccess, async_free(ptr, stream_));
    
    // Synchronize to ensure operations complete
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 2: Multiple concurrent allocations
TEST_F(AsyncMemoryTest, ConcurrentAllocations) {
    const int num_allocs = 100;
    const size_t size = 10 * 1024 * 1024;  // 10MB each
    std::vector<void*> ptrs(num_allocs);
    
    // Allocate multiple buffers concurrently
    for (int i = 0; i < num_allocs; i++) {
        ptrs[i] = async_allocate(size, stream_);
        ASSERT_NE(nullptr, ptrs[i]);
    }
    
    // Initialize all buffers
    for (int i = 0; i < num_allocs; i++) {
        ASSERT_EQ(cudaSuccess, cudaMemsetAsync(ptrs[i], i, size, stream_));
    }
    
    // Free all buffers
    for (int i = 0; i < num_allocs; i++) {
        ASSERT_EQ(cudaSuccess, async_free(ptrs[i], stream_));
    }
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 3: Memory pool creation and usage
TEST_F(AsyncMemoryTest, MemoryPoolOperations) {
    const size_t initial_size = 100 * 1024 * 1024;  // 100MB
    const size_t max_size = 500 * 1024 * 1024;      // 500MB
    
    // Create a memory pool
    ASSERT_EQ(cudaSuccess, create_memory_pool(initial_size, max_size, stream_));
    
    // Allocate from pool
    void* ptr1 = async_allocate(50 * 1024 * 1024, stream_);
    void* ptr2 = async_allocate(30 * 1024 * 1024, stream_);
    
    ASSERT_NE(nullptr, ptr1);
    ASSERT_NE(nullptr, ptr2);
    
    // Free back to pool
    ASSERT_EQ(cudaSuccess, async_free(ptr1, stream_));
    ASSERT_EQ(cudaSuccess, async_free(ptr2, stream_));
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 4: Batch operations
TEST_F(AsyncMemoryTest, BatchOperations) {
    const int num_ops = 50;
    std::vector<void*> ptrs(num_ops, nullptr);
    std::vector<size_t> sizes(num_ops);
    std::vector<uint8_t> operations(num_ops);
    
    // Prepare batch allocation requests
    for (int i = 0; i < num_ops / 2; i++) {
        sizes[i] = (i + 1) * 1024 * 1024;  // Varying sizes
        operations[i] = 0;  // Allocate
    }
    
    // Execute batch allocation
    ASSERT_EQ(cudaSuccess, batch_async_operations(
        ptrs.data(), sizes.data(), operations.data(), num_ops / 2, stream_));
    
    // Prepare batch free operations
    for (int i = 0; i < num_ops / 2; i++) {
        operations[i] = 1;  // Free
    }
    
    // Execute batch free
    ASSERT_EQ(cudaSuccess, batch_async_operations(
        ptrs.data(), sizes.data(), operations.data(), num_ops / 2, stream_));
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 5: Memory statistics tracking
TEST_F(AsyncMemoryTest, MemoryStatistics) {
    size_t current_allocated = 0;
    size_t peak_allocated = 0;
    
    // Get initial stats
    get_memory_stats(&current_allocated, &peak_allocated);
    size_t initial_allocated = current_allocated;
    
    // Allocate some memory
    const size_t size1 = 100 * 1024 * 1024;  // 100MB
    const size_t size2 = 200 * 1024 * 1024;  // 200MB
    
    void* ptr1 = async_allocate(size1, stream_);
    ASSERT_NE(nullptr, ptr1);
    
    get_memory_stats(&current_allocated, &peak_allocated);
    EXPECT_GE(current_allocated, initial_allocated + size1);
    
    void* ptr2 = async_allocate(size2, stream_);
    ASSERT_NE(nullptr, ptr2);
    
    get_memory_stats(&current_allocated, &peak_allocated);
    EXPECT_GE(current_allocated, initial_allocated + size1 + size2);
    EXPECT_GE(peak_allocated, current_allocated);
    
    // Free memory
    ASSERT_EQ(cudaSuccess, async_free(ptr1, stream_));
    ASSERT_EQ(cudaSuccess, async_free(ptr2, stream_));
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 6: Performance comparison async vs sync
TEST_F(AsyncMemoryTest, PerformanceComparison) {
    const size_t size = 256 * 1024 * 1024;  // 256MB
    const int iterations = 50;
    
    // Measure sync allocation time
    auto sync_start = std::chrono::high_resolution_clock::now();
    std::vector<void*> sync_ptrs(iterations);
    
    for (int i = 0; i < iterations; i++) {
        ASSERT_EQ(cudaSuccess, cudaMalloc(&sync_ptrs[i], size));
    }
    for (int i = 0; i < iterations; i++) {
        ASSERT_EQ(cudaSuccess, cudaFree(sync_ptrs[i]));
    }
    
    auto sync_end = std::chrono::high_resolution_clock::now();
    auto sync_duration = std::chrono::duration_cast<std::chrono::milliseconds>
                        (sync_end - sync_start).count();
    
    // Measure async allocation time
    float async_duration = test_async_memory_performance(size, iterations);
    
    // Async should be faster
    printf("Sync allocation: %ld ms\n", sync_duration);
    printf("Async allocation: %.2f ms\n", async_duration);
    printf("Speedup: %.2fx\n", (float)sync_duration / async_duration);
    
    // Expect at least 1.5x speedup with async
    EXPECT_GT((float)sync_duration / async_duration, 1.5f);
}

// Test 7: Stream ordering
TEST_F(AsyncMemoryTest, StreamOrdering) {
    cudaStream_t stream2;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream2));
    
    const size_t size = 64 * 1024 * 1024;  // 64MB
    
    // Allocate on stream1
    void* ptr1 = async_allocate(size, stream_);
    ASSERT_NE(nullptr, ptr1);
    
    // Initialize on stream1
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(ptr1, 1, size, stream_));
    
    // Allocate on stream2
    void* ptr2 = async_allocate(size, stream2);
    ASSERT_NE(nullptr, ptr2);
    
    // Initialize on stream2
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(ptr2, 2, size, stream2));
    
    // Copy from stream1 to stream2 (requires sync point)
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(ptr2, ptr1, size, 
                                           cudaMemcpyDeviceToDevice, stream2));
    
    // Free on respective streams
    ASSERT_EQ(cudaSuccess, async_free(ptr1, stream_));
    ASSERT_EQ(cudaSuccess, async_free(ptr2, stream2));
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream2));
    
    cudaStreamDestroy(stream2);
}

// Test 8: Error handling
TEST_F(AsyncMemoryTest, ErrorHandling) {
    // Try to allocate too much memory
    const size_t huge_size = 100ULL * 1024 * 1024 * 1024;  // 100GB
    void* ptr = async_allocate(huge_size, stream_);
    
    // Should fail gracefully (might succeed on systems with >100GB)
    if (ptr == nullptr) {
        SUCCEED();  // Expected failure
    } else {
        // If it succeeded, clean up
        ASSERT_EQ(cudaSuccess, async_free(ptr, stream_));
    }
    
    // Try to free null pointer (should handle gracefully)
    ASSERT_EQ(cudaSuccess, async_free(nullptr, stream_));
    
    // Try to free invalid pointer (should handle gracefully)
    void* invalid_ptr = (void*)0xDEADBEEF;
    cudaError_t err = async_free(invalid_ptr, stream_);
    // Should return an error but not crash
    EXPECT_NE(cudaSuccess, err);
}

// Test 9: Memory pool stress test
TEST_F(AsyncMemoryTest, MemoryPoolStress) {
    const size_t pool_size = 1ULL * 1024 * 1024 * 1024;  // 1GB pool
    ASSERT_EQ(cudaSuccess, create_memory_pool(pool_size, pool_size * 2, stream_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(1024 * 1024, 100 * 1024 * 1024);
    
    const int num_iterations = 1000;
    std::vector<void*> active_ptrs;
    
    for (int i = 0; i < num_iterations; i++) {
        if (active_ptrs.size() < 10 || (gen() % 2 == 0 && !active_ptrs.empty())) {
            // Allocate
            size_t size = size_dist(gen);
            void* ptr = async_allocate(size, stream_);
            if (ptr != nullptr) {
                active_ptrs.push_back(ptr);
            }
        } else if (!active_ptrs.empty()) {
            // Free random allocation
            int idx = gen() % active_ptrs.size();
            async_free(active_ptrs[idx], stream_);
            active_ptrs.erase(active_ptrs.begin() + idx);
        }
    }
    
    // Cleanup remaining allocations
    for (void* ptr : active_ptrs) {
        async_free(ptr, stream_);
    }
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

// Test 10: RTX 5090 specific features (if available)
TEST_F(AsyncMemoryTest, RTX5090Features) {
    cudaDeviceProp prop;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&prop, 0));
    
    // Check if we're on RTX 5090 (sm_110)
    bool is_rtx5090 = (prop.major == 11 && prop.minor == 0) ||
                      strstr(prop.name, "RTX 5090") != nullptr;
    
    if (is_rtx5090) {
        printf("RTX 5090 detected - testing specific features\n");
        
        // Test large allocation (RTX 5090 has 32GB)
        const size_t large_size = 20ULL * 1024 * 1024 * 1024;  // 20GB
        void* large_ptr = async_allocate(large_size, stream_);
        
        if (large_ptr != nullptr) {
            printf("Successfully allocated 20GB on RTX 5090\n");
            
            // Test bandwidth with large allocation
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start, stream_);
            cudaMemsetAsync(large_ptr, 0, large_size, stream_);
            cudaEventRecord(stop, stream_);
            cudaStreamSynchronize(stream_);
            
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            double bandwidth_gbps = (large_size / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            
            printf("Memory bandwidth: %.2f GB/s (target: 1536 GB/s)\n", bandwidth_gbps);
            
            // Expect high bandwidth on RTX 5090
            EXPECT_GT(bandwidth_gbps, 1000.0);
            
            async_free(large_ptr, stream_);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    } else {
        printf("Not on RTX 5090 (detected %s with sm_%d%d) - skipping specific tests\n",
               prop.name, prop.major, prop.minor);
    }
    
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found, skipping tests\n";
        return 0;
    }
    
    return RUN_ALL_TESTS();
}