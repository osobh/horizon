// CUDA 13.0 Memory Pool Management Header
// Provides interfaces for async memory pools optimized for RTX 5090

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory pool configuration
typedef struct {
    size_t initial_size;          // Initial pool size in bytes
    size_t max_size;              // Maximum pool size
    size_t release_threshold;     // Threshold for releasing unused memory
    size_t block_size;            // Allocation block size
    int device_id;                // CUDA device ID
    bool enable_p2p;              // Enable peer-to-peer access
} MemoryPoolConfig;

// Pool statistics
typedef struct {
    size_t current_allocated;     // Currently allocated bytes
    size_t peak_allocated;        // Peak allocation
    size_t total_allocations;     // Total number of allocations
    size_t total_deallocations;   // Total number of deallocations
    float fragmentation_ratio;    // Memory fragmentation (0.0 - 1.0)
    double avg_allocation_time;   // Average allocation time in ms
} MemoryPoolStats;

// Allocation request
typedef struct {
    void* ptr;                    // Allocated pointer
    size_t size;                  // Size in bytes
    cudaStream_t stream;          // Associated stream
    uint64_t timestamp;           // Allocation timestamp
} AllocationRequest;

// Initialize memory pool system
cudaError_t memory_pool_init(const MemoryPoolConfig* config);

// Cleanup memory pool system
cudaError_t memory_pool_cleanup(void);

// Async allocation from pool
void* memory_pool_alloc_async(size_t size, cudaStream_t stream);

// Async deallocation to pool
cudaError_t memory_pool_free_async(void* ptr, cudaStream_t stream);

// Batch allocation
cudaError_t memory_pool_batch_alloc(
    AllocationRequest* requests,
    int num_requests,
    cudaStream_t stream
);

// Batch deallocation
cudaError_t memory_pool_batch_free(
    void** ptrs,
    int num_ptrs,
    cudaStream_t stream
);

// Get pool statistics
cudaError_t memory_pool_get_stats(MemoryPoolStats* stats);

// Optimize pool configuration
cudaError_t memory_pool_optimize(void);

// Set memory pool attributes
cudaError_t memory_pool_set_attribute(
    cudaMemPoolAttr attr,
    void* value
);

// Get memory pool attributes
cudaError_t memory_pool_get_attribute(
    cudaMemPoolAttr attr,
    void* value
);

// Export pool for IPC
cudaError_t memory_pool_export_pointer(
    void* ptr,
    cudaMemPoolPtrExportData* export_data
);

// Import pool from IPC
cudaError_t memory_pool_import_pointer(
    cudaMemPool_t pool,
    cudaMemPoolPtrExportData* export_data,
    void** ptr
);

// Trim unused memory
cudaError_t memory_pool_trim(size_t min_bytes_to_keep);

// RTX 5090 specific optimizations
cudaError_t memory_pool_enable_blackwell_features(void);

// Performance benchmarking
float memory_pool_benchmark(size_t allocation_size, int num_iterations);

#ifdef __cplusplus
}
#endif

// C++ convenience wrapper
#ifdef __cplusplus

#include <memory>
#include <vector>

namespace stratoswarm {
namespace cuda {

class MemoryPool {
private:
    cudaMemPool_t pool_;
    MemoryPoolConfig config_;
    MemoryPoolStats stats_;
    bool initialized_;
    
public:
    MemoryPool(const MemoryPoolConfig& config);
    ~MemoryPool();
    
    // Disable copy
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Enable move
    MemoryPool(MemoryPool&& other) noexcept;
    MemoryPool& operator=(MemoryPool&& other) noexcept;
    
    // Allocation methods
    template<typename T>
    T* allocate(size_t count, cudaStream_t stream = 0) {
        size_t size = count * sizeof(T);
        return static_cast<T*>(memory_pool_alloc_async(size, stream));
    }
    
    template<typename T>
    cudaError_t deallocate(T* ptr, cudaStream_t stream = 0) {
        return memory_pool_free_async(ptr, stream);
    }
    
    // Batch operations
    template<typename T>
    std::vector<T*> batchAllocate(
        const std::vector<size_t>& counts,
        cudaStream_t stream = 0
    ) {
        std::vector<AllocationRequest> requests(counts.size());
        for (size_t i = 0; i < counts.size(); ++i) {
            requests[i].size = counts[i] * sizeof(T);
            requests[i].stream = stream;
        }
        
        memory_pool_batch_alloc(requests.data(), requests.size(), stream);
        
        std::vector<T*> result;
        for (const auto& req : requests) {
            result.push_back(static_cast<T*>(req.ptr));
        }
        return result;
    }
    
    // Statistics
    MemoryPoolStats getStats() const {
        MemoryPoolStats stats;
        memory_pool_get_stats(&stats);
        return stats;
    }
    
    // Optimization
    void optimize() {
        memory_pool_optimize();
    }
    
    // RTX 5090 features
    void enableBlackwellOptimizations() {
        memory_pool_enable_blackwell_features();
    }
};

// Smart pointer with custom deleter for pool allocations
template<typename T>
using PoolUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

template<typename T>
PoolUniquePtr<T> makePoolUnique(MemoryPool& pool, size_t count = 1, cudaStream_t stream = 0) {
    T* ptr = pool.allocate<T>(count, stream);
    return PoolUniquePtr<T>(ptr, [&pool, stream](T* p) {
        pool.deallocate(p, stream);
    });
}

} // namespace cuda
} // namespace stratoswarm

#endif // __cplusplus

#endif // MEMORY_POOL_H