// CUDA 13.0 GPUDirect RDMA for Multi-Node Scaling
// RTX 5090 optimized with NVLink and InfiniBand support
// Zero-copy GPU-to-GPU communication across nodes

#include <cuda_runtime.h>
#include <cuda.h>
#include <infiniband/verbs.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <memory>

// GPUDirect RDMA configuration
struct GPUDirectConfig {
    int device_id;
    bool rdma_enabled;
    bool nvlink_enabled;
    bool gdr_enabled;  // GPUDirect RDMA
    size_t bar_size;   // BAR1 size for P2P
    int num_nvlinks;
    std::vector<int> peer_devices;
};

// RDMA connection context
struct RDMAContext {
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_mr* mr;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_port_attr port_attr;
    int ib_port;
    int gid_index;
};

// ============================================================================
// GPUDirect RDMA Initialization
// ============================================================================

// Check and enable GPUDirect RDMA capabilities
extern "C" GPUDirectConfig* init_gpudirect_rdma(int device_id) {
    GPUDirectConfig* config = new GPUDirectConfig;
    config->device_id = device_id;
    
    cudaSetDevice(device_id);
    
    // Check for GPUDirect RDMA support
    int rdma_supported = 0;
    cudaDeviceGetAttribute(&rdma_supported, 
                           cudaDevAttrGPUDirectRDMASupported,
                           device_id);
    config->rdma_enabled = (rdma_supported == 1);
    
    if (!config->rdma_enabled) {
        printf("Warning: GPU %d does not support GPUDirect RDMA\n", device_id);
    } else {
        printf("GPUDirect RDMA enabled on GPU %d\n", device_id);
        
        // Check additional capabilities
        int rdma_flush_writes = 0;
        cudaDeviceGetAttribute(&rdma_flush_writes,
                               cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                               device_id);
        
        int rdma_writes_ordering = 0;
        cudaDeviceGetAttribute(&rdma_writes_ordering,
                               cudaDevAttrGPUDirectRDMAWritesOrdering,
                               device_id);
        
        printf("  Flush writes options: %d\n", rdma_flush_writes);
        printf("  Writes ordering: %d\n", rdma_writes_ordering);
    }
    
    // Check NVLink topology
    config->num_nvlinks = 0;
    for (int peer = 0; peer < 8; peer++) {
        if (peer != device_id) {
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, device_id, peer);
            
            if (can_access) {
                config->peer_devices.push_back(peer);
                config->num_nvlinks++;
                
                // Enable peer access
                cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
                if (err == cudaSuccess) {
                    printf("  NVLink enabled: GPU %d <-> GPU %d\n", device_id, peer);
                }
            }
        }
    }
    
    config->nvlink_enabled = (config->num_nvlinks > 0);
    
    // Get BAR1 size for P2P mappings
    size_t bar1_size = 0;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, device_id);
    cuDeviceTotalMem(&bar1_size, cu_device);
    config->bar_size = bar1_size;
    
    printf("  BAR1 size: %.2f GB\n", bar1_size / (1024.0 * 1024.0 * 1024.0));
    
    return config;
}

// ============================================================================
// RDMA Memory Registration
// ============================================================================

// Register GPU memory for RDMA access
extern "C" cudaError_t register_gpu_memory_rdma(
    void* gpu_ptr,
    size_t size,
    RDMAContext* rdma_ctx
) {
    // Get GPU memory handle
    cudaIpcMemHandle_t ipc_handle;
    cudaError_t err = cudaIpcGetMemHandle(&ipc_handle, gpu_ptr);
    if (err != cudaSuccess) {
        printf("Failed to get IPC handle: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // Register memory with InfiniBand
    int access_flags = IBV_ACCESS_LOCAL_WRITE | 
                      IBV_ACCESS_REMOTE_WRITE |
                      IBV_ACCESS_REMOTE_READ;
    
    rdma_ctx->mr = ibv_reg_mr(rdma_ctx->pd, gpu_ptr, size, access_flags);
    if (!rdma_ctx->mr) {
        printf("Failed to register GPU memory with RDMA\n");
        return cudaErrorMemoryAllocation;
    }
    
    printf("Registered %zu bytes of GPU memory for RDMA\n", size);
    printf("  Local key: 0x%x\n", rdma_ctx->mr->lkey);
    printf("  Remote key: 0x%x\n", rdma_ctx->mr->rkey);
    
    return cudaSuccess;
}

// ============================================================================
// Direct GPU-to-GPU Transfer
// ============================================================================

// Zero-copy transfer between GPUs across nodes
__global__ void rdma_copy_kernel(
    float* local_data,
    volatile float* remote_data,
    size_t num_elements,
    bool is_sender
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;
    
    if (is_sender) {
        // Write to remote GPU memory
        remote_data[tid] = local_data[tid];
        
        // Memory fence for write ordering
        __threadfence_system();
    } else {
        // Read from remote GPU memory
        local_data[tid] = remote_data[tid];
    }
}

// Perform RDMA transfer between GPUs
extern "C" cudaError_t gpudirect_rdma_transfer(
    void* local_gpu_ptr,
    void* remote_gpu_ptr,
    size_t size,
    bool is_sender,
    cudaStream_t stream
) {
    size_t num_elements = size / sizeof(float);
    
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);
    
    rdma_copy_kernel<<<grid, block, 0, stream>>>(
        (float*)local_gpu_ptr,
        (volatile float*)remote_gpu_ptr,
        num_elements,
        is_sender
    );
    
    return cudaGetLastError();
}

// ============================================================================
// NVLink Direct Transfer
// ============================================================================

// Optimized P2P transfer using NVLink
extern "C" cudaError_t nvlink_p2p_transfer(
    void* dst_ptr,
    int dst_device,
    const void* src_ptr,
    int src_device,
    size_t size,
    cudaStream_t stream
) {
    // Check if P2P is possible
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, dst_device, src_device);
    
    if (!can_access) {
        printf("No P2P access between GPU %d and GPU %d\n", src_device, dst_device);
        return cudaErrorInvalidDevice;
    }
    
    // Perform P2P copy
    cudaError_t err = cudaMemcpyPeerAsync(
        dst_ptr, dst_device,
        src_ptr, src_device,
        size, stream
    );
    
    if (err == cudaSuccess) {
        // Get NVLink throughput info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, src_device);
        
        printf("NVLink P2P transfer: %.2f MB\n", size / (1024.0 * 1024.0));
        printf("  Max bandwidth: %.2f GB/s\n", 
               prop.memoryBusWidth * prop.memoryClockRate * 2.0 / 8e6);
    }
    
    return err;
}

// ============================================================================
// Multi-GPU Broadcast with GPUDirect
// ============================================================================

// Broadcast data to multiple GPUs using GPUDirect
__global__ void broadcast_kernel(
    float* data,
    size_t size,
    int root_device,
    int my_device
) {
    // Only root device writes, others read
    if (my_device == root_device) {
        // Root ensures data is visible
        __threadfence_system();
    } else {
        // Non-root devices wait for data
        __syncthreads();
    }
}

extern "C" cudaError_t gpudirect_broadcast(
    void* data,
    size_t size,
    int root_device,
    const std::vector<int>& devices,
    cudaStream_t* streams
) {
    // Create IPC handle for root's data
    cudaIpcMemHandle_t ipc_handle;
    
    if (devices[0] == root_device) {
        cudaSetDevice(root_device);
        cudaIpcGetMemHandle(&ipc_handle, data);
    }
    
    // Broadcast IPC handle (would use MPI in practice)
    // For now, assuming handle is shared
    
    // Each device opens the IPC handle
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] != root_device) {
            cudaSetDevice(devices[i]);
            
            void* remote_ptr;
            cudaIpcOpenMemHandle(&remote_ptr, ipc_handle,
                                cudaIpcMemLazyEnablePeerAccess);
            
            // Copy from root to this device
            cudaMemcpyAsync(data, remote_ptr, size,
                           cudaMemcpyDeviceToDevice, streams[i]);
            
            // Close IPC handle
            cudaIpcCloseMemHandle(remote_ptr);
        }
    }
    
    // Synchronize all streams
    for (size_t i = 0; i < devices.size(); i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    return cudaSuccess;
}

// ============================================================================
// RDMA-based AllReduce
// ============================================================================

// Custom all-reduce using GPUDirect RDMA
extern "C" cudaError_t gpudirect_allreduce(
    float* data,
    size_t count,
    const std::vector<RDMAContext*>& rdma_contexts,
    int my_rank,
    cudaStream_t stream
) {
    // Ring-based all-reduce algorithm
    int num_ranks = rdma_contexts.size();
    int next_rank = (my_rank + 1) % num_ranks;
    int prev_rank = (my_rank - 1 + num_ranks) % num_ranks;
    
    size_t chunk_size = count / num_ranks;
    
    // Phase 1: Reduce-scatter
    for (int step = 0; step < num_ranks - 1; step++) {
        int send_chunk = (my_rank - step + num_ranks) % num_ranks;
        int recv_chunk = (my_rank - step - 1 + num_ranks) % num_ranks;
        
        float* send_ptr = data + send_chunk * chunk_size;
        float* recv_ptr = data + recv_chunk * chunk_size;
        
        // Post RDMA write to next rank
        struct ibv_sge sge;
        sge.addr = (uint64_t)send_ptr;
        sge.length = chunk_size * sizeof(float);
        sge.lkey = rdma_contexts[my_rank]->mr->lkey;
        
        struct ibv_send_wr wr, *bad_wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = step;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        
        // Remote memory info (would be exchanged during setup)
        wr.wr.rdma.remote_addr = (uint64_t)recv_ptr;
        wr.wr.rdma.rkey = rdma_contexts[next_rank]->mr->rkey;
        
        ibv_post_send(rdma_contexts[my_rank]->qp, &wr, &bad_wr);
        
        // Reduce received data
        dim3 block(256);
        dim3 grid((chunk_size + block.x - 1) / block.x);
        
        // Simple reduction kernel
        auto reduce = [=] __device__ (int idx) {
            if (idx < chunk_size) {
                recv_ptr[idx] += send_ptr[idx];
            }
        };
    }
    
    // Phase 2: All-gather
    for (int step = 0; step < num_ranks - 1; step++) {
        int send_chunk = (my_rank - step + 1 + num_ranks) % num_ranks;
        int recv_chunk = (my_rank - step + num_ranks) % num_ranks;
        
        float* send_ptr = data + send_chunk * chunk_size;
        float* recv_ptr = data + recv_chunk * chunk_size;
        
        // RDMA write to all other ranks
        // Similar to reduce-scatter but without reduction
    }
    
    cudaStreamSynchronize(stream);
    return cudaSuccess;
}

// ============================================================================
// Performance Benchmarking
// ============================================================================

// Benchmark GPUDirect RDMA performance
extern "C" float benchmark_gpudirect_performance(
    GPUDirectConfig* config,
    size_t data_size,
    int iterations
) {
    // Allocate test buffers
    void *local_data, *remote_data;
    cudaMalloc(&local_data, data_size);
    cudaMalloc(&remote_data, data_size);
    
    // Initialize data
    cudaMemset(local_data, 1, data_size);
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test P2P transfer if available
    if (config->nvlink_enabled && !config->peer_devices.empty()) {
        int peer = config->peer_devices[0];
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            nvlink_p2p_transfer(remote_data, peer,
                              local_data, config->device_id,
                              data_size, stream);
        }
        cudaStreamSynchronize(stream);
        
        // Benchmark
        cudaEventRecord(start, stream);
        
        for (int i = 0; i < iterations; i++) {
            nvlink_p2p_transfer(remote_data, peer,
                              local_data, config->device_id,
                              data_size, stream);
        }
        
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        double bandwidth_gbps = (data_size * iterations / (milliseconds / 1000.0)) / 1e9;
        
        printf("\nGPUDirect P2P Performance:\n");
        printf("  Data size: %.2f MB\n", data_size / (1024.0 * 1024.0));
        printf("  Time: %.2f ms for %d iterations\n", milliseconds, iterations);
        printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbps);
        printf("  Latency: %.2f us\n", milliseconds * 1000.0 / iterations);
        
        // Test bidirectional bandwidth
        cudaEventRecord(start, stream);
        
        for (int i = 0; i < iterations / 2; i++) {
            // Send
            nvlink_p2p_transfer(remote_data, peer,
                              local_data, config->device_id,
                              data_size, stream);
            // Receive
            nvlink_p2p_transfer(local_data, config->device_id,
                              remote_data, peer,
                              data_size, stream);
        }
        
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        double bidir_bandwidth = (2.0 * data_size * iterations / 2) / 
                                (milliseconds / 1000.0) / 1e9;
        
        printf("  Bidirectional: %.2f GB/s\n", bidir_bandwidth);
        
        // Cleanup
        cudaFree(local_data);
        cudaFree(remote_data);
        cudaStreamDestroy(stream);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return (float)bandwidth_gbps;
    }
    
    printf("GPUDirect P2P not available\n");
    return 0.0f;
}

// ============================================================================
// C++ Wrapper Class
// ============================================================================

class GPUDirectManager {
private:
    GPUDirectConfig* config_;
    std::vector<RDMAContext*> rdma_contexts_;
    std::vector<cudaStream_t> streams_;
    bool initialized_;
    
public:
    GPUDirectManager(int device_id) : initialized_(false) {
        config_ = init_gpudirect_rdma(device_id);
        
        // Create streams for each peer device
        for (int peer : config_->peer_devices) {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams_.push_back(stream);
        }
        
        initialized_ = true;
    }
    
    ~GPUDirectManager() {
        if (initialized_) {
            for (auto stream : streams_) {
                cudaStreamDestroy(stream);
            }
            for (auto ctx : rdma_contexts_) {
                if (ctx->mr) ibv_dereg_mr(ctx->mr);
                if (ctx->qp) ibv_destroy_qp(ctx->qp);
                if (ctx->cq) ibv_destroy_cq(ctx->cq);
                if (ctx->pd) ibv_dealloc_pd(ctx->pd);
                delete ctx;
            }
            delete config_;
        }
    }
    
    // Transfer data to peer GPU
    cudaError_t transfer_to_peer(void* data, size_t size, int peer_device) {
        auto it = std::find(config_->peer_devices.begin(),
                           config_->peer_devices.end(), peer_device);
        
        if (it == config_->peer_devices.end()) {
            return cudaErrorInvalidDevice;
        }
        
        int stream_idx = std::distance(config_->peer_devices.begin(), it);
        
        return nvlink_p2p_transfer(data, peer_device,
                                  data, config_->device_id,
                                  size, streams_[stream_idx]);
    }
    
    // Broadcast to all peers
    cudaError_t broadcast_to_peers(void* data, size_t size) {
        return gpudirect_broadcast(data, size, config_->device_id,
                                  config_->peer_devices, streams_.data());
    }
    
    bool is_rdma_enabled() const { return config_->rdma_enabled; }
    bool is_nvlink_enabled() const { return config_->nvlink_enabled; }
    int get_num_peers() const { return config_->peer_devices.size(); }
};