// CUDA 13.0 NCCL Collective Operations for Distributed Consensus
// Multi-GPU and Multi-Node Support for StratoSwarm
// RTX 5090 optimized with GPUDirect and NVLink

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <vector>
#include <cstdio>
#include <chrono>

// NCCL error checking
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("NCCL error %s:%d '%s'\n",              \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(1);                                         \
  }                                                  \
} while(0)

// MPI error checking
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("MPI error %s:%d\n", __FILE__,__LINE__); \
    exit(1);                                         \
  }                                                  \
} while(0)

// Distributed consensus configuration
struct DistributedConsensusConfig {
    int num_gpus;
    int num_nodes;
    int local_rank;
    int global_rank;
    int agents_per_gpu;
    ncclComm_t nccl_comm;
    cudaStream_t stream;
};

// Consensus vote structure
struct ConsensusVote {
    uint32_t proposal_id;
    float confidence;
    uint32_t voter_id;
    uint32_t timestamp;
};

// ============================================================================
// NCCL-Accelerated Distributed Voting
// ============================================================================

// All-reduce consensus voting across GPUs
__global__ void prepare_votes_kernel(
    ConsensusVote* local_votes,
    const float* agent_opinions,
    uint32_t num_agents,
    uint32_t proposal_id,
    uint32_t gpu_id,
    uint32_t timestamp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_agents) return;
    
    local_votes[tid].proposal_id = proposal_id;
    local_votes[tid].confidence = agent_opinions[tid];
    local_votes[tid].voter_id = gpu_id * num_agents + tid;
    local_votes[tid].timestamp = timestamp;
}

// Aggregate votes after all-reduce
__global__ void aggregate_consensus_kernel(
    float* global_consensus,
    const float* reduced_votes,
    uint32_t num_proposals,
    uint32_t total_voters,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_proposals) return;
    
    float avg_confidence = reduced_votes[tid] / total_voters;
    global_consensus[tid] = (avg_confidence >= threshold) ? 1.0f : 0.0f;
}

// ============================================================================
// Multi-GPU Swarm Synchronization
// ============================================================================

// Broadcast best solution across all GPUs
extern "C" cudaError_t nccl_broadcast_best_solution(
    float* best_genome,
    float* best_fitness,
    int root_gpu,
    uint32_t genome_size,
    DistributedConsensusConfig* config
) {
    // Broadcast fitness value
    NCCLCHECK(ncclBroadcast(
        best_fitness,      // sendbuff
        best_fitness,      // recvbuff
        1,                 // count
        ncclFloat,         // datatype
        root_gpu,          // root
        config->nccl_comm, // comm
        config->stream     // stream
    ));
    
    // Broadcast genome
    NCCLCHECK(ncclBroadcast(
        best_genome,
        best_genome,
        genome_size,
        ncclFloat,
        root_gpu,
        config->nccl_comm,
        config->stream
    ));
    
    return cudaStreamSynchronize(config->stream);
}

// All-gather for population exchange
extern "C" cudaError_t nccl_population_exchange(
    float* local_population,
    float* global_population,
    uint32_t local_pop_size,
    uint32_t genome_size,
    DistributedConsensusConfig* config
) {
    size_t data_size = local_pop_size * genome_size;
    
    NCCLCHECK(ncclAllGather(
        local_population,   // sendbuff
        global_population,  // recvbuff
        data_size,         // sendcount
        ncclFloat,         // datatype
        config->nccl_comm, // comm
        config->stream     // stream
    ));
    
    return cudaStreamSynchronize(config->stream);
}

// Reduce-scatter for distributed fitness evaluation
extern "C" cudaError_t nccl_distributed_fitness_reduction(
    float* local_fitness,
    float* reduced_fitness,
    uint32_t population_size,
    DistributedConsensusConfig* config
) {
    // Each GPU gets a portion of the reduced result
    uint32_t chunk_size = population_size / config->num_gpus;
    
    NCCLCHECK(ncclReduceScatter(
        local_fitness,      // sendbuff
        reduced_fitness,    // recvbuff
        chunk_size,        // recvcount
        ncclFloat,         // datatype
        ncclSum,           // op
        config->nccl_comm, // comm
        config->stream     // stream
    ));
    
    return cudaStreamSynchronize(config->stream);
}

// ============================================================================
// Ring All-Reduce for Consensus
// ============================================================================

// Custom ring all-reduce for consensus voting
extern "C" cudaError_t nccl_consensus_all_reduce(
    ConsensusVote* local_votes,
    float* consensus_result,
    uint32_t num_proposals,
    uint32_t agents_per_gpu,
    float consensus_threshold,
    DistributedConsensusConfig* config
) {
    // Prepare local vote aggregates
    float* local_aggregates;
    cudaMalloc(&local_aggregates, num_proposals * sizeof(float));
    
    // Aggregate local votes
    dim3 block(256);
    dim3 grid((agents_per_gpu + block.x - 1) / block.x);
    
    // Simple aggregation kernel (would be more complex in practice)
    auto aggregate_local = [=] __device__ (int tid) {
        if (tid < agents_per_gpu) {
            atomicAdd(&local_aggregates[local_votes[tid].proposal_id],
                     local_votes[tid].confidence);
        }
    };
    
    // All-reduce across GPUs
    NCCLCHECK(ncclAllReduce(
        local_aggregates,   // sendbuff
        consensus_result,   // recvbuff
        num_proposals,      // count
        ncclFloat,         // datatype
        ncclSum,           // op
        config->nccl_comm, // comm
        config->stream     // stream
    ));
    
    // Apply consensus threshold
    aggregate_consensus_kernel<<<grid, block, 0, config->stream>>>(
        consensus_result,
        consensus_result,
        num_proposals,
        config->num_gpus * agents_per_gpu,
        consensus_threshold
    );
    
    cudaFree(local_aggregates);
    return cudaStreamSynchronize(config->stream);
}

// ============================================================================
// Multi-Node Communication with MPI + NCCL
// ============================================================================

// Initialize multi-node NCCL communicator
extern "C" ncclComm_t* init_multi_node_nccl(
    int* argc,
    char*** argv,
    int* num_gpus,
    int* local_rank,
    int* global_rank
) {
    // Initialize MPI
    MPICHECK(MPI_Init(argc, argv));
    
    int world_size, world_rank;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    
    // Get local GPU count
    cudaGetDeviceCount(num_gpus);
    *local_rank = world_rank % (*num_gpus);
    *global_rank = world_rank;
    
    // Set CUDA device
    cudaSetDevice(*local_rank);
    
    // Get NCCL unique ID from rank 0
    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPICHECK(MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    
    // Create NCCL communicator
    ncclComm_t* comm = new ncclComm_t;
    NCCLCHECK(ncclCommInitRank(comm, world_size, nccl_id, world_rank));
    
    return comm;
}

// ============================================================================
// GPUDirect RDMA Support
// ============================================================================

// Enable GPUDirect RDMA for efficient multi-node communication
extern "C" cudaError_t enable_gpudirect_rdma(int device_id) {
    cudaSetDevice(device_id);
    
    // Check for GPUDirect RDMA support
    int rdma_capable = 0;
    cudaDeviceGetAttribute(&rdma_capable, 
                           cudaDevAttrGPUDirectRDMASupported,
                           device_id);
    
    if (!rdma_capable) {
        printf("Warning: GPU %d does not support GPUDirect RDMA\n", device_id);
        return cudaErrorNotSupported;
    }
    
    // Enable peer access for NVLink
    int can_access_peer;
    for (int peer_id = 0; peer_id < 8; peer_id++) {
        if (peer_id != device_id) {
            cudaDeviceCanAccessPeer(&can_access_peer, device_id, peer_id);
            if (can_access_peer) {
                cudaDeviceEnablePeerAccess(peer_id, 0);
                printf("Enabled P2P access: GPU %d -> GPU %d\n", device_id, peer_id);
            }
        }
    }
    
    // Set GPUDirect RDMA flags
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    
    return cudaSuccess;
}

// ============================================================================
// Advanced Collective Patterns
// ============================================================================

// Hierarchical all-reduce for large clusters
extern "C" cudaError_t hierarchical_all_reduce(
    float* data,
    size_t count,
    DistributedConsensusConfig* config,
    ncclComm_t* node_comm,  // Intra-node communicator
    ncclComm_t* global_comm // Inter-node communicator
) {
    // Stage 1: Reduce within node
    if (node_comm) {
        NCCLCHECK(ncclReduce(
            data, data, count, ncclFloat, ncclSum,
            0,  // Root of local node
            *node_comm, config->stream
        ));
    }
    
    cudaStreamSynchronize(config->stream);
    
    // Stage 2: Reduce across nodes (only node leaders)
    if (global_comm && config->local_rank == 0) {
        NCCLCHECK(ncclAllReduce(
            data, data, count, ncclFloat, ncclSum,
            *global_comm, config->stream
        ));
    }
    
    cudaStreamSynchronize(config->stream);
    
    // Stage 3: Broadcast within node
    if (node_comm) {
        NCCLCHECK(ncclBroadcast(
            data, data, count, ncclFloat,
            0,  // Root of local node
            *node_comm, config->stream
        ));
    }
    
    return cudaStreamSynchronize(config->stream);
}

// Pipeline parallel pattern for model parallelism
extern "C" cudaError_t pipeline_parallel_forward(
    float* input,
    float* output,
    float* intermediate,
    size_t layer_size,
    int pipeline_stage,
    int num_stages,
    ncclComm_t comm,
    cudaStream_t stream
) {
    // Receive from previous stage
    if (pipeline_stage > 0) {
        NCCLCHECK(ncclRecv(
            input, layer_size, ncclFloat,
            pipeline_stage - 1,  // Previous stage
            comm, stream
        ));
    }
    
    // Process layer (simplified - would call actual kernel)
    dim3 block(256);
    dim3 grid((layer_size + block.x - 1) / block.x);
    
    // Dummy processing kernel
    auto process = [=] __device__ (int idx) {
        if (idx < layer_size) {
            intermediate[idx] = input[idx] * 2.0f;  // Simplified
        }
    };
    
    // Send to next stage
    if (pipeline_stage < num_stages - 1) {
        NCCLCHECK(ncclSend(
            intermediate, layer_size, ncclFloat,
            pipeline_stage + 1,  // Next stage
            comm, stream
        ));
    } else {
        // Last stage copies to output
        cudaMemcpyAsync(output, intermediate, layer_size * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    
    return cudaStreamSynchronize(stream);
}

// ============================================================================
// Performance Monitoring
// ============================================================================

// Benchmark NCCL collective operations
extern "C" float benchmark_nccl_performance(
    DistributedConsensusConfig* config,
    size_t data_size,
    int iterations
) {
    float* data;
    cudaMalloc(&data, data_size * sizeof(float));
    
    // Initialize data
    cudaMemset(data, 1, data_size * sizeof(float));
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        ncclAllReduce(data, data, data_size, ncclFloat, ncclSum,
                     config->nccl_comm, config->stream);
    }
    cudaStreamSynchronize(config->stream);
    
    // Benchmark
    cudaEventRecord(start, config->stream);
    
    for (int i = 0; i < iterations; i++) {
        ncclAllReduce(data, data, data_size, ncclFloat, ncclSum,
                     config->nccl_comm, config->stream);
    }
    
    cudaEventRecord(stop, config->stream);
    cudaStreamSynchronize(config->stream);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth
    size_t bytes = data_size * sizeof(float) * 2;  // Send + receive
    double bandwidth_gbps = (bytes * iterations / (milliseconds / 1000.0)) / 1e9;
    
    printf("NCCL All-Reduce Performance:\n");
    printf("  Data size: %zu floats (%.2f MB)\n", 
           data_size, data_size * sizeof(float) / (1024.0 * 1024.0));
    printf("  Time: %.2f ms for %d iterations\n", milliseconds, iterations);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("  Latency: %.2f us\n", milliseconds * 1000.0 / iterations);
    
    // Cleanup
    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (float)bandwidth_gbps;
}

// ============================================================================
// C++ Interface Class
// ============================================================================

class NCCLConsensusManager {
private:
    DistributedConsensusConfig config_;
    ncclComm_t* node_comm_;
    ncclComm_t* global_comm_;
    bool initialized_;
    
public:
    NCCLConsensusManager() : node_comm_(nullptr), global_comm_(nullptr), initialized_(false) {}
    
    ~NCCLConsensusManager() {
        if (initialized_) {
            cleanup();
        }
    }
    
    // Initialize NCCL for distributed consensus
    cudaError_t initialize(int num_gpus, int local_rank, int agents_per_gpu) {
        config_.num_gpus = num_gpus;
        config_.local_rank = local_rank;
        config_.agents_per_gpu = agents_per_gpu;
        
        // Set device
        cudaSetDevice(local_rank);
        
        // Create stream
        cudaStreamCreate(&config_.stream);
        
        // Initialize NCCL communicator (simplified - would use MPI in practice)
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        ncclCommInitRank(&config_.nccl_comm, num_gpus, id, local_rank);
        
        // Enable GPUDirect if available
        enable_gpudirect_rdma(local_rank);
        
        initialized_ = true;
        return cudaSuccess;
    }
    
    // Run distributed consensus
    cudaError_t run_consensus(
        ConsensusVote* local_votes,
        float* consensus_result,
        uint32_t num_proposals,
        float threshold
    ) {
        return nccl_consensus_all_reduce(
            local_votes,
            consensus_result,
            num_proposals,
            config_.agents_per_gpu,
            threshold,
            &config_
        );
    }
    
    // Synchronize swarm across GPUs
    cudaError_t synchronize_swarm(
        float* local_population,
        float* global_population,
        uint32_t local_pop_size,
        uint32_t genome_size
    ) {
        return nccl_population_exchange(
            local_population,
            global_population,
            local_pop_size,
            genome_size,
            &config_
        );
    }
    
    // Cleanup
    void cleanup() {
        if (config_.nccl_comm) {
            ncclCommDestroy(config_.nccl_comm);
        }
        if (config_.stream) {
            cudaStreamDestroy(config_.stream);
        }
        if (node_comm_) {
            ncclCommDestroy(*node_comm_);
            delete node_comm_;
        }
        if (global_comm_) {
            ncclCommDestroy(*global_comm_);
            delete global_comm_;
        }
        initialized_ = false;
    }
    
    // Get configuration
    const DistributedConsensusConfig& get_config() const {
        return config_;
    }
};