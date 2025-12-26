//! C++ FFI wrapper for consensus CUDA kernels
//! 
//! This file provides the interface between Rust and CUDA kernels

#include <cuda_runtime.h>
#include <cstdint>

// Vote structure matching Rust definition
struct Vote {
    uint32_t agent_id;
    uint32_t proposal_id;
    uint32_t value;
    uint64_t timestamp;
};

// Proposal structure matching Rust definition
struct Proposal {
    uint32_t id;
    uint32_t proposer_id;
    uint32_t value;
    uint32_t round;
};

// Consensus rules structure
struct ConsensusRules {
    uint32_t min_votes;
    uint32_t quorum_percentage;
    uint32_t max_round_time;
    uint32_t max_proposal_value;
};

// External kernel declarations (from consensus_kernel.cu)
extern "C" {
    // Vote aggregation kernel
    __global__ void aggregate_votes_kernel(
        const Vote* votes,
        uint32_t* vote_counts,
        uint32_t num_votes,
        uint32_t num_options
    );
    
    // Proposal validation kernel  
    __global__ void validate_proposals_kernel(
        const Proposal* proposals,
        uint32_t* results,
        const ConsensusRules* rules,
        uint32_t num_proposals
    );
    
    // Leader election kernel
    __global__ void elect_leader_kernel(
        const uint64_t* heartbeats,
        const uint32_t* agent_fitness,
        uint32_t* leader_id,
        uint32_t num_agents,
        uint32_t round
    );
    
    // Proposal conflict checking kernel
    __global__ void check_proposal_conflicts_kernel(
        const Proposal* proposals,
        uint8_t* conflict_matrix,
        uint32_t num_proposals
    );
    
    // Vote verification kernel
    __global__ void verify_votes_kernel(
        const Vote* votes,
        const uint8_t* agent_authorized,
        uint8_t* vote_valid,
        uint32_t num_votes,
        uint32_t num_agents
    );
}

// Helper function to calculate launch configuration
inline dim3 calculate_grid_dim(uint32_t num_threads, uint32_t block_size) {
    uint32_t num_blocks = (num_threads + block_size - 1) / block_size;
    return dim3(num_blocks, 1, 1);
}

// FFI wrapper for vote aggregation
extern "C" void launch_aggregate_votes(
    const Vote* votes,
    uint32_t* vote_counts,
    uint32_t num_votes,
    uint32_t num_options
) {
    const uint32_t BLOCK_SIZE = 256;
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim = calculate_grid_dim(num_votes, BLOCK_SIZE);
    
    // Launch kernel synchronously
    aggregate_votes_kernel<<<grid_dim, block_dim>>>(
        votes, vote_counts, num_votes, num_options
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
}

// FFI wrapper for proposal validation
extern "C" void launch_validate_proposals(
    const Proposal* proposals,
    uint32_t* results,
    const ConsensusRules* rules,
    uint32_t num_proposals
) {
    const uint32_t BLOCK_SIZE = 256;
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim = calculate_grid_dim(num_proposals, BLOCK_SIZE);
    
    // Launch kernel synchronously
    validate_proposals_kernel<<<grid_dim, block_dim>>>(
        proposals, results, rules, num_proposals
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
}

// FFI wrapper for leader election
extern "C" void launch_elect_leader(
    const uint64_t* heartbeats,
    const uint32_t* agent_fitness,
    uint32_t* leader_id,
    uint32_t num_agents,
    uint32_t round
) {
    // Use single block for leader election to avoid coordination overhead
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim(1, 1, 1);
    
    // Launch kernel synchronously
    elect_leader_kernel<<<grid_dim, block_dim>>>(
        heartbeats, agent_fitness, leader_id, num_agents, round
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
}

// FFI wrapper for proposal conflict checking
extern "C" void launch_check_proposal_conflicts(
    const Proposal* proposals,
    uint8_t* conflict_matrix,
    uint32_t num_proposals
) {
    // Use 2D grid for N x N conflict matrix
    const uint32_t BLOCK_SIZE = 16; // 16x16 threads per block
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    uint32_t num_blocks_x = (num_proposals + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t num_blocks_y = (num_proposals + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    
    // Launch conflict checking kernel
    
    check_proposal_conflicts_kernel<<<grid_dim, block_dim>>>(
        proposals, conflict_matrix, num_proposals
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
}

// FFI wrapper for vote verification
extern "C" void launch_verify_votes(
    const Vote* votes,
    const uint8_t* agent_authorized,
    uint8_t* vote_valid,
    uint32_t num_votes,
    uint32_t num_agents
) {
    const uint32_t BLOCK_SIZE = 256;
    dim3 block_dim(BLOCK_SIZE, 1, 1);
    dim3 grid_dim = calculate_grid_dim(num_votes, BLOCK_SIZE);
    
    // Launch vote verification kernel
    
    verify_votes_kernel<<<grid_dim, block_dim>>>(
        votes, agent_authorized, vote_valid, num_votes, num_agents
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
}

// Error checking utilities
extern "C" const char* get_last_cuda_error() {
    cudaError_t error = cudaGetLastError();
    return cudaGetErrorString(error);
}

extern "C" bool cuda_check_error() {
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess;
}