// GPU Consensus Kernels
// High-performance voting and proposal validation using atomic operations

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

// Consensus rules for validation
struct ConsensusRules {
    uint32_t min_votes;
    uint32_t quorum_percentage;
    uint32_t max_round_time;
    uint32_t max_proposal_value;
};

/**
 * Aggregate votes using atomic operations
 * Each thread processes one vote and atomically increments the corresponding counter
 */
extern "C" __global__ void aggregate_votes_kernel(
    const Vote* votes,
    uint32_t* vote_counts,
    uint32_t num_votes,
    uint32_t num_options
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_votes) return;
    
    // Get the vote
    Vote vote = votes[tid];
    
    // Ensure vote value is within bounds
    if (vote.value < num_options) {
        // Atomically increment the vote count for this option
        atomicAdd(&vote_counts[vote.value], 1);
    }
}

/**
 * Validate proposals according to consensus rules
 * Each thread validates one proposal
 */
extern "C" __global__ void validate_proposals_kernel(
    const Proposal* proposals,
    uint32_t* results,  // 0 = invalid, 1 = valid
    const ConsensusRules* rules,
    uint32_t num_proposals
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_proposals) return;
    
    Proposal proposal = proposals[tid];
    ConsensusRules rule = *rules;
    
    // Validate proposal
    uint32_t is_valid = 1;
    
    // Check proposal value is within limits
    if (proposal.value > rule.max_proposal_value) {
        is_valid = 0;
    }
    
    // Check round is reasonable (not too high)
    if (proposal.round > 1000000) {  // Sanity check
        is_valid = 0;
    }
    
    // Store validation result
    results[tid] = is_valid;
}

/**
 * Fast leader election kernel
 * Uses deterministic algorithm based on round and agent fitness
 */
extern "C" __global__ void elect_leader_kernel(
    const uint64_t* heartbeats,
    const uint32_t* agent_fitness,
    uint32_t* leader_id,
    uint32_t num_agents,
    uint32_t round
) {
    // Use a single thread block for leader election
    if (blockIdx.x > 0) return;
    
    __shared__ uint32_t best_agent;
    __shared__ uint64_t best_score;
    
    if (threadIdx.x == 0) {
        best_agent = 0;
        best_score = 0;
    }
    __syncthreads();
    
    // Each thread checks a subset of agents
    uint32_t agents_per_thread = (num_agents + blockDim.x - 1) / blockDim.x;
    uint32_t start = threadIdx.x * agents_per_thread;
    uint32_t end = min(start + agents_per_thread, num_agents);
    
    uint32_t local_best_agent = 0;
    uint64_t local_best_score = 0;
    
    for (uint32_t i = start; i < end; i++) {
        // Calculate score based on heartbeat recency and fitness
        // Use round as a tie-breaker for determinism
        uint64_t score = heartbeats[i] + (uint64_t)agent_fitness[i] * 1000000 + (round * 31 + i) % 100;
        
        if (score > local_best_score) {
            local_best_score = score;
            local_best_agent = i;
        }
    }
    
    // Reduce within block to find global best
    __syncthreads();
    
    // Simple reduction - in production would use more efficient algorithm
    if (local_best_score > atomicMax((unsigned long long*)&best_score, local_best_score)) {
        atomicExch(&best_agent, local_best_agent);
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        *leader_id = best_agent;
    }
}

/**
 * Check proposal conflicts using parallel comparison
 * Detects if multiple proposals conflict with each other
 */
extern "C" __global__ void check_proposal_conflicts_kernel(
    const Proposal* proposals,
    uint8_t* conflict_matrix,  // NxN matrix where 1 indicates conflict
    uint32_t num_proposals
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_proposals || col >= num_proposals) return;
    
    // Don't check proposal against itself
    if (row == col) {
        conflict_matrix[row * num_proposals + col] = 0;
        return;
    }
    
    Proposal p1 = proposals[row];
    Proposal p2 = proposals[col];
    
    // Check for conflicts
    uint8_t has_conflict = 0;
    
    // Same round but different values = conflict
    if (p1.round == p2.round && p1.value != p2.value) {
        has_conflict = 1;
    }
    
    // Same proposer in same round = conflict
    if (p1.round == p2.round && p1.proposer_id == p2.proposer_id && p1.id != p2.id) {
        has_conflict = 1;
    }
    
    conflict_matrix[row * num_proposals + col] = has_conflict;
}

/**
 * Parallel vote verification
 * Ensures votes are valid and from authorized agents
 */
extern "C" __global__ void verify_votes_kernel(
    const Vote* votes,
    const uint8_t* agent_authorized,  // Bitmap of authorized agents
    uint8_t* vote_valid,             // Output: 1 if vote is valid
    uint32_t num_votes,
    uint32_t num_agents
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_votes) return;
    
    Vote vote = votes[tid];
    uint8_t is_valid = 1;
    
    // Check agent ID is within bounds
    if (vote.agent_id >= num_agents) {
        is_valid = 0;
    } else {
        // Check if agent is authorized
        uint32_t byte_idx = vote.agent_id / 8;
        uint32_t bit_idx = vote.agent_id % 8;
        uint8_t is_authorized = (agent_authorized[byte_idx] >> bit_idx) & 1;
        
        if (!is_authorized) {
            is_valid = 0;
        }
    }
    
    // Check timestamp is reasonable (not in future, not too old)
    // In production, would compare against current GPU time
    
    vote_valid[tid] = is_valid;
}