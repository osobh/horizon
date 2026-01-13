//! GPU Consensus Module
//!
//! High-performance consensus implementation using CUDA for distributed GPU agents.
//! Achieves <100μs consensus latency through atomic operations and GPU parallelism.

pub mod leader;
pub mod multi_gpu;
pub mod persistence;
pub mod scale;
pub mod voting;

#[cfg(test)]
mod tests;

use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig};
use std::sync::Arc;

// Define key types here instead of importing from tests
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vote {
    pub agent_id: u32,
    pub proposal_id: u32,
    pub value: u32,
    pub timestamp: u64,
}

// Make Vote safe to use with GPU
// SAFETY: Vote is #[repr(C)] with only primitive types (u32, u64).
// All bit patterns are valid and fields are naturally aligned.
unsafe impl bytemuck::Pod for Vote {}
// SAFETY: Zero-initialization is safe for all numeric primitive types.
unsafe impl bytemuck::Zeroable for Vote {}
// SAFETY: All-zeros is a valid Vote value (agent_id=0, proposal_id=0, value=0, timestamp=0).
unsafe impl cudarc::driver::ValidAsZeroBits for Vote {}

// SAFETY: Vote has a stable C ABI layout (#[repr(C)]) with only types safe
// to pass across the FFI boundary to CUDA kernels. Copy is required by DeviceRepr.
unsafe impl cudarc::driver::DeviceRepr for Vote {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proposal {
    pub id: u32,
    pub proposer_id: u32,
    pub value: u32,
    pub round: u32,
}

// SAFETY: Proposal is #[repr(C)] with only u32 fields. All bit patterns valid.
unsafe impl bytemuck::Pod for Proposal {}
// SAFETY: Zero-initialization is safe for u32 types.
unsafe impl bytemuck::Zeroable for Proposal {}

// SAFETY: Proposal has a stable C ABI layout (#[repr(C)]) matching the CUDA
// Proposal struct. Copy is required by DeviceRepr.
unsafe impl cudarc::driver::DeviceRepr for Proposal {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConsensusState {
    pub current_round: u32,
    pub leader_id: u32,
    pub vote_count: u32,
    pub decision: u32,
}

// SAFETY: ConsensusState is #[repr(C)] with only u32 fields. All bit patterns valid.
unsafe impl bytemuck::Pod for ConsensusState {}
// SAFETY: Zero-initialization is safe for u32 types.
unsafe impl bytemuck::Zeroable for ConsensusState {}

// SAFETY: ConsensusState has a stable C ABI layout (#[repr(C)]) matching the CUDA
// ConsensusState struct. Copy is required by DeviceRepr.
unsafe impl cudarc::driver::DeviceRepr for ConsensusState {}

/// Main GPU Consensus Module
pub struct GpuConsensusModule {
    device: Arc<CudaContext>,
    voting: voting::GpuVoting,
    leader: leader::GpuLeaderElection,
    persistence: persistence::GpuConsensusPersistence,
    num_agents: usize,
}

impl GpuConsensusModule {
    /// Create a new GPU consensus module
    pub fn new(device: Arc<CudaContext>, num_agents: usize) -> Result<Self> {
        let voting = voting::GpuVoting::new(Arc::clone(&device), num_agents)?;
        let leader = leader::GpuLeaderElection::new(Arc::clone(&device), num_agents)?;

        // Use default path for persistence
        let persistence = persistence::GpuConsensusPersistence::new(
            Arc::clone(&device),
            "/tmp/gpu_consensus", // Will be /magikdev/gpu/consensus/ in production
            10 * 1024 * 1024,     // 10MB log
        )?;

        Ok(Self {
            device,
            voting,
            leader,
            persistence,
            num_agents,
        })
    }

    /// Run a complete consensus round
    pub fn run_consensus_round(
        &mut self,
        proposal: Proposal,
        votes: &[Vote],
    ) -> Result<ConsensusState> {
        // 1. Elect leader for this round
        let leader_id = self.leader.elect_leader(proposal.round)?;

        // 2. Aggregate votes
        let vote_counts = self.voting.aggregate_votes(votes, 2)?; // Binary vote: yes/no

        // 3. Determine decision
        let yes_votes = vote_counts[1];
        let no_votes = vote_counts[0];
        let decision = if yes_votes > no_votes { 1 } else { 0 };

        // 4. Create consensus state
        let state = ConsensusState {
            current_round: proposal.round,
            leader_id,
            vote_count: votes.len() as u32,
            decision,
        };

        // 5. Persist state
        self.persistence.checkpoint_state(&state)?;

        Ok(state)
    }
}

// External C functions for CUDA kernels
unsafe extern "C" {
    pub fn launch_aggregate_votes(
        votes: *const Vote,
        vote_counts: *mut u32,
        num_votes: u32,
        num_options: u32,
    );

    pub fn launch_validate_proposals(
        proposals: *const Proposal,
        results: *mut u32,
        rules: *const u32,
        num_proposals: u32,
    );
}

/// Launch configuration for consensus kernels
pub fn get_consensus_launch_config(num_elements: usize) -> LaunchConfig {
    const BLOCK_SIZE: u32 = 256;
    let num_blocks = ((num_elements as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);

    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}
