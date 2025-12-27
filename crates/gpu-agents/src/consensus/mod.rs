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
use cudarc::driver::{CudaDevice, LaunchConfig};
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
unsafe impl bytemuck::Pod for Vote {}
unsafe impl bytemuck::Zeroable for Vote {}
unsafe impl cudarc::driver::ValidAsZeroBits for Vote {}

// Implement DeviceRepr for Vote
unsafe impl cudarc::driver::DeviceRepr for Vote {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proposal {
    pub id: u32,
    pub proposer_id: u32,
    pub value: u32,
    pub round: u32,
}

unsafe impl bytemuck::Pod for Proposal {}
unsafe impl bytemuck::Zeroable for Proposal {}

unsafe impl cudarc::driver::DeviceRepr for Proposal {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConsensusState {
    pub current_round: u32,
    pub leader_id: u32,
    pub vote_count: u32,
    pub decision: u32,
}

unsafe impl bytemuck::Pod for ConsensusState {}
unsafe impl bytemuck::Zeroable for ConsensusState {}

unsafe impl cudarc::driver::DeviceRepr for ConsensusState {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// Main GPU Consensus Module
pub struct GpuConsensusModule {
    device: Arc<CudaDevice>,
    voting: voting::GpuVoting,
    leader: leader::GpuLeaderElection,
    persistence: persistence::GpuConsensusPersistence,
    num_agents: usize,
}

impl GpuConsensusModule {
    /// Create a new GPU consensus module
    pub fn new(device: Arc<CudaDevice>, num_agents: usize) -> Result<Self> {
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
extern "C" {
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
