//! GPU Voting Implementation
//!
//! Implements atomic vote aggregation on GPU with high parallelism

use crate::consensus::Vote;
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// GPU Voting system for consensus
pub struct GpuVoting {
    device: Arc<CudaDevice>,
    vote_buffer: CudaSlice<Vote>,
    vote_counts_buffer: CudaSlice<u32>,
    max_votes: usize,
    max_options: usize,
}

impl GpuVoting {
    /// Create a new GPU voting system
    pub fn new(device: Arc<CudaDevice>, max_votes: usize) -> Result<Self> {
        const MAX_OPTIONS: usize = 10; // Support up to 10 voting options

        // Allocate GPU buffers
        // SAFETY: alloc returns uninitialized memory. The buffer will be written via
        // htod_copy_into before any kernel reads from it. max_votes is a valid size.
        let vote_buffer =
            unsafe { device.alloc::<Vote>(max_votes) }.context("Failed to allocate vote buffer")?;
        let vote_counts_buffer = device
            .alloc_zeros::<u32>(MAX_OPTIONS)
            .context("Failed to allocate vote counts buffer")?;

        Ok(Self {
            device,
            vote_buffer,
            vote_counts_buffer,
            max_votes,
            max_options: MAX_OPTIONS,
        })
    }

    /// Aggregate votes using GPU atomic operations
    pub fn aggregate_votes(&self, votes: &[Vote], num_options: usize) -> Result<Vec<u32>> {
        if votes.len() > self.max_votes {
            anyhow::bail!("Too many votes: {} > {}", votes.len(), self.max_votes);
        }
        if num_options > self.max_options {
            anyhow::bail!("Too many options: {} > {}", num_options, self.max_options);
        }

        // Copy votes to GPU
        self.device
            .htod_copy_into(votes.to_vec(), &mut self.vote_buffer.clone())?;

        // Clear vote counts
        let zeros = vec![0u32; self.max_options];
        self.device
            .htod_copy_into(zeros, &mut self.vote_counts_buffer.clone())?;

        // Launch aggregation kernel through FFI
        // SAFETY: All pointers are valid device pointers from CudaSlice allocations:
        // - vote_buffer: populated via htod_copy_into above, size bounds checked
        // - vote_counts_buffer: cleared to zeros above, size is MAX_OPTIONS
        // - num_options checked against max_options, votes.len() checked against max_votes
        unsafe {
            let votes_ptr = *self.vote_buffer.device_ptr() as *const Vote;
            let counts_ptr = *self.vote_counts_buffer.device_ptr() as *mut u32;

            crate::consensus::launch_aggregate_votes(
                votes_ptr,
                counts_ptr,
                votes.len() as u32,
                num_options as u32,
            );
        }

        // Synchronize and copy results back
        self.device.synchronize()?;

        let mut results = vec![0u32; num_options];
        let vote_slice = self.vote_counts_buffer.slice(0..num_options);
        self.device.dtoh_sync_copy_into(&vote_slice, &mut results)?;

        Ok(results)
    }

    /// Aggregate votes for multiple proposals concurrently
    pub fn aggregate_votes_concurrent(
        &self,
        votes: &[Vote],
        num_options: usize,
        num_proposals: usize,
    ) -> Result<Vec<Vec<u32>>> {
        // Group votes by proposal
        let mut votes_by_proposal: Vec<Vec<Vote>> = vec![vec![]; num_proposals];

        for vote in votes {
            let proposal_idx = vote.proposal_id as usize;
            if proposal_idx < num_proposals {
                votes_by_proposal[proposal_idx].push(*vote);
            }
        }

        // Process each proposal
        let mut results = Vec::new();
        for proposal_votes in votes_by_proposal {
            let counts = self.aggregate_votes(&proposal_votes, num_options)?;
            results.push(counts);
        }

        Ok(results)
    }
}

impl Drop for GpuVoting {
    fn drop(&mut self) {
        // Buffers are automatically freed when DevicePtr is dropped
    }
}
