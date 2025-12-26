//! GPU Leader Election Implementation
//!
//! Implements deterministic leader election with heartbeat tracking on GPU

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// GPU Leader Election system
pub struct GpuLeaderElection {
    device: Arc<CudaDevice>,
    heartbeat_buffer: CudaSlice<u64>,
    leader_state_buffer: CudaSlice<u32>,
    num_agents: usize,
}

impl GpuLeaderElection {
    /// Create a new GPU leader election system
    pub fn new(device: Arc<CudaDevice>, num_agents: usize) -> Result<Self> {
        // Allocate GPU buffers
        let heartbeat_buffer = device
            .alloc_zeros::<u64>(num_agents)
            .context("Failed to allocate heartbeat buffer")?;

        // Leader state: [current_leader_id, round, last_update_time]
        let leader_state_buffer = device
            .alloc_zeros::<u32>(3)
            .context("Failed to allocate leader state buffer")?;

        Ok(Self {
            device,
            heartbeat_buffer,
            leader_state_buffer,
            num_agents,
        })
    }

    /// Elect a leader for the given round using deterministic algorithm
    pub fn elect_leader(&self, round: u32) -> Result<u32> {
        // Simple deterministic leader election based on round
        // In production, this would use a more sophisticated algorithm
        let leader_id = (round * 31 + 17) % self.num_agents as u32;

        // Update leader state on GPU
        let state = vec![leader_id, round, 0];
        self.device
            .htod_copy_into(state, &mut self.leader_state_buffer.clone())?;

        Ok(leader_id)
    }

    /// Update heartbeat timestamps for agents
    pub fn update_heartbeats(&self, heartbeats: &[u64]) -> Result<()> {
        if heartbeats.len() != self.num_agents {
            anyhow::bail!(
                "Heartbeat count mismatch: {} != {}",
                heartbeats.len(),
                self.num_agents
            );
        }

        // Copy heartbeats to GPU
        self.device
            .htod_copy_into(heartbeats.to_vec(), &mut self.heartbeat_buffer.clone())?;

        Ok(())
    }

    /// Check leader health and elect new leader if current one failed
    pub fn check_leader_health(&self, current_leader: u32, timeout_threshold: u64) -> Result<u32> {
        // Get heartbeats from GPU
        let mut heartbeats = vec![0u64; self.num_agents];
        self.device
            .dtoh_sync_copy_into(&self.heartbeat_buffer, &mut heartbeats)?;

        // Check if current leader is healthy
        let leader_heartbeat = heartbeats[current_leader as usize];

        if leader_heartbeat < timeout_threshold {
            // Leader has failed, elect new one
            // Find agent with highest heartbeat (most recently active)
            let mut new_leader = 0u32;
            let mut max_heartbeat = 0u64;

            for (idx, &hb) in heartbeats.iter().enumerate() {
                if hb > max_heartbeat && idx != current_leader as usize {
                    max_heartbeat = hb;
                    new_leader = idx as u32;
                }
            }

            // Update leader state
            let mut state = vec![0u32; 3];
            self.device
                .dtoh_sync_copy_into(&self.leader_state_buffer, &mut state)?;
            state[0] = new_leader;
            state[2] = (max_heartbeat & 0xFFFFFFFF) as u32; // Store lower 32 bits
            self.device
                .htod_copy_into(state, &mut self.leader_state_buffer.clone())?;

            Ok(new_leader)
        } else {
            Ok(current_leader)
        }
    }

    /// Get current leader state
    pub fn get_leader_state(&self) -> Result<(u32, u32, u32)> {
        let mut state = vec![0u32; 3];
        self.device
            .dtoh_sync_copy_into(&self.leader_state_buffer, &mut state)?;
        Ok((state[0], state[1], state[2]))
    }
}

impl Drop for GpuLeaderElection {
    fn drop(&mut self) {
        // Buffers are automatically freed when DevicePtr is dropped
    }
}
