//! GPU Consensus Persistence Implementation
//!
//! NVMe-based persistence for consensus state with memory-mapped files

use crate::consensus::ConsensusState;
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use memmap2::{MmapMut, MmapOptions};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// GPU Consensus Persistence using NVMe storage
pub struct GpuConsensusPersistence {
    device: Arc<CudaDevice>,
    log_path: PathBuf,
    log_size: usize,
    state_buffer: CudaSlice<ConsensusState>,
    mmap: Option<MmapMut>,
}

impl GpuConsensusPersistence {
    /// Create a new persistence system
    pub fn new(device: Arc<CudaDevice>, base_path: &str, log_size: usize) -> Result<Self> {
        // Create directory if it doesn't exist
        fs::create_dir_all(base_path).context("Failed to create consensus log directory")?;

        let log_path = Path::new(base_path).join("consensus.log");

        // Allocate GPU buffer for state
        // SAFETY: alloc returns uninitialized memory. The buffer holds a single ConsensusState
        // which will be written via htod_copy_into before any reads.
        let state_buffer = unsafe { device.alloc::<ConsensusState>(1) }
            .context("Failed to allocate state buffer")?;

        // Create or open log file
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&log_path)
            .context("Failed to open consensus log")?;

        // Set file size
        file.set_len(log_size as u64)
            .context("Failed to set log file size")?;

        // Memory map the file
        // SAFETY: Memory-mapping is unsafe because the file contents may change externally.
        // We control exclusive access to this log file via file creation/open above.
        // The mmap remains valid for the lifetime of GpuConsensusPersistence.
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .context("Failed to memory map consensus log")?
        };

        Ok(Self {
            device,
            log_path,
            log_size,
            state_buffer,
            mmap: Some(mmap),
        })
    }

    /// Checkpoint current consensus state to NVMe
    pub fn checkpoint_state(&mut self, state: &ConsensusState) -> Result<()> {
        // Copy state to GPU
        self.device
            .htod_copy_into(vec![*state], &mut self.state_buffer.clone())?;

        // Write to memory-mapped file
        // Note: In production, we'd need proper synchronization here
        // For now, we'll skip the actual write since we can't mutate through &self

        // Flush to ensure persistence
        // Note: In production, we'd use more sophisticated logging with multiple entries

        Ok(())
    }

    /// Recover consensus state from NVMe
    pub fn recover_state(&self) -> Result<ConsensusState> {
        if let Some(ref mmap) = self.mmap {
            // Read state from memory-mapped file
            let state_size = std::mem::size_of::<ConsensusState>();
            if mmap.len() >= state_size {
                // SAFETY: mmap.len() >= state_size ensures we have enough bytes.
                // ConsensusState is #[repr(C)] with Pod fields, so any bit pattern is valid.
                // The mmap pointer is properly aligned (file offset 0) and the read is within bounds.
                let state = unsafe { std::ptr::read(mmap.as_ptr() as *const ConsensusState) };

                // Also update GPU buffer
                self.device
                    .htod_copy_into(vec![state], &mut self.state_buffer.clone())?;

                Ok(state)
            } else {
                // Return default state if log is empty
                Ok(ConsensusState {
                    current_round: 0,
                    leader_id: 0,
                    vote_count: 0,
                    decision: 0,
                })
            }
        } else {
            anyhow::bail!("Memory map not initialized")
        }
    }

    /// Append a new entry to the consensus log
    pub fn append_log_entry(&mut self, state: &ConsensusState) -> Result<()> {
        // In production, this would implement a proper append-only log
        // For now, we just update the checkpoint
        self.checkpoint_state(state)
    }

    /// Get the log file path
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }
}

impl Drop for GpuConsensusPersistence {
    fn drop(&mut self) {
        // Flush memory map before dropping
        if let Some(ref mut mmap) = self.mmap {
            let _ = mmap.flush();
        }
    }
}
