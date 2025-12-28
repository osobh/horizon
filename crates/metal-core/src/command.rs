//! Command encoding and submission.
//!
//! Metal uses command buffers to batch GPU work and command
//! queues to submit them for execution.

use crate::buffer::MetalBuffer;
use crate::compute::MetalComputePipeline;
use crate::error::Result;

/// Trait for Metal command queues.
///
/// Command queues manage the submission of command buffers
/// to the GPU for execution.
pub trait MetalCommandQueue: Send + Sync {
    /// Command buffer type for this queue.
    type CommandBuffer: MetalCommandBuffer;

    /// Create a new command buffer.
    fn create_command_buffer(&self) -> Result<Self::CommandBuffer>;

    /// Submit a command buffer for execution.
    ///
    /// The buffer is committed and scheduled for execution.
    /// Use `wait_until_completed` to block until done.
    fn submit(&self, buffer: &mut Self::CommandBuffer) -> Result<()>;

    /// Submit and wait for completion.
    fn submit_and_wait(&self, mut buffer: Self::CommandBuffer) -> Result<()> {
        self.submit(&mut buffer)?;
        buffer.wait_until_completed()
    }

    /// Wait for all submitted work to complete.
    fn wait_until_completed(&self) -> Result<()>;
}

/// Trait for Metal command buffers.
///
/// Command buffers contain encoded GPU commands that can be
/// submitted to a command queue.
pub trait MetalCommandBuffer: Send {
    /// Compute encoder type for this buffer.
    type ComputeEncoder<'a>: MetalComputeEncoder<'a>
    where
        Self: 'a;

    /// Create a compute command encoder.
    ///
    /// The encoder is used to record compute commands.
    fn compute_encoder(&mut self) -> Result<Self::ComputeEncoder<'_>>;

    /// Encode compute commands using a closure.
    fn encode_compute<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self::ComputeEncoder<'_>) -> Result<R>,
    {
        let mut encoder = self.compute_encoder()?;
        let result = f(&mut encoder)?;
        encoder.end_encoding()?;
        Ok(result)
    }

    /// Commit the command buffer for execution.
    ///
    /// After committing, no more commands can be added.
    fn commit(&mut self) -> Result<()>;

    /// Wait for the command buffer to complete.
    fn wait_until_completed(&self) -> Result<()>;

    /// Check if the command buffer has completed.
    fn is_completed(&self) -> bool;
}

/// Trait for compute command encoders.
///
/// Encoders record compute commands into a command buffer.
pub trait MetalComputeEncoder<'a>: Send {
    /// Buffer type for binding.
    type Buffer: MetalBuffer;
    /// Pipeline type for binding.
    type Pipeline: MetalComputePipeline;

    /// Set the compute pipeline state.
    fn set_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<()>;

    /// Bind a buffer at the given index.
    fn set_buffer(&mut self, index: u32, buffer: &Self::Buffer, offset: usize) -> Result<()>;

    /// Bind a buffer at index 0 with no offset.
    fn set_buffer_simple(&mut self, index: u32, buffer: &Self::Buffer) -> Result<()> {
        self.set_buffer(index, buffer, 0)
    }

    /// Set bytes directly (for small constant data).
    fn set_bytes(&mut self, index: u32, data: &[u8]) -> Result<()>;

    /// Set a typed value as bytes.
    fn set_value<T: bytemuck::Pod>(&mut self, index: u32, value: &T) -> Result<()> {
        self.set_bytes(index, bytemuck::bytes_of(value))
    }

    /// Dispatch compute threads.
    ///
    /// # Arguments
    /// * `threads` - Total number of threads to dispatch
    fn dispatch_threads(&mut self, threads: u64) -> Result<()>;

    /// Dispatch compute threads with explicit dimensions.
    ///
    /// # Arguments
    /// * `threads` - Thread grid dimensions (x, y, z)
    /// * `threads_per_threadgroup` - Threadgroup dimensions (x, y, z)
    fn dispatch_threads_3d(
        &mut self,
        threads: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()>;

    /// Dispatch threadgroups.
    ///
    /// # Arguments
    /// * `threadgroups` - Number of threadgroups (x, y, z)
    /// * `threads_per_threadgroup` - Threads per threadgroup (x, y, z)
    fn dispatch_threadgroups(
        &mut self,
        threadgroups: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()>;

    /// Insert a memory barrier.
    fn memory_barrier(&mut self, scope: BarrierScope) -> Result<()>;

    /// End encoding.
    fn end_encoding(&mut self) -> Result<()>;
}

/// Scope for memory barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarrierScope {
    /// Barrier for buffer resources.
    #[default]
    Buffers,
    /// Barrier for texture resources.
    Textures,
    /// Barrier for all resources.
    All,
}

/// Thread dispatch configuration.
#[derive(Debug, Clone, Copy)]
pub struct DispatchConfig {
    /// Total threads in X dimension.
    pub threads_x: u64,
    /// Total threads in Y dimension.
    pub threads_y: u64,
    /// Total threads in Z dimension.
    pub threads_z: u64,
    /// Threads per threadgroup in X.
    pub group_x: u64,
    /// Threads per threadgroup in Y.
    pub group_y: u64,
    /// Threads per threadgroup in Z.
    pub group_z: u64,
}

impl DispatchConfig {
    /// Create a 1D dispatch configuration.
    pub fn new_1d(total_threads: u64, threads_per_group: u64) -> Self {
        Self {
            threads_x: total_threads,
            threads_y: 1,
            threads_z: 1,
            group_x: threads_per_group,
            group_y: 1,
            group_z: 1,
        }
    }

    /// Create a 2D dispatch configuration.
    pub fn new_2d(threads: (u64, u64), group: (u64, u64)) -> Self {
        Self {
            threads_x: threads.0,
            threads_y: threads.1,
            threads_z: 1,
            group_x: group.0,
            group_y: group.1,
            group_z: 1,
        }
    }

    /// Create a 3D dispatch configuration.
    pub fn new_3d(threads: (u64, u64, u64), group: (u64, u64, u64)) -> Self {
        Self {
            threads_x: threads.0,
            threads_y: threads.1,
            threads_z: threads.2,
            group_x: group.0,
            group_y: group.1,
            group_z: group.2,
        }
    }

    /// Calculate the number of threadgroups needed.
    pub fn threadgroups(&self) -> (u64, u64, u64) {
        (
            (self.threads_x + self.group_x - 1) / self.group_x,
            (self.threads_y + self.group_y - 1) / self.group_y,
            (self.threads_z + self.group_z - 1) / self.group_z,
        )
    }
}
