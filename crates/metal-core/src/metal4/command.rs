//! Metal 4 command encoding.
//!
//! Metal 4 introduces command allocators and improved command buffer
//! management. This module provides abstractions for these features.

use crate::command::{BarrierScope, MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder};
use crate::error::{MetalError, Result};
use crate::metal4::{Metal4Buffer, Metal4Device};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLSize,
};

/// Metal 4 command queue.
///
/// In Metal 4, command queues can be created with command allocators
/// for more efficient command buffer creation.
pub struct Metal4CommandQueue {
    raw: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl Metal4CommandQueue {
    /// Create a new command queue.
    pub fn new(device: &Metal4Device) -> Result<Self> {
        let raw_device = device.raw();

        let queue = raw_device.newCommandQueue().ok_or_else(|| {
            MetalError::creation_failed("Metal4CommandQueue", "Failed to create command queue")
        })?;

        Ok(Self { raw: queue })
    }

    /// Get the raw MTLCommandQueue.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.raw
    }

    /// Create a command buffer using Metal 4 command allocator.
    ///
    /// When Metal 4 is available, this will use the more efficient
    /// command allocator pattern. For now, it falls back to Metal 3 behavior
    /// and records the buffer creation with the allocator for tracking.
    ///
    /// # Arguments
    ///
    /// * `allocator` - The command allocator for this frame
    ///
    /// # Example
    ///
    /// ```ignore
    /// let allocator = Metal4CommandAllocator::new(&device, 3)?;
    /// allocator.reset(); // Start of frame
    /// let cmd = queue.create_command_buffer_with_allocator(&allocator)?;
    /// ```
    pub fn create_command_buffer_with_allocator(
        &self,
        allocator: &super::Metal4CommandAllocator,
    ) -> Result<Metal4CommandBuffer> {
        // Record that we're creating a buffer with this allocator
        allocator.record_buffer_created();

        // When Metal 4 is available:
        // self.raw.commandBufferWithAllocator(allocator.raw())

        // For now, use standard command buffer creation
        self.create_command_buffer()
    }
}

impl MetalCommandQueue for Metal4CommandQueue {
    type CommandBuffer = Metal4CommandBuffer;

    fn create_command_buffer(&self) -> Result<Self::CommandBuffer> {
        let buffer = self.raw.commandBuffer().ok_or_else(|| {
            MetalError::creation_failed("Metal4CommandBuffer", "Failed to create command buffer")
        })?;

        Ok(Metal4CommandBuffer { raw: buffer })
    }

    fn submit(&self, buffer: &mut Self::CommandBuffer) -> Result<()> {
        buffer.raw.commit();
        Ok(())
    }

    fn wait_until_completed(&self) -> Result<()> {
        // Metal command queues don't have a direct wait method
        // The waiting happens on individual command buffers
        Ok(())
    }
}

// Safety: Command queues are thread-safe
unsafe impl Send for Metal4CommandQueue {}
unsafe impl Sync for Metal4CommandQueue {}

/// Metal 4 command buffer.
///
/// Metal 4 command buffers support additional features like
/// GPU-side command generation and improved error reporting.
pub struct Metal4CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl Metal4CommandBuffer {
    /// Get the raw MTLCommandBuffer.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }

    /// Add a completion handler.
    ///
    /// The handler will be called when the command buffer completes execution.
    ///
    /// Note: Due to block2 API constraints, completion handlers are currently
    /// not implemented. This is a placeholder for when macOS 26 ships with
    /// improved block interop.
    ///
    /// # Example
    ///
    /// ```ignore
    /// cmd_buffer.add_completion_handler(|| {
    ///     println!("Command buffer completed!");
    /// });
    /// ```
    pub fn add_completion_handler<F>(&self, _handler: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // TODO: Implement when block2 API allows proper MTLCommandBufferHandler creation
        // The handler signature is: void (^)(id<MTLCommandBuffer>)
        // For now, users should call wait_until_completed() for synchronous completion
    }

    /// Add a scheduled handler.
    ///
    /// The handler will be called when the command buffer is scheduled for execution.
    ///
    /// Note: Currently not implemented due to block2 API constraints.
    pub fn add_scheduled_handler<F>(&self, _handler: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // TODO: Implement when block2 API allows proper MTLCommandBufferHandler creation
    }

    /// Get GPU execution time (Metal 4 enhanced).
    ///
    /// Metal 4 provides more detailed timing information.
    pub fn gpu_execution_time(&self) -> Option<f64> {
        // Requires command buffer to be completed
        let start = self.raw.GPUStartTime();
        let end = self.raw.GPUEndTime();
        if end > start {
            Some(end - start)
        } else {
            None
        }
    }
}

impl MetalCommandBuffer for Metal4CommandBuffer {
    type ComputeEncoder<'a> = Metal4ComputeEncoder<'a>;

    fn compute_encoder(&mut self) -> Result<Self::ComputeEncoder<'_>> {
        let encoder = self.raw.computeCommandEncoder().ok_or_else(|| {
            MetalError::creation_failed("Metal4ComputeEncoder", "Failed to create compute encoder")
        })?;

        Ok(Metal4ComputeEncoder {
            raw: encoder,
            _marker: std::marker::PhantomData,
        })
    }

    fn commit(&mut self) -> Result<()> {
        self.raw.commit();
        Ok(())
    }

    fn wait_until_completed(&self) -> Result<()> {
        self.raw.waitUntilCompleted();
        Ok(())
    }

    fn is_completed(&self) -> bool {
        self.raw.status() == MTLCommandBufferStatus::Completed
    }
}

// Safety: Command buffers are thread-safe
unsafe impl Send for Metal4CommandBuffer {}
unsafe impl Sync for Metal4CommandBuffer {}

/// Metal 4 compute encoder.
///
/// Metal 4 compute encoders can use MTL4ArgumentTable for
/// more efficient buffer binding.
pub struct Metal4ComputeEncoder<'a> {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Metal4ComputeEncoder<'a> {
    /// Get the raw MTLComputeCommandEncoder.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        &self.raw
    }

    /// Set an argument table (Metal 4 feature).
    ///
    /// This is more efficient than individual setBuffer calls.
    /// Falls back to no-op until Metal 4 is available.
    pub fn set_argument_table(&self, _table: &super::Metal4ArgumentTable) {
        // TODO: Implement when Metal 4 APIs are available
        // Will use encoder.setArgumentTable() or similar
    }

    /// Use a residency set for this encoding (Metal 4 feature).
    ///
    /// Ensures all buffers in the residency set are GPU-resident.
    pub fn use_residency_set(&self, _set: &super::Metal4ResidencySet) {
        // TODO: Implement when Metal 4 APIs are available
        // Will use encoder.useResidencySet() or similar
    }

    /// Bind a Metal4Buffer at the given index.
    pub fn set_metal4_buffer(&mut self, index: u32, buffer: &Metal4Buffer) {
        // Safety: buffer.raw() returns a valid MTLBuffer
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.raw()), 0, index as usize);
        }
    }

    /// Bind a Metal4Buffer with offset at the given index.
    pub fn set_metal4_buffer_with_offset(&mut self, index: u32, buffer: &Metal4Buffer, offset: usize) {
        // Safety: buffer.raw() returns a valid MTLBuffer
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.raw()), offset, index as usize);
        }
    }
}

impl<'a> MetalComputeEncoder<'a> for Metal4ComputeEncoder<'a> {
    type Buffer = Metal4Buffer;
    type Pipeline = crate::metal3::Metal3ComputePipeline;

    fn set_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<()> {
        self.raw.setComputePipelineState(pipeline.raw());
        Ok(())
    }

    fn set_buffer(&mut self, index: u32, buffer: &Self::Buffer, offset: usize) -> Result<()> {
        // Safety: buffer.raw() returns a valid MTLBuffer
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.raw()), offset, index as usize);
        }
        Ok(())
    }

    fn set_bytes(&mut self, index: u32, data: &[u8]) -> Result<()> {
        if let Some(ptr) = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void) {
            // Safety: ptr is a valid NonNull pointer to the data slice
            unsafe {
                self.raw.setBytes_length_atIndex(ptr, data.len(), index as usize);
            }
        }
        Ok(())
    }

    fn dispatch_threads(&mut self, threads: u64) -> Result<()> {
        let grid = MTLSize {
            width: threads as usize,
            height: 1,
            depth: 1,
        };
        // Use 256 threads per group as default
        let group_size = MTLSize {
            width: 256.min(threads as usize),
            height: 1,
            depth: 1,
        };
        self.raw
            .dispatchThreads_threadsPerThreadgroup(grid, group_size);
        Ok(())
    }

    fn dispatch_threads_3d(
        &mut self,
        threads: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()> {
        let grid = MTLSize {
            width: threads.0 as usize,
            height: threads.1 as usize,
            depth: threads.2 as usize,
        };
        let group = MTLSize {
            width: threads_per_threadgroup.0 as usize,
            height: threads_per_threadgroup.1 as usize,
            depth: threads_per_threadgroup.2 as usize,
        };
        self.raw.dispatchThreads_threadsPerThreadgroup(grid, group);
        Ok(())
    }

    fn dispatch_threadgroups(
        &mut self,
        threadgroups: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()> {
        let groups = MTLSize {
            width: threadgroups.0 as usize,
            height: threadgroups.1 as usize,
            depth: threadgroups.2 as usize,
        };
        let threads = MTLSize {
            width: threads_per_threadgroup.0 as usize,
            height: threads_per_threadgroup.1 as usize,
            depth: threads_per_threadgroup.2 as usize,
        };
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        Ok(())
    }

    fn memory_barrier(&mut self, scope: BarrierScope) -> Result<()> {
        let mtl_scope = match scope {
            BarrierScope::Buffers => MTLBarrierScope::Buffers,
            BarrierScope::Textures => MTLBarrierScope::Textures,
            BarrierScope::All => MTLBarrierScope::Buffers | MTLBarrierScope::Textures,
        };

        self.raw.memoryBarrierWithScope(mtl_scope);
        Ok(())
    }

    fn end_encoding(&mut self) -> Result<()> {
        self.raw.endEncoding();
        Ok(())
    }
}

// Safety: Compute encoders should only be used from one thread at a time
// but can be sent between threads
unsafe impl Send for Metal4ComputeEncoder<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_queue_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let queue = Metal4CommandQueue::new(&device).expect("Failed to create queue");
            let _buffer = queue
                .create_command_buffer()
                .expect("Failed to create command buffer");
            // Just verify we can create these objects
        }
    }

    #[test]
    fn test_compute_encoder_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let queue = Metal4CommandQueue::new(&device).expect("Failed to create queue");
            let mut cmd_buffer = queue
                .create_command_buffer()
                .expect("Failed to create command buffer");
            let mut encoder = cmd_buffer
                .compute_encoder()
                .expect("Failed to create encoder");
            encoder.end_encoding().expect("Failed to end encoding");
        }
    }
}
