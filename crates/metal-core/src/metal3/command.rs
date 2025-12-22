//! Metal 3 command encoding implementation.

use crate::command::{BarrierScope, MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder};
use crate::error::{MetalError, Result};
use crate::metal3::{Metal3Buffer, Metal3ComputePipeline, Metal3Device};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLBarrierScope, MTLSize,
};

use std::marker::PhantomData;

/// Metal 3 command queue.
pub struct Metal3CommandQueue {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

// SAFETY: MTLCommandQueue is thread-safe
unsafe impl Send for Metal3CommandQueue {}
unsafe impl Sync for Metal3CommandQueue {}

impl Metal3CommandQueue {
    /// Create a new command queue.
    pub fn new(device: &Metal3Device) -> Result<Self> {
        let queue = device
            .raw()
            .newCommandQueue()
            .ok_or_else(|| MetalError::creation_failed("command queue", "Failed to create command queue"))?;

        Ok(Self { queue })
    }

    /// Get the raw MTLCommandQueue.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandQueue>> {
        &self.queue
    }
}

impl MetalCommandQueue for Metal3CommandQueue {
    type CommandBuffer = Metal3CommandBuffer;

    fn create_command_buffer(&self) -> Result<Self::CommandBuffer> {
        Metal3CommandBuffer::new(&self.queue)
    }

    fn submit(&self, buffer: &mut Self::CommandBuffer) -> Result<()> {
        buffer.commit()
    }

    fn wait_until_completed(&self) -> Result<()> {
        // Metal 3 doesn't have a direct queue-level wait
        // We rely on individual command buffer waits
        Ok(())
    }
}

/// Metal 3 command buffer.
pub struct Metal3CommandBuffer {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    committed: bool,
}

// SAFETY: MTLCommandBuffer is thread-safe before commit
unsafe impl Send for Metal3CommandBuffer {}

impl Metal3CommandBuffer {
    /// Create a new command buffer.
    pub fn new(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self> {
        let buffer = queue
            .commandBuffer()
            .ok_or_else(|| MetalError::creation_failed("command buffer", "Failed to create command buffer"))?;

        Ok(Self {
            buffer,
            committed: false,
        })
    }

    /// Get the raw MTLCommandBuffer.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        &self.buffer
    }
}

impl MetalCommandBuffer for Metal3CommandBuffer {
    type ComputeEncoder<'a> = Metal3ComputeEncoder<'a>;

    fn compute_encoder(&mut self) -> Result<Self::ComputeEncoder<'_>> {
        let encoder = self
            .buffer
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::creation_failed("compute encoder", "Failed to create compute encoder"))?;

        Ok(Metal3ComputeEncoder {
            encoder,
            _marker: PhantomData,
        })
    }

    fn commit(&mut self) -> Result<()> {
        if !self.committed {
            self.buffer.commit();
            self.committed = true;
        }
        Ok(())
    }

    fn wait_until_completed(&self) -> Result<()> {
        self.buffer.waitUntilCompleted();
        Ok(())
    }

    fn is_completed(&self) -> bool {
        // Check status
        use objc2_metal::MTLCommandBufferStatus;
        let status = self.buffer.status();
        status == MTLCommandBufferStatus::Completed
    }
}

/// Metal 3 compute command encoder.
pub struct Metal3ComputeEncoder<'a> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _marker: PhantomData<&'a ()>,
}

// SAFETY: Encoder is only used from one thread at a time
unsafe impl<'a> Send for Metal3ComputeEncoder<'a> {}

impl<'a> MetalComputeEncoder<'a> for Metal3ComputeEncoder<'a> {
    type Buffer = Metal3Buffer;
    type Pipeline = Metal3ComputePipeline;

    fn set_pipeline(&mut self, pipeline: &Self::Pipeline) -> Result<()> {
        self.encoder.setComputePipelineState(pipeline.raw());
        Ok(())
    }

    fn set_buffer(&mut self, index: u32, buffer: &Self::Buffer, offset: usize) -> Result<()> {
        unsafe {
            self.encoder.setBuffer_offset_atIndex(
                Some(buffer.raw()),
                offset,
                index as usize,
            );
        }
        Ok(())
    }

    fn set_bytes(&mut self, index: u32, data: &[u8]) -> Result<()> {
        use std::ptr::NonNull;

        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .expect("data pointer should not be null");

        unsafe {
            self.encoder.setBytes_length_atIndex(ptr, data.len(), index as usize);
        }
        Ok(())
    }

    fn dispatch_threads(&mut self, threads: u64) -> Result<()> {
        // Use dispatchThreads for non-uniform thread counts
        let thread_size = MTLSize {
            width: threads as usize,
            height: 1,
            depth: 1,
        };

        // Use a reasonable threadgroup size
        let group_size = MTLSize {
            width: (threads.min(256)) as usize,
            height: 1,
            depth: 1,
        };

        self.encoder.dispatchThreads_threadsPerThreadgroup(thread_size, group_size);
        Ok(())
    }

    fn dispatch_threads_3d(
        &mut self,
        threads: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()> {
        let thread_size = MTLSize {
            width: threads.0 as usize,
            height: threads.1 as usize,
            depth: threads.2 as usize,
        };

        let group_size = MTLSize {
            width: threads_per_threadgroup.0 as usize,
            height: threads_per_threadgroup.1 as usize,
            depth: threads_per_threadgroup.2 as usize,
        };

        self.encoder.dispatchThreads_threadsPerThreadgroup(thread_size, group_size);
        Ok(())
    }

    fn dispatch_threadgroups(
        &mut self,
        threadgroups: (u64, u64, u64),
        threads_per_threadgroup: (u64, u64, u64),
    ) -> Result<()> {
        let threadgroup_count = MTLSize {
            width: threadgroups.0 as usize,
            height: threadgroups.1 as usize,
            depth: threadgroups.2 as usize,
        };

        let group_size = MTLSize {
            width: threads_per_threadgroup.0 as usize,
            height: threads_per_threadgroup.1 as usize,
            depth: threads_per_threadgroup.2 as usize,
        };

        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroup_count, group_size);
        Ok(())
    }

    fn memory_barrier(&mut self, scope: BarrierScope) -> Result<()> {
        let mtl_scope = match scope {
            BarrierScope::Buffers => MTLBarrierScope::Buffers,
            BarrierScope::Textures => MTLBarrierScope::Textures,
            BarrierScope::All => MTLBarrierScope::Buffers | MTLBarrierScope::Textures,
        };

        self.encoder.memoryBarrierWithScope(mtl_scope);
        Ok(())
    }

    fn end_encoding(&mut self) -> Result<()> {
        self.encoder.endEncoding();
        Ok(())
    }
}
