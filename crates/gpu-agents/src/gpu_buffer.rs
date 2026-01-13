//! GPU Buffer wrapper for CudaSlice with additional metadata
//!
//! This module provides a wrapper around cudarc's CudaSlice that tracks
//! additional information like length and provides a more convenient API.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::marker::PhantomData;
use std::sync::Arc;

/// GPU buffer wrapper that tracks size and provides convenient methods
pub struct GpuBuffer<T> {
    /// The underlying CudaSlice
    slice: CudaSlice<T>,
    /// Number of elements in the buffer
    len: usize,
    /// Marker for the element type
    _phantom: PhantomData<T>,
}

impl<T: cudarc::driver::DeviceRepr> GpuBuffer<T> {
    /// Create a new uninitialized GPU buffer
    ///
    /// # Safety
    /// The buffer contents are uninitialized and must be written before reading
    pub unsafe fn new_uninit(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        debug_assert!(len > 0, "GPU buffer length must be positive");
        // SAFETY: Caller guarantees the buffer will be initialized before reading
        let slice = unsafe {
            stream
                .alloc::<T>(len)
                .context("Failed to allocate GPU buffer")?
        };
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Create a new zero-initialized GPU buffer
    pub fn new_zeros(stream: &Arc<CudaStream>, len: usize) -> Result<Self>
    where
        T: cudarc::driver::ValidAsZeroBits,
    {
        debug_assert!(len > 0, "GPU buffer length must be positive");
        let slice = stream
            .alloc_zeros::<T>(len)
            .context("Failed to allocate GPU buffer")?;
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Create a GPU buffer from host data
    pub fn from_host_slice(stream: &Arc<CudaStream>, data: &[T]) -> Result<Self>
    where
        T: Clone + Unpin,
    {
        let len = data.len();
        let slice = stream
            .clone_htod(data)
            .context("Failed to copy data to GPU")?;
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Get the number of elements in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a reference to the underlying CudaSlice
    #[inline]
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get a mutable reference to the underlying CudaSlice
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Get the raw device pointer value (CUdeviceptr)
    ///
    /// This returns the CUDA device pointer as a u64 value that can be used
    /// in kernel launches or other CUDA operations.
    /// The guard must be kept alive while the pointer is in use.
    pub fn device_ptr_raw<'a>(&'a self, stream: &'a Arc<CudaStream>) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
        use cudarc::driver::DevicePtr;
        self.slice.device_ptr(stream)
    }

    /// Copy data from GPU to host
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Vec<T>> {
        stream
            .clone_dtoh(&self.slice)
            .context("Failed to copy data from GPU to host")
    }

    /// Copy data from host to this GPU buffer
    pub fn copy_from_host(&mut self, stream: &Arc<CudaStream>, data: &[T]) -> Result<()>
    where
        T: Clone,
    {
        debug_assert_eq!(
            data.len(),
            self.len,
            "Data length must match buffer length for GPU copy"
        );
        if data.len() != self.len {
            anyhow::bail!(
                "Data length {} doesn't match buffer length {}",
                data.len(),
                self.len
            );
        }
        stream
            .memcpy_htod(data, &mut self.slice)
            .context("Failed to copy data from host to GPU")
    }

    /// Create a GpuBuffer from an existing CudaSlice
    pub fn from_cuda_slice(slice: CudaSlice<T>, len: usize) -> Result<Self> {
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }
}

/// GPU buffer for u8 data (commonly used for string operations)
pub type GpuByteBuffer = GpuBuffer<u8>;

/// GPU buffer for f32 data (commonly used for ML operations)
pub type GpuFloatBuffer = GpuBuffer<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_creation() -> Result<()> {
        // CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Test zero-initialized buffer
        let buffer = GpuFloatBuffer::new_zeros(&stream, 1024)?;
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());

        // Test buffer from host data
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = GpuFloatBuffer::from_host_slice(&stream, &data)?;
        assert_eq!(buffer.len(), 4);

        // Verify data roundtrip
        let retrieved = buffer.to_host(&stream)?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_empty_buffer() -> Result<()> {
        // CudaContext::new returns Arc<CudaContext>
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let buffer = GpuFloatBuffer::new_zeros(&stream, 0)?;
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        Ok(())
    }
}
