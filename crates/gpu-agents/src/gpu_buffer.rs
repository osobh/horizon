//! GPU Buffer wrapper for CudaSlice with additional metadata
//!
//! This module provides a wrapper around cudarc's CudaSlice that tracks
//! additional information like length and provides a more convenient API.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
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
    pub unsafe fn new_uninit(device: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        let slice = device
            .alloc::<T>(len)
            .context("Failed to allocate GPU buffer")?;
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Create a new zero-initialized GPU buffer
    pub fn new_zeros(device: &Arc<CudaDevice>, len: usize) -> Result<Self>
    where
        T: cudarc::driver::ValidAsZeroBits,
    {
        let slice = device
            .alloc_zeros::<T>(len)
            .context("Failed to allocate GPU buffer")?;
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Create a GPU buffer from host data
    pub fn from_host_slice(device: &Arc<CudaDevice>, data: &[T]) -> Result<Self>
    where
        T: Clone + Unpin,
    {
        let len = data.len();
        let slice = device
            .htod_copy(data.to_vec())
            .context("Failed to copy data to GPU")?;
        Ok(Self {
            slice,
            len,
            _phantom: PhantomData,
        })
    }

    /// Get the number of elements in the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a reference to the underlying CudaSlice
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get a mutable reference to the underlying CudaSlice
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Get the device pointer as a raw pointer
    ///
    /// # Safety
    /// The returned pointer is only valid as long as the GpuBuffer exists
    pub unsafe fn device_ptr(&self) -> *const T {
        // CudaSlice can be used as a pointer in kernel launches
        // We'll cast it through the slice reference
        &self.slice as *const CudaSlice<T> as *const T
    }

    /// Get the device pointer as a mutable raw pointer
    ///
    /// # Safety
    /// The returned pointer is only valid as long as the GpuBuffer exists
    pub unsafe fn device_ptr_mut(&mut self) -> *mut T {
        &mut self.slice as *mut CudaSlice<T> as *mut T
    }

    /// Copy data from GPU to host
    pub fn to_host(&self, device: &Arc<CudaDevice>) -> Result<Vec<T>> {
        device
            .dtoh_sync_copy(&self.slice)
            .context("Failed to copy data from GPU to host")
    }

    /// Copy data from host to this GPU buffer
    pub fn copy_from_host(&mut self, device: &Arc<CudaDevice>, data: &[T]) -> Result<()>
    where
        T: Clone,
    {
        if data.len() != self.len {
            anyhow::bail!(
                "Data length {} doesn't match buffer length {}",
                data.len(),
                self.len
            );
        }
        device
            .htod_sync_copy_into(data, &mut self.slice)
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
        let device = Arc::new(CudaDevice::new(0)?);

        // Test zero-initialized buffer
        let buffer = GpuFloatBuffer::new_zeros(&device, 1024)?;
        assert_eq!(buffer.len(), 1024);
        assert!(!buffer.is_empty());

        // Test buffer from host data
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = GpuFloatBuffer::from_host_slice(&device, &data)?;
        assert_eq!(buffer.len(), 4);

        // Verify data roundtrip
        let retrieved = buffer.to_host(&device)?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_empty_buffer() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let buffer = GpuFloatBuffer::new_zeros(&device, 0)?;
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        Ok(())
    }
}
