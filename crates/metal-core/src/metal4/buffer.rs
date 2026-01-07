//! Metal 4 buffer abstraction.
//!
//! Metal 4 buffers are similar to Metal 3 buffers but can integrate
//! with MTLResidencySet for explicit memory management.

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal4::Metal4Device;

use bytemuck::Pod;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use std::ptr::NonNull;

/// Metal 4 buffer wrapper.
///
/// Provides the same functionality as Metal 3 buffers with additional
/// support for Metal 4 residency management when available.
pub struct Metal4Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

impl Metal4Buffer {
    /// Create a new uninitialized buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device to create the buffer on
    /// * `size` - Size in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails.
    pub fn new(device: &Metal4Device, size: usize) -> Result<Self> {
        let raw_device = device.raw();
        let options = MTLResourceOptions::StorageModeShared;

        let buffer = raw_device
            .newBufferWithLength_options(size, options)
            .ok_or_else(|| {
                MetalError::creation_failed("Metal4Buffer", "Failed to create buffer")
            })?;

        Ok(Self { raw: buffer, size })
    }

    /// Create a buffer initialized with data.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    /// * `data` - Initial data to copy into the buffer
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails.
    pub fn with_data<T: Pod>(device: &Metal4Device, data: &[T]) -> Result<Self> {
        let size = std::mem::size_of_val(data);
        let raw_device = device.raw();
        let options = MTLResourceOptions::StorageModeShared;

        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .ok_or_else(|| MetalError::creation_failed("Metal4Buffer", "Null data pointer"))?;

        // SAFETY: ptr is a NonNull created from a valid &[T] slice and points to
        // size bytes of readable memory. The Metal API copies the data during buffer
        // creation, so the pointer only needs to be valid for this call.
        let buffer = unsafe { raw_device.newBufferWithBytes_length_options(ptr, size, options) }
            .ok_or_else(|| {
                MetalError::creation_failed("Metal4Buffer", "Failed to create buffer with data")
            })?;

        Ok(Self { raw: buffer, size })
    }

    /// Get the raw MTLBuffer.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Mark this buffer for residency in a residency set.
    ///
    /// **Current Status:** No-op (Metal 3 fallback mode).
    ///
    /// In Metal 3, residency is managed automatically by the driver. This method
    /// is provided for forward compatibility with Metal 4's explicit residency API.
    pub fn mark_for_residency(&self) {
        // FALLBACK: No-op in Metal 3 mode. Residency is automatic.
        //
        // TODO(metal4): Implement explicit residency management when Metal 4 APIs
        // are available. This will add the buffer to an MTLResidencySet for
        // fine-grained GPU memory management.
    }
}

impl MetalBuffer for Metal4Buffer {
    fn byte_len(&self) -> usize {
        self.size
    }

    fn len(&self) -> usize {
        self.size
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn contents<T: Pod>(&self) -> &[T] {
        let ptr = self.raw.contents();
        let count = self.size / std::mem::size_of::<T>();
        // SAFETY: MTLBuffer.contents() returns a valid pointer to self.size bytes
        // of GPU-accessible memory. T: Pod ensures safe reinterpretation. The
        // slice lifetime is tied to &self, ensuring the buffer remains valid.
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const T, count) }
    }

    fn contents_mut<T: Pod>(&mut self) -> &mut [T] {
        let ptr = self.raw.contents();
        let count = self.size / std::mem::size_of::<T>();
        // SAFETY: MTLBuffer.contents() returns a valid pointer to self.size bytes
        // of GPU-accessible memory. T: Pod ensures safe reinterpretation. &mut self
        // ensures exclusive access, preventing data races. The slice lifetime is
        // tied to &mut self, ensuring the buffer remains valid.
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, count) }
    }

    fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
    }
}

// SAFETY: Metal4Buffer is Send because:
// 1. `Retained<ProtocolObject<dyn MTLBuffer>>` wraps an Objective-C MTLBuffer which is
//    thread-safe per Apple's Metal documentation (GPU resources are thread-safe)
// 2. The buffer uses `StorageModeShared` which provides coherent unified memory access
//    from any thread on Apple Silicon's unified memory architecture
// 3. The `size` field is a trivially Send primitive
// 4. Ownership of the buffer can safely be transferred between threads
// 5. Metal's reference counting (via `Retained`) is thread-safe
unsafe impl Send for Metal4Buffer {}

// SAFETY: Metal4Buffer is Sync because:
// 1. MTLBuffer objects are thread-safe for concurrent read access per Metal documentation
// 2. Shared storage mode buffers have coherent memory views from all threads
// 3. The `contents()` method returns a raw pointer that can be safely read concurrently
// 4. Mutation through `contents_mut()` requires `&mut self`, preventing data races
// 5. GPU operations on the buffer are synchronized through Metal command encoders
// 6. Metal 4 residency management (when implemented) is thread-safe
unsafe impl Sync for Metal4Buffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            assert_eq!(buffer.len(), 1024);
            assert!(!buffer.is_empty());
        }
    }

    #[test]
    fn test_buffer_with_data() {
        if let Ok(device) = Metal4Device::system_default() {
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let buffer = Metal4Buffer::with_data(&device, &data).expect("Failed to create buffer");

            assert_eq!(buffer.len(), 16); // 4 floats * 4 bytes
            let contents: &[f32] = buffer.contents();
            assert_eq!(contents[0], 1.0);
            assert_eq!(contents[3], 4.0);
        }
    }

    #[test]
    fn test_buffer_gpu_address() {
        if let Ok(device) = Metal4Device::system_default() {
            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            let addr = buffer.gpu_address();
            // GPU address should be non-zero for valid buffers
            assert!(addr > 0);
        }
    }
}
