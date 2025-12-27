//! Metal 3 buffer implementation.

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal3::Metal3Device;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Metal 3 GPU buffer.
///
/// Uses shared storage mode for unified memory access on Apple Silicon.
pub struct Metal3Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    byte_len: usize,
    element_count: usize,
    element_size: usize,
}

// SAFETY: Metal3Buffer is Send because:
// 1. `Retained<ProtocolObject<dyn MTLBuffer>>` wraps an Objective-C MTLBuffer which is
//    thread-safe per Apple's Metal documentation (GPU resources are thread-safe)
// 2. The buffer uses `StorageModeShared` which provides coherent unified memory access
//    from any thread on Apple Silicon's unified memory architecture
// 3. All other fields (byte_len, element_count, element_size) are trivially Send primitives
// 4. Ownership of the buffer can safely be transferred between threads
// 5. Metal's reference counting (via `Retained`) is thread-safe
unsafe impl Send for Metal3Buffer {}

// SAFETY: Metal3Buffer is Sync because:
// 1. MTLBuffer objects are thread-safe for concurrent read access per Metal documentation
// 2. Shared storage mode buffers have coherent memory views from all threads
// 3. The `contents()` method returns a raw pointer that can be safely read concurrently
// 4. Mutation through `contents_mut()` requires `&mut self`, preventing data races
// 5. GPU operations on the buffer are synchronized through Metal command encoders
// 6. All metadata fields are immutable after construction
unsafe impl Sync for Metal3Buffer {}

impl Metal3Buffer {
    /// Create a new buffer with the given size in bytes.
    pub fn new(device: &Metal3Device, size_bytes: usize) -> Result<Self> {
        Self::new_with_element_size(device, size_bytes, 1)
    }

    /// Create a new buffer with element count and size.
    pub fn new_with_element_size(
        device: &Metal3Device,
        size_bytes: usize,
        element_size: usize,
    ) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared;

        let buffer = device
            .raw()
            .newBufferWithLength_options(size_bytes, options)
            .ok_or_else(|| MetalError::creation_failed("buffer", "Failed to allocate GPU buffer"))?;

        let element_count = if element_size > 0 {
            size_bytes / element_size
        } else {
            size_bytes
        };

        Ok(Self {
            buffer,
            byte_len: size_bytes,
            element_count,
            element_size,
        })
    }

    /// Create a buffer initialized with data.
    pub fn with_data<T: bytemuck::Pod>(device: &Metal3Device, data: &[T]) -> Result<Self> {
        use std::ptr::NonNull;

        let size_bytes = data.len() * std::mem::size_of::<T>();
        let options = MTLResourceOptions::StorageModeShared;

        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .expect("data pointer should not be null");

        let buffer = unsafe {
            device
                .raw()
                .newBufferWithBytes_length_options(ptr, size_bytes, options)
        }
        .ok_or_else(|| MetalError::creation_failed("buffer", "Failed to allocate GPU buffer"))?;

        Ok(Self {
            buffer,
            byte_len: size_bytes,
            element_count: data.len(),
            element_size: std::mem::size_of::<T>(),
        })
    }

    /// Get the raw MTLBuffer.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }

    /// Get a raw pointer to the buffer contents.
    pub fn contents_ptr(&self) -> *mut u8 {
        self.buffer.contents().as_ptr() as *mut u8
    }
}

impl MetalBuffer for Metal3Buffer {
    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn len(&self) -> usize {
        self.element_count
    }

    fn contents<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.buffer.contents().as_ptr() as *const T;
        let len = self.byte_len / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    fn contents_mut<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let ptr = self.buffer.contents().as_ptr() as *mut T;
        let len = self.byte_len / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    fn gpu_address(&self) -> u64 {
        self.buffer.gpuAddress()
    }
}

impl std::fmt::Debug for Metal3Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal3Buffer")
            .field("byte_len", &self.byte_len)
            .field("element_count", &self.element_count)
            .field("gpu_address", &format!("0x{:x}", self.gpu_address()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_buffer_creation() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let buffer = Metal3Buffer::new(&device, 1024).unwrap();

        assert_eq!(buffer.byte_len(), 1024);
        assert!(buffer.gpu_address() != 0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_buffer_contents() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let mut buffer = Metal3Buffer::new(&device, 16 * std::mem::size_of::<f32>()).unwrap();

        // Write data
        {
            let data = buffer.contents_mut::<f32>();
            for i in 0..16 {
                data[i] = i as f32;
            }
        }

        // Read back
        {
            let data = buffer.contents::<f32>();
            for i in 0..16 {
                assert_eq!(data[i], i as f32);
            }
        }
    }
}
