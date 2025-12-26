//! Metal buffer abstraction.
//!
//! Provides unified memory access for Apple Silicon,
//! where CPU and GPU share the same physical memory.

use std::ops::{Deref, DerefMut};

/// Trait for Metal GPU buffers.
///
/// On Apple Silicon, buffers use unified memory, meaning
/// both CPU and GPU can access the same data without copies.
pub trait MetalBuffer: Send + Sync {
    /// Get the buffer length in bytes.
    fn byte_len(&self) -> usize;

    /// Get the element count (for typed buffers).
    fn len(&self) -> usize;

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a slice of the buffer contents.
    ///
    /// # Safety
    /// The type `T` must match the buffer's data type.
    /// On Apple Silicon with unified memory, this provides
    /// direct CPU access to GPU memory.
    fn contents<T: bytemuck::Pod>(&self) -> &[T];

    /// Get a mutable slice of the buffer contents.
    ///
    /// # Safety
    /// The type `T` must match the buffer's data type.
    /// Modifications are immediately visible to the GPU
    /// on Apple Silicon.
    fn contents_mut<T: bytemuck::Pod>(&mut self) -> &mut [T];

    /// Get the GPU address for this buffer.
    ///
    /// Used for argument buffer binding in Metal 3+.
    fn gpu_address(&self) -> u64;

    /// Copy data from a slice into the buffer.
    fn copy_from_slice<T: bytemuck::Pod>(&mut self, data: &[T]) {
        let contents = self.contents_mut::<T>();
        let len = data.len().min(contents.len());
        contents[..len].copy_from_slice(&data[..len]);
    }

    /// Copy data from the buffer into a slice.
    fn copy_to_slice<T: bytemuck::Pod>(&self, data: &mut [T]) {
        let contents = self.contents::<T>();
        let len = data.len().min(contents.len());
        data[..len].copy_from_slice(&contents[..len]);
    }
}

/// Buffer storage mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageMode {
    /// Shared between CPU and GPU (default for Apple Silicon).
    #[default]
    Shared,
    /// Private to GPU only.
    Private,
    /// Managed with explicit synchronization.
    Managed,
}

/// Buffer usage hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage {
    /// Buffer will be read by shaders.
    pub shader_read: bool,
    /// Buffer will be written by shaders.
    pub shader_write: bool,
    /// Buffer will be used for argument binding.
    pub argument_buffer: bool,
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self {
            shader_read: true,
            shader_write: true,
            argument_buffer: false,
        }
    }
}

/// A view into a portion of a buffer.
pub struct BufferView<'a, T: bytemuck::Pod> {
    data: &'a [T],
}

impl<'a, T: bytemuck::Pod> BufferView<'a, T> {
    /// Create a new buffer view.
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }
}

impl<'a, T: bytemuck::Pod> Deref for BufferView<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// A mutable view into a portion of a buffer.
pub struct BufferViewMut<'a, T: bytemuck::Pod> {
    data: &'a mut [T],
}

impl<'a, T: bytemuck::Pod> BufferViewMut<'a, T> {
    /// Create a new mutable buffer view.
    pub fn new(data: &'a mut [T]) -> Self {
        Self { data }
    }
}

impl<'a, T: bytemuck::Pod> Deref for BufferViewMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T: bytemuck::Pod> DerefMut for BufferViewMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}
