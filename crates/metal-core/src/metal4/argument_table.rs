//! Metal 4 argument table abstraction.
//!
//! Provides efficient batch resource binding that mirrors MTL4ArgumentTable.
//! On Metal 4 (macOS 26+), this will wrap the native argument table.
//! On earlier systems, it uses a GPU address buffer for batch binding.
//!
//! # Why Argument Tables?
//!
//! Traditional Metal binding requires calling `setBuffer` for each resource:
//! ```ignore
//! encoder.setBuffer(buf1, 0, 0);
//! encoder.setBuffer(buf2, 0, 1);
//! encoder.setBuffer(buf3, 0, 2);
//! ```
//!
//! Argument tables batch these bindings:
//! ```ignore
//! table.set_buffer(0, &buf1)?;
//! table.set_buffer(1, &buf2)?;
//! table.set_buffer(2, &buf3)?;
//! table.sync();
//! encoder.set_argument_table(&table);
//! ```
//!
//! This is faster because:
//! - Single encoder call instead of many
//! - Table can be reused across multiple encoders
//! - GPU addresses are batched in contiguous memory

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal4::{Metal4Buffer, Metal4Device, Metal4Tensor};
use crate::tensor::MetalTensor;

use std::sync::Arc;

/// Configuration for creating an argument table.
#[derive(Debug, Clone)]
pub struct ArgumentTableDescriptor {
    /// Maximum number of buffer bindings.
    pub max_buffer_bind_count: usize,
    /// Maximum number of texture bindings (not yet implemented).
    pub max_texture_bind_count: usize,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl Default for ArgumentTableDescriptor {
    fn default() -> Self {
        Self {
            max_buffer_bind_count: 31, // Metal's common limit
            max_texture_bind_count: 31,
            label: None,
        }
    }
}

impl ArgumentTableDescriptor {
    /// Create a descriptor with the specified buffer capacity.
    pub fn with_buffer_capacity(count: usize) -> Self {
        Self {
            max_buffer_bind_count: count,
            ..Default::default()
        }
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Argument table for efficient batch resource binding.
///
/// On Metal 4: Wraps MTL4ArgumentTable
/// On Metal 3: Uses a backing buffer with GPU addresses
///
/// # Thread Safety
///
/// The table itself is not thread-safe for concurrent modification.
/// Use separate tables per thread, or synchronize access externally.
pub struct Metal4ArgumentTable {
    /// Reference to the device
    device: Arc<Metal4Device>,
    /// Backing buffer for GPU addresses
    backing_buffer: Metal4Buffer,
    /// Cached GPU addresses for each slot
    buffer_addresses: Vec<u64>,
    /// Number of buffer slots currently bound
    buffer_count: usize,
    /// Maximum buffer slots
    max_buffers: usize,
    /// Whether the table needs to be synced to GPU memory
    dirty: bool,
}

impl Metal4ArgumentTable {
    /// Create a new argument table.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    /// * `desc` - Configuration for the table
    ///
    /// # Errors
    ///
    /// Returns an error if the backing buffer cannot be created.
    pub fn new(device: &Arc<Metal4Device>, desc: ArgumentTableDescriptor) -> Result<Self> {
        // Each GPU address is 8 bytes (u64)
        let backing_size = desc.max_buffer_bind_count * 8;
        let backing_buffer = Metal4Buffer::new(device, backing_size)?;

        Ok(Self {
            device: Arc::clone(device),
            backing_buffer,
            buffer_addresses: vec![0; desc.max_buffer_bind_count],
            buffer_count: 0,
            max_buffers: desc.max_buffer_bind_count,
            dirty: false,
        })
    }

    /// Create an argument table with default settings.
    pub fn with_capacity(device: &Arc<Metal4Device>, buffer_capacity: usize) -> Result<Self> {
        let desc = ArgumentTableDescriptor::with_buffer_capacity(buffer_capacity);
        Self::new(device, desc)
    }

    /// Set a buffer at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The binding index (must be < max_buffer_bind_count)
    /// * `buffer` - The buffer to bind
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds.
    pub fn set_buffer(&mut self, index: u32, buffer: &Metal4Buffer) -> Result<()> {
        let idx = index as usize;
        if idx >= self.max_buffers {
            return Err(MetalError::creation_failed(
                "ArgumentTable",
                &format!("Buffer index {} exceeds max {}", idx, self.max_buffers),
            ));
        }

        self.buffer_addresses[idx] = buffer.gpu_address();
        self.buffer_count = self.buffer_count.max(idx + 1);
        self.dirty = true;
        Ok(())
    }

    /// Set a tensor at the given index.
    ///
    /// This binds the tensor's underlying buffer.
    pub fn set_tensor(&mut self, index: u32, tensor: &Metal4Tensor) -> Result<()> {
        let idx = index as usize;
        if idx >= self.max_buffers {
            return Err(MetalError::creation_failed(
                "ArgumentTable",
                &format!("Tensor index {} exceeds max {}", idx, self.max_buffers),
            ));
        }

        self.buffer_addresses[idx] = tensor.gpu_address();
        self.buffer_count = self.buffer_count.max(idx + 1);
        self.dirty = true;
        Ok(())
    }

    /// Set a raw GPU address at the given index.
    ///
    /// Use this for advanced scenarios where you have the address directly.
    pub fn set_address(&mut self, index: u32, address: u64) -> Result<()> {
        let idx = index as usize;
        if idx >= self.max_buffers {
            return Err(MetalError::creation_failed(
                "ArgumentTable",
                &format!("Index {} exceeds max {}", idx, self.max_buffers),
            ));
        }

        self.buffer_addresses[idx] = address;
        self.buffer_count = self.buffer_count.max(idx + 1);
        self.dirty = true;
        Ok(())
    }

    /// Sync the table to GPU memory.
    ///
    /// Call this after making changes and before using the table with an encoder.
    pub fn sync(&mut self) {
        if self.dirty {
            let contents = self.backing_buffer.contents_mut::<u64>();
            contents[..self.buffer_count]
                .copy_from_slice(&self.buffer_addresses[..self.buffer_count]);
            self.dirty = false;
        }
    }

    /// Clear all bindings.
    pub fn clear(&mut self) {
        self.buffer_addresses.fill(0);
        self.buffer_count = 0;
        self.dirty = true;
    }

    /// Get the backing buffer for encoder binding.
    ///
    /// This is the buffer that should be bound to the encoder.
    pub fn backing_buffer(&self) -> &Metal4Buffer {
        &self.backing_buffer
    }

    /// Get the number of bound buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Get the maximum buffer capacity.
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }

    /// Check if the table has any bindings.
    pub fn is_empty(&self) -> bool {
        self.buffer_count == 0
    }

    /// Check if the table needs to be synced.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the GPU address at a specific index.
    ///
    /// Returns 0 if the index is unbound or out of bounds.
    pub fn get_address(&self, index: u32) -> u64 {
        let idx = index as usize;
        if idx < self.buffer_addresses.len() {
            self.buffer_addresses[idx]
        } else {
            0
        }
    }
}

// SAFETY: Metal4ArgumentTable is Send because:
// 1. Arc<Metal4Device> is Send + Sync
// 2. Metal4Buffer is Send
// 3. Vec<u64> is Send
// 4. Primitive fields are Send
unsafe impl Send for Metal4ArgumentTable {}

// SAFETY: Metal4ArgumentTable is NOT Sync by default because:
// - Mutation methods like set_buffer() and sync() are not atomic
// - Concurrent modification would cause data races
// However, the backing_buffer itself is Sync for reading.
// Users should use external synchronization for concurrent table updates.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argument_table_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let desc = ArgumentTableDescriptor::default();
            let table =
                Metal4ArgumentTable::new(&device, desc).expect("Failed to create argument table");

            assert_eq!(table.buffer_count(), 0);
            assert_eq!(table.max_buffers(), 31);
            assert!(table.is_empty());
        }
    }

    #[test]
    fn test_argument_table_with_capacity() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let table = Metal4ArgumentTable::with_capacity(&device, 16)
                .expect("Failed to create argument table");

            assert_eq!(table.max_buffers(), 16);
        }
    }

    #[test]
    fn test_argument_table_set_buffer() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut table = Metal4ArgumentTable::with_capacity(&device, 16)
                .expect("Failed to create argument table");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            table.set_buffer(0, &buffer).expect("Failed to set buffer");
            assert_eq!(table.buffer_count(), 1);
            assert!(table.is_dirty());

            table.sync();
            assert!(!table.is_dirty());

            // Verify the address was stored
            assert_eq!(table.get_address(0), buffer.gpu_address());
        }
    }

    #[test]
    fn test_argument_table_out_of_bounds() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut table = Metal4ArgumentTable::with_capacity(&device, 4)
                .expect("Failed to create argument table");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            // Index 4 is out of bounds for capacity 4
            let result = table.set_buffer(4, &buffer);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_argument_table_clear() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut table = Metal4ArgumentTable::with_capacity(&device, 16)
                .expect("Failed to create argument table");

            let buffer = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");

            table.set_buffer(0, &buffer).expect("Failed to set buffer");
            table.set_buffer(5, &buffer).expect("Failed to set buffer");
            assert_eq!(table.buffer_count(), 6);

            table.clear();
            assert_eq!(table.buffer_count(), 0);
            assert!(table.is_empty());
            assert!(table.is_dirty());
        }
    }

    #[test]
    fn test_argument_table_multiple_buffers() {
        if let Ok(device) = Metal4Device::system_default() {
            let device = Arc::new(device);
            let mut table = Metal4ArgumentTable::with_capacity(&device, 16)
                .expect("Failed to create argument table");

            let buf1 = Metal4Buffer::new(&device, 1024).expect("Failed to create buffer");
            let buf2 = Metal4Buffer::new(&device, 2048).expect("Failed to create buffer");
            let buf3 = Metal4Buffer::new(&device, 512).expect("Failed to create buffer");

            table.set_buffer(0, &buf1).unwrap();
            table.set_buffer(1, &buf2).unwrap();
            table.set_buffer(2, &buf3).unwrap();

            assert_eq!(table.buffer_count(), 3);

            table.sync();

            // Verify all addresses
            assert_eq!(table.get_address(0), buf1.gpu_address());
            assert_eq!(table.get_address(1), buf2.gpu_address());
            assert_eq!(table.get_address(2), buf3.gpu_address());
        }
    }

    #[test]
    fn test_argument_table_descriptor() {
        let desc = ArgumentTableDescriptor::with_buffer_capacity(64).with_label("MyTable");

        assert_eq!(desc.max_buffer_bind_count, 64);
        assert_eq!(desc.label, Some("MyTable".to_string()));
    }
}
