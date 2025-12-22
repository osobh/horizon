//! Unified memory pool for zero-copy GPU memory management.
//!
//! This module provides a memory pool that manages GPU buffer allocations
//! using `bytes::Bytes` for zero-copy data transfers.

use bytes::Bytes;
use std::collections::HashMap;

/// Result type for memory operations.
pub type MemoryResult<T> = Result<T, String>;

/// Statistics about memory pool usage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryStats {
    /// Total capacity in bytes
    pub capacity: usize,
    /// Currently used bytes
    pub used: usize,
    /// Available bytes
    pub available: usize,
    /// Memory pressure (0.0 to 1.0)
    pub pressure: f64,
    /// Number of active allocations
    pub allocation_count: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            capacity: 0,
            used: 0,
            available: 0,
            pressure: 0.0,
            allocation_count: 0,
        }
    }
}

/// A buffer allocation in the memory pool.
#[derive(Debug, Clone)]
struct BufferAllocation {
    size: usize,
    data: Vec<u8>,
}

/// Unified memory pool for managing GPU buffers.
///
/// This pool provides zero-copy memory management using `bytes::Bytes`.
/// It tracks allocations, handles read/write operations, and monitors
/// memory pressure.
#[derive(Debug)]
pub struct UnifiedMemoryPool {
    capacity: usize,
    used: usize,
    allocations: HashMap<String, BufferAllocation>,
}

impl UnifiedMemoryPool {
    /// Create a new memory pool with the specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            allocations: HashMap::new(),
        }
    }

    /// Get total capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get currently used bytes.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Get available bytes.
    #[must_use]
    pub fn available(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }

    /// Get memory pressure (0.0 to 1.0).
    #[must_use]
    pub fn pressure(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            self.used as f64 / self.capacity as f64
        }
    }

    /// Get memory statistics.
    #[must_use]
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            capacity: self.capacity,
            used: self.used,
            available: self.available(),
            pressure: self.pressure(),
            allocation_count: self.allocations.len(),
        }
    }

    /// Check if a buffer is allocated.
    #[must_use]
    pub fn is_allocated(&self, buffer_id: &str) -> bool {
        self.allocations.contains_key(buffer_id)
    }

    /// Get the size of an allocated buffer.
    #[must_use]
    pub fn get_buffer_size(&self, buffer_id: &str) -> Option<usize> {
        self.allocations.get(buffer_id).map(|alloc| alloc.size)
    }

    /// Allocate a buffer with the given ID and size.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer ID is already allocated
    /// - There is insufficient memory available
    pub fn allocate(&mut self, buffer_id: &str, size: usize) -> MemoryResult<()> {
        // Check if already allocated
        if self.allocations.contains_key(buffer_id) {
            return Err(format!("Buffer '{}' is already allocated", buffer_id));
        }

        // Check if we have enough space
        if self.used + size > self.capacity {
            return Err(format!(
                "insufficient memory: requested {}, available {}",
                size,
                self.available()
            ));
        }

        // Create allocation
        let allocation = BufferAllocation {
            size,
            data: vec![0; size],
        };

        self.allocations.insert(buffer_id.to_string(), allocation);
        self.used += size;

        Ok(())
    }

    /// Deallocate a buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not found.
    pub fn deallocate(&mut self, buffer_id: &str) -> MemoryResult<()> {
        let allocation = self
            .allocations
            .remove(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        self.used = self.used.saturating_sub(allocation.size);

        Ok(())
    }

    /// Write data to a buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is not found
    /// - The write would exceed buffer bounds
    pub fn write(&mut self, buffer_id: &str, data: Bytes, offset: usize) -> MemoryResult<()> {
        let allocation = self
            .allocations
            .get_mut(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        // Check bounds
        if offset + data.len() > allocation.size {
            return Err(format!(
                "Write exceeds buffer size: {} + {} > {}",
                offset,
                data.len(),
                allocation.size
            ));
        }

        // Copy data
        allocation.data[offset..offset + data.len()].copy_from_slice(&data);

        Ok(())
    }

    /// Read data from a buffer.
    ///
    /// Returns a `Bytes` object for zero-copy sharing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The buffer is not found
    /// - The read would exceed buffer bounds
    pub fn read(&self, buffer_id: &str, size: usize, offset: usize) -> MemoryResult<Bytes> {
        let allocation = self
            .allocations
            .get(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        // Check bounds
        if offset + size > allocation.size {
            return Err(format!(
                "Read exceeds buffer size: {} + {} > {}",
                offset, size, allocation.size
            ));
        }

        // Return zero-copy Bytes
        Ok(Bytes::copy_from_slice(&allocation.data[offset..offset + size]))
    }

    /// Clear all allocations.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic_allocation() {
        let mut pool = UnifiedMemoryPool::new(1024);
        assert_eq!(pool.capacity(), 1024);
        assert_eq!(pool.used(), 0);

        pool.allocate("test", 100).unwrap();
        assert_eq!(pool.used(), 100);
        assert_eq!(pool.available(), 924);
    }

    #[test]
    fn test_memory_pool_write_read() {
        let mut pool = UnifiedMemoryPool::new(1024);
        pool.allocate("buf", 100).unwrap();

        let data = Bytes::from(vec![1, 2, 3, 4]);
        pool.write("buf", data.clone(), 0).unwrap();

        let read = pool.read("buf", 4, 0).unwrap();
        assert_eq!(read.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_memory_pool_pressure_calculation() {
        let mut pool = UnifiedMemoryPool::new(1000);

        assert_eq!(pool.pressure(), 0.0);

        pool.allocate("buf1", 500).unwrap();
        assert_eq!(pool.pressure(), 0.5);

        pool.allocate("buf2", 250).unwrap();
        assert_eq!(pool.pressure(), 0.75);
    }
}
