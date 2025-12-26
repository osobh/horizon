//! Memory Pool Implementation for GPU Memory Tier
//!
//! Provides efficient memory pool management with block reuse,
//! fragmentation reduction, and pool-specific allocation strategies.

use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use anyhow::Result;

use crate::gpu_memory_tier::{GpuMemoryError, MemoryLocation, AllocationType};

/// Memory allocation from the pool system
#[derive(Debug)]
pub struct MemoryAllocation {
    pub ptr: NonNull<u8>,
    pub size: u64,
    pub location: MemoryLocation,
    pub allocation_type: AllocationType,
    pub gpu_id: Option<u32>,
    pub created_at: Instant,
    pub from_pool: bool,
}

/// A memory pool for efficient allocation and reuse
pub struct MemoryPool {
    name: String,
    total_size: u64,
    used_size: Arc<Mutex<u64>>,
    free_blocks: Arc<Mutex<VecDeque<PoolBlock>>>,
    allocated_blocks: Arc<Mutex<HashMap<*mut u8, PoolBlock>>>,
    allocation_count: Arc<Mutex<u64>>,
    deallocation_count: Arc<Mutex<u64>>,
    created_at: Instant,
}

/// A block within a memory pool
#[derive(Debug, Clone)]
struct PoolBlock {
    ptr: NonNull<u8>,
    size: u64,
    allocated_at: Option<Instant>,
    last_used: Instant,
}

/// An allocation from a memory pool
pub struct PoolAllocation {
    ptr: NonNull<u8>,
    size: u64,
    pool_name: String,
    from_pool: bool,
    allocated_at: Instant,
}

impl MemoryPool {
    /// Create a new memory pool with the specified name and total size
    pub fn new(name: String, total_size: u64) -> Self {
        // Pre-allocate the pool memory
        let layout = std::alloc::Layout::from_size_align(total_size as usize, 8)
            .expect("Invalid layout for memory pool");
        
        let pool_ptr = unsafe { std::alloc::alloc(layout) };
        if pool_ptr.is_null() {
            panic!("Failed to allocate memory pool of size {}", total_size);
        }

        // Initialize with one large free block
        let mut free_blocks = VecDeque::new();
        free_blocks.push_back(PoolBlock {
            ptr: NonNull::new(pool_ptr).unwrap(),
            size: total_size,
            allocated_at: None,
            last_used: Instant::now(),
        });

        Self {
            name,
            total_size,
            used_size: Arc::new(Mutex::new(0)),
            free_blocks: Arc::new(Mutex::new(free_blocks)),
            allocated_blocks: Arc::new(Mutex::new(HashMap::new())),
            allocation_count: Arc::new(Mutex::new(0)),
            deallocation_count: Arc::new(Mutex::new(0)),
            created_at: Instant::now(),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: u64) -> Result<PoolAllocation> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut allocated_blocks = self.allocated_blocks.lock().unwrap();
        let mut used_size = self.used_size.lock().unwrap();

        // Find a suitable free block (first-fit strategy)
        let mut block_index = None;
        for (i, block) in free_blocks.iter().enumerate() {
            if block.size >= size {
                block_index = Some(i);
                break;
            }
        }

        let block_index = block_index.ok_or_else(|| {
            GpuMemoryError::AllocationFailed { size }
        })?;

        let mut block = free_blocks.remove(block_index).unwrap();
        
        // If block is larger than needed, split it
        if block.size > size {
            let remaining_size = block.size - size;
            let remaining_ptr = unsafe {
                NonNull::new(block.ptr.as_ptr().add(size as usize)).unwrap()
            };
            
            free_blocks.push_back(PoolBlock {
                ptr: remaining_ptr,
                size: remaining_size,
                allocated_at: None,
                last_used: Instant::now(),
            });
        }

        // Update block for allocation
        block.size = size;
        block.allocated_at = Some(Instant::now());
        block.last_used = Instant::now();

        // Track the allocated block
        allocated_blocks.insert(block.ptr.as_ptr(), block.clone());
        *used_size += size;
        *self.allocation_count.lock().unwrap() += 1;

        Ok(PoolAllocation {
            ptr: block.ptr,
            size,
            pool_name: self.name.clone(),
            from_pool: true,
            allocated_at: Instant::now(),
        })
    }

    /// Return an allocation back to the pool
    pub fn deallocate(&self, allocation: PoolAllocation) {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut allocated_blocks = self.allocated_blocks.lock().unwrap();
        let mut used_size = self.used_size.lock().unwrap();

        // Remove from allocated blocks
        if let Some(block) = allocated_blocks.remove(&allocation.ptr.as_ptr()) {
            *used_size -= block.size;
            *self.deallocation_count.lock().unwrap() += 1;

            // Add back to free blocks
            let free_block = PoolBlock {
                ptr: block.ptr,
                size: block.size,
                allocated_at: None,
                last_used: Instant::now(),
            };

            // Insert in sorted order for coalescing
            let mut insertion_index = None;
            for (i, existing_block) in free_blocks.iter().enumerate() {
                if free_block.ptr.as_ptr() < existing_block.ptr.as_ptr() {
                    insertion_index = Some(i);
                    break;
                }
            }

            if let Some(index) = insertion_index {
                free_blocks.insert(index, free_block);
            } else {
                free_blocks.push_back(free_block);
            }

            // Attempt to coalesce adjacent free blocks
            self.coalesce_free_blocks(&mut free_blocks);
        }
    }

    /// Coalesce adjacent free blocks to reduce fragmentation
    fn coalesce_free_blocks(&self, free_blocks: &mut VecDeque<PoolBlock>) {
        let mut i = 0;
        while i < free_blocks.len().saturating_sub(1) {
            let current_end = unsafe {
                free_blocks[i].ptr.as_ptr().add(free_blocks[i].size as usize)
            };
            
            if current_end == free_blocks[i + 1].ptr.as_ptr() {
                // Coalesce blocks
                free_blocks[i].size += free_blocks[i + 1].size;
                free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Get the total size of the pool
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Get the number of used bytes in the pool
    pub fn used_bytes(&self) -> u64 {
        *self.used_size.lock().unwrap()
    }

    /// Get the number of available bytes in the pool
    pub fn available_bytes(&self) -> u64 {
        self.total_size - self.used_bytes()
    }

    /// Get fragmentation percentage (0.0 to 100.0)
    pub fn fragmentation_percent(&self) -> f32 {
        let free_blocks = self.free_blocks.lock().unwrap();
        if free_blocks.is_empty() {
            return 0.0;
        }

        let largest_block = free_blocks.iter().max_by_key(|b| b.size).unwrap().size;
        let total_free = self.available_bytes();
        
        if total_free == 0 {
            0.0
        } else {
            ((total_free - largest_block) as f32 / total_free as f32) * 100.0
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let free_blocks = self.free_blocks.lock().unwrap();
        let allocated_blocks = self.allocated_blocks.lock().unwrap();

        PoolStats {
            name: self.name.clone(),
            total_size: self.total_size,
            used_bytes: self.used_bytes(),
            available_bytes: self.available_bytes(),
            allocated_blocks: allocated_blocks.len(),
            free_blocks: free_blocks.len(),
            fragmentation_percent: self.fragmentation_percent(),
            allocation_count: *self.allocation_count.lock().unwrap(),
            deallocation_count: *self.deallocation_count.lock().unwrap(),
            uptime: self.created_at.elapsed(),
        }
    }

    /// Get the pool name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for a memory pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub name: String,
    pub total_size: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub fragmentation_percent: f32,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub uptime: std::time::Duration,
}

impl PoolAllocation {
    /// Check if this allocation came from a pool
    pub fn from_pool(&self) -> bool {
        self.from_pool
    }

    /// Get the size of the allocation
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the pool name this allocation came from
    pub fn pool_name(&self) -> &str {
        &self.pool_name
    }

    /// Get the raw pointer (unsafe)
    pub unsafe fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get allocation timestamp
    pub fn allocated_at(&self) -> Instant {
        self.allocated_at
    }

    /// Convert to a memory allocation
    pub fn into_memory_allocation(self) -> MemoryAllocation {
        MemoryAllocation {
            ptr: self.ptr,
            size: self.size,
            location: MemoryLocation::Cpu, // Pool allocations are typically CPU-based
            allocation_type: AllocationType::Standard,
            gpu_id: None,
            created_at: self.allocated_at,
            from_pool: self.from_pool,
        }
    }
}

// Safety implementations
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}
unsafe impl Send for PoolAllocation {}
unsafe impl Sync for PoolAllocation {}
unsafe impl Send for PoolBlock {}
unsafe impl Sync for PoolBlock {}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Free the underlying pool memory
        let layout = std::alloc::Layout::from_size_align(self.total_size as usize, 8)
            .expect("Invalid layout for memory pool");
        
        // Get the base pointer from the first block
        let free_blocks = self.free_blocks.lock().unwrap();
        if let Some(first_block) = free_blocks.front() {
            unsafe {
                std::alloc::dealloc(first_block.ptr.as_ptr(), layout);
            }
        }
    }
}

impl Drop for PoolAllocation {
    fn drop(&mut self) {
        // Pool allocations should be explicitly returned to the pool
        // This is just a safety net in case they're dropped without being returned
        tracing::warn!(
            "PoolAllocation from pool '{}' dropped without being returned to pool", 
            self.pool_name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024);
        assert_eq!(pool.name(), "test_pool");
        assert_eq!(pool.total_size(), 1024);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.available_bytes(), 1024);
    }

    #[test]
    fn test_pool_allocation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024);
        
        let alloc = pool.allocate(256).expect("Failed to allocate");
        assert_eq!(alloc.size(), 256);
        assert!(alloc.from_pool());
        assert_eq!(pool.used_bytes(), 256);
        assert_eq!(pool.available_bytes(), 768);
    }

    #[test]
    fn test_pool_deallocation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024);
        
        let alloc = pool.allocate(256).expect("Failed to allocate");
        assert_eq!(pool.used_bytes(), 256);
        
        pool.deallocate(alloc);
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.available_bytes(), 1024);
    }

    #[test]
    fn test_pool_fragmentation() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024);
        
        // Allocate several blocks
        let alloc1 = pool.allocate(100).expect("Failed to allocate");
        let alloc2 = pool.allocate(100).expect("Failed to allocate");
        let alloc3 = pool.allocate(100).expect("Failed to allocate");
        
        // Free middle block to create fragmentation
        pool.deallocate(alloc2);
        
        // Check fragmentation
        assert!(pool.fragmentation_percent() > 0.0);
    }

    #[test]
    fn test_pool_coalescing() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024);
        
        // Allocate and free adjacent blocks
        let alloc1 = pool.allocate(256).expect("Failed to allocate");
        let alloc2 = pool.allocate(256).expect("Failed to allocate");
        
        pool.deallocate(alloc1);
        pool.deallocate(alloc2);
        
        // Should be able to allocate the full size again due to coalescing
        let large_alloc = pool.allocate(512);
        assert!(large_alloc.is_ok());
    }
}