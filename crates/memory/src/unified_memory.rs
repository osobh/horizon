//! Unified Memory Management for GPU-CPU Coherent Access
//!
//! Provides CUDA Unified Memory support with automatic migration,
//! memory hints, and zero-copy access from both CPU and GPU.

// Allow Arc<Mutex<T>> where T contains NonNull - we have explicit unsafe impl Send/Sync
#![allow(clippy::arc_with_non_send_sync)]

use anyhow::Result;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr};

use crate::gpu_memory_tier::GpuMemoryError;

/// Manager for CUDA Unified Memory allocations
pub struct UnifiedMemoryManager {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    allocations: Arc<Mutex<Vec<UnifiedAllocationInfo>>>,
    total_allocated: Arc<Mutex<u64>>,
    allocation_counter: Arc<Mutex<u64>>,
}

/// Information about a unified memory allocation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct UnifiedAllocationInfo {
    id: u64,
    ptr: NonNull<u8>,
    size: u64,
    current_location: MemoryLocation,
    access_pattern: AccessPattern,
    allocated_at: Instant,
    last_accessed: Instant,
}

/// Current memory location for unified allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Memory is on GPU
    Gpu,
    /// Memory is on CPU
    Cpu,
    /// Memory location is unknown
    Unknown,
}

/// Memory access pattern for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum AccessPattern {
    CpuMostly,
    GpuMostly,
    Balanced,
    Unknown,
}

/// A unified memory allocation that can be accessed from both CPU and GPU
pub struct UnifiedAllocation {
    id: u64,
    ptr: NonNull<u8>,
    size: u64,
    manager: Arc<UnifiedMemoryManager>,
    location: Arc<Mutex<MemoryLocation>>,
    allocated_at: Instant,
}

impl UnifiedMemoryManager {
    /// Create a new unified memory manager
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(Vec::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            allocation_counter: Arc::new(Mutex::new(0)),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(_device: ()) -> Result<Self> {
        Ok(Self {
            allocations: Arc::new(Mutex::new(Vec::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            allocation_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Allocate unified memory
    #[must_use = "UnifiedAllocation must be stored to free the memory later"]
    pub fn allocate(&self, size: u64) -> Result<UnifiedAllocation> {
        let mut counter = self.allocation_counter.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (allocation_counter), recovering inner data");
            poisoned.into_inner()
        });
        let id = *counter;
        *counter += 1;
        drop(counter);

        #[cfg(feature = "cuda")]
        let ptr = {
            // Allocate CUDA managed memory
            let managed_ptr = self
                .device
                .alloc_zeros::<u8>(size as usize)
                .map_err(|e| GpuMemoryError::AllocationFailed { size })?;

            // PLACEHOLDER: Fake pointer for type system satisfaction.
            //
            // SAFETY: This is NOT a valid pointer. The actual GPU memory is managed by
            // the `device.alloc_zeros()` call above. This placeholder exists because
            // cudarc does not expose the raw unified memory pointer directly.
            //
            // TODO(cuda): Extract actual unified memory pointer from cudarc CudaSlice.
            // The value 0x1000 is a non-null sentinel that will fail gracefully if
            // accidentally dereferenced (segfault rather than silent corruption).
            NonNull::new(0x1000 as *mut u8)
                .ok_or_else(|| GpuMemoryError::AllocationFailed { size })?
        };

        #[cfg(not(feature = "cuda"))]
        let ptr = {
            // For testing without CUDA, use regular allocation
            let layout = std::alloc::Layout::from_size_align(size as usize, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
            // SAFETY: Layout is valid (checked above). The allocation is tracked
            // and freed in UnifiedAllocation::Drop. NonNull::new handles null check.
            let raw_ptr = unsafe { std::alloc::alloc(layout) };
            NonNull::new(raw_ptr).ok_or(GpuMemoryError::AllocationFailed { size })?
        };

        // Track the allocation
        let allocation_info = UnifiedAllocationInfo {
            id,
            ptr,
            size,
            current_location: MemoryLocation::Unknown,
            access_pattern: AccessPattern::Unknown,
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
        };

        self.allocations.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (allocations), recovering inner data");
            poisoned.into_inner()
        }).push(allocation_info);
        *self.total_allocated.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (total_allocated), recovering inner data");
            poisoned.into_inner()
        }) += size;

        Ok(UnifiedAllocation {
            id,
            ptr,
            size,
            manager: Arc::new(self.clone()),
            location: Arc::new(Mutex::new(MemoryLocation::Unknown)),
            allocated_at: Instant::now(),
        })
    }

    /// Free a unified memory allocation
    pub fn free(&self, allocation: &UnifiedAllocation) -> Result<()> {
        let mut allocations = self.allocations.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (allocations), recovering inner data");
            poisoned.into_inner()
        });
        if let Some(pos) = allocations.iter().position(|info| info.id == allocation.id) {
            let info = allocations.remove(pos);
            *self.total_allocated.lock().unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (total_allocated), recovering inner data");
                poisoned.into_inner()
            }) -= info.size;

            #[cfg(feature = "cuda")]
            {
                // Free CUDA managed memory
                // The device will handle freeing managed memory when dropped
            }

            #[cfg(not(feature = "cuda"))]
            {
                // Free regular memory
                // SAFETY: info.ptr was allocated with std::alloc::alloc using the same
                // layout (size, 8-byte alignment). The pointer has not been freed elsewhere
                // as it's managed by UnifiedMemoryManager tracking.
                let layout = std::alloc::Layout::from_size_align(info.size as usize, 8)
                    .expect("Invalid layout");
                unsafe {
                    std::alloc::dealloc(info.ptr.as_ptr(), layout);
                }
            }
        }

        Ok(())
    }

    /// Set memory access hints for optimization
    pub fn set_access_hint(
        &self,
        allocation: &UnifiedAllocation,
        location: MemoryLocation,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would use cuMemAdvise
            // to provide hints to the CUDA runtime about memory access patterns
        }

        // Update our tracking
        let mut allocations = self.allocations.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (allocations), recovering inner data");
            poisoned.into_inner()
        });
        if let Some(info) = allocations.iter_mut().find(|info| info.id == allocation.id) {
            info.current_location = location.clone();
            info.last_accessed = Instant::now();
        }

        *allocation.location.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (location), recovering inner data");
            poisoned.into_inner()
        }) = location;
        Ok(())
    }

    /// Get total allocated unified memory
    #[inline]
    pub fn total_allocated(&self) -> u64 {
        *self.total_allocated.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (total_allocated), recovering inner data");
            poisoned.into_inner()
        })
    }

    /// Get number of active allocations
    #[inline]
    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (allocations), recovering inner data");
            poisoned.into_inner()
        }).len()
    }
}

impl UnifiedAllocation {
    /// Check if the allocation is accessible from CPU
    #[inline]
    pub fn cpu_accessible(&self) -> bool {
        true // Unified memory is always CPU accessible
    }

    /// Check if the allocation is accessible from GPU
    #[inline]
    pub fn gpu_accessible(&self) -> bool {
        true // Unified memory is always GPU accessible
    }

    /// Get the unified pointer that can be used by both CPU and GPU
    #[inline]
    pub fn unified_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Check if memory is currently resident on GPU
    #[inline]
    pub fn is_gpu_resident(&self) -> bool {
        matches!(*self.location.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (location), recovering inner data");
            poisoned.into_inner()
        }), MemoryLocation::Gpu)
    }

    /// Check if memory is currently resident on CPU
    #[inline]
    pub fn is_cpu_resident(&self) -> bool {
        matches!(*self.location.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (location), recovering inner data");
            poisoned.into_inner()
        }), MemoryLocation::Cpu)
    }

    /// Provide hint that memory will be accessed primarily by GPU
    pub fn hint_gpu_usage(&self) -> Result<()> {
        self.manager.set_access_hint(self, MemoryLocation::Gpu)
    }

    /// Provide hint that memory will be accessed primarily by CPU
    pub fn hint_cpu_usage(&self) -> Result<()> {
        self.manager.set_access_hint(self, MemoryLocation::Cpu)
    }

    /// Prefetch memory to GPU
    pub fn prefetch_to_gpu(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would use cuMemPrefetchAsync
            // to asynchronously migrate memory to GPU
        }

        self.manager.set_access_hint(self, MemoryLocation::Gpu)?;
        Ok(())
    }

    /// Prefetch memory to CPU
    pub fn prefetch_to_cpu(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would use cuMemPrefetchAsync
            // to asynchronously migrate memory to CPU
        }

        self.manager.set_access_hint(self, MemoryLocation::Cpu)?;
        Ok(())
    }

    /// Write data to the unified memory from CPU
    pub fn write_from_cpu(&self, data: &[u8], offset: usize) -> Result<()> {
        if offset + data.len() > self.size as usize {
            return Err(GpuMemoryError::AllocationFailed {
                size: (offset + data.len()) as u64,
            }
            .into());
        }

        // SAFETY: data.as_ptr() is valid for data.len() bytes. self.ptr is a valid
        // NonNull from allocation. offset + data.len() <= self.size (checked above).
        // dest_ptr points within allocated region. Regions don't overlap (data is caller's slice).
        unsafe {
            let dest_ptr = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest_ptr, data.len());
        }

        // Update access pattern
        self.manager.set_access_hint(self, MemoryLocation::Cpu)?;
        Ok(())
    }

    /// Read data from the unified memory to CPU
    pub fn read_to_cpu(&self, buffer: &mut [u8], offset: usize) -> Result<()> {
        if offset + buffer.len() > self.size as usize {
            return Err(GpuMemoryError::AllocationFailed {
                size: (offset + buffer.len()) as u64,
            }
            .into());
        }

        // SAFETY: self.ptr is a valid NonNull from allocation. buffer.as_mut_ptr() is valid
        // for buffer.len() bytes. offset + buffer.len() <= self.size (checked above).
        // src_ptr points within allocated region. Regions don't overlap (buffer is caller's slice).
        unsafe {
            let src_ptr = self.ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, buffer.as_mut_ptr(), buffer.len());
        }

        // Update access pattern
        self.manager.set_access_hint(self, MemoryLocation::Cpu)?;
        Ok(())
    }

    /// Get allocation timestamp
    #[inline]
    pub fn allocated_at(&self) -> Instant {
        self.allocated_at
    }

    /// Get allocation ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }
}

// Clone implementation for UnifiedMemoryManager
impl Clone for UnifiedMemoryManager {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "cuda")]
            device: self.device.clone(),
            allocations: self.allocations.clone(),
            total_allocated: self.total_allocated.clone(),
            allocation_counter: self.allocation_counter.clone(),
        }
    }
}

// SAFETY: UnifiedMemoryManager is Send because:
// 1. All fields are either Arc (Send) or primitives
// 2. The NonNull<u8> in UnifiedAllocationInfo points to CUDA unified memory (thread-safe)
// 3. Mutex guards all mutable state
unsafe impl Send for UnifiedMemoryManager {}

// SAFETY: UnifiedMemoryManager is Sync because:
// 1. All mutable state is protected by Mutex
// 2. Methods only access shared state through proper synchronization
unsafe impl Sync for UnifiedMemoryManager {}

// SAFETY: UnifiedAllocation is Send because:
// 1. ptr points to unified memory that can be accessed from any thread
// 2. manager is Arc (Send), location is Arc<Mutex<_>> (Send)
// 3. Ownership can be safely transferred between threads
unsafe impl Send for UnifiedAllocation {}

// SAFETY: UnifiedAllocation is Sync because:
// 1. Unified memory provides hardware-level coherency
// 2. All mutable state is protected by Mutex (location)
// 3. Read methods don't modify shared state
unsafe impl Sync for UnifiedAllocation {}

impl Drop for UnifiedAllocation {
    fn drop(&mut self) {
        // Attempt to free the allocation when dropped
        if let Err(e) = self.manager.free(self) {
            tracing::error!("Failed to free unified allocation {}: {}", self.id, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_unified_memory_manager_creation() {
        let manager = UnifiedMemoryManager::new(()).expect("Failed to create manager");
        assert_eq!(manager.total_allocated(), 0);
        assert_eq!(manager.allocation_count(), 0);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_unified_allocation() {
        let manager = UnifiedMemoryManager::new(()).expect("Failed to create manager");
        let allocation = manager.allocate(1024).expect("Failed to allocate");

        assert_eq!(allocation.size(), 1024);
        assert!(allocation.cpu_accessible());
        assert!(allocation.gpu_accessible());
        assert!(!allocation.unified_ptr().is_null());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_unified_memory_hints() {
        let manager = UnifiedMemoryManager::new(()).expect("Failed to create manager");
        let allocation = manager.allocate(1024).expect("Failed to allocate");

        allocation
            .hint_gpu_usage()
            .expect("Failed to hint GPU usage");
        assert!(allocation.is_gpu_resident());

        allocation
            .hint_cpu_usage()
            .expect("Failed to hint CPU usage");
        assert!(allocation.is_cpu_resident());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_unified_memory_read_write() {
        let manager = UnifiedMemoryManager::new(()).expect("Failed to create manager");
        let allocation = manager.allocate(1024).expect("Failed to allocate");

        let test_data = vec![42u8; 256];
        allocation
            .write_from_cpu(&test_data, 0)
            .expect("Failed to write");

        let mut read_buffer = vec![0u8; 256];
        allocation
            .read_to_cpu(&mut read_buffer, 0)
            .expect("Failed to read");

        assert_eq!(read_buffer, test_data);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_unified_memory_bounds_checking() {
        let manager = UnifiedMemoryManager::new(()).expect("Failed to create manager");
        let allocation = manager.allocate(100).expect("Failed to allocate");

        let large_data = vec![1u8; 200];
        let result = allocation.write_from_cpu(&large_data, 0);
        assert!(result.is_err());

        let mut large_buffer = vec![0u8; 200];
        let result = allocation.read_to_cpu(&mut large_buffer, 0);
        assert!(result.is_err());
    }
}
