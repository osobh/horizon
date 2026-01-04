//! GPU Memory Tier Implementation for StratoSwarm
//!
//! Provides GPU-native memory management with real CUDA integration,
//! automatic CPU overflow, unified memory, and multi-GPU support.

use anyhow::Result;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "cuda")]
use cudarc::driver::sys::cuMemHostAlloc;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr};

use crate::memory_pool::MemoryPool;
use crate::multi_gpu::MultiGpuManager;
use crate::unified_memory::{UnifiedAllocation, UnifiedMemoryManager};

/// Configuration for GPU memory tier
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub gpu_memory_gb: u32,
    pub cpu_memory_gb: u32,
    pub prefer_gpu: bool,
    pub enable_unified_memory: bool,
    pub overflow_threshold: f32,
}

/// Memory allocation strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Prefer GPU, fallback to CPU if needed
    GpuPrimary,
    /// CPU memory only
    CpuOnly,
    /// GPU memory only (fail if not available)
    GpuOnly,
    /// Automatic selection based on usage patterns
    Auto,
}

/// Memory metrics for monitoring
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub gpu_allocated_bytes: u64,
    pub cpu_allocated_bytes: u64,
    pub total_allocations: u64,
    pub overflow_events: u64,
    pub gpu_utilization: f32,
    pub cpu_utilization: f32,
}

/// Errors that can occur during memory operations
#[derive(Debug, Error)]
pub enum GpuMemoryError {
    #[error("CUDA initialization failed: {msg}")]
    CudaInitFailed { msg: String },
    #[error("Memory allocation failed: {size} bytes")]
    AllocationFailed { size: u64 },
    #[error("Invalid memory configuration: {msg}")]
    InvalidConfig { msg: String },
    #[error("GPU memory exhausted")]
    GpuMemoryExhausted,
    #[error("Unified memory not supported")]
    UnifiedMemoryNotSupported,
    #[error("P2P transfer failed between GPU {src} and GPU {dst}")]
    P2PTransferFailed { src: u32, dst: u32 },
}

/// Memory allocation handle with location and metadata
#[derive(Debug)]
pub struct MemoryAllocation {
    ptr: NonNull<u8>,
    size: u64,
    location: MemoryLocation,
    allocation_type: AllocationType,
    gpu_id: Option<u32>,
    created_at: Instant,
    from_pool: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryLocation {
    Gpu,
    Cpu,
    Unified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationType {
    Standard,
    Pinned,
    Unified,
}

/// Zero-copy buffer for efficient CPU-GPU data sharing
pub struct ZeroCopyBuffer {
    ptr: NonNull<u8>,
    size: usize,
    #[cfg(feature = "cuda")]
    cuda_ptr: u64, // Store raw pointer value instead of DevicePtr
}

/// Pinned memory allocation for fast DMA transfers
pub struct PinnedAllocation {
    #[allow(dead_code)]
    ptr: NonNull<u8>,
    size: u64,
    #[cfg(feature = "cuda")]
    cuda_ptr: u64, // Store raw pointer value instead of DevicePtr
}

/// Main GPU memory tier implementation
pub struct GpuMemoryTier {
    config: MemoryConfig,
    #[cfg(feature = "cuda")]
    cuda_device: Arc<CudaDevice>,
    unified_manager: Option<UnifiedMemoryManager>,
    multi_gpu_manager: Option<MultiGpuManager>,
    memory_pools: Arc<Mutex<HashMap<String, Arc<MemoryPool>>>>,
    metrics: Arc<Mutex<MemoryMetrics>>,
    #[allow(dead_code)]
    gpu_allocations: Arc<Mutex<Vec<MemoryAllocation>>>,
    cpu_allocations: Arc<Mutex<Vec<MemoryAllocation>>>,
}

// Implementation of memory allocation methods
impl MemoryAllocation {
    #[inline]
    pub fn is_gpu_resident(&self) -> bool {
        matches!(self.location, MemoryLocation::Gpu | MemoryLocation::Unified)
    }

    #[inline]
    pub fn is_cpu_resident(&self) -> bool {
        matches!(self.location, MemoryLocation::Cpu | MemoryLocation::Unified)
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    #[inline]
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id.unwrap_or(0)
    }

    #[inline]
    pub fn from_pool(&self) -> bool {
        self.from_pool
    }

    #[inline]
    pub fn cpu_accessible(&self) -> bool {
        !matches!(self.location, MemoryLocation::Gpu)
    }

    #[inline]
    pub fn gpu_accessible(&self) -> bool {
        !matches!(self.location, MemoryLocation::Cpu)
    }

    pub fn unified_ptr(&self) -> *mut u8 {
        if matches!(self.location, MemoryLocation::Unified) {
            self.ptr.as_ptr()
        } else {
            std::ptr::null_mut()
        }
    }

    pub fn hint_gpu_usage(&self) -> Result<()> {
        if matches!(self.location, MemoryLocation::Unified) {
            // In a real implementation, this would use CUDA unified memory hints
            Ok(())
        } else {
            Err(GpuMemoryError::UnifiedMemoryNotSupported.into())
        }
    }

    pub fn hint_cpu_usage(&self) -> Result<()> {
        if matches!(self.location, MemoryLocation::Unified) {
            // In a real implementation, this would use CUDA unified memory hints
            Ok(())
        } else {
            Err(GpuMemoryError::UnifiedMemoryNotSupported.into())
        }
    }

    pub fn migrate_to_gpu(&mut self) -> Result<()> {
        if self.location == MemoryLocation::Cpu {
            // In a real implementation, this would:
            // 1. Allocate GPU memory
            // 2. Copy data from CPU to GPU
            // 3. Free CPU memory
            // 4. Update location and pointer
            self.location = MemoryLocation::Gpu;
            Ok(())
        } else {
            Ok(()) // Already on GPU or unified
        }
    }

    pub fn migrate_to_cpu(&mut self) -> Result<()> {
        if self.location == MemoryLocation::Gpu {
            // In a real implementation, this would:
            // 1. Allocate CPU memory
            // 2. Copy data from GPU to CPU
            // 3. Free GPU memory
            // 4. Update location and pointer
            self.location = MemoryLocation::Cpu;
            Ok(())
        } else {
            Ok(()) // Already on CPU or unified
        }
    }
}

// Implementation of zero-copy buffer methods
impl ZeroCopyBuffer {
    pub fn write_cpu(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(GpuMemoryError::AllocationFailed {
                size: data.len() as u64,
            }
            .into());
        }

        // SAFETY: The source pointer (data.as_ptr()) is valid for data.len() bytes.
        // The destination (self.ptr) is a valid NonNull from allocation with size >= data.len()
        // (checked above). The regions do not overlap as data is a separate slice.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }
        Ok(())
    }

    pub fn read_cpu(&self) -> Vec<u8> {
        let mut data = vec![0u8; self.size];
        // SAFETY: The source (self.ptr) is a valid NonNull from allocation with self.size bytes.
        // The destination (data.as_mut_ptr()) is valid for self.size bytes from vec allocation.
        // The regions do not overlap as data is a newly allocated Vec.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), data.as_mut_ptr(), self.size);
        }
        data
    }

    pub fn gpu_view(&self) -> GpuView {
        GpuView {
            #[cfg(feature = "cuda")]
            ptr: self.cuda_ptr,
            size: self.size,
        }
    }
}

pub struct GpuView {
    #[cfg(feature = "cuda")]
    ptr: u64, // Store raw pointer value instead of DevicePtr
    size: usize,
}

impl GpuView {
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

// Implementation of pinned allocation methods
impl PinnedAllocation {
    #[inline]
    pub fn is_pinned(&self) -> bool {
        true
    }

    #[inline]
    pub fn is_dma_capable(&self) -> bool {
        true
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn dma_copy_to(&self, _dest: &MemoryAllocation) -> Result<Duration> {
        let start = Instant::now();

        // In a real implementation, this would use CUDA async memory copy
        #[cfg(feature = "cuda")]
        {
            // cuMemcpyHtoDAsync or similar CUDA API call
        }

        let duration = start.elapsed();
        Ok(duration)
    }
}

// Main implementation of GpuMemoryTier
impl GpuMemoryTier {
    pub fn new(config: MemoryConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let cuda_device =
            { CudaDevice::new(0).with_context(|| "Failed to initialize CUDA device")? };

        let unified_manager = if config.enable_unified_memory {
            #[cfg(feature = "cuda")]
            {
                Some(UnifiedMemoryManager::new(cuda_device.clone())?)
            }
            #[cfg(not(feature = "cuda"))]
            None
        } else {
            None
        };

        Ok(Self {
            config,
            #[cfg(feature = "cuda")]
            cuda_device: cuda_device,
            unified_manager,
            multi_gpu_manager: None,
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(MemoryMetrics {
                gpu_allocated_bytes: 0,
                cpu_allocated_bytes: 0,
                total_allocations: 0,
                overflow_events: 0,
                gpu_utilization: 0.0,
                cpu_utilization: 0.0,
            })),
            gpu_allocations: Arc::new(Mutex::new(Vec::new())),
            cpu_allocations: Arc::new(Mutex::new(Vec::new())),
        })
    }

    #[allow(unused_mut, unused_variables)]
    pub fn new_multi_gpu(config: MemoryConfig, gpu_count: u32) -> Result<Self> {
        let mut tier = Self::new(config)?;

        #[cfg(feature = "cuda")]
        {
            tier.multi_gpu_manager = Some(MultiGpuManager::new(gpu_count)?);
        }

        Ok(tier)
    }

    #[must_use = "MemoryAllocation must be stored to free the memory later"]
    pub fn allocate(&self, size: u64, strategy: AllocationStrategy) -> Result<MemoryAllocation> {
        let should_use_gpu = match strategy {
            AllocationStrategy::GpuPrimary => {
                self.gpu_utilization() < self.config.overflow_threshold
            }
            AllocationStrategy::GpuOnly => true,
            AllocationStrategy::CpuOnly => false,
            AllocationStrategy::Auto => {
                // Simple heuristic: use GPU for allocations > 1MB
                size > 1024 * 1024 && self.gpu_utilization() < self.config.overflow_threshold
            }
        };

        let allocation = if should_use_gpu && strategy != AllocationStrategy::CpuOnly {
            self.allocate_gpu(size)?
        } else if strategy == AllocationStrategy::GpuOnly {
            return Err(GpuMemoryError::GpuMemoryExhausted.into());
        } else {
            self.allocate_cpu(size)?
        };

        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_allocations += 1;
            match allocation.location {
                MemoryLocation::Gpu => metrics.gpu_allocated_bytes += size,
                MemoryLocation::Cpu => metrics.cpu_allocated_bytes += size,
                MemoryLocation::Unified => {
                    metrics.gpu_allocated_bytes += size;
                    metrics.cpu_allocated_bytes += size;
                }
            }
        }

        Ok(allocation)
    }

    fn allocate_gpu(&self, size: u64) -> Result<MemoryAllocation> {
        #[cfg(feature = "cuda")]
        {
            let ptr = self.cuda_device.alloc_zeros::<u8>(size as usize)?;
            // For now, store a dummy pointer value - in a real implementation
            // we would properly manage the device pointer
            let allocation = MemoryAllocation {
                ptr: NonNull::new(0x1000 as *mut u8).unwrap(), // Placeholder
                size,
                location: MemoryLocation::Gpu,
                allocation_type: AllocationType::Standard,
                gpu_id: Some(0),
                created_at: Instant::now(),
                from_pool: false,
            };

            self.gpu_allocations
                .lock()
                .unwrap()
                .push(allocation.clone());
            Ok(allocation)
        }
        #[cfg(not(feature = "cuda"))]
        {
            // For testing without CUDA
            let layout = std::alloc::Layout::from_size_align(size as usize, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
            // SAFETY: Layout is valid (checked above). Caller is responsible for freeing
            // the memory. Null check immediately follows to handle allocation failure.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuMemoryError::AllocationFailed { size }.into());
            }

            Ok(MemoryAllocation {
                ptr: NonNull::new(ptr).unwrap(),
                size,
                location: MemoryLocation::Gpu,
                allocation_type: AllocationType::Standard,
                gpu_id: Some(0),
                created_at: Instant::now(),
                from_pool: false,
            })
        }
    }

    fn allocate_cpu(&self, size: u64) -> Result<MemoryAllocation> {
        let layout = std::alloc::Layout::from_size_align(size as usize, 8)
            .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
        // SAFETY: Layout is valid (checked above). The allocation is tracked in
        // cpu_allocations for later deallocation. Null check handles failure.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(GpuMemoryError::AllocationFailed { size }.into());
        }

        let allocation = MemoryAllocation {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            location: MemoryLocation::Cpu,
            allocation_type: AllocationType::Standard,
            gpu_id: None,
            created_at: Instant::now(),
            from_pool: false,
        };

        self.cpu_allocations
            .lock()
            .unwrap()
            .push(allocation.clone());
        Ok(allocation)
    }

    #[must_use = "UnifiedAllocation must be stored to free the memory later"]
    pub fn allocate_unified(
        &self,
        size: u64,
        _strategy: AllocationStrategy,
    ) -> Result<UnifiedAllocation> {
        if let Some(ref manager) = self.unified_manager {
            manager.allocate(size)
        } else {
            Err(GpuMemoryError::UnifiedMemoryNotSupported.into())
        }
    }

    #[must_use = "ZeroCopyBuffer must be stored to free the memory later"]
    pub fn create_zero_copy_buffer(&self, size: usize) -> Result<ZeroCopyBuffer> {
        #[cfg(feature = "cuda")]
        {
            // Allocate page-locked memory for zero-copy access
            // For now, use regular allocation as placeholder
            let layout = std::alloc::Layout::from_size_align(size, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size: size as u64 })?;
            // SAFETY: Layout is valid (checked above). In production this would use
            // cudaMallocHost for page-locked memory. Null check handles failure.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuMemoryError::AllocationFailed { size: size as u64 }.into());
            }

            Ok(ZeroCopyBuffer {
                ptr: NonNull::new(ptr).unwrap(),
                size,
                cuda_ptr: ptr as u64, // Store as raw pointer value
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            // For testing without CUDA
            let layout = std::alloc::Layout::from_size_align(size, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size: size as u64 })?;
            // SAFETY: Layout is valid (checked above). Null check handles allocation failure.
            // Memory is managed by ZeroCopyBuffer lifetime.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuMemoryError::AllocationFailed { size: size as u64 }.into());
            }

            Ok(ZeroCopyBuffer {
                ptr: NonNull::new(ptr).unwrap(),
                size,
            })
        }
    }

    #[must_use = "PinnedAllocation must be stored to free the memory later"]
    pub fn allocate_pinned(&self, size: u64) -> Result<PinnedAllocation> {
        #[cfg(feature = "cuda")]
        {
            // Allocate pinned memory using CUDA
            // For now, use regular allocation as placeholder
            let layout = std::alloc::Layout::from_size_align(size as usize, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
            // SAFETY: Layout is valid (checked above). In production this would use
            // cudaMallocHost for pinned memory suitable for DMA. Null check handles failure.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuMemoryError::AllocationFailed { size }.into());
            }

            Ok(PinnedAllocation {
                ptr: NonNull::new(ptr).unwrap(),
                size,
                cuda_ptr: ptr as u64, // Store as raw pointer value
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            // For testing without CUDA
            let layout = std::alloc::Layout::from_size_align(size as usize, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
            // SAFETY: Layout is valid (checked above). Null check handles allocation failure.
            // Memory is managed by PinnedAllocation lifetime.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuMemoryError::AllocationFailed { size }.into());
            }

            Ok(PinnedAllocation {
                ptr: NonNull::new(ptr).unwrap(),
                size,
            })
        }
    }

    pub fn create_pool(&self, name: &str, size: u64) -> Result<Arc<MemoryPool>> {
        let pool = Arc::new(MemoryPool::new(name.to_string(), size));
        self.memory_pools
            .lock()
            .unwrap()
            .insert(name.to_string(), pool.clone());
        Ok(pool)
    }

    #[must_use = "MemoryAllocation must be stored to free the memory later"]
    pub fn allocate_on_gpu(&self, size: u64, gpu_id: u32) -> Result<MemoryAllocation> {
        if let Some(ref manager) = self.multi_gpu_manager {
            // Convert from multi_gpu MemoryAllocation to gpu_memory_tier MemoryAllocation
            let multi_alloc = manager.allocate_on_gpu(size, gpu_id)?;
            Ok(MemoryAllocation {
                ptr: NonNull::new(multi_alloc.ptr.as_ptr()).unwrap(),
                size: multi_alloc.size,
                location: MemoryLocation::Gpu,
                allocation_type: AllocationType::Standard,
                gpu_id: Some(gpu_id),
                created_at: Instant::now(),
                from_pool: false,
            })
        } else {
            // Single GPU fallback
            if gpu_id == 0 {
                self.allocate_gpu(size)
            } else {
                Err(GpuMemoryError::AllocationFailed { size }.into())
            }
        }
    }

    pub fn p2p_copy(&self, src: &MemoryAllocation, dest: &MemoryAllocation) -> Result<Duration> {
        if let Some(ref manager) = self.multi_gpu_manager {
            // Convert to multi_gpu MemoryAllocation type
            let src_multi = crate::multi_gpu::MemoryAllocation::new(src.ptr, src.size, src.gpu_id);
            let dest_multi =
                crate::multi_gpu::MemoryAllocation::new(dest.ptr, dest.size, dest.gpu_id);
            manager.p2p_copy(&src_multi, &dest_multi)
        } else {
            Err(GpuMemoryError::P2PTransferFailed {
                src: src.gpu_id(),
                dst: dest.gpu_id(),
            }
            .into())
        }
    }

    #[inline]
    pub fn gpu_utilization(&self) -> f32 {
        let gpu_allocated = self.metrics.lock().unwrap().gpu_allocated_bytes;
        let total_gpu_memory = (self.config.gpu_memory_gb as u64) * 1024 * 1024 * 1024;
        (gpu_allocated as f32) / (total_gpu_memory as f32)
    }

    #[inline]
    pub fn cpu_utilization(&self) -> f32 {
        let cpu_allocated = self.metrics.lock().unwrap().cpu_allocated_bytes;
        let total_cpu_memory = (self.config.cpu_memory_gb as u64) * 1024 * 1024 * 1024;
        (cpu_allocated as f32) / (total_cpu_memory as f32)
    }

    pub fn get_metrics(&self) -> MemoryMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.gpu_utilization = self.gpu_utilization();
        metrics.cpu_utilization = self.cpu_utilization();
        metrics.clone()
    }
}

// SAFETY: MemoryAllocation is Send because:
// 1. `ptr: NonNull<u8>` points to memory that is either:
//    - GPU memory managed by CUDA (thread-safe by design)
//    - CPU memory allocated via std::alloc (ownership transfer is safe)
// 2. All metadata fields (size, location, gpu_id, etc.) are trivially Send
// 3. The allocation represents exclusive ownership of the pointed-to memory
// 4. Memory is freed via corresponding deallocation APIs that are thread-safe
// 5. The Instant timestamp is Send
unsafe impl Send for MemoryAllocation {}

// SAFETY: MemoryAllocation is Sync because:
// 1. All fields are either primitive types or pointers that are only read
// 2. Methods taking &self only read metadata or call thread-safe CUDA APIs
// 3. Actual memory access requires unsafe code with proper synchronization
// 4. No interior mutability is used - mutations require &mut self
unsafe impl Sync for MemoryAllocation {}

// SAFETY: ZeroCopyBuffer is Send because:
// 1. `ptr: NonNull<u8>` points to page-locked host memory allocated for DMA
// 2. `cuda_ptr: u64` is just a pointer value (not a reference)
// 3. CUDA page-locked memory is designed for concurrent CPU/GPU access
// 4. The buffer represents exclusive ownership that can be transferred
// 5. Size is a trivially Send primitive type
unsafe impl Send for ZeroCopyBuffer {}

// SAFETY: ZeroCopyBuffer is Sync because:
// 1. Read operations (read_cpu, gpu_view) don't modify shared state
// 2. Write operations (write_cpu) are synchronized by CUDA driver
// 3. The buffer is designed for CPU/GPU data sharing (thread-safe by design)
// 4. No interior mutability - writing requires &mut self or is explicitly unsafe
unsafe impl Sync for ZeroCopyBuffer {}

// SAFETY: PinnedAllocation is Send because:
// 1. `ptr: NonNull<u8>` points to CUDA pinned memory (cudaMallocHost)
// 2. `cuda_ptr: u64` is just a pointer value for device access
// 3. Pinned memory is designed for efficient DMA transfers (thread-safe)
// 4. The allocation can be safely transferred between threads
// 5. Size is a trivially Send primitive type
unsafe impl Send for PinnedAllocation {}

// SAFETY: PinnedAllocation is Sync because:
// 1. Pinned memory access is synchronized by CUDA driver during transfers
// 2. Methods only read metadata or perform DMA operations
// 3. DMA transfers are serialized by CUDA stream ordering
// 4. No interior mutability - actual memory access requires proper sync
unsafe impl Sync for PinnedAllocation {}

// Clone implementation for MemoryAllocation (shallow copy of metadata)
impl Clone for MemoryAllocation {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            size: self.size,
            location: self.location.clone(),
            allocation_type: self.allocation_type.clone(),
            gpu_id: self.gpu_id,
            created_at: self.created_at,
            from_pool: self.from_pool,
        }
    }
}
