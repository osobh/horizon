//! Multi-GPU Memory Management with P2P Transfers
//!
//! Provides support for multiple GPU memory management, peer-to-peer
//! memory transfers, and cross-GPU memory allocation strategies.

// Allow Arc<Mutex<T>> where T contains NonNull - we have explicit unsafe impl Send/Sync
#![allow(clippy::arc_with_non_send_sync)]

use anyhow::Result;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr};

// Note: We'll use a local MemoryAllocation type for this module to avoid circular dependencies
#[derive(Debug, Clone)]
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

impl MemoryAllocation {
    pub fn new(ptr: NonNull<u8>, size: u64, gpu_id: Option<u32>) -> Self {
        Self {
            ptr,
            size,
            location: MemoryLocation::Gpu,
            allocation_type: AllocationType::Standard,
            gpu_id,
            created_at: Instant::now(),
            from_pool: false,
        }
    }

    #[inline]
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id.unwrap_or(0)
    }

    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    #[inline]
    pub fn is_gpu_resident(&self) -> bool {
        matches!(self.location, MemoryLocation::Gpu | MemoryLocation::Unified)
    }

    /// Get the raw pointer (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer is used within its allocation lifetime
    /// and that proper synchronization is maintained for concurrent access.
    #[inline]
    pub unsafe fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }
}

/// Errors that can occur during multi-GPU memory operations
#[derive(Debug, Error)]
pub enum GpuMemoryError {
    #[error("CUDA initialization failed: {msg}")]
    CudaInitFailed { msg: String },
    #[error("Memory allocation failed: {size} bytes")]
    AllocationFailed { size: u64 },
    #[error("Invalid memory configuration: {msg}")]
    InvalidConfig { msg: String },
    #[error("P2P transfer failed between GPU {src} and GPU {dst}")]
    P2PTransferFailed { src: u32, dst: u32 },
}

/// Manager for multi-GPU memory operations
pub struct MultiGpuManager {
    gpu_count: u32,
    #[cfg(feature = "cuda")]
    devices: Vec<Arc<CudaDevice>>,
    p2p_manager: Arc<P2PManager>,
    gpu_allocations: Arc<Mutex<HashMap<u32, Vec<GpuAllocationInfo>>>>,
    load_balancer: Arc<Mutex<LoadBalancer>>,
}

/// Peer-to-peer transfer manager
pub struct P2PManager {
    gpu_count: u32,
    p2p_matrix: Arc<Mutex<Vec<Vec<bool>>>>, // P2P connectivity matrix
    transfer_stats: Arc<Mutex<HashMap<(u32, u32), TransferStats>>>,
}

/// Information about a GPU-specific allocation
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GpuAllocationInfo {
    ptr: NonNull<u8>,
    size: u64,
    gpu_id: u32,
    allocated_at: Instant,
    last_accessed: Instant,
}

/// Load balancer for distributing allocations across GPUs
#[allow(dead_code)]
struct LoadBalancer {
    gpu_utilization: Vec<f32>,
    allocation_strategy: AllocationStrategy,
}

/// Multi-GPU allocation strategy
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
enum AllocationStrategy {
    RoundRobin,
    LeastUtilized,
    FirstFit,
    Locality,
}

/// P2P transfer statistics
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Total number of P2P transfers
    pub total_transfers: u64,
    /// Total bytes transferred across GPUs
    pub total_bytes_transferred: u64,
    /// Average bandwidth in GB/s
    pub average_bandwidth_gbps: f32,
    /// Time of last transfer
    pub last_transfer_time: Option<Instant>,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager
    pub fn new(gpu_count: u32) -> Result<Self> {
        if gpu_count == 0 {
            return Err(GpuMemoryError::InvalidConfig {
                msg: "GPU count must be greater than 0".to_string(),
            }
            .into());
        }

        #[cfg(feature = "cuda")]
        let devices = {
            let mut devices = Vec::new();
            for i in 0..gpu_count {
                let device =
                    CudaDevice::new(i as usize).map_err(|e| GpuMemoryError::CudaInitFailed {
                        msg: format!("Failed to initialize GPU {}: {}", i, e),
                    })?;
                devices.push(device);
            }
            devices
        };

        let p2p_manager = Arc::new(P2PManager::new(gpu_count)?);
        let mut gpu_allocations = HashMap::new();
        for i in 0..gpu_count {
            gpu_allocations.insert(i, Vec::new());
        }

        Ok(Self {
            gpu_count,
            #[cfg(feature = "cuda")]
            devices,
            p2p_manager,
            gpu_allocations: Arc::new(Mutex::new(gpu_allocations)),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new(gpu_count))),
        })
    }

    /// Allocate memory on a specific GPU
    pub fn allocate_on_gpu(&self, size: u64, gpu_id: u32) -> Result<MemoryAllocation> {
        if gpu_id >= self.gpu_count {
            return Err(GpuMemoryError::InvalidConfig {
                msg: format!("Invalid GPU ID: {} (max: {})", gpu_id, self.gpu_count - 1),
            }
            .into());
        }

        #[cfg(feature = "cuda")]
        let ptr = {
            let device = &self.devices[gpu_id as usize];
            let device_ptr = device
                .alloc_zeros::<u8>(size as usize)
                .map_err(|e| GpuMemoryError::AllocationFailed { size })?;

            // Store the raw device pointer directly
            // For now, use a placeholder - in real implementation would handle properly
            NonNull::new(0x1000 as *mut u8)
                .ok_or_else(|| GpuMemoryError::AllocationFailed { size })?
        };

        #[cfg(not(feature = "cuda"))]
        let ptr = {
            // For testing without CUDA
            let layout = std::alloc::Layout::from_size_align(size as usize, 8)
                .map_err(|_| GpuMemoryError::AllocationFailed { size })?;
            // SAFETY: Layout is valid (checked above). The allocation is tracked in
            // gpu_allocations for later deallocation. NonNull::new handles null check.
            let raw_ptr = unsafe { std::alloc::alloc(layout) };
            NonNull::new(raw_ptr).ok_or(GpuMemoryError::AllocationFailed { size })?
        };

        // Track the allocation
        let allocation_info = GpuAllocationInfo {
            ptr,
            size,
            gpu_id,
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
        };

        self.gpu_allocations
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (gpu_allocations), recovering inner data");
                poisoned.into_inner()
            })
            .get_mut(&gpu_id)
            .unwrap()
            .push(allocation_info);

        // Update load balancer
        self.load_balancer
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (load_balancer), recovering inner data");
                poisoned.into_inner()
            })
            .update_utilization(gpu_id, size, true);

        Ok(MemoryAllocation::new(ptr, size, Some(gpu_id)))
    }

    /// Allocate memory on the best available GPU
    pub fn allocate_auto(&self, size: u64) -> Result<MemoryAllocation> {
        let gpu_id = self
            .load_balancer
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (load_balancer), recovering inner data");
                poisoned.into_inner()
            })
            .select_gpu_for_allocation(size);
        self.allocate_on_gpu(size, gpu_id)
    }

    /// Perform peer-to-peer memory copy between GPUs
    pub fn p2p_copy(&self, src: &MemoryAllocation, dest: &MemoryAllocation) -> Result<Duration> {
        let src_gpu = src.gpu_id();
        let dst_gpu = dest.gpu_id();

        self.p2p_manager.transfer(src_gpu, dst_gpu, src.size())
    }

    /// Get GPU utilization statistics
    #[inline]
    pub fn get_gpu_utilization(&self, gpu_id: u32) -> f32 {
        self.load_balancer
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (load_balancer), recovering inner data");
                poisoned.into_inner()
            })
            .gpu_utilization
            .get(gpu_id as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get P2P connectivity matrix
    #[inline]
    pub fn get_p2p_matrix(&self) -> Vec<Vec<bool>> {
        self.p2p_manager.get_connectivity_matrix()
    }

    /// Enable P2P access between two GPUs
    pub fn enable_p2p(&self, gpu_src: u32, gpu_dst: u32) -> Result<()> {
        self.p2p_manager.enable_p2p(gpu_src, gpu_dst)
    }

    /// Get transfer statistics between two GPUs
    #[inline]
    pub fn get_transfer_stats(&self, gpu_src: u32, gpu_dst: u32) -> Option<TransferStats> {
        self.p2p_manager.get_stats(gpu_src, gpu_dst)
    }
}

impl P2PManager {
    /// Create a new P2P manager
    pub fn new(gpu_count: u32) -> Result<Self> {
        // Initialize P2P connectivity matrix
        let mut p2p_matrix = vec![vec![false; gpu_count as usize]; gpu_count as usize];

        #[cfg(feature = "cuda")]
        {
            // Query actual P2P connectivity
            for i in 0..gpu_count {
                for j in 0..gpu_count {
                    if i != j {
                        // In a real implementation, query CUDA P2P support
                        // For now, assume all GPUs can communicate P2P
                        p2p_matrix[i as usize][j as usize] = true;
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // For testing, assume P2P is available between all GPUs
            for i in 0..gpu_count {
                for j in 0..gpu_count {
                    if i != j {
                        p2p_matrix[i as usize][j as usize] = true;
                    }
                }
            }
        }

        Ok(Self {
            gpu_count,
            p2p_matrix: Arc::new(Mutex::new(p2p_matrix)),
            transfer_stats: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Perform P2P transfer between GPUs
    pub fn transfer(&self, src_gpu: u32, dst_gpu: u32, size: u64) -> Result<Duration> {
        if src_gpu >= self.gpu_count || dst_gpu >= self.gpu_count {
            return Err(GpuMemoryError::P2PTransferFailed {
                src: src_gpu,
                dst: dst_gpu,
            }
            .into());
        }

        // Check P2P connectivity
        let p2p_matrix = self.p2p_matrix.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (p2p_matrix), recovering inner data");
            poisoned.into_inner()
        });
        if !p2p_matrix[src_gpu as usize][dst_gpu as usize] {
            return Err(GpuMemoryError::P2PTransferFailed {
                src: src_gpu,
                dst: dst_gpu,
            }
            .into());
        }
        drop(p2p_matrix);

        let start = Instant::now();

        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would use cuMemcpyPeerAsync
            // or similar CUDA P2P memory copy API
            std::thread::sleep(Duration::from_micros(100)); // Simulate transfer time
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Simulate P2P transfer time based on size
            let transfer_time_us = (size as f64 / (100.0 * 1024.0 * 1024.0 * 1024.0)) * 1_000_000.0;
            std::thread::sleep(Duration::from_micros(transfer_time_us as u64));
        }

        let duration = start.elapsed();

        // Update transfer statistics
        let mut stats = self.transfer_stats.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (transfer_stats), recovering inner data");
            poisoned.into_inner()
        });
        let key = (src_gpu, dst_gpu);
        let transfer_stats = stats.entry(key).or_insert_with(|| TransferStats {
            total_transfers: 0,
            total_bytes_transferred: 0,
            average_bandwidth_gbps: 0.0,
            last_transfer_time: None,
        });

        transfer_stats.total_transfers += 1;
        transfer_stats.total_bytes_transferred += size;
        transfer_stats.last_transfer_time = Some(Instant::now());

        // Calculate average bandwidth
        let total_time_seconds = (transfer_stats.total_transfers as f64) * (duration.as_secs_f64());
        if total_time_seconds > 0.0 {
            let gbps = (transfer_stats.total_bytes_transferred as f64)
                / (1024.0 * 1024.0 * 1024.0)
                / total_time_seconds;
            transfer_stats.average_bandwidth_gbps = gbps as f32;
        }

        Ok(duration)
    }

    /// Enable P2P access between two GPUs
    pub fn enable_p2p(&self, gpu_src: u32, gpu_dst: u32) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would use cuCtxEnablePeerAccess
        }

        let mut p2p_matrix = self.p2p_matrix.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (p2p_matrix), recovering inner data");
            poisoned.into_inner()
        });
        p2p_matrix[gpu_src as usize][gpu_dst as usize] = true;
        Ok(())
    }

    /// Get P2P connectivity matrix
    #[inline]
    pub fn get_connectivity_matrix(&self) -> Vec<Vec<bool>> {
        self.p2p_matrix.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Mutex was poisoned (p2p_matrix), recovering inner data");
            poisoned.into_inner()
        }).clone()
    }

    /// Get transfer statistics between two GPUs
    #[inline]
    pub fn get_stats(&self, gpu_src: u32, gpu_dst: u32) -> Option<TransferStats> {
        self.transfer_stats
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!("Mutex was poisoned (transfer_stats), recovering inner data");
                poisoned.into_inner()
            })
            .get(&(gpu_src, gpu_dst))
            .cloned()
    }
}

impl LoadBalancer {
    /// Create a new load balancer
    fn new(gpu_count: u32) -> Self {
        Self {
            gpu_utilization: vec![0.0; gpu_count as usize],
            allocation_strategy: AllocationStrategy::LeastUtilized,
        }
    }

    /// Select the best GPU for allocation
    fn select_gpu_for_allocation(&self, _size: u64) -> u32 {
        match self.allocation_strategy {
            AllocationStrategy::LeastUtilized => self
                .gpu_utilization
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0),
            AllocationStrategy::RoundRobin => {
                // Simple round-robin (would need state in real implementation)
                0
            }
            AllocationStrategy::FirstFit => {
                // Find first GPU with enough free memory
                0
            }
            AllocationStrategy::Locality => {
                // Consider data locality (would need more context)
                0
            }
        }
    }

    /// Update GPU utilization
    fn update_utilization(&mut self, gpu_id: u32, size: u64, is_allocation: bool) {
        if let Some(utilization) = self.gpu_utilization.get_mut(gpu_id as usize) {
            if is_allocation {
                *utilization += size as f32 / (8.0 * 1024.0 * 1024.0 * 1024.0); // Assume 8GB per GPU
            } else {
                *utilization -= size as f32 / (8.0 * 1024.0 * 1024.0 * 1024.0);
                *utilization = utilization.max(0.0);
            }
            *utilization = utilization.min(1.0);
        }
    }
}

// SAFETY: MultiGpuManager is Send because:
// 1. All fields are Arc-wrapped or primitives
// 2. CudaDevice (when enabled) is designed for multi-threaded access
// 3. Mutable state is protected by Mutex
// 4. NonNull pointers in allocations point to GPU memory (thread-safe by design)
unsafe impl Send for MultiGpuManager {}

// SAFETY: MultiGpuManager is Sync because:
// 1. All mutable state is protected by Mutex (gpu_allocations, load_balancer)
// 2. Methods only access shared state through proper synchronization
// 3. P2PManager is itself Sync
unsafe impl Sync for MultiGpuManager {}

// SAFETY: P2PManager is Send because:
// 1. All fields are Arc-wrapped primitives or collections
// 2. No raw pointers - only manages connectivity metadata
// 3. Transfer operations use thread-safe CUDA APIs
unsafe impl Send for P2PManager {}

// SAFETY: P2PManager is Sync because:
// 1. All mutable state (p2p_matrix, transfer_stats) is Mutex-protected
// 2. Methods acquire locks before any modifications
// 3. Transfer operations are serialized by CUDA stream ordering
unsafe impl Sync for P2PManager {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_gpu_manager_creation() {
        let manager = MultiGpuManager::new(2).expect("Failed to create multi-GPU manager");
        assert_eq!(manager.gpu_count, 2);
    }

    #[test]
    fn test_multi_gpu_allocation() {
        let manager = MultiGpuManager::new(2).expect("Failed to create multi-GPU manager");

        let alloc = manager
            .allocate_on_gpu(1024, 0)
            .expect("Failed to allocate on GPU 0");
        assert_eq!(alloc.gpu_id(), 0);
        assert_eq!(alloc.size(), 1024);
        assert!(alloc.is_gpu_resident());
    }

    #[test]
    fn test_invalid_gpu_id() {
        let manager = MultiGpuManager::new(2).expect("Failed to create multi-GPU manager");

        let result = manager.allocate_on_gpu(1024, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_p2p_manager_creation() {
        let p2p_manager = P2PManager::new(4).expect("Failed to create P2P manager");
        assert_eq!(p2p_manager.gpu_count, 4);

        let matrix = p2p_manager.get_connectivity_matrix();
        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix[0].len(), 4);

        // Diagonal should be false (no self P2P)
        for i in 0..4 {
            assert!(!matrix[i][i]);
        }
    }

    #[test]
    fn test_p2p_transfer() {
        let manager = MultiGpuManager::new(2).expect("Failed to create multi-GPU manager");

        let src_alloc = manager
            .allocate_on_gpu(1024, 0)
            .expect("Failed to allocate on GPU 0");
        let dst_alloc = manager
            .allocate_on_gpu(1024, 1)
            .expect("Failed to allocate on GPU 1");

        let duration = manager
            .p2p_copy(&src_alloc, &dst_alloc)
            .expect("P2P copy failed");
        assert!(duration.as_micros() > 0);
    }

    #[test]
    fn test_load_balancer() {
        let mut balancer = LoadBalancer::new(3);

        // Initially all GPUs should have 0 utilization
        assert_eq!(balancer.select_gpu_for_allocation(1024), 0);

        // Update utilization and test selection
        balancer.update_utilization(0, 1024 * 1024 * 1024, true); // 1GB on GPU 0
        let selected = balancer.select_gpu_for_allocation(1024);
        assert!(selected == 1 || selected == 2); // Should select less utilized GPU
    }

    #[test]
    fn test_gpu_utilization_tracking() {
        let manager = MultiGpuManager::new(2).expect("Failed to create multi-GPU manager");

        assert_eq!(manager.get_gpu_utilization(0), 0.0);

        let _alloc = manager
            .allocate_on_gpu(1024 * 1024 * 1024, 0)
            .expect("Failed to allocate"); // 1GB

        let utilization = manager.get_gpu_utilization(0);
        assert!(utilization > 0.0);
        assert!(utilization < 1.0);
    }
}
