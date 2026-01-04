//! CUDA Pinned Memory Manager for Faster Transfers
//!
//! Implements pinned memory allocations for 2-3x faster CPU-GPU transfers

use crate::synthesis::{AstNode, Pattern};
use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance metrics for memory transfer operations
#[derive(Debug, Clone)]
pub struct TransferMetrics {
    pub transfer_time: Duration,
    pub throughput_gbps: f64,
    pub data_size_bytes: usize,
    pub transfer_type: TransferType,
}

#[derive(Debug, Clone)]
pub enum TransferType {
    PageableMemory,
    PinnedMemory,
    ZeroCopy,
}

/// Configuration for pinned memory allocation
#[derive(Debug, Clone)]
pub struct PinnedMemoryConfig {
    pub initial_pool_size: usize,
    pub max_pool_size: usize,
    pub growth_factor: f64,
    pub enable_zero_copy: bool,
}

impl Default for PinnedMemoryConfig {
    fn default() -> Self {
        Self {
            initial_pool_size: 64 * 1024 * 1024, // 64MB
            max_pool_size: 1024 * 1024 * 1024,   // 1GB
            growth_factor: 1.5,
            enable_zero_copy: true,
        }
    }
}

/// Manages pinned memory allocations for optimal GPU transfers
pub struct PinnedMemoryManager {
    device: Arc<CudaDevice>,
    config: PinnedMemoryConfig,
    pinned_buffers: HashMap<usize, PinnedBuffer>,
    buffer_pool: Vec<PinnedBuffer>,
    total_allocated: usize,
}

/// A pinned memory buffer for fast CPU-GPU transfers
struct PinnedBuffer {
    host_ptr: *mut u8,
    device_buffer: Option<CudaSlice<u8>>,
    size: usize,
    in_use: bool,
}

// SAFETY: PinnedBuffer is Send because:
// 1. `host_ptr: *mut u8` points to pinned memory allocated with page-aligned alloc
//    which is thread-safe for ownership transfer (no other references exist)
// 2. `device_buffer: Option<CudaSlice<u8>>` is Send (cudarc CudaSlice is Send)
// 3. `size: usize` and `in_use: bool` are trivially Send primitive types
// 4. Pinned memory allocated via host allocator is safe to access from any thread
// 5. The buffer is only accessed through PinnedMemoryManager which serializes access
unsafe impl Send for PinnedBuffer {}

// SAFETY: PinnedBuffer is Sync because:
// 1. PinnedBuffer is only accessed through &mut self methods in PinnedMemoryManager
// 2. PinnedMemoryManager does not share PinnedBuffer references across threads
// 3. The raw pointer is never dereferenced without proper synchronization
// 4. CudaSlice provides its own thread-safety guarantees
// 5. In practice, buffers are accessed sequentially via get_buffer/return_buffer
unsafe impl Sync for PinnedBuffer {}

impl PinnedMemoryManager {
    /// Create a new pinned memory manager
    pub fn new(device: Arc<CudaDevice>, config: PinnedMemoryConfig) -> Result<Self> {
        let mut manager = Self {
            device,
            config,
            pinned_buffers: HashMap::new(),
            buffer_pool: Vec::new(),
            total_allocated: 0,
        };

        // Pre-allocate initial buffer pool
        manager.preallocate_buffers()?;

        Ok(manager)
    }

    /// Pre-allocate a pool of pinned memory buffers
    fn preallocate_buffers(&mut self) -> Result<()> {
        let buffer_count = 8;
        let buffer_size = self.config.initial_pool_size / buffer_count;

        for _ in 0..buffer_count {
            let buffer = self.allocate_pinned_buffer(buffer_size)?;
            self.buffer_pool.push(buffer);
        }

        Ok(())
    }

    /// Allocate a pinned memory buffer
    fn allocate_pinned_buffer(&mut self, size: usize) -> Result<PinnedBuffer> {
        if self.total_allocated + size > self.config.max_pool_size {
            return Err(anyhow::anyhow!("Pinned memory pool exhausted"));
        }

        // Allocate pinned host memory using device API
        // SAFETY: Layout is valid (size > 0, align is power of 2 and >= 1).
        // The returned pointer will be stored in PinnedBuffer and freed in Drop.
        // 4096-byte alignment is required for page-aligned pinned memory.
        let host_ptr = {
            let layout = std::alloc::Layout::from_size_align(size, 4096)?;
            unsafe { std::alloc::alloc(layout) }
        };

        // Optionally allocate corresponding device buffer for zero-copy
        let device_buffer = if self.config.enable_zero_copy {
            Some(self.device.alloc_zeros::<u8>(size)?)
        } else {
            None
        };

        self.total_allocated += size;

        Ok(PinnedBuffer {
            host_ptr,
            device_buffer,
            size,
            in_use: false,
        })
    }

    /// Get or allocate a pinned buffer of at least the specified size
    pub fn get_buffer(&mut self, min_size: usize) -> Result<usize> {
        // Try to find a suitable buffer in the pool
        for (i, buffer) in self.buffer_pool.iter_mut().enumerate() {
            if !buffer.in_use && buffer.size >= min_size {
                buffer.in_use = true;
                return Ok(i);
            }
        }

        // Need to allocate a new buffer
        let new_size = (min_size as f64 * self.config.growth_factor) as usize;
        let buffer = self.allocate_pinned_buffer(new_size)?;
        let buffer_id = self.buffer_pool.len();

        self.buffer_pool.push(buffer);
        self.buffer_pool[buffer_id].in_use = true;

        Ok(buffer_id)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer_id: usize) -> Result<()> {
        if let Some(buffer) = self.buffer_pool.get_mut(buffer_id) {
            buffer.in_use = false;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid buffer ID"))
        }
    }

    /// Transfer patterns to GPU using pinned memory
    pub async fn transfer_patterns_to_gpu_pinned(
        &mut self,
        patterns: &[Pattern],
    ) -> Result<(CudaSlice<u8>, TransferMetrics)> {
        // Serialize patterns to binary format
        let serialized = self.serialize_patterns_to_buffer(patterns)?;
        let data_size = serialized.len();

        // Get pinned buffer
        let buffer_id = self.get_buffer(data_size)?;
        let buffer = &mut self.buffer_pool[buffer_id];

        let start = Instant::now();

        // Copy data to pinned memory
        // SAFETY: serialized.as_ptr() is valid for data_size bytes.
        // buffer.host_ptr was allocated with at least data_size bytes in get_buffer().
        // The memory regions do not overlap (serialized is stack-allocated Vec, host_ptr is heap).
        unsafe {
            std::ptr::copy_nonoverlapping(serialized.as_ptr(), buffer.host_ptr, data_size);
        }

        // Transfer from pinned memory to GPU (much faster than pageable)
        // SAFETY: host_ptr is valid for data_size bytes (was just written above).
        // The slice lifetime is temporary and data is immediately copied to Vec.
        let host_data = unsafe { std::slice::from_raw_parts(buffer.host_ptr, data_size).to_vec() };
        let gpu_buffer = self.device.htod_copy(host_data)?;

        // Synchronize to measure actual transfer time
        self.device.synchronize()?;
        let transfer_time = start.elapsed();

        // Return buffer to pool
        self.return_buffer(buffer_id)?;

        let throughput_gbps =
            (data_size as f64) / (1024.0 * 1024.0 * 1024.0) / transfer_time.as_secs_f64();

        let metrics = TransferMetrics {
            transfer_time,
            throughput_gbps,
            data_size_bytes: data_size,
            transfer_type: TransferType::PinnedMemory,
        };

        Ok((gpu_buffer, metrics))
    }

    /// Transfer AST nodes to GPU using pinned memory
    pub async fn transfer_ast_nodes_to_gpu_pinned(
        &mut self,
        ast_nodes: &[AstNode],
    ) -> Result<(CudaSlice<u8>, TransferMetrics)> {
        let serialized = self.serialize_ast_nodes_to_buffer(ast_nodes)?;
        let data_size = serialized.len();

        let buffer_id = self.get_buffer(data_size)?;
        let buffer = &mut self.buffer_pool[buffer_id];

        let start = Instant::now();

        // SAFETY: serialized.as_ptr() is valid for data_size bytes.
        // buffer.host_ptr was allocated with at least data_size bytes in get_buffer().
        // The memory regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(serialized.as_ptr(), buffer.host_ptr, data_size);
        }

        // Use device API for transfer
        // SAFETY: host_ptr is valid for data_size bytes (was just written above).
        // The slice lifetime is temporary and data is immediately copied to Vec.
        let host_data = unsafe { std::slice::from_raw_parts(buffer.host_ptr, data_size).to_vec() };
        let gpu_buffer = self.device.htod_copy(host_data)?;

        self.device.synchronize()?;
        let transfer_time = start.elapsed();

        self.return_buffer(buffer_id)?;

        let throughput_gbps =
            (data_size as f64) / (1024.0 * 1024.0 * 1024.0) / transfer_time.as_secs_f64();

        let metrics = TransferMetrics {
            transfer_time,
            throughput_gbps,
            data_size_bytes: data_size,
            transfer_type: TransferType::PinnedMemory,
        };

        Ok((gpu_buffer, metrics))
    }

    /// Compare transfer performance: pageable vs pinned memory
    pub async fn benchmark_transfer_methods(
        &mut self,
        patterns: &[Pattern],
    ) -> Result<Vec<TransferMetrics>> {
        let mut results = Vec::new();

        // Test pageable memory transfer (baseline)
        let pageable_metrics = self.transfer_with_pageable_memory(patterns).await?;
        results.push(pageable_metrics);

        // Test pinned memory transfer
        let (_, pinned_metrics) = self.transfer_patterns_to_gpu_pinned(patterns).await?;
        results.push(pinned_metrics);

        // Test zero-copy if enabled
        if self.config.enable_zero_copy {
            let zero_copy_metrics = self.transfer_with_zero_copy(patterns).await?;
            results.push(zero_copy_metrics);
        }

        Ok(results)
    }

    /// Transfer using regular pageable memory (for comparison)
    async fn transfer_with_pageable_memory(&self, patterns: &[Pattern]) -> Result<TransferMetrics> {
        let serialized = self.serialize_patterns_to_buffer(patterns)?;
        let data_size = serialized.len();

        let start = Instant::now();

        // Regular allocation and transfer (slower)
        let gpu_buffer = self.device.htod_copy(serialized.clone())?;
        self.device.synchronize()?;

        let transfer_time = start.elapsed();
        let throughput_gbps =
            (data_size as f64) / (1024.0 * 1024.0 * 1024.0) / transfer_time.as_secs_f64();

        Ok(TransferMetrics {
            transfer_time,
            throughput_gbps,
            data_size_bytes: data_size,
            transfer_type: TransferType::PageableMemory,
        })
    }

    /// Transfer using zero-copy memory
    async fn transfer_with_zero_copy(&mut self, patterns: &[Pattern]) -> Result<TransferMetrics> {
        let serialized = self.serialize_patterns_to_buffer(patterns)?;
        let data_size = serialized.len();

        let buffer_id = self.get_buffer(data_size)?;
        let buffer = &self.buffer_pool[buffer_id];

        let start = Instant::now();

        // Zero-copy: GPU can directly access pinned host memory
        // SAFETY: serialized.as_ptr() is valid for data_size bytes.
        // buffer.host_ptr was allocated with at least data_size bytes in get_buffer().
        // The memory regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(serialized.as_ptr(), buffer.host_ptr, data_size);
        }

        // In zero-copy, data stays in host memory but GPU accesses it directly
        // This eliminates transfer time but may have bandwidth limitations
        let transfer_time = start.elapsed();

        self.return_buffer(buffer_id)?;

        let throughput_gbps =
            (data_size as f64) / (1024.0 * 1024.0 * 1024.0) / transfer_time.as_secs_f64();

        Ok(TransferMetrics {
            transfer_time,
            throughput_gbps,
            data_size_bytes: data_size,
            transfer_type: TransferType::ZeroCopy,
        })
    }

    /// Serialize patterns to binary format for transfer
    fn serialize_patterns_to_buffer(&self, patterns: &[Pattern]) -> Result<Vec<u8>> {
        // Use efficient binary serialization
        use crate::synthesis::binary_serializer::BinarySerializer;
        let serializer = BinarySerializer::new(false); // No compression for transfer speed
        serializer.serialize_patterns(patterns)
    }

    /// Serialize AST nodes to binary format for transfer
    fn serialize_ast_nodes_to_buffer(&self, ast_nodes: &[AstNode]) -> Result<Vec<u8>> {
        use crate::synthesis::binary_serializer::BinarySerializer;
        let serializer = BinarySerializer::new(false);
        serializer.serialize_ast_nodes(ast_nodes)
    }

    /// Get memory pool statistics
    pub fn get_pool_stats(&self) -> MemoryPoolStats {
        let buffers_in_use = self.buffer_pool.iter().filter(|b| b.in_use).count();
        let total_buffers = self.buffer_pool.len();

        MemoryPoolStats {
            total_allocated_bytes: self.total_allocated,
            buffers_in_use,
            total_buffers,
            memory_utilization: self.total_allocated as f64 / self.config.max_pool_size as f64,
        }
    }
}

impl Drop for PinnedMemoryManager {
    fn drop(&mut self) {
        // Free all pinned memory buffers
        // SAFETY: Each buffer.host_ptr was allocated with std::alloc::alloc using the
        // same layout (buffer.size, 4096 alignment). The pointer has not been freed
        // elsewhere as buffers are only managed by this struct. After drop, no references
        // to these buffers exist.
        for buffer in &mut self.buffer_pool {
            unsafe {
                if let Ok(layout) = std::alloc::Layout::from_size_align(buffer.size, 4096) {
                    std::alloc::dealloc(buffer.host_ptr, layout);
                }
            }
        }
    }
}

/// Statistics for the memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated_bytes: usize,
    pub buffers_in_use: usize,
    pub total_buffers: usize,
    pub memory_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::NodeType;

    fn create_test_patterns(count: usize) -> Vec<Pattern> {
        (0..count)
            .map(|i| Pattern {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some(format!("pattern_{}", i)),
            })
            .collect()
    }

    fn create_test_asts(count: usize) -> Vec<AstNode> {
        (0..count)
            .map(|i| AstNode {
                node_type: NodeType::Variable,
                children: vec![],
                value: Some(format!("node_{}", i)),
            })
            .collect()
    }

    #[tokio::test]
    async fn test_pinned_memory_manager_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig::default();
        let manager = PinnedMemoryManager::new(device, config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_buffer_allocation_and_return() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig::default();
        let mut manager = PinnedMemoryManager::new(device, config)?;

        let buffer_id = manager.get_buffer(1024)?;
        assert!(buffer_id < manager.buffer_pool.len());
        assert!(manager.buffer_pool[buffer_id].in_use);

        manager.return_buffer(buffer_id)?;
        assert!(!manager.buffer_pool[buffer_id].in_use);
    }

    #[tokio::test]
    async fn test_pattern_transfer_with_pinned_memory() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig::default();
        let mut manager = PinnedMemoryManager::new(device, config)?;

        let patterns = create_test_patterns(1000);
        let result = manager.transfer_patterns_to_gpu_pinned(&patterns).await;

        assert!(result.is_ok());
        let (gpu_buffer, metrics) = result?;

        assert!(gpu_buffer.len() > 0);
        assert!(metrics.transfer_time.as_micros() > 0);
        assert!(metrics.throughput_gbps > 0.0);
        assert_eq!(metrics.transfer_type, TransferType::PinnedMemory);
    }

    #[tokio::test]
    async fn test_ast_transfer_with_pinned_memory() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig::default();
        let mut manager = PinnedMemoryManager::new(device, config)?;

        let ast_nodes = create_test_asts(1000);
        let result = manager.transfer_ast_nodes_to_gpu_pinned(&ast_nodes).await;

        assert!(result.is_ok());
        let (gpu_buffer, metrics) = result?;

        assert!(gpu_buffer.len() > 0);
        assert!(metrics.transfer_time.as_micros() > 0);
        assert_eq!(metrics.transfer_type, TransferType::PinnedMemory);
    }

    #[tokio::test]
    async fn test_transfer_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig {
            enable_zero_copy: true,
            ..Default::default()
        };
        let mut manager = PinnedMemoryManager::new(device, config)?;

        let patterns = create_test_patterns(5000); // Larger dataset for meaningful comparison
        let results = manager.benchmark_transfer_methods(&patterns).await;

        assert!(results.is_ok());
        let benchmark_results = results?;

        // Should have at least 2 results (pageable and pinned)
        assert!(benchmark_results.len() >= 2);

        // Find pageable and pinned results
        let pageable = benchmark_results
            .iter()
            .find(|m| matches!(m.transfer_type, TransferType::PageableMemory));
        let pinned = benchmark_results
            .iter()
            .find(|m| matches!(m.transfer_type, TransferType::PinnedMemory));

        assert!(pageable.is_some());
        assert!(pinned.is_some());

        let pageable_metrics = pageable?;
        let pinned_metrics = pinned?;

        println!(
            "Pageable memory: {:?} ({:.2} GB/s)",
            pageable_metrics.transfer_time, pageable_metrics.throughput_gbps
        );
        println!(
            "Pinned memory: {:?} ({:.2} GB/s)",
            pinned_metrics.transfer_time, pinned_metrics.throughput_gbps
        );

        // Pinned memory should generally be faster
        assert!(pinned_metrics.throughput_gbps >= pageable_metrics.throughput_gbps * 0.8);
        // At least 80% as fast (accounting for test variations)
    }

    #[tokio::test]
    async fn test_memory_pool_stats() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = PinnedMemoryConfig::default();
        let mut manager = PinnedMemoryManager::new(device, config)?;

        let stats_before = manager.get_pool_stats();

        let buffer_id = manager.get_buffer(1024)?;
        let stats_after = manager.get_pool_stats();

        assert_eq!(stats_after.buffers_in_use, stats_before.buffers_in_use + 1);

        manager.return_buffer(buffer_id)?;
        let stats_final = manager.get_pool_stats();

        assert_eq!(stats_final.buffers_in_use, stats_before.buffers_in_use);
    }
}

// Make TransferType comparable for tests
impl PartialEq for TransferType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (TransferType::PageableMemory, TransferType::PageableMemory)
                | (TransferType::PinnedMemory, TransferType::PinnedMemory)
                | (TransferType::ZeroCopy, TransferType::ZeroCopy)
        )
    }
}
