//! GPU DMA Lock Kernel Module
//!
//! Provides GPU memory protection and DMA access control for StratoSwarm agents.
//!
//! Key features:
//! - Per-agent GPU memory quotas
//! - DMA access control lists
//! - GPU context isolation
//! - Multi-GPU support
//! - GPUDirect RDMA integration

#![no_std]

extern crate alloc;

// Test allocator setup
#[cfg(test)]
extern crate std;

#[cfg(test)]
use std::alloc::System;

#[cfg(test)]
#[global_allocator]
static ALLOCATOR: System = System;

// Panic handler for no_std
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In kernel: trigger kernel panic
    loop {}
}

// Simple allocator for no_std build
#[cfg(not(test))]
#[global_allocator]
static KERNEL_ALLOCATOR: KernelAllocator = KernelAllocator;

#[cfg(not(test))]
struct KernelAllocator;

#[cfg(not(test))]
unsafe impl core::alloc::GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, _layout: core::alloc::Layout) -> *mut u8 {
        // In kernel: would use kmalloc
        core::ptr::null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {
        // In kernel: would use kfree
    }
}

pub mod allocation;
pub mod context;
pub mod dma;
pub mod ffi;
pub mod proc;
pub mod security;
pub mod spin;
pub mod stats;

use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Maximum number of GPUs supported
pub const MAX_GPU_DEVICES: usize = 8;

/// Maximum number of concurrent agents
pub const MAX_AGENTS: usize = 65536;

/// Page size for GPU allocations
pub const GPU_PAGE_SIZE: usize = 64 * 1024; // 64KB

/// DMA access modes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaAccessMode {
    None = 0,
    ReadOnly = 1,
    WriteOnly = 2,
    ReadWrite = 3,
}

impl DmaAccessMode {
    pub fn can_read(&self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    pub fn can_write(&self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }
}

/// GPU device information
#[derive(Debug)]
pub struct GpuDevice {
    id: u32,
    name: [u8; 64],
    total_memory: usize,
    available_memory: AtomicUsize,
    allocated_memory: AtomicUsize,
    allocation_count: AtomicU64,
}

impl GpuDevice {
    pub fn new(id: u32, name: &str, total_memory: usize) -> Self {
        let mut name_buf = [0u8; 64];
        let name_bytes = name.as_bytes();
        let len = name_bytes.len().min(63);
        name_buf[..len].copy_from_slice(&name_bytes[..len]);

        Self {
            id,
            name: name_buf,
            total_memory,
            available_memory: AtomicUsize::new(total_memory),
            allocated_memory: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn name(&self) -> &str {
        let len = self.name.iter().position(|&b| b == 0).unwrap_or(64);
        core::str::from_utf8(&self.name[..len]).unwrap_or("Unknown")
    }

    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    pub fn available_memory(&self) -> usize {
        self.available_memory.load(Ordering::Relaxed)
    }

    pub fn allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }

    pub fn try_allocate(&self, size: usize) -> bool {
        let mut available = self.available_memory.load(Ordering::Acquire);
        loop {
            if available < size {
                return false;
            }

            match self.available_memory.compare_exchange_weak(
                available,
                available - size,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocated_memory.fetch_add(size, Ordering::Relaxed);
                    self.allocation_count.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(current) => available = current,
            }
        }
    }

    pub fn deallocate(&self, size: usize) {
        self.available_memory.fetch_add(size, Ordering::Relaxed);
        self.allocated_memory.fetch_sub(size, Ordering::Relaxed);
    }
}

/// Agent GPU quota information
#[derive(Debug)]
pub struct AgentGpuQuota {
    agent_id: u64,
    memory_limit: usize,
    current_usage: AtomicUsize,
    allocation_count: AtomicU32,
}

impl AgentGpuQuota {
    pub fn new(agent_id: u64, memory_limit: usize) -> Self {
        Self {
            agent_id,
            memory_limit,
            current_usage: AtomicUsize::new(0),
            allocation_count: AtomicU32::new(0),
        }
    }

    pub fn agent_id(&self) -> u64 {
        self.agent_id
    }

    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    pub fn can_allocate(&self, size: usize) -> bool {
        self.current_usage.load(Ordering::Relaxed) + size <= self.memory_limit
    }

    pub fn try_allocate(&self, size: usize) -> bool {
        let mut current = self.current_usage.load(Ordering::Acquire);
        loop {
            if current + size > self.memory_limit {
                return false;
            }

            match self.current_usage.compare_exchange_weak(
                current,
                current + size,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.allocation_count.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(c) => current = c,
            }
        }
    }

    pub fn deallocate(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }
}

/// Memory pressure levels
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    Normal = 0,
    Warning = 1,
    Critical = 2,
}

/// Result type for kernel operations
pub type KernelResult<T> = Result<T, KernelError>;

/// Kernel error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// Out of memory
    OutOfMemory,
    /// Agent not found
    AgentNotFound,
    /// Quota exceeded
    QuotaExceeded,
    /// Invalid GPU device
    InvalidDevice,
    /// DMA access denied
    DmaAccessDenied,
    /// Invalid argument
    InvalidArgument,
    /// Operation not supported
    NotSupported,
    /// Context creation failed
    ContextError,
    /// Already exists
    AlreadyExists,
}

/// Global GPU DMA lock statistics
///
/// Cache-line aligned (64 bytes) to prevent false sharing when
/// multiple threads update these counters concurrently.
#[repr(C, align(64))]
pub struct GpuDmaLockStats {
    /// Total allocations
    pub total_allocations: AtomicU64,
    /// Total deallocations
    pub total_deallocations: AtomicU64,
    /// Total bytes allocated
    pub total_bytes_allocated: AtomicU64,
    /// DMA access checks
    pub dma_checks: AtomicU64,
    /// DMA access denials
    pub dma_denials: AtomicU64,
    /// Context switches
    pub context_switches: AtomicU64,
    // Padding to fill cache line (6 * 8 = 48 bytes, need 16 more)
    _padding: [u8; 16],
}

impl GpuDmaLockStats {
    pub const fn new() -> Self {
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_bytes_allocated: AtomicU64::new(0),
            dma_checks: AtomicU64::new(0),
            dma_denials: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            _padding: [0; 16],
        }
    }

    pub fn record_allocation(&self, size: usize) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(size as u64, Ordering::Relaxed);
    }

    pub fn record_deallocation(&self) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_dma_check(&self, allowed: bool) {
        self.dma_checks.fetch_add(1, Ordering::Relaxed);
        if !allowed {
            self.dma_denials.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_context_switch(&self) {
        self.context_switches.fetch_add(1, Ordering::Relaxed);
    }
}

/// Global statistics instance
pub static GPU_DMA_STATS: GpuDmaLockStats = GpuDmaLockStats::new();

/// Initialize the GPU DMA lock module
pub fn init() -> KernelResult<()> {
    // Initialize subsystems
    allocation::init()?;
    dma::init()?;
    context::init()?;
    security::init()?;
    stats::init()?;
    proc::init()?;

    Ok(())
}

/// Cleanup the GPU DMA lock module
pub fn cleanup() {
    proc::cleanup();
    stats::cleanup();
    security::cleanup();
    context::cleanup();
    dma::cleanup();
    allocation::cleanup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dma_access_mode() {
        assert!(DmaAccessMode::ReadOnly.can_read());
        assert!(!DmaAccessMode::ReadOnly.can_write());
        assert!(!DmaAccessMode::WriteOnly.can_read());
        assert!(DmaAccessMode::WriteOnly.can_write());
        assert!(DmaAccessMode::ReadWrite.can_read());
        assert!(DmaAccessMode::ReadWrite.can_write());
        assert!(!DmaAccessMode::None.can_read());
        assert!(!DmaAccessMode::None.can_write());
    }

    #[test]
    fn test_gpu_device() {
        let device = GpuDevice::new(0, "Test GPU", 1 << 30);
        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "Test GPU");
        assert_eq!(device.total_memory(), 1 << 30);
        assert_eq!(device.available_memory(), 1 << 30);

        // Test allocation
        assert!(device.try_allocate(512 << 20)); // 512MB
        assert_eq!(device.available_memory(), 512 << 20);
        assert_eq!(device.allocated_memory(), 512 << 20);

        // Test deallocation
        device.deallocate(512 << 20);
        assert_eq!(device.available_memory(), 1 << 30);
        assert_eq!(device.allocated_memory(), 0);
    }

    #[test]
    fn test_agent_quota() {
        let quota = AgentGpuQuota::new(100, 1 << 30); // 1GB
        assert_eq!(quota.agent_id(), 100);
        assert_eq!(quota.memory_limit(), 1 << 30);
        assert_eq!(quota.current_usage(), 0);

        // Test allocation within quota
        assert!(quota.try_allocate(512 << 20)); // 512MB
        assert_eq!(quota.current_usage(), 512 << 20);

        // Test allocation exceeding quota
        assert!(!quota.try_allocate(600 << 20)); // 600MB more would exceed
        assert_eq!(quota.current_usage(), 512 << 20); // Unchanged

        // Test deallocation
        quota.deallocate(512 << 20);
        assert_eq!(quota.current_usage(), 0);
    }
}
