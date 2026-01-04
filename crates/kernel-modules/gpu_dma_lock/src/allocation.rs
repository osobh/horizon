//! GPU memory allocation tracking and management
//!
//! This module handles GPU memory allocations, quota enforcement,
//! and tracks allocations per agent.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{spin, AgentGpuQuota, GpuDevice, KernelError, KernelResult, GPU_DMA_STATS};

/// Allocation information
#[derive(Debug, Clone)]
pub struct Allocation {
    pub id: u64,
    pub agent_id: u64,
    pub size: usize,
    pub device_id: u32,
    pub gpu_address: u64,
    pub timestamp: u64,
}

/// Allocation tracker
pub struct AllocationTracker {
    /// Next allocation ID
    pub next_id: AtomicU64,
    /// Allocations by ID
    pub allocations: spin::RwLock<BTreeMap<u64, Allocation>>,
    /// Allocations by agent
    pub agent_allocations: spin::RwLock<BTreeMap<u64, Vec<u64>>>,
}

impl AllocationTracker {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            allocations: spin::RwLock::new(BTreeMap::new()),
            agent_allocations: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn track_allocation(&self, agent_id: u64, size: usize, device_id: u32) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let allocation = Allocation {
            id,
            agent_id,
            size,
            device_id,
            gpu_address: 0, // Would be set by actual GPU allocation
            timestamp: get_current_time(),
        };

        // Track allocation
        {
            let mut allocations = self.allocations.write();
            allocations.insert(id, allocation);
        }

        // Track by agent
        {
            let mut agent_allocs = self.agent_allocations.write();
            agent_allocs
                .entry(agent_id)
                .or_insert_with(Vec::new)
                .push(id);
        }

        // Update statistics
        GPU_DMA_STATS.record_allocation(size);

        id
    }

    pub fn track_deallocation(&self, alloc_id: u64) {
        let agent_id = {
            let mut allocations = self.allocations.write();
            allocations.remove(&alloc_id).map(|a| a.agent_id)
        };

        if let Some(agent_id) = agent_id {
            let mut agent_allocs = self.agent_allocations.write();
            if let Some(allocs) = agent_allocs.get_mut(&agent_id) {
                allocs.retain(|&id| id != alloc_id);
            }
        }

        GPU_DMA_STATS.record_deallocation();
    }

    pub fn get_allocation(&self, alloc_id: u64) -> Option<Allocation> {
        self.allocations.read().get(&alloc_id).cloned()
    }

    pub fn get_agent_allocations(&self, agent_id: u64) -> Vec<Allocation> {
        let agent_allocs = self.agent_allocations.read();
        if let Some(alloc_ids) = agent_allocs.get(&agent_id) {
            let allocations = self.allocations.read();
            alloc_ids
                .iter()
                .filter_map(|id| allocations.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn remove_agent_allocations(&self, agent_id: u64) {
        let alloc_ids: Vec<u64> = {
            let mut agent_allocs = self.agent_allocations.write();
            agent_allocs.remove(&agent_id).unwrap_or_default()
        };

        let mut allocations = self.allocations.write();
        for id in alloc_ids {
            allocations.remove(&id);
        }
    }
}

/// Quota enforcer
pub struct QuotaEnforcer {
    /// Agent quotas
    pub quotas: spin::RwLock<BTreeMap<u64, AgentGpuQuota>>,
}

impl QuotaEnforcer {
    pub fn new() -> Self {
        Self {
            quotas: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn set_agent_quota(&self, agent_id: u64, limit: usize) {
        let quota = AgentGpuQuota::new(agent_id, limit);
        self.quotas.write().insert(agent_id, quota);
    }

    pub fn remove_agent_quota(&self, agent_id: u64) {
        self.quotas.write().remove(&agent_id);
    }

    pub fn check_allocation(&self, agent_id: u64, size: usize) -> bool {
        self.quotas
            .read()
            .get(&agent_id)
            .map(|quota| quota.can_allocate(size))
            .unwrap_or(false)
    }

    pub fn record_allocation(&self, agent_id: u64, size: usize) -> bool {
        self.quotas
            .read()
            .get(&agent_id)
            .map(|quota| quota.try_allocate(size))
            .unwrap_or(false)
    }

    pub fn record_deallocation(&self, agent_id: u64, size: usize) {
        if let Some(quota) = self.quotas.read().get(&agent_id) {
            quota.deallocate(size);
        }
    }

    pub fn get_agent_usage(&self, agent_id: u64) -> Option<(usize, usize)> {
        self.quotas
            .read()
            .get(&agent_id)
            .map(|quota| (quota.current_usage(), quota.memory_limit()))
    }
}

/// GPU manager
pub struct GpuManager {
    /// GPU devices
    pub devices: spin::RwLock<Vec<GpuDevice>>,
    /// Allocation tracker
    pub tracker: AllocationTracker,
    /// Quota enforcer
    pub enforcer: QuotaEnforcer,
}

impl GpuManager {
    pub fn new() -> Self {
        Self {
            devices: spin::RwLock::new(Vec::new()),
            tracker: AllocationTracker::new(),
            enforcer: QuotaEnforcer::new(),
        }
    }

    pub fn register_device(&self, id: u32, name: &str, total_memory: usize) {
        let device = GpuDevice::new(id, name, total_memory);
        self.devices.write().push(device);
    }

    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }

    pub fn total_memory(&self) -> usize {
        self.devices.read().iter().map(|d| d.total_memory()).sum()
    }

    pub fn allocate(
        &self,
        agent_id: u64,
        size: usize,
        device_id: Option<u32>,
    ) -> KernelResult<u64> {
        // Check quota first
        if !self.enforcer.check_allocation(agent_id, size) {
            return Err(KernelError::QuotaExceeded);
        }

        // Find device
        let devices = self.devices.read();
        let device = if let Some(id) = device_id {
            devices.iter().find(|d| d.id() == id)
        } else {
            // Find device with available memory
            devices.iter().find(|d| d.available_memory() >= size)
        };

        let device = device.ok_or(KernelError::InvalidDevice)?;

        // Try to allocate on device
        if !device.try_allocate(size) {
            return Err(KernelError::OutOfMemory);
        }

        // Record in quota
        if !self.enforcer.record_allocation(agent_id, size) {
            // Rollback device allocation
            device.deallocate(size);
            return Err(KernelError::QuotaExceeded);
        }

        // Track allocation
        let alloc_id = self.tracker.track_allocation(agent_id, size, device.id());

        Ok(alloc_id)
    }

    pub fn deallocate(&self, alloc_id: u64) -> KernelResult<()> {
        let allocation = self
            .tracker
            .get_allocation(alloc_id)
            .ok_or(KernelError::InvalidArgument)?;

        // Find device
        let devices = self.devices.read();
        let device = devices
            .iter()
            .find(|d| d.id() == allocation.device_id)
            .ok_or(KernelError::InvalidDevice)?;

        // Deallocate from device
        device.deallocate(allocation.size);

        // Update quota
        self.enforcer
            .record_deallocation(allocation.agent_id, allocation.size);

        // Remove tracking
        self.tracker.track_deallocation(alloc_id);

        Ok(())
    }

    pub fn create_agent(&self, agent_id: u64, quota: usize) -> KernelResult<()> {
        // Check if agent already exists
        if self.enforcer.get_agent_usage(agent_id).is_some() {
            return Err(KernelError::AlreadyExists);
        }

        self.enforcer.set_agent_quota(agent_id, quota);
        Ok(())
    }

    pub fn remove_agent(&self, agent_id: u64) -> KernelResult<()> {
        // Get all allocations for this agent
        let allocations = self.tracker.get_agent_allocations(agent_id);

        // Deallocate all
        for allocation in allocations {
            self.deallocate(allocation.id)?;
        }

        // Remove quota
        self.enforcer.remove_agent_quota(agent_id);

        Ok(())
    }
}

/// Fast allocator for performance testing
pub struct FastAllocator {
    allocations: spin::Mutex<Vec<(u64, usize, u32)>>,
}

impl FastAllocator {
    pub fn new() -> Self {
        Self {
            allocations: spin::Mutex::new(Vec::with_capacity(100000)),
        }
    }

    pub fn allocate(&self, agent_id: u64, size: usize, device_id: u32) {
        let mut allocs = self.allocations.lock();
        allocs.push((agent_id, size, device_id));
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.lock().len()
    }
}

/// Concurrent GPU manager for thread-safe operations
pub struct ConcurrentGpuManager {
    manager: GpuManager,
    allocation_count: AtomicU64,
}

impl ConcurrentGpuManager {
    pub fn new() -> Self {
        Self {
            manager: GpuManager::new(),
            allocation_count: AtomicU64::new(0),
        }
    }

    pub fn allocate(&self, agent_id: u64, size: usize, device_id: u32) -> KernelResult<()> {
        // First create agent if needed
        let _ = self.manager.create_agent(agent_id, 10 << 30); // 10GB default quota

        // Register device if needed
        if self.manager.device_count() <= device_id as usize {
            self.manager.register_device(device_id, "GPU", 24 << 30); // 24GB
        }

        // Allocate
        self.manager.allocate(agent_id, size, Some(device_id))?;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    pub fn total_allocations(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed) as usize
    }
}

/// Memory oversubscription manager
pub struct OversubscriptionManager {
    /// Physical memory per device
    physical_memory: spin::RwLock<BTreeMap<u32, usize>>,
    /// Virtual memory limit (as ratio of physical)
    oversubscription_ratio: spin::RwLock<f64>,
    /// Current virtual allocations
    virtual_allocations: spin::RwLock<BTreeMap<u32, usize>>,
}

impl OversubscriptionManager {
    pub fn new() -> Self {
        Self {
            physical_memory: spin::RwLock::new(BTreeMap::new()),
            oversubscription_ratio: spin::RwLock::new(1.0),
            virtual_allocations: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn set_oversubscription_ratio(&self, ratio: f64) {
        *self.oversubscription_ratio.write() = ratio;
    }

    pub fn allocate(&self, _agent_id: u64, size: usize, device_id: u32) -> KernelResult<()> {
        // Set physical memory if not set
        self.physical_memory
            .write()
            .entry(device_id)
            .or_insert(24 << 30);

        let physical = self
            .physical_memory
            .read()
            .get(&device_id)
            .copied()
            .unwrap_or(0);
        let ratio = *self.oversubscription_ratio.read();
        let virtual_limit = (physical as f64 * ratio) as usize;

        let mut virtual_allocs = self.virtual_allocations.write();
        let current = virtual_allocs.entry(device_id).or_insert(0);

        if *current + size > virtual_limit {
            return Err(KernelError::OutOfMemory);
        }

        *current += size;
        Ok(())
    }
}

/// Memory migration manager
pub struct MemoryMigrationManager {
    allocations: spin::RwLock<BTreeMap<u64, Allocation>>,
    next_id: AtomicU64,
}

impl MemoryMigrationManager {
    pub fn new() -> Self {
        Self {
            allocations: spin::RwLock::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn allocate(&self, agent_id: u64, size: usize, device_id: u32) -> KernelResult<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let allocation = Allocation {
            id,
            agent_id,
            size,
            device_id,
            gpu_address: 0,
            timestamp: get_current_time(),
        };

        self.allocations.write().insert(id, allocation);
        Ok(id)
    }

    pub fn migrate(&self, alloc_id: u64, new_device_id: u32) -> KernelResult<()> {
        let mut allocations = self.allocations.write();
        if let Some(alloc) = allocations.get_mut(&alloc_id) {
            alloc.device_id = new_device_id;
            Ok(())
        } else {
            Err(KernelError::InvalidArgument)
        }
    }

    pub fn get_allocation_info(&self, alloc_id: u64) -> Option<Allocation> {
        self.allocations.read().get(&alloc_id).cloned()
    }

    pub fn get_device_stats(&self, device_id: u32) -> DeviceStats {
        let allocations = self.allocations.read();
        let allocated: usize = allocations
            .values()
            .filter(|a| a.device_id == device_id)
            .map(|a| a.size)
            .sum();

        DeviceStats { allocated }
    }
}

#[derive(Debug)]
pub struct DeviceStats {
    pub allocated: usize,
}

/// GPU DMA lock system for integration tests
pub struct GpuDmaLockSystem {
    manager: GpuManager,
}

impl GpuDmaLockSystem {
    pub fn new() -> Self {
        Self {
            manager: GpuManager::new(),
        }
    }

    pub fn add_gpu(&self, id: u32, name: &str, memory: usize) {
        self.manager.register_device(id, name, memory);
    }

    pub fn create_agent(&self, agent_id: u64, quota: usize) -> KernelResult<()> {
        self.manager.create_agent(agent_id, quota)
    }

    pub fn allocate(&self, agent_id: u64, size: usize, gpu: u32) -> KernelResult<u64> {
        self.manager.allocate(agent_id, size, Some(gpu))
    }

    pub fn get_gpu_stats(&self, gpu: u32) -> GpuStats {
        let devices = self.manager.devices.read();
        if let Some(device) = devices.iter().find(|d| d.id() == gpu) {
            GpuStats {
                allocated: device.allocated_memory(),
            }
        } else {
            GpuStats { allocated: 0 }
        }
    }
}

#[derive(Debug)]
pub struct GpuStats {
    pub allocated: usize,
}

/// Global allocation manager instance
static mut ALLOCATION_MANAGER: Option<GpuManager> = None;

/// Get global allocation manager
pub fn get_manager() -> &'static GpuManager {
    // SAFETY: This function is only called after init() has been called during
    // kernel module initialization. The static ALLOCATION_MANAGER is initialized
    // once at module load time and never modified until module cleanup, at which
    // point no more calls to this function should occur. Single-threaded init
    // ensures no data races during the initialization sequence.
    unsafe {
        ALLOCATION_MANAGER
            .as_ref()
            .expect("Allocation manager not initialized")
    }
}

/// Initialize allocation subsystem
pub fn init() -> KernelResult<()> {
    // SAFETY: This function is called exactly once during kernel module
    // initialization, before any other threads can access ALLOCATION_MANAGER.
    // The kernel module init sequence is single-threaded, ensuring no data
    // races during this write to the static mutable.
    unsafe {
        ALLOCATION_MANAGER = Some(GpuManager::new());
    }
    Ok(())
}

/// Cleanup allocation subsystem
pub fn cleanup() {
    // SAFETY: This function is called exactly once during kernel module
    // unload, after all other operations have completed and no threads are
    // accessing ALLOCATION_MANAGER. The kernel module exit sequence ensures
    // exclusive access to module globals during cleanup.
    unsafe {
        ALLOCATION_MANAGER = None;
    }
}

/// Get current time (mock)
fn get_current_time() -> u64 {
    // In real kernel: ktime_get()
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_tracker() {
        let tracker = AllocationTracker::new();

        let id1 = tracker.track_allocation(100, 1 << 20, 0);
        let id2 = tracker.track_allocation(100, 2 << 20, 0);

        assert!(tracker.get_allocation(id1).is_some());
        assert!(tracker.get_allocation(id2).is_some());

        let agent_allocs = tracker.get_agent_allocations(100);
        assert_eq!(agent_allocs.len(), 2);

        tracker.track_deallocation(id1);
        assert!(tracker.get_allocation(id1).is_none());
    }

    #[test]
    fn test_quota_enforcer() {
        let enforcer = QuotaEnforcer::new();

        enforcer.set_agent_quota(200, 2 << 30); // 2GB

        assert!(enforcer.check_allocation(200, 1 << 30)); // 1GB OK
        assert!(enforcer.record_allocation(200, 1 << 30));

        assert!(enforcer.check_allocation(200, 500 << 20)); // 500MB OK
        assert!(!enforcer.check_allocation(200, 2 << 30)); // 2GB would exceed

        enforcer.record_deallocation(200, 1 << 30);
        assert!(enforcer.check_allocation(200, 2 << 30)); // Now OK
    }
}
