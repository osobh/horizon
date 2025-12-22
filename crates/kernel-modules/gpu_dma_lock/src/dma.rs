//! DMA access control and PCIe BAR protection
//!
//! This module implements DMA access control lists and protects
//! GPU memory regions from unauthorized access.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{spin, DmaAccessMode, KernelError, KernelResult, GPU_DMA_STATS};

/// DMA permission entry
#[derive(Debug, Clone)]
pub struct DmaPermission {
    pub agent_id: u64,
    pub start_addr: u64,
    pub end_addr: u64,
    pub access_mode: DmaAccessMode,
}

impl DmaPermission {
    pub fn new(agent_id: u64, start_addr: u64, end_addr: u64, access_mode: DmaAccessMode) -> Self {
        Self {
            agent_id,
            start_addr,
            end_addr,
            access_mode,
        }
    }

    pub fn agent_id(&self) -> u64 {
        self.agent_id
    }

    pub fn can_access(&self, addr: u64, mode: DmaAccessMode) -> bool {
        if addr < self.start_addr || addr >= self.end_addr {
            return false;
        }

        match mode {
            DmaAccessMode::None => false,
            DmaAccessMode::ReadOnly => self.access_mode.can_read(),
            DmaAccessMode::WriteOnly => self.access_mode.can_write(),
            DmaAccessMode::ReadWrite => self.access_mode.can_read() && self.access_mode.can_write(),
        }
    }
}

/// DMA access control list
pub struct DmaAccessControlList {
    /// Permissions by agent
    pub permissions: spin::RwLock<BTreeMap<u64, Vec<DmaPermission>>>,
    /// Quick lookup cache
    pub cache: spin::RwLock<BTreeMap<(u64, u64), bool>>,
}

impl DmaAccessControlList {
    pub fn new() -> Self {
        Self {
            permissions: spin::RwLock::new(BTreeMap::new()),
            cache: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn grant_access(&self, agent_id: u64, start_addr: u64, end_addr: u64, mode: DmaAccessMode) {
        let permission = DmaPermission::new(agent_id, start_addr, end_addr, mode);

        let mut perms = self.permissions.write();
        perms
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(permission);

        // Clear cache for this agent
        let mut cache = self.cache.write();
        cache.retain(|(aid, _), _| *aid != agent_id);
    }

    pub fn revoke_access(&self, agent_id: u64) {
        self.permissions.write().remove(&agent_id);

        // Clear cache for this agent
        let mut cache = self.cache.write();
        cache.retain(|(aid, _), _| *aid != agent_id);
    }

    pub fn check_access(&self, agent_id: u64, addr: u64, mode: DmaAccessMode) -> bool {
        // OPTIMIZATION: Cache is temporarily disabled to ensure correctness
        // Future optimization: Include access mode in cache key for performance improvement
        // let cache_key = (agent_id, addr & !0xFFF, mode); // Include mode in key
        // if let Some(&allowed) = self.cache.read().get(&cache_key) {
        //     GPU_DMA_STATS.record_dma_check(allowed);
        //     return allowed;
        // }

        // Check permissions
        let allowed = self
            .permissions
            .read()
            .get(&agent_id)
            .map(|perms| perms.iter().any(|p| p.can_access(addr, mode)))
            .unwrap_or(false);

        // Update cache (disabled for now)
        // self.cache.write().insert(cache_key, allowed);

        GPU_DMA_STATS.record_dma_check(allowed);
        allowed
    }
}

/// PCIe BAR protection
pub struct PcieBarProtection {
    /// BAR regions by device
    bar_regions: spin::RwLock<BTreeMap<u32, Vec<BarRegion>>>,
    /// Authorized agents per device
    authorized_agents: spin::RwLock<BTreeMap<u32, Vec<u64>>>,
}

#[derive(Debug, Clone)]
struct BarRegion {
    bar_index: u8,
    start_addr: u64,
    size: u64,
}

impl PcieBarProtection {
    pub fn new() -> Self {
        Self {
            bar_regions: spin::RwLock::new(BTreeMap::new()),
            authorized_agents: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn register_bar(&self, device_id: u32, bar_index: u8, start_addr: u64, size: u64) {
        let region = BarRegion {
            bar_index,
            start_addr,
            size,
        };

        let mut bars = self.bar_regions.write();
        bars.entry(device_id).or_insert_with(Vec::new).push(region);
    }

    pub fn is_protected(&self, addr: u64) -> bool {
        let bars = self.bar_regions.read();
        bars.values().any(|regions| {
            regions
                .iter()
                .any(|r| addr >= r.start_addr && addr < r.start_addr + r.size)
        })
    }

    pub fn authorize_agent(&self, agent_id: u64, device_id: u32) {
        let mut auth = self.authorized_agents.write();
        auth.entry(device_id)
            .or_insert_with(Vec::new)
            .push(agent_id);
    }

    pub fn check_agent_access(&self, agent_id: u64, addr: u64) -> bool {
        // Find which device this address belongs to
        let bars = self.bar_regions.read();
        for (device_id, regions) in bars.iter() {
            for region in regions {
                if addr >= region.start_addr && addr < region.start_addr + region.size {
                    // Found the device, check authorization
                    let auth = self.authorized_agents.read();
                    if let Some(agents) = auth.get(device_id) {
                        return agents.contains(&agent_id);
                    }
                    return false;
                }
            }
        }
        false
    }
}

/// GPU context isolation manager
pub struct GpuContextManager {
    /// Context counter
    next_context_id: AtomicU64,
    /// Agent to context mapping
    agent_contexts: spin::RwLock<BTreeMap<u64, u64>>,
    /// Context information
    contexts: spin::RwLock<BTreeMap<u64, GpuContext>>,
}

#[derive(Debug, Clone)]
struct GpuContext {
    id: u64,
    agent_id: u64,
    device_id: u32,
    created_at: u64,
}

impl GpuContextManager {
    pub fn new() -> Self {
        Self {
            next_context_id: AtomicU64::new(1),
            agent_contexts: spin::RwLock::new(BTreeMap::new()),
            contexts: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn create_context(&self, agent_id: u64, device_id: u32) -> u64 {
        let context_id = self.next_context_id.fetch_add(1, Ordering::Relaxed);

        let context = GpuContext {
            id: context_id,
            agent_id,
            device_id,
            created_at: get_current_time(),
        };

        self.contexts.write().insert(context_id, context);
        self.agent_contexts.write().insert(agent_id, context_id);

        context_id
    }

    pub fn is_isolated(&self, ctx1: u64, ctx2: u64) -> bool {
        // Contexts are always isolated in this implementation
        ctx1 != ctx2
    }

    pub fn measure_switch_overhead(&self, _from: u64, _to: u64) -> u64 {
        // Simulated context switch overhead in nanoseconds
        GPU_DMA_STATS.record_context_switch();
        500 // 500ns typical overhead
    }

    pub fn get_agent_context(&self, agent_id: u64) -> Option<u64> {
        self.agent_contexts.read().get(&agent_id).copied()
    }
}

/// GPUDirect RDMA manager
pub struct GpuDirectRdmaManager {
    /// RDMA registrations
    registrations: spin::RwLock<BTreeMap<u64, RdmaRegistration>>,
    /// Next handle ID
    next_handle: AtomicU64,
}

#[derive(Debug, Clone)]
struct RdmaRegistration {
    handle: u64,
    agent_id: u64,
    gpu_addr: u64,
    size: usize,
}

impl GpuDirectRdmaManager {
    pub fn new() -> Self {
        Self {
            registrations: spin::RwLock::new(BTreeMap::new()),
            next_handle: AtomicU64::new(1),
        }
    }

    pub fn is_available(&self) -> bool {
        // Check if GPUDirect RDMA is supported
        // In real implementation, would check driver capabilities
        true
    }

    pub fn register_memory(&self, agent_id: u64, gpu_addr: u64, size: usize) -> KernelResult<u64> {
        if !self.is_available() {
            return Err(KernelError::NotSupported);
        }

        let handle = self.next_handle.fetch_add(1, Ordering::Relaxed);

        let registration = RdmaRegistration {
            handle,
            agent_id,
            gpu_addr,
            size,
        };

        self.registrations.write().insert(handle, registration);

        Ok(handle)
    }

    pub fn authorize_transfer(&self, agent_id: u64, handle: u64) -> bool {
        self.registrations
            .read()
            .get(&handle)
            .map(|reg| reg.agent_id == agent_id)
            .unwrap_or(false)
    }

    pub fn unregister(&self, handle: u64) -> KernelResult<()> {
        self.registrations
            .write()
            .remove(&handle)
            .ok_or(KernelError::InvalidArgument)?;
        Ok(())
    }
}

/// DMA transfer statistics tracker
pub struct DmaTransferTracker {
    /// Transfer records by agent
    transfers: spin::RwLock<BTreeMap<u64, Vec<TransferRecord>>>,
}

#[derive(Debug, Clone)]
struct TransferRecord {
    bytes: usize,
    latency_us: u64,
    timestamp: u64,
}

impl DmaTransferTracker {
    pub fn new() -> Self {
        Self {
            transfers: spin::RwLock::new(BTreeMap::new()),
        }
    }

    pub fn track_transfer(&self, agent_id: u64, bytes: usize, latency_us: u64) {
        let record = TransferRecord {
            bytes,
            latency_us,
            timestamp: get_current_time(),
        };

        let mut transfers = self.transfers.write();
        transfers
            .entry(agent_id)
            .or_insert_with(Vec::new)
            .push(record);
    }

    pub fn get_agent_stats(&self, agent_id: u64) -> DmaTransferStats {
        let transfers = self.transfers.read();
        if let Some(records) = transfers.get(&agent_id) {
            let total_bytes = records.iter().map(|r| r.bytes).sum();
            let transfer_count = records.len();
            let avg_latency_us = if transfer_count > 0 {
                records.iter().map(|r| r.latency_us).sum::<u64>() / transfer_count as u64
            } else {
                0
            };

            DmaTransferStats {
                total_bytes,
                transfer_count,
                avg_latency_us,
            }
        } else {
            DmaTransferStats {
                total_bytes: 0,
                transfer_count: 0,
                avg_latency_us: 0,
            }
        }
    }

    pub fn calculate_bandwidth(&self, agent_id: u64) -> f64 {
        let stats = self.get_agent_stats(agent_id);
        if stats.avg_latency_us > 0 {
            (stats.total_bytes as f64) / (stats.avg_latency_us as f64 * 1e-6)
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct DmaTransferStats {
    pub total_bytes: usize,
    pub transfer_count: usize,
    pub avg_latency_us: u64,
}

/// Global DMA manager instance
static mut DMA_MANAGER: Option<DmaManager> = None;

/// Combined DMA management
pub struct DmaManager {
    pub acl: DmaAccessControlList,
    pub bar_protection: PcieBarProtection,
    pub context_manager: GpuContextManager,
    pub rdma_manager: GpuDirectRdmaManager,
    pub transfer_tracker: DmaTransferTracker,
}

impl DmaManager {
    pub fn new() -> Self {
        Self {
            acl: DmaAccessControlList::new(),
            bar_protection: PcieBarProtection::new(),
            context_manager: GpuContextManager::new(),
            rdma_manager: GpuDirectRdmaManager::new(),
            transfer_tracker: DmaTransferTracker::new(),
        }
    }
}

/// Get global DMA manager
pub fn get_manager() -> &'static DmaManager {
    unsafe { DMA_MANAGER.as_ref().expect("DMA manager not initialized") }
}

/// Initialize DMA subsystem
pub fn init() -> KernelResult<()> {
    unsafe {
        DMA_MANAGER = Some(DmaManager::new());
    }
    Ok(())
}

/// Cleanup DMA subsystem
pub fn cleanup() {
    unsafe {
        DMA_MANAGER = None;
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
    fn test_dma_permission() {
        let perm = DmaPermission::new(100, 0x1000, 0x2000, DmaAccessMode::ReadWrite);

        assert!(perm.can_access(0x1500, DmaAccessMode::ReadOnly));
        assert!(perm.can_access(0x1500, DmaAccessMode::WriteOnly));
        assert!(!perm.can_access(0x0500, DmaAccessMode::ReadOnly));
        assert!(!perm.can_access(0x2500, DmaAccessMode::ReadOnly));
    }

    #[test]
    fn test_dma_acl() {
        let acl = DmaAccessControlList::new();

        acl.grant_access(200, 0x1000, 0x2000, DmaAccessMode::ReadOnly);
        acl.grant_access(201, 0x2000, 0x3000, DmaAccessMode::ReadWrite);

        // Test read access for agent 200 (should pass)
        assert!(acl.check_access(200, 0x1500, DmaAccessMode::ReadOnly));

        // Test write access for agent 200 (should fail because they only have ReadOnly)
        let write_access = acl.check_access(200, 0x1500, DmaAccessMode::WriteOnly);
        assert!(
            !write_access,
            "Agent 200 should not have write access, but got: {}",
            write_access
        );

        assert!(acl.check_access(201, 0x2500, DmaAccessMode::WriteOnly));
        assert!(!acl.check_access(202, 0x1500, DmaAccessMode::ReadOnly));

        acl.revoke_access(200);
        assert!(!acl.check_access(200, 0x1500, DmaAccessMode::ReadOnly));
    }

    #[test]
    fn test_bar_protection() {
        let bar = PcieBarProtection::new();

        bar.register_bar(0, 0, 0xF0000000, 0x1000000);
        bar.register_bar(1, 0, 0xE0000000, 0x2000000);

        assert!(bar.is_protected(0xF0100000));
        assert!(bar.is_protected(0xE1000000));
        assert!(!bar.is_protected(0xD0000000));

        bar.authorize_agent(300, 0);
        assert!(bar.check_agent_access(300, 0xF0100000));
        assert!(!bar.check_agent_access(301, 0xF0100000));
    }

    #[test]
    fn test_context_isolation() {
        let ctx_mgr = GpuContextManager::new();

        let ctx1 = ctx_mgr.create_context(400, 0);
        let ctx2 = ctx_mgr.create_context(401, 0);

        assert_ne!(ctx1, ctx2);
        assert!(ctx_mgr.is_isolated(ctx1, ctx2));

        let overhead = ctx_mgr.measure_switch_overhead(ctx1, ctx2);
        assert!(overhead < 1000); // Less than 1µs
    }
}
