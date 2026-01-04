//! Page fault handling and tracking
//!
//! This module hooks into the kernel's page fault handler to track
//! access patterns across the memory hierarchy.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{KernelResult, MemoryTier};

/// Page information for tracking
#[derive(Debug, Clone, Copy)]
pub struct PageInfo {
    /// Physical page frame number
    pub pfn: u64,
    /// Virtual address that triggered fault
    pub vaddr: u64,
    /// Current memory tier
    pub tier: MemoryTier,
    /// Access count for this page
    pub access_count: u32,
    /// Last access timestamp (in jiffies)
    pub last_access: u64,
    /// Agent ID that owns this page
    pub agent_id: Option<u64>,
    /// NUMA node ID
    pub numa_node: u8,
}

/// Compact page tracking structure (5 bytes)
#[repr(packed)]
#[derive(Debug, Clone, Copy)]
pub struct CompactPageInfo {
    /// Tier (3 bits) and flags (5 bits)
    pub tier_and_flags: u8,
    /// Access count (capped at 65535)
    pub access_count: u16,
    /// Agent ID (0 = unassigned)
    pub agent_id: u16,
}

impl CompactPageInfo {
    /// Create new compact page info
    pub fn new(tier: MemoryTier, agent_id: u16) -> Self {
        Self {
            tier_and_flags: (tier as u8) & 0x07,
            access_count: 0,
            agent_id,
        }
    }

    /// Get tier from compact representation
    pub fn tier(&self) -> MemoryTier {
        match self.tier_and_flags & 0x07 {
            0 => MemoryTier::Gpu,
            1 => MemoryTier::Cpu,
            2 => MemoryTier::Nvme,
            3 => MemoryTier::Ssd,
            4 => MemoryTier::Hdd,
            _ => MemoryTier::Cpu, // Default
        }
    }

    /// Set tier in compact representation
    pub fn set_tier(&mut self, tier: MemoryTier) {
        self.tier_and_flags = (self.tier_and_flags & 0xF8) | ((tier as u8) & 0x07);
    }

    /// Increment access count (with saturation)
    pub fn increment_access(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }

    /// Check if page is hot (frequently accessed)
    pub fn is_hot(&self) -> bool {
        self.access_count > 100
    }

    /// Check if page is cold (rarely accessed)
    pub fn is_cold(&self) -> bool {
        self.access_count < 10
    }
}

/// Page fault type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultType {
    /// Page not present (major fault)
    Major,
    /// Page present but not accessible (minor fault)
    Minor,
    /// Copy-on-write fault
    CoW,
    /// Permission fault
    Permission,
}

/// Page fault statistics
pub struct FaultStats {
    /// Total faults by type
    pub major_faults: AtomicU64,
    pub minor_faults: AtomicU64,
    pub cow_faults: AtomicU64,
    pub perm_faults: AtomicU64,
    /// Faults by tier
    pub tier_faults: [AtomicU64; 5],
}

impl FaultStats {
    const fn new() -> Self {
        Self {
            major_faults: AtomicU64::new(0),
            minor_faults: AtomicU64::new(0),
            cow_faults: AtomicU64::new(0),
            perm_faults: AtomicU64::new(0),
            tier_faults: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    fn record_fault(&self, fault_type: FaultType, tier: MemoryTier) {
        match fault_type {
            FaultType::Major => self.major_faults.fetch_add(1, Ordering::Relaxed),
            FaultType::Minor => self.minor_faults.fetch_add(1, Ordering::Relaxed),
            FaultType::CoW => self.cow_faults.fetch_add(1, Ordering::Relaxed),
            FaultType::Permission => self.perm_faults.fetch_add(1, Ordering::Relaxed),
        };

        self.tier_faults[tier as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Global fault statistics
static FAULT_STATS: FaultStats = FaultStats::new();

/// Page tracking storage
static mut PAGE_TRACKER: Option<PageTracker> = None;

/// Efficient page tracking with hierarchical structure
pub struct PageTracker {
    /// Level 1: Buckets for page groups
    buckets: Vec<PageBucket>,
    /// Total pages tracked
    total_pages: AtomicUsize,
}

/// Page bucket for efficient tracking
struct PageBucket {
    /// Pages in this bucket
    pages: Vec<CompactPageInfo>,
    /// Bucket statistics
    hot_pages: AtomicUsize,
    cold_pages: AtomicUsize,
}

impl PageTracker {
    /// Create new page tracker
    fn new(num_buckets: usize) -> Self {
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(PageBucket {
                pages: Vec::with_capacity(10_000),
                hot_pages: AtomicUsize::new(0),
                cold_pages: AtomicUsize::new(0),
            });
        }

        Self {
            buckets,
            total_pages: AtomicUsize::new(0),
        }
    }

    /// Track a page access
    fn track_access(&mut self, pfn: u64, tier: MemoryTier, agent_id: u16) {
        let bucket_idx = (pfn as usize) % self.buckets.len();
        let page_idx = (pfn as usize) / self.buckets.len();

        let bucket = &mut self.buckets[bucket_idx];

        // Ensure bucket has enough pages
        while bucket.pages.len() <= page_idx {
            bucket.pages.push(CompactPageInfo::new(MemoryTier::Cpu, 0));
        }

        // Update page info
        let page = &mut bucket.pages[page_idx];
        let was_hot = page.is_hot();
        let was_cold = page.is_cold();

        page.increment_access();
        page.set_tier(tier);
        if agent_id > 0 {
            page.agent_id = agent_id;
        }

        // Update bucket statistics
        if !was_hot && page.is_hot() {
            bucket.hot_pages.fetch_add(1, Ordering::Relaxed);
        }
        if was_cold && !page.is_cold() {
            bucket.cold_pages.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get page info
    fn get_page_info(&self, pfn: u64) -> Option<CompactPageInfo> {
        let bucket_idx = (pfn as usize) % self.buckets.len();
        let page_idx = (pfn as usize) / self.buckets.len();

        self.buckets.get(bucket_idx)?.pages.get(page_idx).copied()
    }
}

/// Handle a page fault
pub fn handle_page_fault(
    _vaddr: u64,
    pfn: u64,
    fault_type: FaultType,
    agent_id: Option<u64>,
) -> KernelResult<()> {
    // Determine which tier this page belongs to
    let tier = determine_page_tier(pfn)?;

    // Record fault statistics
    FAULT_STATS.record_fault(fault_type, tier);

    // Update tier statistics
    if let Some(tier_stats) = crate::tiers::get_tier_stats(tier) {
        tier_stats.record_fault(fault_type == FaultType::Major);
    }

    // Track page access
    // SAFETY: PAGE_TRACKER is a module-private static that is only accessed from
    // kernel fault handling context. In kernel modules, fault handlers run with
    // interrupts disabled or proper locking. The Option check ensures we only
    // access after init() and before cleanup().
    unsafe {
        if let Some(tracker) = &mut PAGE_TRACKER {
            tracker.track_access(pfn, tier, agent_id.unwrap_or(0) as u16);
        }
    }

    // Record per-CPU fault
    let cpu_id = get_current_cpu();
    crate::tiers::record_cpu_fault(tier, cpu_id);

    Ok(())
}

/// Determine which tier a page belongs to
fn determine_page_tier(pfn: u64) -> KernelResult<MemoryTier> {
    // In real kernel, would check page flags and memory zones
    // For now, use simple heuristic based on PFN ranges

    // Assume GPU memory is in high PFNs
    if pfn > 0x1000000 {
        Ok(MemoryTier::Gpu)
    }
    // Regular RAM
    else if pfn < 0x100000 {
        Ok(MemoryTier::Cpu)
    }
    // Everything else is storage-backed
    else {
        Ok(MemoryTier::Nvme)
    }
}

/// Get current CPU ID (mock for kernel module)
fn get_current_cpu() -> usize {
    // In real kernel: smp_processor_id()
    0
}

/// Get hot pages that should be promoted
pub fn get_hot_pages(max_pages: usize) -> Vec<(u64, MemoryTier)> {
    let mut hot_pages = Vec::new();

    // SAFETY: PAGE_TRACKER is only modified during init/cleanup which are called
    // during module load/unload. This read-only access during normal operation
    // does not race with modifications. The Option check handles pre-init state.
    unsafe {
        if let Some(tracker) = &PAGE_TRACKER {
            for (bucket_idx, bucket) in tracker.buckets.iter().enumerate() {
                for (page_idx, page) in bucket.pages.iter().enumerate() {
                    if page.is_hot() && page.tier() > MemoryTier::Gpu {
                        let pfn = (bucket_idx + page_idx * tracker.buckets.len()) as u64;
                        hot_pages.push((pfn, page.tier()));

                        if hot_pages.len() >= max_pages {
                            return hot_pages;
                        }
                    }
                }
            }
        }
    }

    hot_pages
}

/// Get cold pages that should be demoted
pub fn get_cold_pages(max_pages: usize) -> Vec<(u64, MemoryTier)> {
    let mut cold_pages = Vec::new();

    // SAFETY: PAGE_TRACKER is only modified during init/cleanup which are called
    // during module load/unload. This read-only access during normal operation
    // does not race with modifications. The Option check handles pre-init state.
    unsafe {
        if let Some(tracker) = &PAGE_TRACKER {
            for (bucket_idx, bucket) in tracker.buckets.iter().enumerate() {
                for (page_idx, page) in bucket.pages.iter().enumerate() {
                    if page.is_cold() && page.tier() < MemoryTier::Hdd {
                        let pfn = (bucket_idx + page_idx * tracker.buckets.len()) as u64;
                        cold_pages.push((pfn, page.tier()));

                        if cold_pages.len() >= max_pages {
                            return cold_pages;
                        }
                    }
                }
            }
        }
    }

    cold_pages
}

/// Initialize fault handling
pub fn init() -> KernelResult<()> {
    // SAFETY: init() is called exactly once during module load, before any
    // fault handling occurs. No concurrent access to PAGE_TRACKER is possible
    // at module initialization time.
    unsafe {
        // Initialize page tracker with 1000 buckets
        // This allows tracking 10M pages with 10K pages per bucket
        PAGE_TRACKER = Some(PageTracker::new(1000));
    }

    // In real kernel: hook page fault handler

    Ok(())
}

/// Cleanup fault handling
pub fn cleanup() {
    // SAFETY: cleanup() is called exactly once during module unload, after all
    // fault handling has stopped. No concurrent access to PAGE_TRACKER is
    // possible at module cleanup time.
    unsafe {
        PAGE_TRACKER = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_page_info() {
        let mut page = CompactPageInfo::new(MemoryTier::Cpu, 42);

        // Copy fields to avoid unaligned access
        let tier = page.tier();
        let agent_id = page.agent_id;
        let access_count = page.access_count;

        assert_eq!(tier, MemoryTier::Cpu);
        assert_eq!(agent_id, 42);
        assert_eq!(access_count, 0);
        assert!(page.is_cold());

        // Increment access
        for _ in 0..150 {
            page.increment_access();
        }

        assert!(page.is_hot());
        // Copy field to avoid unaligned access
        let access_count = page.access_count;
        assert_eq!(access_count, 150);

        // Change tier
        page.set_tier(MemoryTier::Gpu);
        assert_eq!(page.tier(), MemoryTier::Gpu);
    }

    #[test]
    fn test_fault_type_recording() {
        FAULT_STATS.record_fault(FaultType::Major, MemoryTier::Cpu);
        FAULT_STATS.record_fault(FaultType::Minor, MemoryTier::Cpu);
        FAULT_STATS.record_fault(FaultType::Minor, MemoryTier::Gpu);

        assert!(FAULT_STATS.major_faults.load(Ordering::Relaxed) > 0);
        assert!(FAULT_STATS.minor_faults.load(Ordering::Relaxed) > 0);
        assert!(FAULT_STATS.tier_faults[MemoryTier::Cpu as usize].load(Ordering::Relaxed) > 0);
        assert!(FAULT_STATS.tier_faults[MemoryTier::Gpu as usize].load(Ordering::Relaxed) > 0);
    }
}
