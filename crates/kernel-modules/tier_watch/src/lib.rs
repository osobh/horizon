//! TierWatch Kernel Module
//!
//! Monitors the 5-tier memory hierarchy:
//! 1. GPU (32GB) - 200ns latency
//! 2. CPU (96GB) - 50ns latency  
//! 3. NVMe (3.2TB) - 20μs latency
//! 4. SSD (4.5TB) - 100μs latency
//! 5. HDD (3.7TB) - 10ms latency
//!
//! Provides:
//! - Page fault tracking across all tiers
//! - Memory pressure monitoring
//! - Migration candidate detection
//! - NUMA-aware optimization
//! - Per-agent memory tracking

#![no_std]

extern crate alloc;

pub mod fault;
pub mod ffi;
pub mod migration;
pub mod numa;
pub mod proc;
pub mod stats;
pub mod tiers;

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Maximum number of pages we can track
pub const MAX_TRACKED_PAGES: usize = 10_000_000; // 10M pages = 40GB at 4KB/page

/// Memory tier enumeration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryTier {
    Gpu = 0,
    Cpu = 1,
    Nvme = 2,
    Ssd = 3,
    Hdd = 4,
}

impl MemoryTier {
    /// All available tiers
    pub const ALL: [Self; 5] = [Self::Gpu, Self::Cpu, Self::Nvme, Self::Ssd, Self::Hdd];

    /// Get tier name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
            Self::Nvme => "nvme",
            Self::Ssd => "ssd",
            Self::Hdd => "hdd",
        }
    }

    /// Get tier capacity in bytes
    pub fn capacity_bytes(&self) -> usize {
        match self {
            Self::Gpu => 32 * (1 << 30),    // 32GB
            Self::Cpu => 96 * (1 << 30),    // 96GB
            Self::Nvme => 3200 * (1 << 30), // 3.2TB
            Self::Ssd => 4500 * (1 << 30),  // 4.5TB
            Self::Hdd => 3700 * (1 << 30),  // 3.7TB
        }
    }

    /// Get tier latency in nanoseconds
    pub fn latency_ns(&self) -> u64 {
        match self {
            Self::Gpu => 200,        // 200ns
            Self::Cpu => 50,         // 50ns
            Self::Nvme => 20_000,    // 20μs
            Self::Ssd => 100_000,    // 100μs
            Self::Hdd => 10_000_000, // 10ms
        }
    }

    /// Get next (slower) tier
    pub fn next_tier(&self) -> Option<Self> {
        match self {
            Self::Gpu => Some(Self::Cpu),
            Self::Cpu => Some(Self::Nvme),
            Self::Nvme => Some(Self::Ssd),
            Self::Ssd => Some(Self::Hdd),
            Self::Hdd => None,
        }
    }

    /// Get previous (faster) tier
    pub fn prev_tier(&self) -> Option<Self> {
        match self {
            Self::Gpu => None,
            Self::Cpu => Some(Self::Gpu),
            Self::Nvme => Some(Self::Cpu),
            Self::Ssd => Some(Self::Nvme),
            Self::Hdd => Some(Self::Ssd),
        }
    }
}

/// Global tier monitoring statistics
pub struct TierWatchStats {
    /// Total pages tracked
    pub pages_tracked: AtomicUsize,
    /// Total page faults intercepted
    pub total_faults: AtomicU64,
    /// Total migrations performed
    pub total_migrations: AtomicU64,
    /// Total migration latency in nanoseconds
    pub total_migration_ns: AtomicU64,
}

impl TierWatchStats {
    pub const fn new() -> Self {
        Self {
            pages_tracked: AtomicUsize::new(0),
            total_faults: AtomicU64::new(0),
            total_migrations: AtomicU64::new(0),
            total_migration_ns: AtomicU64::new(0),
        }
    }

    pub fn record_fault(&self) {
        self.total_faults.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_migration(&self, latency_ns: u64) {
        self.total_migrations.fetch_add(1, Ordering::Relaxed);
        self.total_migration_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
    }

    pub fn avg_migration_latency_ns(&self) -> u64 {
        let migrations = self.total_migrations.load(Ordering::Relaxed);
        if migrations == 0 {
            0
        } else {
            self.total_migration_ns.load(Ordering::Relaxed) / migrations
        }
    }
}

/// Global statistics instance
pub static TIER_WATCH_STATS: TierWatchStats = TierWatchStats::new();

/// Result type for kernel operations
pub type KernelResult<T> = Result<T, KernelError>;

/// Kernel error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// Out of memory
    OutOfMemory,
    /// Invalid tier specified
    InvalidTier,
    /// Page not found
    PageNotFound,
    /// Migration failed
    MigrationFailed,
    /// NUMA node error
    NumaError,
    /// Invalid argument
    InvalidArgument,
    /// Operation not supported
    NotSupported,
}

/// Initialize the TierWatch module
pub fn init() -> KernelResult<()> {
    // Initialize subsystems
    tiers::init()?;
    fault::init()?;
    migration::init()?;
    numa::init()?;
    proc::init()?;

    Ok(())
}

/// Cleanup the TierWatch module
pub fn cleanup() {
    proc::cleanup();
    numa::cleanup();
    migration::cleanup();
    fault::cleanup();
    tiers::cleanup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_ordering() {
        assert!(MemoryTier::Gpu < MemoryTier::Cpu);
        assert!(MemoryTier::Cpu < MemoryTier::Nvme);
        assert!(MemoryTier::Nvme < MemoryTier::Ssd);
        assert!(MemoryTier::Ssd < MemoryTier::Hdd);
    }

    #[test]
    fn test_tier_transitions() {
        assert_eq!(MemoryTier::Gpu.next_tier(), Some(MemoryTier::Cpu));
        assert_eq!(MemoryTier::Hdd.next_tier(), None);
        assert_eq!(MemoryTier::Gpu.prev_tier(), None);
        assert_eq!(MemoryTier::Cpu.prev_tier(), Some(MemoryTier::Gpu));
    }

    #[test]
    fn test_tier_properties() {
        // Verify tier capacities match expected values
        assert_eq!(MemoryTier::Gpu.capacity_bytes(), 32 * (1 << 30));
        assert_eq!(MemoryTier::Cpu.capacity_bytes(), 96 * (1 << 30));
        assert_eq!(MemoryTier::Nvme.capacity_bytes(), 3200 * (1 << 30));
        assert_eq!(MemoryTier::Ssd.capacity_bytes(), 4500 * (1 << 30));
        assert_eq!(MemoryTier::Hdd.capacity_bytes(), 3700 * (1 << 30));

        // Verify latencies increase down the hierarchy (excluding GPU)
        let mut prev_latency = MemoryTier::Cpu.latency_ns();
        for tier in [MemoryTier::Nvme, MemoryTier::Ssd, MemoryTier::Hdd] {
            let latency = tier.latency_ns();
            assert!(latency > prev_latency);
            prev_latency = latency;
        }

        // GPU has higher latency than CPU but is still faster tier
        assert!(MemoryTier::Gpu.latency_ns() > MemoryTier::Cpu.latency_ns());
    }
}
