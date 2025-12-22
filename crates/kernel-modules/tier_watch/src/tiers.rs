//! Tier management and statistics tracking
//!
//! This module manages the 5-tier memory hierarchy and tracks
//! statistics for each tier.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{KernelResult, MemoryTier};

/// Statistics for a single memory tier
pub struct TierStats {
    /// Tier identifier
    pub tier: MemoryTier,
    /// Total pages in this tier
    pub total_pages: AtomicUsize,
    /// Used bytes in this tier
    pub used_bytes: AtomicUsize,
    /// Major page faults
    pub major_faults: AtomicU64,
    /// Minor page faults
    pub minor_faults: AtomicU64,
    /// Pages migrated into this tier
    pub migrations_in: AtomicU64,
    /// Pages migrated out of this tier
    pub migrations_out: AtomicU64,
    /// Total access count
    pub access_count: AtomicU64,
    /// Total latency for operations (ns)
    pub total_latency_ns: AtomicU64,
    /// Number of latency samples
    pub latency_samples: AtomicU64,
}

impl TierStats {
    /// Create new tier statistics
    pub fn new(tier: MemoryTier) -> Self {
        Self {
            tier,
            total_pages: AtomicUsize::new(0),
            used_bytes: AtomicUsize::new(0),
            major_faults: AtomicU64::new(0),
            minor_faults: AtomicU64::new(0),
            migrations_in: AtomicU64::new(0),
            migrations_out: AtomicU64::new(0),
            access_count: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            latency_samples: AtomicU64::new(0),
        }
    }

    /// Calculate memory pressure as percentage (0-100)
    pub fn pressure_percent(&self) -> u8 {
        let used = self.used_bytes.load(Ordering::Relaxed);
        let capacity = self.tier.capacity_bytes();

        if capacity == 0 {
            return 0;
        }

        ((used as f64 / capacity as f64) * 100.0).min(100.0) as u8
    }

    /// Get average latency in nanoseconds
    pub fn avg_latency_ns(&self) -> u64 {
        let samples = self.latency_samples.load(Ordering::Relaxed);
        if samples == 0 {
            self.tier.latency_ns() // Return theoretical latency
        } else {
            self.total_latency_ns.load(Ordering::Relaxed) / samples
        }
    }

    /// Record a page fault
    pub fn record_fault(&self, is_major: bool) {
        if is_major {
            self.major_faults.fetch_add(1, Ordering::Relaxed);
        } else {
            self.minor_faults.fetch_add(1, Ordering::Relaxed);
        }
        crate::TIER_WATCH_STATS.record_fault();
    }

    /// Record a migration in
    pub fn record_migration_in(&self) {
        self.migrations_in.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a migration out
    pub fn record_migration_out(&self) {
        self.migrations_out.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an access with latency
    pub fn record_access(&self, latency_ns: u64) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Update used bytes
    pub fn update_used_bytes(&self, delta: isize) {
        if delta > 0 {
            self.used_bytes.fetch_add(delta as usize, Ordering::Relaxed);
        } else {
            self.used_bytes
                .fetch_sub((-delta) as usize, Ordering::Relaxed);
        }
    }
}

/// Per-CPU tier statistics to reduce contention
#[repr(align(64))] // Cache line aligned
pub struct PerCpuTierStats {
    /// Per-tier counters
    tier_counters: [PerCpuCounter; 5],
}

#[repr(align(64))]
struct PerCpuCounter {
    faults: AtomicU64,
    accesses: AtomicU64,
}

impl PerCpuTierStats {
    const fn new() -> Self {
        Self {
            tier_counters: [
                PerCpuCounter {
                    faults: AtomicU64::new(0),
                    accesses: AtomicU64::new(0),
                },
                PerCpuCounter {
                    faults: AtomicU64::new(0),
                    accesses: AtomicU64::new(0),
                },
                PerCpuCounter {
                    faults: AtomicU64::new(0),
                    accesses: AtomicU64::new(0),
                },
                PerCpuCounter {
                    faults: AtomicU64::new(0),
                    accesses: AtomicU64::new(0),
                },
                PerCpuCounter {
                    faults: AtomicU64::new(0),
                    accesses: AtomicU64::new(0),
                },
            ],
        }
    }

    fn record_fault(&self, tier: MemoryTier) {
        self.tier_counters[tier as usize]
            .faults
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_access(&self, tier: MemoryTier) {
        self.tier_counters[tier as usize]
            .accesses
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Global tier statistics array
static mut TIER_STATS: Option<Vec<TierStats>> = None;

/// Per-CPU statistics (in real kernel, would use per_cpu macro)
static mut PER_CPU_STATS: Option<Vec<PerCpuTierStats>> = None;

/// Get tier statistics
pub fn get_tier_stats(tier: MemoryTier) -> Option<&'static TierStats> {
    unsafe { TIER_STATS.as_ref()?.get(tier as usize) }
}

/// Get mutable tier statistics
pub fn get_tier_stats_mut(tier: MemoryTier) -> Option<&'static mut TierStats> {
    unsafe { TIER_STATS.as_mut()?.get_mut(tier as usize) }
}

/// Record a fault on current CPU
pub fn record_cpu_fault(tier: MemoryTier, cpu_id: usize) {
    unsafe {
        if let Some(per_cpu) = &PER_CPU_STATS {
            if let Some(cpu_stats) = per_cpu.get(cpu_id) {
                cpu_stats.record_fault(tier);
            }
        }
    }
}

/// Record an access on current CPU
pub fn record_cpu_access(tier: MemoryTier, cpu_id: usize) {
    unsafe {
        if let Some(per_cpu) = &PER_CPU_STATS {
            if let Some(cpu_stats) = per_cpu.get(cpu_id) {
                cpu_stats.record_access(tier);
            }
        }
    }
}

/// Consolidate per-CPU statistics into global stats
pub fn consolidate_stats() {
    unsafe {
        if let (Some(global), Some(per_cpu)) = (&mut TIER_STATS, &PER_CPU_STATS) {
            for (tier_idx, tier_stats) in global.iter_mut().enumerate() {
                let mut total_faults = 0u64;
                let mut total_accesses = 0u64;

                // Sum up all per-CPU counters
                for cpu_stats in per_cpu.iter() {
                    total_faults += cpu_stats.tier_counters[tier_idx]
                        .faults
                        .load(Ordering::Relaxed);
                    total_accesses += cpu_stats.tier_counters[tier_idx]
                        .accesses
                        .load(Ordering::Relaxed);
                }

                // Update global stats
                tier_stats
                    .minor_faults
                    .store(total_faults, Ordering::Relaxed);
                tier_stats
                    .access_count
                    .store(total_accesses, Ordering::Relaxed);
            }
        }
    }
}

/// Check if any tier is under pressure
pub fn check_memory_pressure() -> Vec<(MemoryTier, u8)> {
    let mut pressure_tiers = Vec::new();

    unsafe {
        if let Some(stats) = &TIER_STATS {
            for tier_stats in stats.iter() {
                let pressure = tier_stats.pressure_percent();
                if pressure > 80 {
                    pressure_tiers.push((tier_stats.tier, pressure));
                }
            }
        }
    }

    pressure_tiers
}

/// Initialize tier management
pub fn init() -> KernelResult<()> {
    unsafe {
        // Initialize global tier statistics
        let mut stats = Vec::with_capacity(5);
        for tier in MemoryTier::ALL {
            stats.push(TierStats::new(tier));
        }
        TIER_STATS = Some(stats);

        // Initialize per-CPU statistics
        // In real kernel, would get actual CPU count
        let num_cpus = 32; // Assume 32 CPUs for now
        let mut per_cpu = Vec::with_capacity(num_cpus);
        for _ in 0..num_cpus {
            per_cpu.push(PerCpuTierStats::new());
        }
        PER_CPU_STATS = Some(per_cpu);
    }

    Ok(())
}

/// Cleanup tier management
pub fn cleanup() {
    unsafe {
        TIER_STATS = None;
        PER_CPU_STATS = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_calculation() {
        let stats = TierStats::new(MemoryTier::Gpu);

        // Set 80% usage
        let capacity = MemoryTier::Gpu.capacity_bytes();
        stats
            .used_bytes
            .store(capacity * 80 / 100, Ordering::Relaxed);

        let pressure = stats.pressure_percent();
        assert!(pressure >= 79 && pressure <= 80); // Allow for rounding
    }

    #[test]
    fn test_average_latency() {
        let stats = TierStats::new(MemoryTier::Cpu);

        // Record some accesses
        for _ in 0..100 {
            stats.record_access(100); // 100ns each
        }

        assert_eq!(stats.avg_latency_ns(), 100);
    }
}
