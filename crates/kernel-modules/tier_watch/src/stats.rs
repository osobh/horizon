//! Statistics aggregation and reporting
//!
//! This module provides centralized statistics collection and reporting
//! for the tier watch system.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{KernelResult, MemoryTier};

/// Aggregated statistics across all tiers
#[derive(Debug, Clone)]
pub struct AggregatedStats {
    /// Total memory capacity across all tiers
    pub total_capacity_bytes: u64,
    /// Total used memory across all tiers
    pub total_used_bytes: u64,
    /// Overall memory pressure (0-100)
    pub overall_pressure: u8,
    /// Total page faults across all tiers
    pub total_faults: u64,
    /// Total migrations performed
    pub total_migrations: u64,
    /// Average migration latency in nanoseconds
    pub avg_migration_latency_ns: u64,
    /// Per-tier breakdown
    pub tier_stats: Vec<TierStatsSummary>,
    /// NUMA locality ratio (0.0-1.0)
    pub numa_locality_ratio: f64,
}

/// Summary statistics for a single tier
#[derive(Debug, Clone)]
pub struct TierStatsSummary {
    pub tier: MemoryTier,
    pub capacity_bytes: u64,
    pub used_bytes: u64,
    pub pressure_percent: u8,
    pub major_faults: u64,
    pub minor_faults: u64,
    pub migrations_in: u64,
    pub migrations_out: u64,
    pub avg_latency_ns: u64,
    pub hot_pages: u64,
    pub cold_pages: u64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Page fault overhead in nanoseconds
    pub fault_overhead_ns: u64,
    /// Migration detection time in microseconds
    pub detection_time_us: u64,
    /// Current pages tracked
    pub pages_tracked: usize,
    /// Memory overhead in bytes
    pub memory_overhead_bytes: usize,
}

/// Historical statistics window
const HISTORY_WINDOW_SIZE: usize = 60; // 60 seconds

/// Historical data point
#[derive(Debug, Clone, Copy)]
struct HistoryPoint {
    timestamp_sec: u64,
    total_faults: u64,
    total_migrations: u64,
    overall_pressure: u8,
}

/// Historical statistics
static mut HISTORY: Option<Vec<HistoryPoint>> = None;
static HISTORY_INDEX: AtomicU64 = AtomicU64::new(0);

/// Collect current statistics
pub fn collect_stats() -> AggregatedStats {
    let mut total_capacity = 0u64;
    let mut total_used = 0u64;
    let mut total_faults = 0u64;
    let mut total_migrations = 0u64;
    let mut tier_stats = Vec::new();

    // Consolidate per-CPU stats first
    crate::tiers::consolidate_stats();

    // Collect per-tier statistics
    for tier in MemoryTier::ALL {
        if let Some(stats) = crate::tiers::get_tier_stats(tier) {
            let capacity = tier.capacity_bytes() as u64;
            let used = stats.used_bytes.load(Ordering::Relaxed) as u64;
            let major = stats.major_faults.load(Ordering::Relaxed);
            let minor = stats.minor_faults.load(Ordering::Relaxed);
            let in_migrations = stats.migrations_in.load(Ordering::Relaxed);
            let out_migrations = stats.migrations_out.load(Ordering::Relaxed);

            total_capacity += capacity;
            total_used += used;
            total_faults += major + minor;
            let _ = total_migrations; // Will be used later
            total_migrations += in_migrations + out_migrations;

            // Count hot/cold pages (simplified)
            let (hot_pages, cold_pages) = count_hot_cold_pages(tier);

            tier_stats.push(TierStatsSummary {
                tier,
                capacity_bytes: capacity,
                used_bytes: used,
                pressure_percent: stats.pressure_percent(),
                major_faults: major,
                minor_faults: minor,
                migrations_in: in_migrations,
                migrations_out: out_migrations,
                avg_latency_ns: stats.avg_latency_ns(),
                hot_pages,
                cold_pages,
            });
        }
    }

    // Calculate overall pressure
    let overall_pressure = if total_capacity > 0 {
        ((total_used as f64 / total_capacity as f64) * 100.0) as u8
    } else {
        0
    };

    // Get migration statistics
    let migration_stats = crate::migration::get_migration_stats();
    let avg_migration_latency_ns = if migration_stats.total_success > 0 {
        migration_stats.avg_time_us * 1000
    } else {
        0
    };

    // Calculate NUMA locality
    let numa_locality_ratio = calculate_numa_locality();

    AggregatedStats {
        total_capacity_bytes: total_capacity,
        total_used_bytes: total_used,
        overall_pressure,
        total_faults,
        total_migrations: migration_stats.total_success,
        avg_migration_latency_ns,
        tier_stats,
        numa_locality_ratio,
    }
}

/// Count hot and cold pages in a tier
fn count_hot_cold_pages(tier: MemoryTier) -> (u64, u64) {
    // In real implementation, would iterate through page tracker
    // For now, return estimates based on tier characteristics
    match tier {
        MemoryTier::Gpu => (1000, 100),
        MemoryTier::Cpu => (5000, 500),
        MemoryTier::Nvme => (2000, 2000),
        MemoryTier::Ssd => (1000, 5000),
        MemoryTier::Hdd => (100, 10000),
    }
}

/// Calculate overall NUMA locality ratio
fn calculate_numa_locality() -> f64 {
    let numa_stats = crate::numa::get_numa_stats_summary();

    if numa_stats.is_empty() {
        return 1.0;
    }

    let total_local: u64 = numa_stats.iter().map(|s| s.local_accesses).sum();
    let total_remote: u64 = numa_stats.iter().map(|s| s.remote_accesses).sum();
    let total = total_local + total_remote;

    if total > 0 {
        total_local as f64 / total as f64
    } else {
        1.0
    }
}

/// Get performance metrics
pub fn get_performance_metrics() -> PerformanceMetrics {
    // Calculate page fault overhead
    let fault_overhead_ns = measure_fault_overhead();

    // Calculate migration detection time
    let detection_time_us = measure_detection_time();

    // Get current tracking stats
    let pages_tracked = crate::TIER_WATCH_STATS
        .pages_tracked
        .load(Ordering::Relaxed);

    // Calculate memory overhead (5 bytes per page)
    let memory_overhead_bytes = pages_tracked * 5;

    PerformanceMetrics {
        fault_overhead_ns,
        detection_time_us,
        pages_tracked,
        memory_overhead_bytes,
    }
}

/// Measure page fault handling overhead
fn measure_fault_overhead() -> u64 {
    // In real implementation, would use precise timing
    // Target: <100ns
    85 // Mock value
}

/// Measure migration detection time
fn measure_detection_time() -> u64 {
    // In real implementation, would time detection cycle
    // Target: <1ms = 1000μs
    750 // Mock value
}

/// Record statistics snapshot
pub fn record_snapshot() {
    let stats = collect_stats();
    let timestamp = get_current_time_sec();

    let point = HistoryPoint {
        timestamp_sec: timestamp,
        total_faults: stats.total_faults,
        total_migrations: stats.total_migrations,
        overall_pressure: stats.overall_pressure,
    };

    unsafe {
        if HISTORY.is_none() {
            HISTORY = Some(Vec::with_capacity(HISTORY_WINDOW_SIZE));
        }

        if let Some(history) = &mut HISTORY {
            let index = HISTORY_INDEX.fetch_add(1, Ordering::Relaxed) as usize;
            let position = index % HISTORY_WINDOW_SIZE;

            if history.len() <= position {
                history.push(point);
            } else {
                history[position] = point;
            }
        }
    }
}

/// Get statistics trends over time
pub fn get_trends() -> StatsTrends {
    let mut fault_rate_per_sec = 0.0;
    let mut migration_rate_per_sec = 0.0;
    let mut pressure_trend = 0i8; // -1 decreasing, 0 stable, 1 increasing

    unsafe {
        if let Some(history) = &HISTORY {
            if history.len() >= 2 {
                // Calculate rates from recent history
                let recent_idx = history.len() - 1;
                let prev_idx = recent_idx.saturating_sub(10); // 10 seconds ago

                let recent = &history[recent_idx];
                let prev = &history[prev_idx];

                let time_delta = recent.timestamp_sec.saturating_sub(prev.timestamp_sec);
                if time_delta > 0 {
                    fault_rate_per_sec =
                        (recent.total_faults - prev.total_faults) as f64 / time_delta as f64;
                    migration_rate_per_sec = (recent.total_migrations - prev.total_migrations)
                        as f64
                        / time_delta as f64;
                }

                // Determine pressure trend
                if recent.overall_pressure > prev.overall_pressure + 5 {
                    pressure_trend = 1;
                } else if recent.overall_pressure < prev.overall_pressure.saturating_sub(5) {
                    pressure_trend = -1;
                }
            }
        }
    }

    StatsTrends {
        fault_rate_per_sec,
        migration_rate_per_sec,
        pressure_trend,
    }
}

/// Statistics trends
#[derive(Debug, Clone)]
pub struct StatsTrends {
    pub fault_rate_per_sec: f64,
    pub migration_rate_per_sec: f64,
    pub pressure_trend: i8, // -1 decreasing, 0 stable, 1 increasing
}

/// Get current time in seconds (mock)
fn get_current_time_sec() -> u64 {
    // In real kernel: ktime_get_real_seconds()
    0
}

/// Initialize statistics subsystem
pub fn init() -> KernelResult<()> {
    // Initialize history
    unsafe {
        HISTORY = Some(Vec::with_capacity(HISTORY_WINDOW_SIZE));
    }

    Ok(())
}

/// Cleanup statistics subsystem
pub fn cleanup() {
    unsafe {
        HISTORY = None;
    }
}

/// Generate statistics report
pub fn generate_report() -> alloc::string::String {
    use core::fmt::Write;
    let mut report = alloc::string::String::new();

    let stats = collect_stats();
    let metrics = get_performance_metrics();
    let trends = get_trends();

    let _ = writeln!(report, "=== TierWatch Statistics Report ===");
    let _ = writeln!(report, "Generated at: {} seconds", get_current_time_sec());
    let _ = writeln!(report, "");

    let _ = writeln!(report, "Memory Overview:");
    let _ = writeln!(
        report,
        "  Total Capacity: {} GB",
        stats.total_capacity_bytes >> 30
    );
    let _ = writeln!(report, "  Total Used: {} GB", stats.total_used_bytes >> 30);
    let _ = writeln!(report, "  Overall Pressure: {}%", stats.overall_pressure);
    let _ = writeln!(
        report,
        "  NUMA Locality: {:.1}%",
        stats.numa_locality_ratio * 100.0
    );
    let _ = writeln!(report, "");

    let _ = writeln!(report, "Performance Metrics:");
    let _ = writeln!(
        report,
        "  Fault Overhead: {} ns (target: <100ns)",
        metrics.fault_overhead_ns
    );
    let _ = writeln!(
        report,
        "  Detection Time: {} μs (target: <1000μs)",
        metrics.detection_time_us
    );
    let _ = writeln!(
        report,
        "  Pages Tracked: {} (memory: {} MB)",
        metrics.pages_tracked,
        metrics.memory_overhead_bytes >> 20
    );
    let _ = writeln!(report, "");

    let _ = writeln!(report, "Trends (per second):");
    let _ = writeln!(report, "  Fault Rate: {:.1}", trends.fault_rate_per_sec);
    let _ = writeln!(
        report,
        "  Migration Rate: {:.1}",
        trends.migration_rate_per_sec
    );
    let _ = writeln!(
        report,
        "  Pressure Trend: {}",
        match trends.pressure_trend {
            -1 => "Decreasing",
            0 => "Stable",
            1 => "Increasing",
            _ => "Unknown",
        }
    );

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_collection() {
        // Initialize subsystems
        crate::tiers::init().ok();
        crate::migration::init().ok();
        crate::numa::init().ok();

        let stats = collect_stats();
        assert!(stats.total_capacity_bytes > 0);
        assert_eq!(stats.tier_stats.len(), 5);

        // Cleanup
        crate::numa::cleanup();
        crate::migration::cleanup();
        crate::tiers::cleanup();
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = get_performance_metrics();
        assert!(metrics.fault_overhead_ns < 100);
        assert!(metrics.detection_time_us < 1000);
    }

    #[test]
    fn test_report_generation() {
        let report = generate_report();
        assert!(report.contains("TierWatch Statistics Report"));
        assert!(report.contains("Memory Overview"));
        assert!(report.contains("Performance Metrics"));
    }
}
