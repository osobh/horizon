//! Page migration detection and coordination
//!
//! This module detects hot/cold pages and coordinates their migration
//! between memory tiers.

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;

use crate::{KernelError, KernelResult, MemoryTier, TIER_WATCH_STATS};

/// Migration request for a page
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Physical page frame number
    pub pfn: u64,
    /// Source tier
    pub from_tier: MemoryTier,
    /// Target tier
    pub to_tier: MemoryTier,
    /// Priority (0=low, 100=high)
    pub priority: u8,
    /// Agent ID that owns this page
    pub agent_id: Option<u64>,
    /// Reason for migration
    pub reason: MigrationReason,
}

/// Reason for page migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationReason {
    /// Page is hot and should be promoted
    HotPromotion,
    /// Page is cold and should be demoted
    ColdDemotion,
    /// Memory pressure in current tier
    MemoryPressure,
    /// NUMA optimization
    NumaBalance,
    /// Agent requested migration
    AgentRequest,
}

/// Migration statistics
pub struct MigrationStats {
    /// Successful migrations by reason
    pub success_by_reason: [AtomicU64; 5],
    /// Failed migrations by reason
    pub failed_by_reason: [AtomicU64; 5],
    /// Total migration time in microseconds
    pub total_time_us: AtomicU64,
    /// Number of migrations in progress
    pub in_progress: AtomicU64,
}

impl MigrationStats {
    const fn new() -> Self {
        Self {
            success_by_reason: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            failed_by_reason: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            total_time_us: AtomicU64::new(0),
            in_progress: AtomicU64::new(0),
        }
    }

    fn record_success(&self, reason: MigrationReason, duration_us: u64) {
        self.success_by_reason[reason as usize].fetch_add(1, Ordering::Relaxed);
        self.total_time_us.fetch_add(duration_us, Ordering::Relaxed);
        self.in_progress.fetch_sub(1, Ordering::Relaxed);
    }

    fn record_failure(&self, reason: MigrationReason) {
        self.failed_by_reason[reason as usize].fetch_add(1, Ordering::Relaxed);
        self.in_progress.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Global migration statistics
static MIGRATION_STATS: MigrationStats = MigrationStats::new();

/// Migration detector state
static DETECTOR_ENABLED: AtomicBool = AtomicBool::new(true);

/// Hot page threshold (accesses per period)
const HOT_PAGE_THRESHOLD: u16 = 100;

/// Cold page threshold (accesses per period)
const COLD_PAGE_THRESHOLD: u16 = 10;

/// Maximum concurrent migrations
const MAX_CONCURRENT_MIGRATIONS: u64 = 16;

/// Check if migration detector is enabled
pub fn is_detector_enabled() -> bool {
    DETECTOR_ENABLED.load(Ordering::Relaxed)
}

/// Enable/disable migration detector
pub fn set_detector_enabled(enabled: bool) {
    DETECTOR_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Detect pages that need migration
pub fn detect_migration_candidates() -> Vec<MigrationRequest> {
    if !is_detector_enabled() {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // Check for hot pages that should be promoted
    let hot_pages = crate::fault::get_hot_pages(100);
    for (pfn, current_tier) in hot_pages {
        if let Some(target_tier) = current_tier.prev_tier() {
            // Check if target tier has space
            if let Some(stats) = crate::tiers::get_tier_stats(target_tier) {
                if stats.pressure_percent() < 90 {
                    candidates.push(MigrationRequest {
                        pfn,
                        from_tier: current_tier,
                        to_tier: target_tier,
                        priority: 80,
                        agent_id: None,
                        reason: MigrationReason::HotPromotion,
                    });
                }
            }
        }
    }

    // Check for cold pages that should be demoted
    let cold_pages = crate::fault::get_cold_pages(100);
    for (pfn, current_tier) in cold_pages {
        if let Some(target_tier) = current_tier.next_tier() {
            candidates.push(MigrationRequest {
                pfn,
                from_tier: current_tier,
                to_tier: target_tier,
                priority: 20,
                agent_id: None,
                reason: MigrationReason::ColdDemotion,
            });
        }
    }

    // Check for memory pressure
    let pressure_tiers = crate::tiers::check_memory_pressure();
    for (tier, pressure) in pressure_tiers {
        if pressure > 90 {
            // Find cold pages in this tier to migrate out
            // This is simplified - real implementation would be more sophisticated
            candidates.extend(generate_pressure_migrations(tier, 50));
        }
    }

    // Sort by priority (highest first)
    candidates.sort_by(|a, b| b.priority.cmp(&a.priority));

    candidates
}

/// Generate migration requests due to memory pressure
fn generate_pressure_migrations(_tier: MemoryTier, _count: usize) -> Vec<MigrationRequest> {
    let migrations = Vec::new();

    // In real implementation, would scan tier for coldest pages
    // For now, return empty vec

    migrations
}

/// Perform a page migration
pub fn migrate_page(request: &MigrationRequest) -> KernelResult<Duration> {
    // Check if we can take another migration
    let in_progress = MIGRATION_STATS.in_progress.load(Ordering::Relaxed);
    if in_progress >= MAX_CONCURRENT_MIGRATIONS {
        return Err(KernelError::MigrationFailed);
    }

    MIGRATION_STATS.in_progress.fetch_add(1, Ordering::Relaxed);

    let start_time = get_current_time_us();

    // Validate migration request
    if request.from_tier == request.to_tier {
        MIGRATION_STATS.record_failure(request.reason);
        return Err(KernelError::InvalidArgument);
    }

    // Update tier statistics
    if let Some(from_stats) = crate::tiers::get_tier_stats(request.from_tier) {
        from_stats.record_migration_out();
    }

    if let Some(to_stats) = crate::tiers::get_tier_stats(request.to_tier) {
        to_stats.record_migration_in();
    }

    // In real kernel:
    // 1. Lock the page
    // 2. Update page tables
    // 3. Copy page content
    // 4. Update tier tracking
    // 5. Unlock page

    // Simulate migration time based on tier latencies
    let migration_time_ns = request.from_tier.latency_ns() + request.to_tier.latency_ns();

    let end_time = get_current_time_us();
    let duration_us = end_time - start_time;

    // Record success
    MIGRATION_STATS.record_success(request.reason, duration_us);
    TIER_WATCH_STATS.record_migration(migration_time_ns);

    Ok(Duration::from_nanos(migration_time_ns))
}

/// Batch migrate multiple pages
pub fn batch_migrate(requests: &[MigrationRequest]) -> Vec<KernelResult<Duration>> {
    requests.iter().map(|req| migrate_page(req)).collect()
}

/// Get migration statistics
pub fn get_migration_stats() -> MigrationStatsSummary {
    let total_success: u64 = MIGRATION_STATS
        .success_by_reason
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .sum();

    let total_failed: u64 = MIGRATION_STATS
        .failed_by_reason
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .sum();

    let avg_time_us = if total_success > 0 {
        MIGRATION_STATS.total_time_us.load(Ordering::Relaxed) / total_success
    } else {
        0
    };

    MigrationStatsSummary {
        total_success,
        total_failed,
        avg_time_us,
        in_progress: MIGRATION_STATS.in_progress.load(Ordering::Relaxed),
    }
}

/// Migration statistics summary
#[derive(Debug, Clone)]
pub struct MigrationStatsSummary {
    pub total_success: u64,
    pub total_failed: u64,
    pub avg_time_us: u64,
    pub in_progress: u64,
}

/// Get current time in microseconds (mock)
fn get_current_time_us() -> u64 {
    // In real kernel: ktime_get() / 1000
    0
}

/// Migration worker thread entry point
pub fn migration_worker() {
    loop {
        // Check if we should run
        if !is_detector_enabled() {
            // Sleep for a bit
            kernel_sleep_ms(100);
            continue;
        }

        // Detect migration candidates
        let candidates = detect_migration_candidates();

        // Migrate top candidates
        for candidate in candidates.iter().take(10) {
            if let Err(_e) = migrate_page(candidate) {
                // Log error in real kernel
            }
        }

        // Sleep before next detection cycle
        kernel_sleep_ms(10);
    }
}

/// Mock kernel sleep function
fn kernel_sleep_ms(_ms: u64) {
    // In real kernel: msleep(ms)
}

/// Initialize migration subsystem
pub fn init() -> KernelResult<()> {
    // In real kernel:
    // 1. Create migration worker thread
    // 2. Register with memory management subsystem
    // 3. Set up migration queues

    Ok(())
}

/// Cleanup migration subsystem
pub fn cleanup() {
    // Disable detector
    set_detector_enabled(false);

    // Wait for in-progress migrations
    while MIGRATION_STATS.in_progress.load(Ordering::Relaxed) > 0 {
        kernel_sleep_ms(10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_request_priority() {
        let mut requests = vec![
            MigrationRequest {
                pfn: 1,
                from_tier: MemoryTier::Cpu,
                to_tier: MemoryTier::Gpu,
                priority: 50,
                agent_id: None,
                reason: MigrationReason::HotPromotion,
            },
            MigrationRequest {
                pfn: 2,
                from_tier: MemoryTier::Nvme,
                to_tier: MemoryTier::Cpu,
                priority: 80,
                agent_id: None,
                reason: MigrationReason::HotPromotion,
            },
            MigrationRequest {
                pfn: 3,
                from_tier: MemoryTier::Cpu,
                to_tier: MemoryTier::Nvme,
                priority: 20,
                agent_id: None,
                reason: MigrationReason::ColdDemotion,
            },
        ];

        requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        assert_eq!(requests[0].priority, 80);
        assert_eq!(requests[1].priority, 50);
        assert_eq!(requests[2].priority, 20);
    }

    #[test]
    fn test_detector_enable_disable() {
        set_detector_enabled(false);
        assert!(!is_detector_enabled());

        let candidates = detect_migration_candidates();
        assert!(candidates.is_empty());

        set_detector_enabled(true);
        assert!(is_detector_enabled());
    }
}
