//! Statistics collection and reporting for GPU DMA lock
//!
//! This module collects detailed performance metrics and usage statistics.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{security::SecurityEvent, spin};

/// Performance counter
#[derive(Debug)]
pub struct PerfCounter {
    count: AtomicU64,
    total_time_ns: AtomicU64,
    min_time_ns: AtomicU64,
    max_time_ns: AtomicU64,
}

impl PerfCounter {
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            min_time_ns: AtomicU64::new(u64::MAX),
            max_time_ns: AtomicU64::new(0),
        }
    }

    pub fn record(&self, duration_ns: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(duration_ns, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_time_ns.load(Ordering::Relaxed);
        while duration_ns < current_min {
            match self.min_time_ns.compare_exchange_weak(
                current_min,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max
        let mut current_max = self.max_time_ns.load(Ordering::Relaxed);
        while duration_ns > current_max {
            match self.max_time_ns.compare_exchange_weak(
                current_max,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    pub fn get_stats(&self) -> PerfStats {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_time_ns.load(Ordering::Relaxed);
        let min = self.min_time_ns.load(Ordering::Relaxed);
        let max = self.max_time_ns.load(Ordering::Relaxed);

        PerfStats {
            count,
            total_time_ns: total,
            avg_time_ns: if count > 0 { total / count } else { 0 },
            min_time_ns: if min == u64::MAX { 0 } else { min },
            max_time_ns: max,
        }
    }

    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
        self.min_time_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_time_ns.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct PerfStats {
    pub count: u64,
    pub total_time_ns: u64,
    pub avg_time_ns: u64,
    pub min_time_ns: u64,
    pub max_time_ns: u64,
}

/// Detailed statistics collector
pub struct StatsCollector {
    /// Allocation performance
    pub alloc_perf: PerfCounter,
    /// Deallocation performance
    pub dealloc_perf: PerfCounter,
    /// DMA check performance
    pub dma_check_perf: PerfCounter,
    /// Context switch performance
    pub context_switch_perf: PerfCounter,
    /// Allocation sizes histogram
    alloc_sizes: spin::RwLock<BTreeMap<usize, u64>>,
    /// Agent statistics
    agent_stats: spin::RwLock<BTreeMap<u64, AgentStats>>,
    /// Device utilization
    device_util: spin::RwLock<BTreeMap<u32, DeviceUtilization>>,
}

#[derive(Debug, Clone)]
struct AgentStats {
    allocations: u64,
    deallocations: u64,
    total_allocated: u64,
    current_allocated: u64,
    dma_checks: u64,
    dma_denials: u64,
    context_switches: u64,
}

#[derive(Debug, Clone)]
struct DeviceUtilization {
    allocated_memory: u64,
    allocation_count: u64,
    last_update: u64,
}

impl StatsCollector {
    pub fn new() -> Self {
        Self {
            alloc_perf: PerfCounter::new(),
            dealloc_perf: PerfCounter::new(),
            dma_check_perf: PerfCounter::new(),
            context_switch_perf: PerfCounter::new(),
            alloc_sizes: spin::RwLock::new(BTreeMap::new()),
            agent_stats: spin::RwLock::new(BTreeMap::new()),
            device_util: spin::RwLock::new(BTreeMap::new()),
        }
    }

    /// Record allocation
    pub fn record_allocation(&self, agent_id: u64, size: usize, device_id: u32, duration_ns: u64) {
        self.alloc_perf.record(duration_ns);

        // Update size histogram
        let size_bucket = get_size_bucket(size);
        *self.alloc_sizes.write().entry(size_bucket).or_insert(0) += 1;

        // Update agent stats
        let mut agent_stats = self.agent_stats.write();
        let stats = agent_stats.entry(agent_id).or_insert(AgentStats {
            allocations: 0,
            deallocations: 0,
            total_allocated: 0,
            current_allocated: 0,
            dma_checks: 0,
            dma_denials: 0,
            context_switches: 0,
        });
        stats.allocations += 1;
        stats.total_allocated += size as u64;
        stats.current_allocated += size as u64;

        // Update device utilization
        let mut device_util = self.device_util.write();
        let util = device_util.entry(device_id).or_insert(DeviceUtilization {
            allocated_memory: 0,
            allocation_count: 0,
            last_update: get_current_time(),
        });
        util.allocated_memory += size as u64;
        util.allocation_count += 1;
        util.last_update = get_current_time();
    }

    /// Record deallocation
    pub fn record_deallocation(
        &self,
        agent_id: u64,
        size: usize,
        device_id: u32,
        duration_ns: u64,
    ) {
        self.dealloc_perf.record(duration_ns);

        // Update agent stats
        if let Some(stats) = self.agent_stats.write().get_mut(&agent_id) {
            stats.deallocations += 1;
            stats.current_allocated = stats.current_allocated.saturating_sub(size as u64);
        }

        // Update device utilization
        if let Some(util) = self.device_util.write().get_mut(&device_id) {
            util.allocated_memory = util.allocated_memory.saturating_sub(size as u64);
            util.last_update = get_current_time();
        }
    }

    /// Record DMA check
    pub fn record_dma_check(&self, agent_id: u64, allowed: bool, duration_ns: u64) {
        self.dma_check_perf.record(duration_ns);

        if let Some(stats) = self.agent_stats.write().get_mut(&agent_id) {
            stats.dma_checks += 1;
            if !allowed {
                stats.dma_denials += 1;
            }
        }
    }

    /// Record context switch
    pub fn record_context_switch(&self, from_agent: u64, to_agent: u64, duration_ns: u64) {
        self.context_switch_perf.record(duration_ns);

        let mut agent_stats = self.agent_stats.write();
        if let Some(stats) = agent_stats.get_mut(&from_agent) {
            stats.context_switches += 1;
        }
        if let Some(stats) = agent_stats.get_mut(&to_agent) {
            stats.context_switches += 1;
        }
    }

    /// Get agent statistics
    pub fn get_agent_stats(&self, agent_id: u64) -> Option<AgentStats> {
        self.agent_stats.read().get(&agent_id).cloned()
    }

    /// Get allocation size distribution
    pub fn get_size_distribution(&self) -> Vec<(usize, u64)> {
        self.alloc_sizes
            .read()
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    /// Get device utilization
    pub fn get_device_utilization(&self, device_id: u32) -> Option<DeviceUtilization> {
        self.device_util.read().get(&device_id).cloned()
    }

    /// Generate detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Performance Metrics ===\n");

        let alloc_stats = self.alloc_perf.get_stats();
        report.push_str(&format!(
            "Allocations: {} ops, avg: {} ns, min: {} ns, max: {} ns\n",
            alloc_stats.count,
            alloc_stats.avg_time_ns,
            alloc_stats.min_time_ns,
            alloc_stats.max_time_ns
        ));

        let dealloc_stats = self.dealloc_perf.get_stats();
        report.push_str(&format!(
            "Deallocations: {} ops, avg: {} ns, min: {} ns, max: {} ns\n",
            dealloc_stats.count,
            dealloc_stats.avg_time_ns,
            dealloc_stats.min_time_ns,
            dealloc_stats.max_time_ns
        ));

        let dma_stats = self.dma_check_perf.get_stats();
        report.push_str(&format!(
            "DMA Checks: {} ops, avg: {} ns, min: {} ns, max: {} ns\n",
            dma_stats.count, dma_stats.avg_time_ns, dma_stats.min_time_ns, dma_stats.max_time_ns
        ));

        let ctx_stats = self.context_switch_perf.get_stats();
        report.push_str(&format!(
            "Context Switches: {} ops, avg: {} ns, min: {} ns, max: {} ns\n",
            ctx_stats.count, ctx_stats.avg_time_ns, ctx_stats.min_time_ns, ctx_stats.max_time_ns
        ));

        report.push_str("\n=== Allocation Size Distribution ===\n");
        for (size, count) in self.get_size_distribution() {
            report.push_str(&format!("{} bytes: {} allocations\n", size, count));
        }

        report.push_str("\n=== Top Agents by Memory Usage ===\n");
        let agent_stats = self.agent_stats.read();
        let mut sorted_agents: Vec<_> = agent_stats
            .iter()
            .map(|(&id, stats)| (id, stats.current_allocated))
            .collect();
        sorted_agents.sort_by_key(|&(_, allocated)| core::cmp::Reverse(allocated));

        for (agent_id, allocated) in sorted_agents.iter().take(10) {
            report.push_str(&format!("Agent {}: {} MB\n", agent_id, allocated >> 20));
        }

        report
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.alloc_perf.reset();
        self.dealloc_perf.reset();
        self.dma_check_perf.reset();
        self.context_switch_perf.reset();
        self.alloc_sizes.write().clear();
        self.agent_stats.write().clear();
        self.device_util.write().clear();
    }
}

/// Get size bucket for histogram
fn get_size_bucket(size: usize) -> usize {
    // Round to nearest power of 2
    if size == 0 {
        return 0;
    }
    if size.is_power_of_two() {
        return size;
    }
    // Get the next power of 2
    1 << (64 - size.leading_zeros())
}

/// Debug logging
static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);
static mut DEBUG_BUFFER: Option<spin::Mutex<Vec<String>>> = None;

/// Enable/disable debug logging
pub fn enable_debug(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::Relaxed);

    if enabled {
        unsafe {
            if DEBUG_BUFFER.is_none() {
                DEBUG_BUFFER = Some(spin::Mutex::new(Vec::new()));
            }
        }
    }
}

/// Log debug message
pub fn log_debug(msg: &str) {
    if !DEBUG_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    unsafe {
        if let Some(buffer) = &DEBUG_BUFFER {
            let mut buf = buffer.lock();
            if buf.len() < 10000 {
                // Limit buffer size
                buf.push(format!("[{}] {}", get_current_time(), msg));
            }
        }
    }

    // In real kernel: printk
}

/// Log security event
pub fn log_security_event(event: &SecurityEvent) {
    log_debug(&format!(
        "SECURITY: agent={} type={:?} details={:?}",
        event.agent_id, event.violation_type, event.details
    ));
}

/// Get debug log buffer
pub fn get_debug_logs() -> Vec<String> {
    unsafe {
        if let Some(buffer) = &DEBUG_BUFFER {
            buffer.lock().clone()
        } else {
            Vec::new()
        }
    }
}

/// Global stats collector instance
static mut STATS_COLLECTOR: Option<StatsCollector> = None;

/// Get global stats collector
pub fn get_collector() -> &'static StatsCollector {
    unsafe {
        STATS_COLLECTOR
            .as_ref()
            .expect("Stats collector not initialized")
    }
}

/// Get detailed statistics string
pub fn get_detailed_stats() -> Option<String> {
    unsafe { STATS_COLLECTOR.as_ref().map(|c| c.generate_report()) }
}

/// Reset all statistics
pub fn reset_stats() {
    unsafe {
        if let Some(collector) = STATS_COLLECTOR.as_ref() {
            collector.reset();
        }
    }

    // Also reset global counters
    crate::GPU_DMA_STATS
        .total_allocations
        .store(0, Ordering::Relaxed);
    crate::GPU_DMA_STATS
        .total_deallocations
        .store(0, Ordering::Relaxed);
    crate::GPU_DMA_STATS
        .total_bytes_allocated
        .store(0, Ordering::Relaxed);
    crate::GPU_DMA_STATS.dma_checks.store(0, Ordering::Relaxed);
    crate::GPU_DMA_STATS.dma_denials.store(0, Ordering::Relaxed);
    crate::GPU_DMA_STATS
        .context_switches
        .store(0, Ordering::Relaxed);
}

/// Initialize stats subsystem
pub fn init() -> crate::KernelResult<()> {
    unsafe {
        STATS_COLLECTOR = Some(StatsCollector::new());
        DEBUG_BUFFER = Some(spin::Mutex::new(Vec::new()));
    }
    Ok(())
}

/// Cleanup stats subsystem
pub fn cleanup() {
    unsafe {
        STATS_COLLECTOR = None;
        DEBUG_BUFFER = None;
    }
}

/// Get current time (mock)
fn get_current_time() -> u64 {
    // In real kernel: ktime_get() / 1000
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_counter() {
        let counter = PerfCounter::new();

        counter.record(100);
        counter.record(200);
        counter.record(50);
        counter.record(300);

        let stats = counter.get_stats();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.total_time_ns, 650);
        assert_eq!(stats.avg_time_ns, 162);
        assert_eq!(stats.min_time_ns, 50);
        assert_eq!(stats.max_time_ns, 300);
    }

    #[test]
    fn test_size_buckets() {
        assert_eq!(get_size_bucket(0), 0);
        assert_eq!(get_size_bucket(1), 1);
        assert_eq!(get_size_bucket(2), 2);
        assert_eq!(get_size_bucket(3), 4);
        assert_eq!(get_size_bucket(1000), 1024);
        assert_eq!(get_size_bucket(1 << 20), 1 << 20);
    }

    #[test]
    fn test_stats_collector() {
        let collector = StatsCollector::new();

        collector.record_allocation(100, 1 << 20, 0, 1000);
        collector.record_allocation(100, 2 << 20, 0, 2000);
        collector.record_allocation(101, 1 << 20, 0, 1500);

        let agent_stats = collector.get_agent_stats(100).unwrap();
        assert_eq!(agent_stats.allocations, 2);
        assert_eq!(agent_stats.total_allocated, 3 << 20);

        let report = collector.generate_report();
        assert!(report.contains("Performance Metrics"));
        assert!(report.contains("Allocation Size Distribution"));
    }

    #[test]
    fn test_debug_logging() {
        enable_debug(true);

        log_debug("Test message 1");
        log_debug("Test message 2");

        let logs = get_debug_logs();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].contains("Test message 1"));
        assert!(logs[1].contains("Test message 2"));

        enable_debug(false);
    }
}
