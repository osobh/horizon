//! FFI bridge for C kernel integration
//!
//! This module provides C-compatible interfaces for kernel integration.

use core::ffi::{c_char, c_int, c_long, c_uint, c_ulong};
use core::ptr;

use crate::MemoryTier;

/// C-compatible tier enum
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CTier {
    Gpu = 0,
    Cpu = 1,
    Nvme = 2,
    Ssd = 3,
    Hdd = 4,
}

impl From<CTier> for MemoryTier {
    fn from(tier: CTier) -> Self {
        match tier {
            CTier::Gpu => MemoryTier::Gpu,
            CTier::Cpu => MemoryTier::Cpu,
            CTier::Nvme => MemoryTier::Nvme,
            CTier::Ssd => MemoryTier::Ssd,
            CTier::Hdd => MemoryTier::Hdd,
        }
    }
}

impl From<MemoryTier> for CTier {
    fn from(tier: MemoryTier) -> Self {
        match tier {
            MemoryTier::Gpu => CTier::Gpu,
            MemoryTier::Cpu => CTier::Cpu,
            MemoryTier::Nvme => CTier::Nvme,
            MemoryTier::Ssd => CTier::Ssd,
            MemoryTier::Hdd => CTier::Hdd,
        }
    }
}

/// C-compatible fault type
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum CFaultType {
    Major = 0,
    Minor = 1,
    CoW = 2,
    Permission = 3,
}

impl From<CFaultType> for crate::fault::FaultType {
    fn from(fault: CFaultType) -> Self {
        match fault {
            CFaultType::Major => crate::fault::FaultType::Major,
            CFaultType::Minor => crate::fault::FaultType::Minor,
            CFaultType::CoW => crate::fault::FaultType::CoW,
            CFaultType::Permission => crate::fault::FaultType::Permission,
        }
    }
}

/// C-compatible tier statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CTierStats {
    pub tier: CTier,
    pub total_pages: c_ulong,
    pub used_bytes: c_ulong,
    pub pressure_percent: c_uint,
    pub major_faults: c_ulong,
    pub minor_faults: c_ulong,
    pub migrations_in: c_ulong,
    pub migrations_out: c_ulong,
    pub avg_latency_ns: c_ulong,
}

/// Initialize tier watch module
#[no_mangle]
pub extern "C" fn tier_watch_init() -> c_int {
    match crate::init() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Cleanup tier watch module
#[no_mangle]
pub extern "C" fn tier_watch_cleanup() {
    crate::cleanup();
}

/// Handle page fault from kernel
#[no_mangle]
pub extern "C" fn tier_watch_handle_fault(
    vaddr: c_ulong,
    pfn: c_ulong,
    fault_type: CFaultType,
    agent_id: c_ulong,
) -> c_int {
    let agent = if agent_id == 0 {
        None
    } else {
        Some(agent_id as u64)
    };

    match crate::fault::handle_page_fault(vaddr as u64, pfn as u64, fault_type.into(), agent) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Get tier statistics
#[no_mangle]
pub extern "C" fn tier_watch_get_stats(tier: CTier, stats: *mut CTierStats) -> c_int {
    if stats.is_null() {
        return -1;
    }

    let tier_enum: MemoryTier = tier.into();

    if let Some(tier_stats) = crate::tiers::get_tier_stats(tier_enum) {
        unsafe {
            (*stats).tier = tier;
            (*stats).total_pages = tier_stats
                .total_pages
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).used_bytes = tier_stats
                .used_bytes
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).pressure_percent = tier_stats.pressure_percent() as c_uint;
            (*stats).major_faults = tier_stats
                .major_faults
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).minor_faults = tier_stats
                .minor_faults
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).migrations_in = tier_stats
                .migrations_in
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).migrations_out = tier_stats
                .migrations_out
                .load(core::sync::atomic::Ordering::Relaxed)
                as c_ulong;
            (*stats).avg_latency_ns = tier_stats.avg_latency_ns() as c_ulong;
        }
        0
    } else {
        -1
    }
}

/// Check memory pressure
#[no_mangle]
pub extern "C" fn tier_watch_check_pressure(
    tiers_out: *mut CTier,
    pressures_out: *mut c_uint,
    max_tiers: c_uint,
) -> c_int {
    if tiers_out.is_null() || pressures_out.is_null() || max_tiers == 0 {
        return -1;
    }

    let pressure_tiers = crate::tiers::check_memory_pressure();
    let count = pressure_tiers.len().min(max_tiers as usize);

    unsafe {
        for (i, (tier, pressure)) in pressure_tiers.iter().take(count).enumerate() {
            *tiers_out.add(i) = (*tier).into();
            *pressures_out.add(i) = *pressure as c_uint;
        }
    }

    count as c_int
}

/// Enable/disable migration detector
#[no_mangle]
pub extern "C" fn tier_watch_set_detector_enabled(enabled: c_int) {
    crate::migration::set_detector_enabled(enabled != 0);
}

/// Get NUMA node for CPU
#[no_mangle]
pub extern "C" fn tier_watch_get_numa_node(cpu_id: c_uint) -> c_int {
    match crate::numa::get_node_for_cpu(cpu_id as u32) {
        Some(node) => node as c_int,
        None => -1,
    }
}

/// Record NUMA access
#[no_mangle]
pub extern "C" fn tier_watch_record_numa_access(cpu_id: c_uint, pfn: c_ulong) {
    crate::numa::record_numa_access(cpu_id as u32, pfn as u64);
}

/// Get performance metrics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CPerformanceMetrics {
    pub fault_overhead_ns: c_ulong,
    pub detection_time_us: c_ulong,
    pub pages_tracked: c_ulong,
    pub memory_overhead_bytes: c_ulong,
}

#[no_mangle]
pub extern "C" fn tier_watch_get_performance(metrics: *mut CPerformanceMetrics) -> c_int {
    if metrics.is_null() {
        return -1;
    }

    let perf = crate::stats::get_performance_metrics();

    unsafe {
        (*metrics).fault_overhead_ns = perf.fault_overhead_ns as c_ulong;
        (*metrics).detection_time_us = perf.detection_time_us as c_ulong;
        (*metrics).pages_tracked = perf.pages_tracked as c_ulong;
        (*metrics).memory_overhead_bytes = perf.memory_overhead_bytes as c_ulong;
    }

    0
}

/// Generate statistics report
#[no_mangle]
pub extern "C" fn tier_watch_generate_report(buffer: *mut c_char, buffer_size: c_ulong) -> c_long {
    if buffer.is_null() || buffer_size == 0 {
        return -1;
    }

    let report = crate::stats::generate_report();
    let report_bytes = report.as_bytes();
    let copy_len = report_bytes.len().min(buffer_size as usize - 1);

    unsafe {
        ptr::copy_nonoverlapping(report_bytes.as_ptr(), buffer as *mut u8, copy_len);
        *buffer.add(copy_len) = 0; // Null terminate
    }

    copy_len as c_long
}

/// Module version information
#[no_mangle]
pub static TIER_WATCH_VERSION_MAJOR: c_uint = 1;

#[no_mangle]
pub static TIER_WATCH_VERSION_MINOR: c_uint = 0;

#[no_mangle]
pub static TIER_WATCH_VERSION_PATCH: c_uint = 0;

/// Get version string
#[no_mangle]
pub extern "C" fn tier_watch_get_version() -> *const c_char {
    b"tier_watch 1.0.0\0".as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_conversion() {
        let c_tier = CTier::Gpu;
        let rust_tier: MemoryTier = c_tier.into();
        assert_eq!(rust_tier, MemoryTier::Gpu);

        let back: CTier = rust_tier.into();
        assert_eq!(back as u8, c_tier as u8);
    }

    #[test]
    fn test_version_string() {
        let version = tier_watch_get_version();
        assert!(!version.is_null());
    }
}
