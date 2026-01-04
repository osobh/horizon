//! /proc interface for tier statistics
//!
//! Provides the /proc/swarm/tiers/<tier>/stats interface for monitoring
//! memory tier statistics.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::{KernelResult, MemoryTier, TIER_WATCH_STATS};

/// Proc file operations
#[repr(C)]
pub struct ProcOps {
    pub open: Option<extern "C" fn(*mut ProcFile) -> i32>,
    pub read: Option<extern "C" fn(*mut ProcFile, *mut u8, usize) -> isize>,
    pub write: Option<extern "C" fn(*mut ProcFile, *const u8, usize) -> isize>,
    pub release: Option<extern "C" fn(*mut ProcFile) -> i32>,
}

/// Proc file handle
#[repr(C)]
pub struct ProcFile {
    pub private_data: *mut u8,
    pub pos: u64,
}

/// Proc directory entries
static mut PROC_ENTRIES: Option<Vec<ProcEntry>> = None;

/// Individual proc entry
struct ProcEntry {
    path: String,
    tier: Option<MemoryTier>,
    ops: ProcOps,
}

/// Format tier statistics as string
fn format_tier_stats(tier: MemoryTier) -> String {
    let mut output = String::new();

    if let Some(stats) = crate::tiers::get_tier_stats(tier) {
        let _ = writeln!(output, "Tier: {}", tier.name());
        let _ = writeln!(output, "Capacity: {} MB", tier.capacity_bytes() >> 20);
        let _ = writeln!(
            output,
            "Used: {} MB",
            stats.used_bytes.load(core::sync::atomic::Ordering::Relaxed) >> 20
        );
        let _ = writeln!(output, "Pressure: {}%", stats.pressure_percent());
        let _ = writeln!(output, "Theoretical Latency: {} ns", tier.latency_ns());
        let _ = writeln!(output, "Average Latency: {} ns", stats.avg_latency_ns());
        let _ = writeln!(
            output,
            "Major Faults: {}",
            stats
                .major_faults
                .load(core::sync::atomic::Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "Minor Faults: {}",
            stats
                .minor_faults
                .load(core::sync::atomic::Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "Migrations In: {}",
            stats
                .migrations_in
                .load(core::sync::atomic::Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "Migrations Out: {}",
            stats
                .migrations_out
                .load(core::sync::atomic::Ordering::Relaxed)
        );
        let _ = writeln!(
            output,
            "Access Count: {}",
            stats
                .access_count
                .load(core::sync::atomic::Ordering::Relaxed)
        );

        // Add per-CPU statistics
        crate::tiers::consolidate_stats();
    } else {
        let _ = writeln!(output, "No statistics available for tier {}", tier.name());
    }

    output
}

/// Format global statistics
fn format_global_stats() -> String {
    let mut output = String::new();

    let _ = writeln!(output, "TierWatch Global Statistics");
    let _ = writeln!(output, "===========================");
    let _ = writeln!(
        output,
        "Pages Tracked: {}",
        TIER_WATCH_STATS
            .pages_tracked
            .load(core::sync::atomic::Ordering::Relaxed)
    );
    let _ = writeln!(
        output,
        "Total Faults: {}",
        TIER_WATCH_STATS
            .total_faults
            .load(core::sync::atomic::Ordering::Relaxed)
    );
    let _ = writeln!(
        output,
        "Total Migrations: {}",
        TIER_WATCH_STATS
            .total_migrations
            .load(core::sync::atomic::Ordering::Relaxed)
    );
    let _ = writeln!(
        output,
        "Avg Migration Latency: {} ns",
        TIER_WATCH_STATS.avg_migration_latency_ns()
    );
    let _ = writeln!(output, "");

    // Memory pressure summary
    let _ = writeln!(output, "Memory Pressure:");
    for tier in MemoryTier::ALL {
        if let Some(stats) = crate::tiers::get_tier_stats(tier) {
            let pressure = stats.pressure_percent();
            let status = if pressure > 90 {
                "CRITICAL"
            } else if pressure > 80 {
                "HIGH"
            } else if pressure > 60 {
                "MODERATE"
            } else {
                "LOW"
            };
            let _ = writeln!(output, "  {}: {}% ({})", tier.name(), pressure, status);
        }
    }

    // Migration statistics
    let migration_stats = crate::migration::get_migration_stats();
    let _ = writeln!(output, "");
    let _ = writeln!(output, "Migration Statistics:");
    let _ = writeln!(output, "  Successful: {}", migration_stats.total_success);
    let _ = writeln!(output, "  Failed: {}", migration_stats.total_failed);
    let _ = writeln!(output, "  In Progress: {}", migration_stats.in_progress);
    let _ = writeln!(output, "  Avg Time: {} μs", migration_stats.avg_time_us);

    // NUMA statistics
    let _ = writeln!(output, "");
    let _ = writeln!(output, "NUMA Statistics:");
    for numa_stats in crate::numa::get_numa_stats_summary() {
        let _ = writeln!(
            output,
            "  Node {}: {:.1}% locality ({} local, {} remote)",
            numa_stats.node_id,
            numa_stats.locality_ratio * 100.0,
            numa_stats.local_accesses,
            numa_stats.remote_accesses
        );
    }

    output
}

/// Read handler for tier stats
extern "C" fn read_tier_stats(file: *mut ProcFile, buf: *mut u8, count: usize) -> isize {
    // SAFETY: This is an FFI callback from the kernel's proc filesystem.
    // - file pointer validity is checked before dereferencing
    // - buf pointer is provided by the kernel and is valid for 'count' bytes
    // - The kernel guarantees these pointers are valid during the callback
    // - copy_nonoverlapping is safe because buf is kernel-provided user buffer
    unsafe {
        if file.is_null() || buf.is_null() {
            return -1;
        }

        let file_ref = &mut *file;
        let tier = file_ref.private_data as usize;

        if tier >= 5 {
            return -1;
        }

        let tier_enum = match tier {
            0 => MemoryTier::Gpu,
            1 => MemoryTier::Cpu,
            2 => MemoryTier::Nvme,
            3 => MemoryTier::Ssd,
            4 => MemoryTier::Hdd,
            _ => return -1,
        };

        let content = format_tier_stats(tier_enum);
        let content_bytes = content.as_bytes();

        // Handle offset
        let offset = file_ref.pos as usize;
        if offset >= content_bytes.len() {
            return 0; // EOF
        }

        let available = content_bytes.len() - offset;
        let to_copy = count.min(available);

        // Copy to user buffer
        let src = content_bytes.as_ptr().add(offset);
        core::ptr::copy_nonoverlapping(src, buf, to_copy);

        file_ref.pos += to_copy as u64;

        to_copy as isize
    }
}

/// Read handler for global stats
extern "C" fn read_global_stats(file: *mut ProcFile, buf: *mut u8, count: usize) -> isize {
    // SAFETY: This is an FFI callback from the kernel's proc filesystem.
    // - file pointer validity is checked before dereferencing
    // - buf pointer is provided by the kernel and is valid for 'count' bytes
    // - The kernel guarantees these pointers are valid during the callback
    // - copy_nonoverlapping is safe because buf is kernel-provided user buffer
    unsafe {
        if file.is_null() || buf.is_null() {
            return -1;
        }

        let file_ref = &mut *file;
        let content = format_global_stats();
        let content_bytes = content.as_bytes();

        // Handle offset
        let offset = file_ref.pos as usize;
        if offset >= content_bytes.len() {
            return 0; // EOF
        }

        let available = content_bytes.len() - offset;
        let to_copy = count.min(available);

        // Copy to user buffer
        let src = content_bytes.as_ptr().add(offset);
        core::ptr::copy_nonoverlapping(src, buf, to_copy);

        file_ref.pos += to_copy as u64;

        to_copy as isize
    }
}

/// Open handler for proc files
extern "C" fn proc_open(file: *mut ProcFile) -> i32 {
    // SAFETY: This is an FFI callback from the kernel's proc filesystem.
    // The file pointer is provided by the kernel and is guaranteed valid
    // when non-null. We check for null before dereferencing.
    unsafe {
        if !file.is_null() {
            (*file).pos = 0;
        }
    }
    0
}

/// Create proc entries
pub fn create_proc_entries() -> KernelResult<()> {
    // SAFETY: This function is called exactly once during kernel module
    // initialization, before any other threads can access PROC_ENTRIES.
    // The kernel module init sequence is single-threaded.
    unsafe {
        let mut entries = Vec::new();

        // Create /proc/swarm/tiers/stats
        entries.push(ProcEntry {
            path: String::from("/proc/swarm/tiers/stats"),
            tier: None,
            ops: ProcOps {
                open: Some(proc_open),
                read: Some(read_global_stats),
                write: None,
                release: None,
            },
        });

        // Create per-tier entries
        for (_idx, tier) in MemoryTier::ALL.iter().enumerate() {
            entries.push(ProcEntry {
                path: format!("/proc/swarm/tiers/{}/stats", tier.name()),
                tier: Some(*tier),
                ops: ProcOps {
                    open: Some(proc_open),
                    read: Some(read_tier_stats),
                    write: None,
                    release: None,
                },
            });
        }

        PROC_ENTRIES = Some(entries);
    }

    // In real kernel: create actual proc entries

    Ok(())
}

/// Remove proc entries
pub fn remove_proc_entries() {
    // SAFETY: This function is called exactly once during kernel module
    // unload, after all other operations have completed.
    unsafe {
        PROC_ENTRIES = None;
    }

    // In real kernel: remove actual proc entries
}

/// Initialize proc interface
pub fn init() -> KernelResult<()> {
    create_proc_entries()?;
    Ok(())
}

/// Cleanup proc interface
pub fn cleanup() {
    remove_proc_entries();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tier_stats() {
        // Initialize tiers first
        crate::tiers::init().ok();

        let stats = format_tier_stats(MemoryTier::Gpu);
        assert!(stats.contains("Tier: gpu"));
        assert!(stats.contains("Capacity:"));
        assert!(stats.contains("Pressure:"));

        // Cleanup
        crate::tiers::cleanup();
    }

    #[test]
    fn test_format_global_stats() {
        // Initialize subsystems
        crate::tiers::init().ok();
        crate::migration::init().ok();
        crate::numa::init().ok();

        let stats = format_global_stats();
        assert!(stats.contains("TierWatch Global Statistics"));
        assert!(stats.contains("Pages Tracked:"));
        assert!(stats.contains("Memory Pressure:"));
        assert!(stats.contains("NUMA Statistics:"));

        // Cleanup
        crate::numa::cleanup();
        crate::migration::cleanup();
        crate::tiers::cleanup();
    }
}
