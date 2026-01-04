//! NUMA-aware memory optimization
//!
//! This module provides NUMA node awareness for optimal memory placement
//! and migration decisions.

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::{KernelResult, MemoryTier};

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: u8,
    /// CPUs in this node
    pub cpus: Vec<u32>,
    /// Memory zones in this node
    pub memory_zones: Vec<MemoryZone>,
    /// Distance to other nodes (0-255, lower is closer)
    pub distances: Vec<u8>,
}

/// Memory zone within a NUMA node
#[derive(Debug)]
pub struct MemoryZone {
    /// Start physical address
    pub start_addr: u64,
    /// End physical address
    pub end_addr: u64,
    /// Memory tier
    pub tier: MemoryTier,
    /// Free pages in this zone
    pub free_pages: AtomicUsize,
    /// Total pages in this zone
    pub total_pages: usize,
}

impl Clone for MemoryZone {
    fn clone(&self) -> Self {
        Self {
            start_addr: self.start_addr,
            end_addr: self.end_addr,
            tier: self.tier,
            free_pages: AtomicUsize::new(self.free_pages.load(Ordering::Relaxed)),
            total_pages: self.total_pages,
        }
    }
}

/// NUMA statistics per node
pub struct NumaStats {
    /// Local memory accesses
    pub local_accesses: AtomicU64,
    /// Remote memory accesses
    pub remote_accesses: AtomicU64,
    /// Interleave hits
    pub interleave_hits: AtomicU64,
    /// Migrations between nodes
    pub migrations: AtomicU64,
}

impl NumaStats {
    const fn new() -> Self {
        Self {
            local_accesses: AtomicU64::new(0),
            remote_accesses: AtomicU64::new(0),
            interleave_hits: AtomicU64::new(0),
            migrations: AtomicU64::new(0),
        }
    }

    fn record_access(&self, is_local: bool) {
        if is_local {
            self.local_accesses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.remote_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn locality_ratio(&self) -> f64 {
        let local = self.local_accesses.load(Ordering::Relaxed) as f64;
        let remote = self.remote_accesses.load(Ordering::Relaxed) as f64;
        let total = local + remote;

        if total > 0.0 {
            local / total
        } else {
            1.0 // Assume perfect locality if no accesses
        }
    }
}

/// Maximum NUMA nodes supported
const MAX_NUMA_NODES: usize = 8;

/// NUMA node information
static mut NUMA_NODES: Option<Vec<NumaNode>> = None;

/// NUMA statistics per node
static mut NUMA_STATS: Option<Vec<NumaStats>> = None;

/// Default NUMA distance (same node)
const LOCAL_DISTANCE: u8 = 10;

/// Remote NUMA distance
const REMOTE_DISTANCE: u8 = 20;

/// Get number of NUMA nodes
pub fn get_num_nodes() -> usize {
    // SAFETY: NUMA_NODES is read-only here; initialized at module load.
    unsafe { NUMA_NODES.as_ref().map(|nodes| nodes.len()).unwrap_or(1) }
}

/// Get NUMA node for a CPU
pub fn get_node_for_cpu(cpu_id: u32) -> Option<u8> {
    // SAFETY: NUMA_NODES is read-only here; initialized at module load
    // and never modified afterward.
    unsafe {
        if let Some(nodes) = &NUMA_NODES {
            for node in nodes {
                if node.cpus.contains(&cpu_id) {
                    return Some(node.id);
                }
            }
        }
    }
    None
}

/// Get NUMA node for a physical address
pub fn get_node_for_pfn(pfn: u64) -> Option<u8> {
    let addr = pfn << 12; // Convert PFN to physical address

    // SAFETY: NUMA_NODES is read-only here; initialized at module load.
    // We only iterate and read the node configuration.
    unsafe {
        if let Some(nodes) = &NUMA_NODES {
            for node in nodes {
                for zone in &node.memory_zones {
                    if addr >= zone.start_addr && addr < zone.end_addr {
                        return Some(node.id);
                    }
                }
            }
        }
    }
    None
}

/// Get distance between two NUMA nodes
pub fn get_node_distance(from_node: u8, to_node: u8) -> u8 {
    if from_node == to_node {
        return LOCAL_DISTANCE;
    }

    // SAFETY: NUMA_NODES is read-only here; initialized at module load.
    unsafe {
        if let Some(nodes) = &NUMA_NODES {
            if let Some(node) = nodes.iter().find(|n| n.id == from_node) {
                if let Some(distance) = node.distances.get(to_node as usize) {
                    return *distance;
                }
            }
        }
    }

    REMOTE_DISTANCE
}

/// Check if a memory access is local
pub fn is_local_access(cpu_id: u32, pfn: u64) -> bool {
    let cpu_node = get_node_for_cpu(cpu_id);
    let mem_node = get_node_for_pfn(pfn);

    match (cpu_node, mem_node) {
        (Some(cn), Some(mn)) => cn == mn,
        _ => true, // Assume local if we can't determine
    }
}

/// Record a NUMA access
pub fn record_numa_access(cpu_id: u32, pfn: u64) {
    let is_local = is_local_access(cpu_id, pfn);

    if let Some(cpu_node) = get_node_for_cpu(cpu_id) {
        // SAFETY: NUMA_STATS is initialized at module load. NumaStats uses
        // atomics for all counter updates, ensuring thread-safe access.
        unsafe {
            if let Some(stats) = &NUMA_STATS {
                if let Some(node_stats) = stats.get(cpu_node as usize) {
                    node_stats.record_access(is_local);
                }
            }
        }
    }
}

/// Get preferred NUMA node for an agent
pub fn get_preferred_node(agent_id: u64) -> u8 {
    // Simple hash-based distribution
    (agent_id % get_num_nodes() as u64) as u8
}

/// Find best tier and node for page allocation
pub fn find_best_allocation(
    tier: MemoryTier,
    preferred_node: u8,
    size_pages: usize,
) -> Option<(u8, MemoryZone)> {
    // SAFETY: NUMA_NODES is read-only here. MemoryZone.free_pages uses atomics
    // for thread-safe access. We clone zones before returning to avoid holding
    // references to the static data.
    unsafe {
        if let Some(nodes) = &NUMA_NODES {
            // First try preferred node
            if let Some(node) = nodes.iter().find(|n| n.id == preferred_node) {
                for zone in &node.memory_zones {
                    if zone.tier == tier {
                        let free = zone.free_pages.load(Ordering::Relaxed);
                        if free >= size_pages {
                            return Some((node.id, zone.clone()));
                        }
                    }
                }
            }

            // Try other nodes sorted by distance
            let mut node_distances: Vec<(u8, u8)> = nodes
                .iter()
                .filter(|n| n.id != preferred_node)
                .map(|n| (n.id, get_node_distance(preferred_node, n.id)))
                .collect();

            node_distances.sort_by_key(|&(_, dist)| dist);

            for (node_id, _) in node_distances {
                if let Some(node) = nodes.iter().find(|n| n.id == node_id) {
                    for zone in &node.memory_zones {
                        if zone.tier == tier {
                            let free = zone.free_pages.load(Ordering::Relaxed);
                            if free >= size_pages {
                                return Some((node.id, zone.clone()));
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Get NUMA statistics summary
pub fn get_numa_stats_summary() -> Vec<NumaStatsSummary> {
    let mut summaries = Vec::new();

    // SAFETY: NUMA_STATS is read-only here. NumaStats uses atomics for all
    // counter access, ensuring thread-safe reads.
    unsafe {
        if let Some(stats) = &NUMA_STATS {
            for (node_id, node_stats) in stats.iter().enumerate() {
                summaries.push(NumaStatsSummary {
                    node_id: node_id as u8,
                    local_accesses: node_stats.local_accesses.load(Ordering::Relaxed),
                    remote_accesses: node_stats.remote_accesses.load(Ordering::Relaxed),
                    locality_ratio: node_stats.locality_ratio(),
                    interleave_hits: node_stats.interleave_hits.load(Ordering::Relaxed),
                    migrations: node_stats.migrations.load(Ordering::Relaxed),
                });
            }
        }
    }

    summaries
}

/// NUMA statistics summary
#[derive(Debug, Clone)]
pub struct NumaStatsSummary {
    pub node_id: u8,
    pub local_accesses: u64,
    pub remote_accesses: u64,
    pub locality_ratio: f64,
    pub interleave_hits: u64,
    pub migrations: u64,
}

/// Initialize NUMA subsystem
pub fn init() -> KernelResult<()> {
    // SAFETY: This function is called exactly once during kernel module
    // initialization, before any other threads can access NUMA_NODES or
    // NUMA_STATS. The kernel module init sequence is single-threaded.
    unsafe {
        // In real kernel, would parse ACPI SRAT/SLIT tables
        // For now, create a simple 2-node system

        let mut nodes = Vec::new();

        // Node 0
        nodes.push(NumaNode {
            id: 0,
            cpus: (0..16).collect(),
            memory_zones: vec![
                MemoryZone {
                    start_addr: 0,
                    end_addr: 48 << 30, // 48GB
                    tier: MemoryTier::Cpu,
                    free_pages: AtomicUsize::new(48 << 18), // 48GB / 4KB
                    total_pages: 48 << 18,
                },
                MemoryZone {
                    start_addr: 0x100000000,
                    end_addr: 0x100000000 + (16 << 30), // 16GB GPU
                    tier: MemoryTier::Gpu,
                    free_pages: AtomicUsize::new(16 << 18),
                    total_pages: 16 << 18,
                },
            ],
            distances: vec![LOCAL_DISTANCE, REMOTE_DISTANCE],
        });

        // Node 1
        nodes.push(NumaNode {
            id: 1,
            cpus: (16..32).collect(),
            memory_zones: vec![
                MemoryZone {
                    start_addr: 48 << 30,
                    end_addr: 96 << 30, // 48GB
                    tier: MemoryTier::Cpu,
                    free_pages: AtomicUsize::new(48 << 18),
                    total_pages: 48 << 18,
                },
                MemoryZone {
                    start_addr: 0x200000000,
                    end_addr: 0x200000000 + (16 << 30), // 16GB GPU
                    tier: MemoryTier::Gpu,
                    free_pages: AtomicUsize::new(16 << 18),
                    total_pages: 16 << 18,
                },
            ],
            distances: vec![REMOTE_DISTANCE, LOCAL_DISTANCE],
        });

        NUMA_NODES = Some(nodes);

        // Initialize statistics
        let mut stats = Vec::new();
        for _ in 0..2 {
            stats.push(NumaStats::new());
        }
        NUMA_STATS = Some(stats);
    }

    Ok(())
}

/// Cleanup NUMA subsystem
pub fn cleanup() {
    // SAFETY: This function is called exactly once during kernel module
    // unload, after all other operations have completed.
    unsafe {
        NUMA_NODES = None;
        NUMA_STATS = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_distance() {
        assert_eq!(get_node_distance(0, 0), LOCAL_DISTANCE);
        assert_eq!(get_node_distance(0, 1), REMOTE_DISTANCE);
        assert_eq!(get_node_distance(1, 0), REMOTE_DISTANCE);
        assert_eq!(get_node_distance(1, 1), LOCAL_DISTANCE);
    }

    #[test]
    fn test_preferred_node_distribution() {
        // Test that agents are distributed across nodes
        let mut node_counts = vec![0u32; MAX_NUMA_NODES];

        for agent_id in 0..1000 {
            let node = get_preferred_node(agent_id) as usize;
            if node < MAX_NUMA_NODES {
                node_counts[node] += 1;
            }
        }

        // Verify roughly even distribution
        let expected_per_node = 1000 / get_num_nodes();
        for count in node_counts.iter().take(get_num_nodes()) {
            assert!(*count > (expected_per_node * 8 / 10) as u32); // Within 20%
            assert!(*count < (expected_per_node * 12 / 10) as u32);
        }
    }
}
