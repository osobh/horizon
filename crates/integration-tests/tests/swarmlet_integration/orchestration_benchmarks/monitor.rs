//! Resource monitoring for orchestration benchmarks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Resource monitor for tracking utilization
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    pub node_states: HashMap<String, NodeResourceState>,
    pub workload_states: HashMap<Uuid, WorkloadResourceState>,
    pub last_update: Instant,
}

/// Resource snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: Instant,
    pub total_cpu_cores: f32,
    pub used_cpu_cores: f32,
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub total_gpu_count: u32,
    pub used_gpu_count: u32,
}

/// Node resource state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResourceState {
    pub node_id: String,
    pub total_cpu_cores: f32,
    pub allocatable_cpu_cores: f32,
    pub used_cpu_cores: f32,
    pub total_memory_mb: u64,
    pub allocatable_memory_mb: u64,
    pub used_memory_mb: u64,
    pub gpu_count: u32,
    pub available_gpu_count: u32,
    pub running_workloads: Vec<Uuid>,
}

/// Workload resource state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadResourceState {
    pub workload_id: Uuid,
    pub node_id: String,
    pub allocated_cpu_cores: f32,
    pub actual_cpu_usage: f32,
    pub allocated_memory_mb: u64,
    pub actual_memory_mb: u64,
    pub allocated_gpu_count: u32,
    pub actual_gpu_usage_percent: f32,
    pub start_time: Instant,
    pub state: WorkloadState,
}

/// Workload state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadState {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Terminated,
}

impl ResourceMonitor {
    /// Create new resource monitor
    pub fn new() -> Self {
        Self {
            node_states: HashMap::new(),
            workload_states: HashMap::new(),
            last_update: Instant::now(),
        }
    }

    /// Update node state
    pub fn update_node(&mut self, state: NodeResourceState) {
        self.node_states.insert(state.node_id.clone(), state);
        self.last_update = Instant::now();
    }

    /// Update workload state
    pub fn update_workload(&mut self, state: WorkloadResourceState) {
        self.workload_states.insert(state.workload_id, state);
        self.last_update = Instant::now();
    }

    /// Get current resource snapshot
    pub fn get_snapshot(&self) -> ResourceSnapshot {
        let mut total_cpu = 0.0;
        let mut used_cpu = 0.0;
        let mut total_memory = 0;
        let mut used_memory = 0;
        let mut total_gpu = 0;
        let mut used_gpu = 0;

        for node in self.node_states.values() {
            total_cpu += node.total_cpu_cores;
            used_cpu += node.used_cpu_cores;
            total_memory += node.total_memory_mb;
            used_memory += node.used_memory_mb;
            total_gpu += node.gpu_count;
            used_gpu += node.gpu_count - node.available_gpu_count;
        }

        ResourceSnapshot {
            timestamp: Instant::now(),
            total_cpu_cores: total_cpu,
            used_cpu_cores: used_cpu,
            total_memory_mb: total_memory,
            used_memory_mb: used_memory,
            total_gpu_count: total_gpu,
            used_gpu_count: used_gpu,
        }
    }

    /// Calculate resource utilization percentage
    pub fn calculate_utilization(&self) -> (f32, f32, f32) {
        let snapshot = self.get_snapshot();
        
        let cpu_util = if snapshot.total_cpu_cores > 0.0 {
            (snapshot.used_cpu_cores / snapshot.total_cpu_cores) * 100.0
        } else {
            0.0
        };

        let mem_util = if snapshot.total_memory_mb > 0 {
            (snapshot.used_memory_mb as f32 / snapshot.total_memory_mb as f32) * 100.0
        } else {
            0.0
        };

        let gpu_util = if snapshot.total_gpu_count > 0 {
            (snapshot.used_gpu_count as f32 / snapshot.total_gpu_count as f32) * 100.0
        } else {
            0.0
        };

        (cpu_util, mem_util, gpu_util)
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}