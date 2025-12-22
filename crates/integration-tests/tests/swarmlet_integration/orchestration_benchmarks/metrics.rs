//! Performance metrics for orchestration benchmarks

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Container lifecycle performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerLifecycleMetrics {
    pub image_pull_time: Duration,
    pub container_creation_time: Duration,
    pub initialization_time: Duration,
    pub ready_time: Duration,
    pub teardown_time: Duration,
    pub total_lifecycle_time: Duration,
    pub restart_count: u32,
    pub failure_count: u32,
}

/// Concurrent execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentExecutionMetrics {
    pub max_concurrent_workloads: u32,
    pub avg_queue_length: f64,
    pub queue_wait_time_p50: Duration,
    pub queue_wait_time_p95: Duration,
    pub queue_wait_time_p99: Duration,
    pub rejection_rate: f32,
    pub timeout_rate: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub cpu_utilization_percent: f32,
    pub memory_utilization_percent: f32,
    pub gpu_utilization_percent: f32,
    pub disk_io_throughput_mbps: f64,
    pub network_io_throughput_mbps: f64,
    pub resource_fragmentation_percent: f32,
    pub oversubscription_ratio: f32,
}

/// Performance expectations for workloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    pub expected_duration_ms: u64,
    pub expected_cpu_usage_percent: f32,
    pub expected_memory_mb: u64,
    pub expected_gpu_usage_percent: f32,
    pub expected_disk_io_mbps: f64,
    pub expected_network_io_mbps: f64,
}

impl Default for ContainerLifecycleMetrics {
    fn default() -> Self {
        Self {
            image_pull_time: Duration::ZERO,
            container_creation_time: Duration::ZERO,
            initialization_time: Duration::ZERO,
            ready_time: Duration::ZERO,
            teardown_time: Duration::ZERO,
            total_lifecycle_time: Duration::ZERO,
            restart_count: 0,
            failure_count: 0,
        }
    }
}

impl Default for ConcurrentExecutionMetrics {
    fn default() -> Self {
        Self {
            max_concurrent_workloads: 0,
            avg_queue_length: 0.0,
            queue_wait_time_p50: Duration::ZERO,
            queue_wait_time_p95: Duration::ZERO,
            queue_wait_time_p99: Duration::ZERO,
            rejection_rate: 0.0,
            timeout_rate: 0.0,
        }
    }
}

impl Default for ResourceUtilizationMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization_percent: 0.0,
            memory_utilization_percent: 0.0,
            gpu_utilization_percent: 0.0,
            disk_io_throughput_mbps: 0.0,
            network_io_throughput_mbps: 0.0,
            resource_fragmentation_percent: 0.0,
            oversubscription_ratio: 1.0,
        }
    }
}

impl Default for PerformanceExpectations {
    fn default() -> Self {
        Self {
            expected_duration_ms: 1000,
            expected_cpu_usage_percent: 10.0,
            expected_memory_mb: 256,
            expected_gpu_usage_percent: 0.0,
            expected_disk_io_mbps: 10.0,
            expected_network_io_mbps: 1.0,
        }
    }
}