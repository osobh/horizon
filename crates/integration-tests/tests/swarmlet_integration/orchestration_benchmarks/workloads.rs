//! Workload definitions for orchestration benchmarks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Benchmark workload definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    pub workload_id: Uuid,
    pub workload_type: WorkloadType,
    pub container_spec: ContainerSpec,
    pub performance_expectations: super::metrics::PerformanceExpectations,
}

/// Type of workload for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkloadType {
    /// Simple CPU-bound computation
    CpuIntensive {
        iterations: u64,
        complexity: u32,
    },
    /// Memory-intensive workload
    MemoryIntensive {
        allocation_mb: u64,
        access_pattern: MemoryAccessPattern,
    },
    /// GPU computation workload
    GpuCompute {
        kernel_type: GpuKernelType,
        data_size_mb: u64,
    },
    /// I/O-bound workload
    IoIntensive {
        read_mb: u64,
        write_mb: u64,
        pattern: IoPattern,
    },
    /// Network-intensive workload
    NetworkIntensive {
        connections: u32,
        bandwidth_mbps: f64,
    },
    /// Mixed realistic workload
    Mixed {
        cpu_percent: f32,
        memory_mb: u64,
        io_ops: u64,
    },
}

/// Memory access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
}

/// GPU kernel type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuKernelType {
    MatrixMultiply,
    Convolution,
    Reduction,
    Custom,
}

/// I/O pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoPattern {
    Sequential,
    Random,
    Burst,
}

/// Container specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    pub image: String,
    pub command: Vec<String>,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub working_dir: String,
    pub image_pull_policy: ImagePullPolicy,
    pub security_context: Option<SecurityContext>,
    pub resource_requirements: ResourceRequirements,
}

/// Image pull policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImagePullPolicy {
    Always,
    IfNotPresent,
    Never,
}

/// Security context for container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub run_as_user: Option<u32>,
    pub run_as_group: Option<u32>,
    pub privileged: bool,
    pub read_only_root_filesystem: bool,
    pub allow_privilege_escalation: bool,
}

/// Resource requirements for container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f32,
    pub memory_mb: u64,
    pub gpu_requirements: Option<GpuRequirements>,
    pub ephemeral_storage_mb: u64,
    pub persistent_volumes: Vec<VolumeRequirement>,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    pub gpu_count: u32,
    pub gpu_memory_mb: u64,
    pub gpu_type: Option<String>,
}

/// Volume requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeRequirement {
    pub name: String,
    pub size_mb: u64,
    pub access_mode: VolumeAccessMode,
}

/// Volume access mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolumeAccessMode {
    ReadWriteOnce,
    ReadOnlyMany,
    ReadWriteMany,
}

impl Default for ContainerSpec {
    fn default() -> Self {
        Self {
            image: "alpine:latest".to_string(),
            command: vec!["echo".to_string()],
            args: vec!["test".to_string()],
            env: HashMap::new(),
            working_dir: "/app".to_string(),
            image_pull_policy: ImagePullPolicy::IfNotPresent,
            security_context: None,
            resource_requirements: ResourceRequirements::default(),
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 0.1,
            memory_mb: 128,
            gpu_requirements: None,
            ephemeral_storage_mb: 100,
            persistent_volumes: Vec::new(),
        }
    }
}