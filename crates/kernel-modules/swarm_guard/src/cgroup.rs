//! Cgroup v2 management for resource enforcement
//!
//! This module handles the creation and management of cgroups for
//! agent containers to enforce resource limits.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;

use crate::{agent::AgentId, KernelError, KernelResult};

/// Cgroup v2 controller types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgroupController {
    /// CPU controller for CPU time limits
    Cpu,
    /// Memory controller for memory limits
    Memory,
    /// I/O controller for disk bandwidth
    Io,
    /// Process number controller
    Pids,
    /// RDMA controller
    Rdma,
    /// HugeTLB controller
    HugeTlb,
}

impl CgroupController {
    /// Get controller name for cgroup filesystem
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Memory => "memory",
            Self::Io => "io",
            Self::Pids => "pids",
            Self::Rdma => "rdma",
            Self::HugeTlb => "hugetlb",
        }
    }

    /// Get all available controllers
    pub fn all() -> Vec<Self> {
        vec![Self::Cpu, Self::Memory, Self::Io, Self::Pids]
    }
}

/// CPU resource configuration
#[derive(Debug, Clone, Copy)]
pub struct CpuConfig {
    /// CPU quota in microseconds per period
    pub quota_us: u64,
    /// CPU period in microseconds (default: 100000)
    pub period_us: u64,
    /// CPU weight (1-10000, default: 100)
    pub weight: u32,
    /// Nice value for scheduling priority
    pub nice: i32,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            quota_us: 25000,   // 25ms = 25%
            period_us: 100000, // 100ms period
            weight: 100,       // Default weight
            nice: 0,           // Normal priority
        }
    }
}

impl CpuConfig {
    /// Create from percentage (0-100)
    pub fn from_percentage(percent: u32) -> Self {
        let quota_us = (percent as u64 * 1000) as u64; // percent * 1000us
        Self {
            quota_us,
            ..Default::default()
        }
    }
}

/// Memory resource configuration
#[derive(Debug, Clone, Copy)]
pub struct MemoryConfig {
    /// Hard memory limit in bytes
    pub limit_bytes: usize,
    /// Soft memory limit (for reclaim)
    pub soft_limit_bytes: Option<usize>,
    /// Swap limit in bytes (None = same as memory limit)
    pub swap_limit_bytes: Option<usize>,
    /// Whether to account kernel memory
    pub kernel_memory: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            limit_bytes: 256 << 20, // 256MB
            soft_limit_bytes: None, // No soft limit
            swap_limit_bytes: None, // Same as memory
            kernel_memory: true,    // Account kernel memory
        }
    }
}

/// I/O resource configuration
#[derive(Debug, Clone)]
pub struct IoConfig {
    /// Read bandwidth limit in bytes/sec
    pub read_bps: Option<u64>,
    /// Write bandwidth limit in bytes/sec
    pub write_bps: Option<u64>,
    /// Read IOPS limit
    pub read_iops: Option<u64>,
    /// Write IOPS limit
    pub write_iops: Option<u64>,
    /// Device restrictions
    pub device_weights: Vec<DeviceWeight>,
}

/// Device weight for I/O scheduling
#[derive(Debug, Clone, Copy)]
pub struct DeviceWeight {
    /// Device major number
    pub major: u32,
    /// Device minor number
    pub minor: u32,
    /// Weight (1-10000)
    pub weight: u32,
}

/// Complete cgroup configuration for an agent
#[derive(Debug, Clone)]
pub struct CgroupConfig {
    /// CPU resource limits
    pub cpu: CpuConfig,
    /// Memory resource limits
    pub memory: MemoryConfig,
    /// I/O resource limits
    pub io: Option<IoConfig>,
    /// Maximum number of processes
    pub max_pids: Option<u32>,
}

impl Default for CgroupConfig {
    fn default() -> Self {
        Self {
            cpu: CpuConfig::default(),
            memory: MemoryConfig::default(),
            io: None,
            max_pids: Some(1000), // Reasonable default
        }
    }
}

/// Cgroup v2 manager
pub struct CgroupManager {
    /// Root cgroup path (usually /sys/fs/cgroup)
    root_path: String,
    /// StratoSwarm cgroup hierarchy root
    swarm_root: String,
}

impl CgroupManager {
    /// Create a new cgroup manager
    pub fn new() -> Self {
        Self {
            root_path: String::from("/sys/fs/cgroup"),
            swarm_root: String::from("swarm"),
        }
    }

    /// Create cgroup for an agent
    pub fn create_agent_cgroup(
        &self,
        agent_id: AgentId,
        config: &CgroupConfig,
    ) -> KernelResult<String> {
        let cgroup_path = format!(
            "{}/{}/agent-{}",
            self.root_path, self.swarm_root, agent_id.0
        );

        // In kernel: mkdir cgroup directory
        self.create_cgroup_dir(&cgroup_path)?;

        // Enable controllers
        self.enable_controllers(&cgroup_path)?;

        // Configure CPU limits
        self.configure_cpu(&cgroup_path, &config.cpu)?;

        // Configure memory limits
        self.configure_memory(&cgroup_path, &config.memory)?;

        // Configure I/O limits if specified
        if let Some(io_config) = &config.io {
            self.configure_io(&cgroup_path, io_config)?;
        }

        // Configure PID limits if specified
        if let Some(max_pids) = config.max_pids {
            self.configure_pids(&cgroup_path, max_pids)?;
        }

        Ok(cgroup_path)
    }

    /// Add a process to an agent's cgroup
    pub fn add_process_to_cgroup(&self, agent_id: AgentId, pid: u32) -> KernelResult<()> {
        let cgroup_path = format!(
            "{}/{}/agent-{}",
            self.root_path, self.swarm_root, agent_id.0
        );

        // In kernel: write PID to cgroup.procs
        self.write_cgroup_file(&cgroup_path, "cgroup.procs", &pid.to_string())?;

        Ok(())
    }

    /// Remove an agent's cgroup
    pub fn remove_agent_cgroup(&self, agent_id: AgentId) -> KernelResult<()> {
        let cgroup_path = format!(
            "{}/{}/agent-{}",
            self.root_path, self.swarm_root, agent_id.0
        );

        // In kernel: rmdir cgroup directory (must be empty)
        self.remove_cgroup_dir(&cgroup_path)?;

        Ok(())
    }

    /// Create cgroup directory
    fn create_cgroup_dir(&self, path: &str) -> KernelResult<()> {
        // In kernel: kernfs_create_dir()
        Ok(())
    }

    /// Remove cgroup directory
    fn remove_cgroup_dir(&self, path: &str) -> KernelResult<()> {
        // In kernel: kernfs_remove()
        Ok(())
    }

    /// Enable controllers for a cgroup
    fn enable_controllers(&self, cgroup_path: &str) -> KernelResult<()> {
        let controllers = "+cpu +memory +io +pids";
        self.write_cgroup_file(cgroup_path, "cgroup.subtree_control", controllers)?;
        Ok(())
    }

    /// Configure CPU controller
    fn configure_cpu(&self, cgroup_path: &str, config: &CpuConfig) -> KernelResult<()> {
        // Set CPU quota and period
        let cpu_max = if config.quota_us >= config.period_us {
            String::from("max")
        } else {
            format!("{} {}", config.quota_us, config.period_us)
        };
        self.write_cgroup_file(cgroup_path, "cpu.max", &cpu_max)?;

        // Set CPU weight
        self.write_cgroup_file(cgroup_path, "cpu.weight", &config.weight.to_string())?;

        // Nice value would be set via sched_setattr in kernel

        Ok(())
    }

    /// Configure memory controller
    fn configure_memory(&self, cgroup_path: &str, config: &MemoryConfig) -> KernelResult<()> {
        // Set memory limit
        self.write_cgroup_file(cgroup_path, "memory.max", &config.limit_bytes.to_string())?;

        // Set soft limit if specified
        if let Some(soft_limit) = config.soft_limit_bytes {
            self.write_cgroup_file(cgroup_path, "memory.high", &soft_limit.to_string())?;
        }

        // Set swap limit if specified
        if let Some(swap_limit) = config.swap_limit_bytes {
            self.write_cgroup_file(cgroup_path, "memory.swap.max", &swap_limit.to_string())?;
        }

        Ok(())
    }

    /// Configure I/O controller
    fn configure_io(&self, cgroup_path: &str, config: &IoConfig) -> KernelResult<()> {
        let mut io_max = String::new();

        // In real implementation, would detect actual devices
        let device = "8:0"; // Example: sda

        if let Some(read_bps) = config.read_bps {
            writeln!(&mut io_max, "{} rbps={}", device, read_bps).unwrap();
        }

        if let Some(write_bps) = config.write_bps {
            writeln!(&mut io_max, "{} wbps={}", device, write_bps).unwrap();
        }

        if let Some(read_iops) = config.read_iops {
            writeln!(&mut io_max, "{} riops={}", device, read_iops).unwrap();
        }

        if let Some(write_iops) = config.write_iops {
            writeln!(&mut io_max, "{} wiops={}", device, write_iops).unwrap();
        }

        if !io_max.is_empty() {
            self.write_cgroup_file(cgroup_path, "io.max", &io_max)?;
        }

        // Set device weights
        for dev_weight in &config.device_weights {
            let weight_str = format!(
                "{}:{} {}",
                dev_weight.major, dev_weight.minor, dev_weight.weight
            );
            self.write_cgroup_file(cgroup_path, "io.weight", &weight_str)?;
        }

        Ok(())
    }

    /// Configure PID controller
    fn configure_pids(&self, cgroup_path: &str, max_pids: u32) -> KernelResult<()> {
        self.write_cgroup_file(cgroup_path, "pids.max", &max_pids.to_string())?;
        Ok(())
    }

    /// Write to a cgroup file
    fn write_cgroup_file(&self, cgroup_path: &str, file: &str, value: &str) -> KernelResult<()> {
        // In kernel: kernfs_fop_write()
        Ok(())
    }

    /// Read from a cgroup file
    fn read_cgroup_file(&self, cgroup_path: &str, file: &str) -> KernelResult<String> {
        // In kernel: kernfs_fop_read()
        Ok(String::new())
    }

    /// Get current resource usage for an agent
    pub fn get_agent_stats(&self, agent_id: AgentId) -> KernelResult<ResourceStats> {
        let cgroup_path = format!(
            "{}/{}/agent-{}",
            self.root_path, self.swarm_root, agent_id.0
        );

        // Read CPU stats
        let cpu_stat = self.read_cgroup_file(&cgroup_path, "cpu.stat")?;

        // Read memory stats
        let memory_current = self.read_cgroup_file(&cgroup_path, "memory.current")?;
        let memory_stat = self.read_cgroup_file(&cgroup_path, "memory.stat")?;

        // Parse and return stats
        Ok(ResourceStats {
            cpu_usage_us: 0,       // Parse from cpu_stat
            memory_usage_bytes: 0, // Parse from memory_current
            io_bytes_read: 0,
            io_bytes_written: 0,
            pid_count: 0,
        })
    }
}

/// Resource usage statistics
#[derive(Debug, Clone, Copy)]
pub struct ResourceStats {
    /// Total CPU time used in microseconds
    pub cpu_usage_us: u64,
    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Total bytes read from disk
    pub io_bytes_read: u64,
    /// Total bytes written to disk
    pub io_bytes_written: u64,
    /// Current number of processes
    pub pid_count: u32,
}

/// Initialize cgroup subsystem
pub fn init() -> KernelResult<()> {
    // Create StratoSwarm root cgroup
    let manager = CgroupManager::new();
    let swarm_root = format!("{}/{}", manager.root_path, manager.swarm_root);
    manager.create_cgroup_dir(&swarm_root)?;
    manager.enable_controllers(&swarm_root)?;

    Ok(())
}

/// Cleanup cgroup subsystem
pub fn cleanup() {
    // Remove StratoSwarm root cgroup if empty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_config_from_percentage() {
        let config = CpuConfig::from_percentage(50);
        assert_eq!(config.quota_us, 50000); // 50% = 50ms
        assert_eq!(config.period_us, 100000); // 100ms period

        let config = CpuConfig::from_percentage(100);
        assert_eq!(config.quota_us, 100000); // 100% = full period
    }

    #[test]
    fn test_cgroup_path_generation() {
        let manager = CgroupManager::new();
        let agent_id = AgentId(12345);

        let expected = "/sys/fs/cgroup/swarm/agent-12345";
        let path = format!(
            "{}/{}/agent-{}",
            manager.root_path, manager.swarm_root, agent_id.0
        );

        assert_eq!(path, expected);
    }
}
