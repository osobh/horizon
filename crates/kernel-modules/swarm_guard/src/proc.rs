//! /proc/swarm interface for monitoring and control
//!
//! This module provides the /proc filesystem interface for SwarmGuard,
//! allowing userspace to monitor and control agent containers.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::agent::{Agent, AgentId, ResourceLimits};
use crate::syscall::INTERCEPTION_STATS;
use crate::{KernelError, KernelResult, AGENT_STATS};

/// /proc/swarm file types
#[derive(Debug, Clone, Copy)]
pub enum ProcFile {
    /// Global status information
    Status,
    /// List of all agents
    Agents,
    /// Create new agent
    Create,
    /// Per-agent status
    AgentStatus(AgentId),
    /// System call statistics
    Syscalls,
    /// Resource usage
    Resources,
}

/// /proc/swarm file operations
pub struct ProcOps;

impl ProcOps {
    /// Read from a proc file
    pub fn read(file: ProcFile, offset: usize, buffer: &mut [u8]) -> KernelResult<usize> {
        let content = match file {
            ProcFile::Status => Self::read_status()?,
            ProcFile::Agents => Self::read_agents()?,
            ProcFile::Create => {
                return Err(KernelError::PermissionDenied); // Write-only
            }
            ProcFile::AgentStatus(id) => Self::read_agent_status(id)?,
            ProcFile::Syscalls => Self::read_syscalls()?,
            ProcFile::Resources => Self::read_resources()?,
        };

        // Handle offset and copy to buffer
        let bytes = content.as_bytes();
        if offset >= bytes.len() {
            return Ok(0);
        }

        let to_copy = core::cmp::min(buffer.len(), bytes.len() - offset);
        buffer[..to_copy].copy_from_slice(&bytes[offset..offset + to_copy]);

        Ok(to_copy)
    }

    /// Write to a proc file
    pub fn write(file: ProcFile, data: &[u8]) -> KernelResult<usize> {
        match file {
            ProcFile::Create => Self::write_create(data),
            _ => Err(KernelError::PermissionDenied), // Read-only
        }
    }

    /// Read global status
    fn read_status() -> KernelResult<String> {
        let mut output = String::new();

        writeln!(&mut output, "StratoSwarm SwarmGuard Status").unwrap();
        writeln!(&mut output, "=============================").unwrap();
        writeln!(
            &mut output,
            "Active agents: {}",
            AGENT_STATS
                .active_count
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "Total created: {}",
            AGENT_STATS
                .total_created
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "Total destroyed: {}",
            AGENT_STATS
                .total_destroyed
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "Policy violations: {}",
            AGENT_STATS
                .violations
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "System Call Interception:").unwrap();
        writeln!(
            &mut output,
            "  Total intercepted: {}",
            INTERCEPTION_STATS
                .total_intercepted
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  Allowed: {}",
            INTERCEPTION_STATS
                .allowed
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  Denied: {}",
            INTERCEPTION_STATS
                .denied
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  Process creation attempts: {}",
            INTERCEPTION_STATS
                .process_creation_attempts
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  Network attempts: {}",
            INTERCEPTION_STATS
                .network_attempts
                .load(core::sync::atomic::Ordering::Relaxed)
        )
        .unwrap();

        Ok(output)
    }

    /// Read agent list
    fn read_agents() -> KernelResult<String> {
        let mut output = String::new();

        writeln!(
            &mut output,
            "ID        PID      Memory    CPU   NS    Status"
        )
        .unwrap();
        writeln!(
            &mut output,
            "------------------------------------------------"
        )
        .unwrap();

        // In real implementation, would iterate through agent registry
        // For now, show example entries
        writeln!(
            &mut output,
            "1         1234     256MB     25%   0x3F  active"
        )
        .unwrap();
        writeln!(
            &mut output,
            "2         5678     512MB     50%   0x3F  active"
        )
        .unwrap();
        writeln!(
            &mut output,
            "3         9012     128MB     10%   0x1F  active"
        )
        .unwrap();

        Ok(output)
    }

    /// Read per-agent status
    fn read_agent_status(id: AgentId) -> KernelResult<String> {
        let mut output = String::new();

        writeln!(&mut output, "Agent {} Status", id.0).unwrap();
        writeln!(&mut output, "===============").unwrap();

        // In real implementation, would look up agent
        writeln!(&mut output, "PID: 1234").unwrap();
        writeln!(&mut output, "Parent: None").unwrap();
        writeln!(&mut output, "Created: 123456789 ns").unwrap();
        writeln!(&mut output, "Last active: 123456999 ns").unwrap();
        writeln!(&mut output, "Trust score: 0.85").unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "Personality:").unwrap();
        writeln!(&mut output, "  Risk tolerance: 0.5").unwrap();
        writeln!(&mut output, "  Cooperation: 0.7").unwrap();
        writeln!(&mut output, "  Exploration: 0.3").unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "Resources:").unwrap();
        writeln!(&mut output, "  Memory limit: 268435456 bytes").unwrap();
        writeln!(&mut output, "  CPU quota: 25%").unwrap();
        writeln!(&mut output, "  GPU memory: 0 bytes").unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "Namespaces: 0x3F (all)").unwrap();
        writeln!(&mut output, "Cgroup: /sys/fs/cgroup/swarm/agent-1").unwrap();

        Ok(output)
    }

    /// Read syscall statistics
    fn read_syscalls() -> KernelResult<String> {
        let mut output = String::new();

        writeln!(&mut output, "System Call Statistics").unwrap();
        writeln!(&mut output, "======================").unwrap();

        // In real implementation, would show per-syscall stats
        writeln!(&mut output, "Syscall   Allowed  Denied  Total").unwrap();
        writeln!(&mut output, "--------------------------------").unwrap();
        writeln!(&mut output, "read      10000    0       10000").unwrap();
        writeln!(&mut output, "write     8000     0       8000").unwrap();
        writeln!(&mut output, "open      500      10      510").unwrap();
        writeln!(&mut output, "clone     50       5       55").unwrap();
        writeln!(&mut output, "execve    20       3       23").unwrap();

        Ok(output)
    }

    /// Read resource usage
    fn read_resources() -> KernelResult<String> {
        let mut output = String::new();

        writeln!(&mut output, "Resource Usage Summary").unwrap();
        writeln!(&mut output, "======================").unwrap();

        writeln!(&mut output, "Total Memory:").unwrap();
        writeln!(&mut output, "  Allocated: 2147483648 bytes (2GB)").unwrap();
        writeln!(&mut output, "  Used: 1610612736 bytes (1.5GB)").unwrap();
        writeln!(&mut output, "  Available: 536870912 bytes (512MB)").unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "CPU Usage:").unwrap();
        writeln!(&mut output, "  Total quota: 800%").unwrap();
        writeln!(&mut output, "  Allocated: 600%").unwrap();
        writeln!(&mut output, "  In use: 450%").unwrap();
        writeln!(&mut output).unwrap();

        writeln!(&mut output, "GPU Memory:").unwrap();
        writeln!(&mut output, "  Allocated: 0 bytes").unwrap();
        writeln!(&mut output, "  Available: 34359738368 bytes (32GB)").unwrap();

        Ok(output)
    }

    /// Write to create file (create new agent)
    fn write_create(data: &[u8]) -> KernelResult<usize> {
        // Parse JSON configuration
        let config = Self::parse_agent_config(data)?;

        // Create agent with configuration
        // In real implementation, would create agent through registry

        Ok(data.len())
    }

    /// Parse agent configuration from JSON
    fn parse_agent_config(data: &[u8]) -> KernelResult<AgentConfig> {
        // Simple JSON parsing (in real kernel module, would use minimal parser)
        // Expected format:
        // {
        //   "memory_limit": 268435456,
        //   "cpu_quota": 25,
        //   "namespace_flags": 63
        // }

        let s = core::str::from_utf8(data).map_err(|_| KernelError::InvalidArgument)?;

        // Very simple parsing for demo
        let mut config = AgentConfig::default();

        for line in s.lines() {
            let line = line.trim();
            if line.contains("memory_limit") {
                if let Some(value) = Self::extract_number(line) {
                    config.memory_limit = value as usize;
                }
            } else if line.contains("cpu_quota") {
                if let Some(value) = Self::extract_number(line) {
                    config.cpu_quota = value as u32;
                }
            } else if line.contains("namespace_flags") {
                if let Some(value) = Self::extract_number(line) {
                    config.namespace_flags = value as u32;
                }
            }
        }

        Ok(config)
    }

    /// Extract number from JSON line (simple parser)
    fn extract_number(line: &str) -> Option<u64> {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 2 {
            parts[1].trim().trim_end_matches(',').parse().ok()
        } else {
            None
        }
    }
}

/// Agent configuration from /proc/swarm/create
#[derive(Debug, Clone)]
struct AgentConfig {
    memory_limit: usize,
    cpu_quota: u32,
    gpu_memory: usize,
    namespace_flags: u32,
    personality: PersonalityConfig,
}

#[derive(Debug, Clone, Copy)]
struct PersonalityConfig {
    risk_tolerance: f32,
    cooperation: f32,
    exploration: f32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            memory_limit: 256 << 20, // 256MB
            cpu_quota: 25,
            gpu_memory: 0,
            namespace_flags: 0x3F,
            personality: PersonalityConfig {
                risk_tolerance: 0.5,
                cooperation: 0.7,
                exploration: 0.3,
            },
        }
    }
}

/// /proc entry structure
pub struct ProcEntry {
    /// File name
    pub name: &'static str,
    /// File mode (permissions)
    pub mode: u16,
    /// Whether it's a directory
    pub is_dir: bool,
}

/// Get all /proc/swarm entries
pub fn get_proc_entries() -> Vec<ProcEntry> {
    vec![
        ProcEntry {
            name: "status",
            mode: 0o444,
            is_dir: false,
        },
        ProcEntry {
            name: "agents",
            mode: 0o444,
            is_dir: false,
        },
        ProcEntry {
            name: "create",
            mode: 0o200,
            is_dir: false,
        },
        ProcEntry {
            name: "syscalls",
            mode: 0o444,
            is_dir: false,
        },
        ProcEntry {
            name: "resources",
            mode: 0o444,
            is_dir: false,
        },
    ]
}

/// Initialize /proc interface
pub fn init() -> KernelResult<()> {
    // In kernel:
    // 1. Create /proc/swarm directory
    // 2. Create files with appropriate operations
    // 3. Set up file operations table

    Ok(())
}

/// Cleanup /proc interface
pub fn cleanup() {
    // Remove /proc/swarm and all entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parsing() {
        let json = r#"{
            "memory_limit": 536870912,
            "cpu_quota": 50,
            "namespace_flags": 63
        }"#;

        let config = ProcOps::parse_agent_config(json.as_bytes()).unwrap();
        assert_eq!(config.memory_limit, 536870912);
        assert_eq!(config.cpu_quota, 50);
        assert_eq!(config.namespace_flags, 63);
    }

    #[test]
    fn test_number_extraction() {
        assert_eq!(
            ProcOps::extract_number("  \"memory_limit\": 12345,"),
            Some(12345)
        );
        assert_eq!(ProcOps::extract_number("  \"cpu_quota\": 50"), Some(50));
        assert_eq!(ProcOps::extract_number("invalid line"), None);
    }
}
