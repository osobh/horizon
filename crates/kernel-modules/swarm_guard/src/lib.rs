//! SwarmGuard Kernel Module
//!
//! Provides resource enforcement and namespace management for StratoSwarm containers.
//! Intercepts system calls to enforce per-agent policies and resource limits.

#![no_std]
#![feature(allocator_api)]

extern crate alloc;

pub mod agent;
pub mod cgroup;
pub mod ffi;
pub mod namespace;
pub mod policy;
pub mod proc;
pub mod syscall;

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of concurrent agents supported
pub const MAX_AGENTS: usize = 200_000;

/// Agent tracking statistics
pub struct AgentStats {
    /// Total number of active agents
    pub active_count: AtomicU64,
    /// Total number of created agents
    pub total_created: AtomicU64,
    /// Total number of destroyed agents
    pub total_destroyed: AtomicU64,
    /// Number of policy violations
    pub violations: AtomicU64,
}

impl AgentStats {
    pub const fn new() -> Self {
        Self {
            active_count: AtomicU64::new(0),
            total_created: AtomicU64::new(0),
            total_destroyed: AtomicU64::new(0),
            violations: AtomicU64::new(0),
        }
    }

    pub fn increment_active(&self) {
        self.active_count.fetch_add(1, Ordering::Relaxed);
        self.total_created.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_active(&self) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        self.total_destroyed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_violation(&self) {
        self.violations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Global agent statistics
pub static AGENT_STATS: AgentStats = AgentStats::new();

/// Result type for kernel operations
pub type KernelResult<T> = Result<T, KernelError>;

/// Kernel error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// Permission denied
    PermissionDenied,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Invalid agent ID
    InvalidAgent,
    /// Namespace creation failed
    NamespaceError,
    /// Cgroup operation failed
    CgroupError,
    /// System call interception failed
    SyscallError,
    /// Out of memory
    OutOfMemory,
    /// Invalid argument
    InvalidArgument,
}

/// Initialize the SwarmGuard module
pub fn init() -> KernelResult<()> {
    // Initialize subsystems
    syscall::init()?;
    namespace::init()?;
    cgroup::init()?;
    proc::init()?;

    Ok(())
}

/// Cleanup the SwarmGuard module
pub fn cleanup() {
    proc::cleanup();
    cgroup::cleanup();
    namespace::cleanup();
    syscall::cleanup();
}
