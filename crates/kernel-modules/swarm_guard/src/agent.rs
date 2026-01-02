//! Agent management and tracking
//!
//! This module provides the core agent data structures and management functions
//! for tracking container agents within the kernel.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{KernelError, KernelResult, MAX_AGENTS};

/// Unique agent identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

impl AgentId {
    /// Generate a new unique agent ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        AgentId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Agent personality traits that affect behavior
#[derive(Debug, Clone, Copy)]
pub struct Personality {
    /// Risk tolerance (0.0 = conservative, 1.0 = aggressive)
    pub risk_tolerance: f32,
    /// Cooperation level (0.0 = selfish, 1.0 = collaborative)
    pub cooperation: f32,
    /// Exploration tendency (0.0 = exploit, 1.0 = explore)
    pub exploration: f32,
    /// Efficiency focus (0.0 = thorough, 1.0 = fast)
    pub efficiency_focus: f32,
    /// Stability preference (0.0 = dynamic, 1.0 = stable)
    pub stability_preference: f32,
}

impl Default for Personality {
    fn default() -> Self {
        Self {
            risk_tolerance: 0.5,
            cooperation: 0.7,
            exploration: 0.3,
            efficiency_focus: 0.6,
            stability_preference: 0.8,
        }
    }
}

/// Resource limits for an agent
#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// Memory limit in bytes
    pub memory_bytes: usize,
    /// CPU quota as percentage (0-100)
    pub cpu_quota: u32,
    /// GPU memory allocation in bytes (0 if no GPU)
    pub gpu_memory_bytes: usize,
    /// Maximum number of file descriptors
    pub max_fds: u32,
    /// Network bandwidth limit in bytes/sec
    pub network_bps: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_bytes: 256 << 20, // 256MB
            cpu_quota: 25,           // 25%
            gpu_memory_bytes: 0,     // No GPU by default
            max_fds: 1024,           // Standard limit
            network_bps: 10 << 20,   // 10MB/s
        }
    }
}

/// Namespace configuration for an agent
#[derive(Debug, Clone, Copy)]
pub struct NamespaceConfig {
    /// Bitmask of namespace types
    pub flags: u32,
    /// User namespace UID mapping
    pub uid_map: (u32, u32),
    /// User namespace GID mapping  
    pub gid_map: (u32, u32),
}

impl Default for NamespaceConfig {
    fn default() -> Self {
        Self {
            flags: 0x3F,           // All namespaces
            uid_map: (1000, 1000), // Map to UID 1000
            gid_map: (1000, 1000), // Map to GID 1000
        }
    }
}

/// Security policy for an agent
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allowed system calls (empty = all allowed)
    pub allowed_syscalls: Vec<u32>,
    /// Allowed device access
    pub allowed_devices: Vec<DeviceAccess>,
    /// SELinux/AppArmor context
    pub security_context: Option<String>,
    /// Capability bounding set
    pub capabilities: u64,
}

/// Device access permission
#[derive(Debug, Clone, Copy)]
pub struct DeviceAccess {
    pub major: u32,
    pub minor: u32,
    pub read: bool,
    pub write: bool,
    pub mknod: bool,
}

/// Agent state and configuration
pub struct Agent {
    /// Unique identifier
    pub id: AgentId,
    /// Process ID (0 if not yet created)
    pub pid: u32,
    /// Parent agent ID (if spawned from another agent)
    pub parent_id: Option<AgentId>,
    /// Personality traits
    pub personality: Personality,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Namespace configuration
    pub namespaces: NamespaceConfig,
    /// Security policy
    pub security: SecurityPolicy,
    /// Whether the agent is active
    pub active: AtomicBool,
    /// Creation timestamp (nanoseconds since boot)
    pub created_at: u64,
    /// Last activity timestamp
    pub last_active: AtomicU64,
    /// Trust score (0.0 = untrusted, 1.0 = fully trusted)
    pub trust_score: f32,
}

impl Agent {
    /// Create a new agent with default configuration
    pub fn new() -> Self {
        Self {
            id: AgentId::new(),
            pid: 0,
            parent_id: None,
            personality: Personality::default(),
            limits: ResourceLimits::default(),
            namespaces: NamespaceConfig::default(),
            security: SecurityPolicy {
                allowed_syscalls: Vec::new(), // All allowed by default
                allowed_devices: Self::default_devices(),
                security_context: None,
                capabilities: 0, // No special capabilities
            },
            active: AtomicBool::new(false),
            created_at: Self::get_timestamp(),
            last_active: AtomicU64::new(0),
            trust_score: 0.5, // Neutral trust initially
        }
    }

    /// Get default allowed devices
    fn default_devices() -> Vec<DeviceAccess> {
        vec![
            DeviceAccess {
                major: 1,
                minor: 3,
                read: true,
                write: true,
                mknod: false,
            }, // /dev/null
            DeviceAccess {
                major: 1,
                minor: 5,
                read: true,
                write: true,
                mknod: false,
            }, // /dev/zero
            DeviceAccess {
                major: 1,
                minor: 7,
                read: true,
                write: true,
                mknod: false,
            }, // /dev/full
            DeviceAccess {
                major: 1,
                minor: 8,
                read: true,
                write: false,
                mknod: false,
            }, // /dev/random
            DeviceAccess {
                major: 1,
                minor: 9,
                read: true,
                write: false,
                mknod: false,
            }, // /dev/urandom
            DeviceAccess {
                major: 5,
                minor: 0,
                read: true,
                write: true,
                mknod: false,
            }, // /dev/tty
            DeviceAccess {
                major: 5,
                minor: 2,
                read: true,
                write: true,
                mknod: false,
            }, // /dev/ptmx
        ]
    }

    /// Get current timestamp in nanoseconds
    fn get_timestamp() -> u64 {
        // In kernel, this would use ktime_get_ns()
        // For now, return a mock value
        0
    }

    /// Check if a system call is allowed for this agent
    pub fn is_syscall_allowed(&self, syscall_nr: u32) -> bool {
        if self.security.allowed_syscalls.is_empty() {
            // Empty list means all allowed
            true
        } else {
            self.security.allowed_syscalls.contains(&syscall_nr)
        }
    }

    /// Check if device access is allowed
    pub fn is_device_allowed(&self, major: u32, minor: u32, write: bool) -> bool {
        self.security
            .allowed_devices
            .iter()
            .any(|dev| dev.major == major && dev.minor == minor && (!write || dev.write))
    }

    /// Update trust score based on behavior
    pub fn update_trust(&mut self, delta: f32) {
        self.trust_score = (self.trust_score + delta).clamp(0.0, 1.0);
    }

    /// Mark agent as active with given PID
    pub fn activate(&self, pid: u32) -> KernelResult<()> {
        if self.active.load(Ordering::Acquire) {
            return Err(KernelError::InvalidAgent);
        }

        // In real kernel, would set up namespaces and cgroups here
        self.active.store(true, Ordering::Release);
        self.last_active
            .store(Self::get_timestamp(), Ordering::Relaxed);

        Ok(())
    }

    /// Deactivate the agent
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Release);
    }
}

/// Global agent registry
pub struct AgentRegistry {
    /// Array of agent slots
    agents: Vec<Option<Agent>>,
    /// Number of active agents
    active_count: AtomicU64,
}

impl AgentRegistry {
    /// Create a new agent registry
    pub fn new() -> Self {
        let mut agents = Vec::with_capacity(MAX_AGENTS);
        agents.resize_with(MAX_AGENTS, || None);

        Self {
            agents,
            active_count: AtomicU64::new(0),
        }
    }

    /// Register a new agent
    pub fn register(&mut self, mut agent: Agent) -> KernelResult<AgentId> {
        // Find free slot
        for slot in self.agents.iter_mut() {
            if slot.is_none() {
                let id = agent.id;
                *slot = Some(agent);
                self.active_count.fetch_add(1, Ordering::Relaxed);

                crate::AGENT_STATS.increment_active();

                return Ok(id);
            }
        }

        Err(KernelError::ResourceLimitExceeded)
    }

    /// Get an agent by ID
    pub fn get(&self, id: AgentId) -> Option<&Agent> {
        self.agents
            .iter()
            .filter_map(|slot| slot.as_ref())
            .find(|agent| agent.id == id)
    }

    /// Get a mutable reference to an agent
    pub fn get_mut(&mut self, id: AgentId) -> Option<&mut Agent> {
        self.agents
            .iter_mut()
            .filter_map(|slot| slot.as_mut())
            .find(|agent| agent.id == id)
    }

    /// Remove an agent
    pub fn unregister(&mut self, id: AgentId) -> KernelResult<()> {
        for slot in self.agents.iter_mut() {
            if let Some(agent) = slot {
                if agent.id == id {
                    agent.deactivate();
                    *slot = None;
                    self.active_count.fetch_sub(1, Ordering::Relaxed);

                    crate::AGENT_STATS.decrement_active();

                    return Ok(());
                }
            }
        }

        Err(KernelError::InvalidAgent)
    }

    /// Get total number of active agents
    pub fn active_count(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Iterate over all active agents
    pub fn iter(&self) -> impl Iterator<Item = &Agent> {
        self.agents
            .iter()
            .filter_map(|slot| slot.as_ref())
            .filter(|agent| agent.active.load(Ordering::Relaxed))
    }
}
