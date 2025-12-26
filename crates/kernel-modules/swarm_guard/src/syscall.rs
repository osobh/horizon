//! System call interception and filtering
//!
//! This module provides system call interception to enforce security
//! policies for agent containers.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{agent::AgentId, KernelError, KernelResult};

/// System call numbers for x86_64
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Syscall {
    Read = 0,
    Write = 1,
    Open = 2,
    Close = 3,
    Stat = 4,
    Fstat = 5,
    Lstat = 6,
    Poll = 7,
    Lseek = 8,
    Mmap = 9,
    Mprotect = 10,
    Munmap = 11,
    Brk = 12,
    Clone = 56,
    Fork = 57,
    Vfork = 58,
    Execve = 59,
    Exit = 60,
    Wait4 = 61,
    Kill = 62,
    Socket = 41,
    Connect = 42,
    Accept = 43,
    Sendto = 44,
    Recvfrom = 45,
    Bind = 49,
    Listen = 50,
    Openat = 257,
    Mkdirat = 258,
    Unlinkat = 263,
    Renameat = 264,
    Faccessat = 269,
}

impl Syscall {
    /// Check if this is a process creation syscall
    pub fn is_process_creation(self) -> bool {
        matches!(self, Self::Clone | Self::Fork | Self::Vfork | Self::Execve)
    }

    /// Check if this is a network syscall
    pub fn is_network(self) -> bool {
        matches!(
            self,
            Self::Socket
                | Self::Connect
                | Self::Accept
                | Self::Sendto
                | Self::Recvfrom
                | Self::Bind
                | Self::Listen
        )
    }

    /// Check if this is a file operation syscall
    pub fn is_file_operation(self) -> bool {
        matches!(
            self,
            Self::Open
                | Self::Openat
                | Self::Read
                | Self::Write
                | Self::Close
                | Self::Stat
                | Self::Fstat
                | Self::Lstat
                | Self::Mkdirat
                | Self::Unlinkat
                | Self::Renameat
        )
    }
}

/// System call interception statistics
pub struct InterceptionStats {
    /// Total syscalls intercepted
    pub total_intercepted: AtomicU64,
    /// Syscalls allowed
    pub allowed: AtomicU64,
    /// Syscalls denied
    pub denied: AtomicU64,
    /// Process creation attempts
    pub process_creation_attempts: AtomicU64,
    /// Network operation attempts
    pub network_attempts: AtomicU64,
}

impl InterceptionStats {
    pub const fn new() -> Self {
        Self {
            total_intercepted: AtomicU64::new(0),
            allowed: AtomicU64::new(0),
            denied: AtomicU64::new(0),
            process_creation_attempts: AtomicU64::new(0),
            network_attempts: AtomicU64::new(0),
        }
    }
}

/// Global interception statistics
pub static INTERCEPTION_STATS: InterceptionStats = InterceptionStats::new();

/// System call filter policy
#[derive(Debug, Clone)]
pub struct SyscallFilter {
    /// Default action (allow or deny)
    pub default_action: FilterAction,
    /// Per-syscall rules
    pub rules: Vec<SyscallRule>,
    /// Special rules for process creation
    pub process_creation_policy: ProcessCreationPolicy,
    /// Special rules for network operations
    pub network_policy: NetworkPolicy,
}

/// Filter action for syscalls
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterAction {
    /// Allow the syscall
    Allow,
    /// Deny with EPERM
    Deny,
    /// Kill the process
    Kill,
    /// Log and allow
    LogAllow,
    /// Log and deny
    LogDeny,
}

/// Rule for a specific syscall
#[derive(Debug, Clone)]
pub struct SyscallRule {
    /// System call number
    pub syscall: u32,
    /// Action to take
    pub action: FilterAction,
    /// Additional conditions
    pub conditions: Vec<FilterCondition>,
}

/// Condition for syscall filtering
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// Check argument value
    ArgEquals { arg_index: usize, value: u64 },
    /// Check argument flags
    ArgHasFlags { arg_index: usize, flags: u64 },
    /// Check file path (for file operations)
    PathMatches { pattern: Vec<u8> },
    /// Check IP address (for network operations)
    IpAddress { addr: [u8; 4], mask: [u8; 4] },
}

/// Policy for process creation syscalls
#[derive(Debug, Clone, Copy)]
pub struct ProcessCreationPolicy {
    /// Maximum number of child processes
    pub max_children: u32,
    /// Allow clone with specific flags only
    pub allowed_clone_flags: u32,
    /// Require new namespaces
    pub require_namespaces: bool,
}

impl Default for ProcessCreationPolicy {
    fn default() -> Self {
        Self {
            max_children: 100,
            allowed_clone_flags: 0,
            require_namespaces: true,
        }
    }
}

/// Policy for network operations
#[derive(Debug, Clone)]
pub struct NetworkPolicy {
    /// Allowed protocols
    pub allowed_protocols: Vec<i32>,
    /// Allowed port ranges
    pub allowed_ports: Vec<PortRange>,
    /// Blocked IP ranges
    pub blocked_ips: Vec<IpRange>,
}

/// Port range for network policy
#[derive(Debug, Clone, Copy)]
pub struct PortRange {
    pub start: u16,
    pub end: u16,
}

/// IP address range
#[derive(Debug, Clone, Copy)]
pub struct IpRange {
    pub addr: [u8; 4],
    pub mask: [u8; 4],
}

/// System call interceptor
pub struct SyscallInterceptor;

impl SyscallInterceptor {
    /// Handle intercepted system call
    pub fn handle_syscall(
        syscall_nr: u32,
        args: &[u64; 6],
        agent_id: Option<AgentId>,
    ) -> KernelResult<FilterAction> {
        INTERCEPTION_STATS
            .total_intercepted
            .fetch_add(1, Ordering::Relaxed);

        // Get filter policy for agent
        let filter = Self::get_filter_policy(agent_id)?;

        // Check for specific syscall rules
        if let Some(rule) = filter.rules.iter().find(|r| r.syscall == syscall_nr) {
            if Self::check_conditions(&rule.conditions, syscall_nr, args) {
                Self::record_action(rule.action, syscall_nr);
                return Ok(rule.action);
            }
        }

        // Apply special policies
        if let Some(syscall) = Self::syscall_from_nr(syscall_nr) {
            if syscall.is_process_creation() {
                INTERCEPTION_STATS
                    .process_creation_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return Self::check_process_creation(
                    syscall,
                    args,
                    &filter.process_creation_policy,
                );
            }

            if syscall.is_network() {
                INTERCEPTION_STATS
                    .network_attempts
                    .fetch_add(1, Ordering::Relaxed);
                return Self::check_network_operation(syscall, args, &filter.network_policy);
            }
        }

        // Apply default action
        Self::record_action(filter.default_action, syscall_nr);
        Ok(filter.default_action)
    }

    /// Get filter policy for an agent
    fn get_filter_policy(agent_id: Option<AgentId>) -> KernelResult<SyscallFilter> {
        // In real implementation, would look up agent's policy
        Ok(SyscallFilter {
            default_action: FilterAction::Allow,
            rules: Vec::new(),
            process_creation_policy: ProcessCreationPolicy::default(),
            network_policy: NetworkPolicy {
                allowed_protocols: vec![6, 17], // TCP, UDP
                allowed_ports: vec![
                    PortRange {
                        start: 1024,
                        end: 65535,
                    }, // Non-privileged ports
                ],
                blocked_ips: vec![],
            },
        })
    }

    /// Convert syscall number to enum
    fn syscall_from_nr(nr: u32) -> Option<Syscall> {
        match nr {
            0 => Some(Syscall::Read),
            1 => Some(Syscall::Write),
            2 => Some(Syscall::Open),
            3 => Some(Syscall::Close),
            56 => Some(Syscall::Clone),
            57 => Some(Syscall::Fork),
            58 => Some(Syscall::Vfork),
            59 => Some(Syscall::Execve),
            41 => Some(Syscall::Socket),
            42 => Some(Syscall::Connect),
            257 => Some(Syscall::Openat),
            _ => None,
        }
    }

    /// Check conditions for a rule
    fn check_conditions(conditions: &[FilterCondition], syscall_nr: u32, args: &[u64; 6]) -> bool {
        conditions.iter().all(|cond| match cond {
            FilterCondition::ArgEquals { arg_index, value } => args
                .get(*arg_index)
                .map(|&arg| arg == *value)
                .unwrap_or(false),
            FilterCondition::ArgHasFlags { arg_index, flags } => args
                .get(*arg_index)
                .map(|&arg| arg & flags == *flags)
                .unwrap_or(false),
            FilterCondition::PathMatches { pattern } => {
                // In kernel: would copy path from user and check
                true
            }
            FilterCondition::IpAddress { addr, mask } => {
                // In kernel: would check socket address
                true
            }
        })
    }

    /// Check process creation syscalls
    fn check_process_creation(
        syscall: Syscall,
        args: &[u64; 6],
        policy: &ProcessCreationPolicy,
    ) -> KernelResult<FilterAction> {
        match syscall {
            Syscall::Clone => {
                let flags = args[0] as u32;

                // Check if namespaces are required
                if policy.require_namespaces {
                    const NAMESPACE_FLAGS: u32 = 0x3F000000;
                    if flags & NAMESPACE_FLAGS == 0 {
                        return Ok(FilterAction::Deny);
                    }
                }

                // Check allowed flags
                if policy.allowed_clone_flags != 0 && flags & !policy.allowed_clone_flags != 0 {
                    return Ok(FilterAction::Deny);
                }

                Ok(FilterAction::Allow)
            }
            Syscall::Fork | Syscall::Vfork => {
                // These create processes without namespaces
                if policy.require_namespaces {
                    Ok(FilterAction::Deny)
                } else {
                    Ok(FilterAction::Allow)
                }
            }
            Syscall::Execve => {
                // Check executable path if needed
                Ok(FilterAction::Allow)
            }
            _ => Ok(FilterAction::Allow),
        }
    }

    /// Check network operations
    fn check_network_operation(
        syscall: Syscall,
        args: &[u64; 6],
        policy: &NetworkPolicy,
    ) -> KernelResult<FilterAction> {
        match syscall {
            Syscall::Socket => {
                let protocol = args[2] as i32;
                if !policy.allowed_protocols.contains(&protocol) {
                    return Ok(FilterAction::Deny);
                }
                Ok(FilterAction::Allow)
            }
            Syscall::Connect | Syscall::Bind => {
                // In kernel: would check address and port
                Ok(FilterAction::Allow)
            }
            _ => Ok(FilterAction::Allow),
        }
    }

    /// Record action taken
    fn record_action(action: FilterAction, syscall_nr: u32) {
        match action {
            FilterAction::Allow | FilterAction::LogAllow => {
                INTERCEPTION_STATS.allowed.fetch_add(1, Ordering::Relaxed);
            }
            FilterAction::Deny | FilterAction::LogDeny | FilterAction::Kill => {
                INTERCEPTION_STATS.denied.fetch_add(1, Ordering::Relaxed);
                crate::AGENT_STATS.record_violation();
            }
        }
    }
}

/// System call table hooks
pub struct SyscallTable {
    /// Original system call table pointer
    original_table: *const usize,
    /// Our hooked table
    hooked_table: Vec<usize>,
}

/// Initialize syscall interception
pub fn init() -> KernelResult<()> {
    // In kernel:
    // 1. Find system call table
    // 2. Make it writable
    // 3. Replace entries with our hooks
    // 4. Make it read-only again

    Ok(())
}

/// Cleanup syscall interception
pub fn cleanup() {
    // Restore original system call table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_classification() {
        assert!(Syscall::Clone.is_process_creation());
        assert!(Syscall::Fork.is_process_creation());
        assert!(Syscall::Execve.is_process_creation());
        assert!(!Syscall::Read.is_process_creation());

        assert!(Syscall::Socket.is_network());
        assert!(Syscall::Connect.is_network());
        assert!(!Syscall::Open.is_network());

        assert!(Syscall::Open.is_file_operation());
        assert!(Syscall::Read.is_file_operation());
        assert!(!Syscall::Socket.is_file_operation());
    }

    #[test]
    fn test_filter_conditions() {
        let args = [10, 20, 30, 40, 50, 60];

        let cond = FilterCondition::ArgEquals {
            arg_index: 1,
            value: 20,
        };
        assert!(SyscallInterceptor::check_conditions(&[cond], 0, &args));

        let cond = FilterCondition::ArgEquals {
            arg_index: 1,
            value: 21,
        };
        assert!(!SyscallInterceptor::check_conditions(&[cond], 0, &args));

        let cond = FilterCondition::ArgHasFlags {
            arg_index: 2,
            value: 0x10,
        };
        assert!(SyscallInterceptor::check_conditions(&[cond], 0, &args)); // 30 & 0x10 = 0x10
    }
}
