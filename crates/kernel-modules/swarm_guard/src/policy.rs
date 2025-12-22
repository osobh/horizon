//! Security and resource policy management
//!
//! This module defines and enforces security policies for agent containers,
//! integrating with all other SwarmGuard subsystems.

use alloc::string::String;
use alloc::vec::Vec;

use crate::agent::{AgentId, DeviceAccess, ResourceLimits, SecurityPolicy};
use crate::cgroup::CgroupConfig;
use crate::namespace::NamespaceSetup;
use crate::syscall::{
    FilterAction, NetworkPolicy, ProcessCreationPolicy, SyscallFilter, SyscallRule,
};
use crate::{KernelError, KernelResult};

/// Complete policy for an agent
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    /// Unique policy identifier
    pub id: PolicyId,
    /// Policy name for identification
    pub name: String,
    /// Resource limits
    pub resources: ResourcePolicy,
    /// Security restrictions
    pub security: SecurityPolicy,
    /// Namespace requirements
    pub namespaces: NamespacePolicy,
    /// System call filtering
    pub syscalls: SyscallPolicy,
    /// Network access control
    pub network: NetworkAccessPolicy,
    /// Evolution constraints
    pub evolution: EvolutionPolicy,
}

/// Policy identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyId(pub u64);

/// Resource allocation policy
#[derive(Debug, Clone)]
pub struct ResourcePolicy {
    /// Memory limits
    pub memory: MemoryPolicy,
    /// CPU limits
    pub cpu: CpuPolicy,
    /// GPU allocation
    pub gpu: GpuPolicy,
    /// Storage limits
    pub storage: StoragePolicy,
}

/// Memory allocation policy
#[derive(Debug, Clone, Copy)]
pub struct MemoryPolicy {
    /// Minimum memory allocation
    pub min_bytes: usize,
    /// Maximum memory allocation
    pub max_bytes: usize,
    /// Whether agent can use swap
    pub allow_swap: bool,
    /// Memory growth strategy
    pub growth_strategy: MemoryGrowthStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryGrowthStrategy {
    /// Fixed allocation
    Fixed,
    /// Grow on demand up to max
    Dynamic,
    /// Elastic with pressure-based shrinking
    Elastic,
}

/// CPU allocation policy
#[derive(Debug, Clone, Copy)]
pub struct CpuPolicy {
    /// Minimum CPU percentage
    pub min_percent: u32,
    /// Maximum CPU percentage
    pub max_percent: u32,
    /// CPU affinity requirements
    pub affinity: CpuAffinity,
    /// Scheduling class
    pub sched_class: SchedulingClass,
}

#[derive(Debug, Clone, Copy)]
pub enum CpuAffinity {
    /// No affinity requirements
    None,
    /// Prefer NUMA local
    NumaLocal,
    /// Pin to specific CPUs
    Pinned { cpu_mask: u64 },
}

#[derive(Debug, Clone, Copy)]
pub enum SchedulingClass {
    /// Normal scheduling
    Normal,
    /// Real-time scheduling
    Realtime { priority: u32 },
    /// Batch processing
    Batch,
    /// Idle only
    Idle,
}

/// GPU allocation policy
#[derive(Debug, Clone)]
pub struct GpuPolicy {
    /// Whether GPU access is allowed
    pub allowed: bool,
    /// Maximum GPU memory in bytes
    pub max_memory_bytes: usize,
    /// Allowed GPU devices
    pub allowed_devices: Vec<u32>,
    /// Compute capability requirements
    pub min_compute_capability: (u32, u32),
}

/// Storage access policy
#[derive(Debug, Clone)]
pub struct StoragePolicy {
    /// Maximum storage in bytes
    pub max_bytes: usize,
    /// I/O bandwidth limits
    pub io_limits: IoLimits,
    /// Allowed mount points
    pub allowed_mounts: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct IoLimits {
    pub read_bps: Option<u64>,
    pub write_bps: Option<u64>,
    pub read_iops: Option<u64>,
    pub write_iops: Option<u64>,
}

/// Namespace isolation policy
#[derive(Debug, Clone)]
pub struct NamespacePolicy {
    /// Required namespace types
    pub required_namespaces: u32,
    /// User namespace configuration
    pub user_mapping: UserMappingPolicy,
    /// Network namespace configuration
    pub network_isolation: NetworkIsolation,
}

#[derive(Debug, Clone, Copy)]
pub enum UserMappingPolicy {
    /// Map to nobody
    Nobody,
    /// Map to specific UID/GID
    Fixed { uid: u32, gid: u32 },
    /// Dynamic allocation
    Dynamic,
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkIsolation {
    /// Complete isolation
    None,
    /// Bridge networking
    Bridge,
    /// Host networking (dangerous)
    Host,
    /// Custom networking
    Custom,
}

/// System call filtering policy
#[derive(Debug, Clone)]
pub struct SyscallPolicy {
    /// Default action for unlisted syscalls
    pub default_action: FilterAction,
    /// Specific syscall rules
    pub rules: Vec<SyscallRule>,
    /// Process creation policy
    pub process_creation: ProcessCreationPolicy,
}

/// Network access control policy
#[derive(Debug, Clone)]
pub struct NetworkAccessPolicy {
    /// Ingress rules
    pub ingress: Vec<NetworkRule>,
    /// Egress rules
    pub egress: Vec<NetworkRule>,
    /// Rate limiting
    pub rate_limits: RateLimits,
}

#[derive(Debug, Clone)]
pub struct NetworkRule {
    /// Protocol (TCP=6, UDP=17, etc)
    pub protocol: Option<u8>,
    /// Source/destination ports
    pub ports: Option<PortRange>,
    /// Source/destination IPs
    pub ips: Option<IpRange>,
    /// Action to take
    pub action: NetworkAction,
}

#[derive(Debug, Clone, Copy)]
pub struct PortRange {
    pub start: u16,
    pub end: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct IpRange {
    pub addr: [u8; 4],
    pub mask: [u8; 4],
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkAction {
    Allow,
    Deny,
    RateLimit { bps: u64 },
}

#[derive(Debug, Clone, Copy)]
pub struct RateLimits {
    /// Maximum bandwidth in bytes/sec
    pub max_bandwidth_bps: u64,
    /// Maximum packets per second
    pub max_pps: u64,
    /// Maximum connections
    pub max_connections: u32,
}

/// Evolution policy constraints
#[derive(Debug, Clone)]
pub struct EvolutionPolicy {
    /// Whether agent can evolve
    pub allow_evolution: bool,
    /// Maximum mutation rate
    pub max_mutation_rate: f32,
    /// Allowed mutation types
    pub allowed_mutations: Vec<MutationType>,
    /// Fitness constraints
    pub fitness_requirements: FitnessRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationType {
    /// Parameter tuning only
    Parameters,
    /// Algorithm selection
    Algorithms,
    /// Code generation
    CodeGeneration,
    /// Architecture changes
    Architecture,
}

#[derive(Debug, Clone)]
pub struct FitnessRequirements {
    /// Minimum fitness score to survive
    pub min_fitness: f32,
    /// Required fitness improvement per generation
    pub improvement_rate: f32,
    /// Maximum generations without improvement
    pub stagnation_limit: u32,
}

/// Policy manager
pub struct PolicyManager {
    /// All registered policies
    policies: Vec<AgentPolicy>,
    /// Default policy for new agents
    default_policy: PolicyId,
}

impl PolicyManager {
    /// Create a new policy manager
    pub fn new() -> Self {
        let mut manager = Self {
            policies: Vec::new(),
            default_policy: PolicyId(0),
        };

        // Create default policies
        manager.create_default_policies();

        manager
    }

    /// Create default policies
    fn create_default_policies(&mut self) {
        // Restrictive default policy
        let restrictive = AgentPolicy {
            id: PolicyId(1),
            name: String::from("restrictive"),
            resources: ResourcePolicy {
                memory: MemoryPolicy {
                    min_bytes: 64 << 20,  // 64MB
                    max_bytes: 256 << 20, // 256MB
                    allow_swap: false,
                    growth_strategy: MemoryGrowthStrategy::Fixed,
                },
                cpu: CpuPolicy {
                    min_percent: 1,
                    max_percent: 25,
                    affinity: CpuAffinity::None,
                    sched_class: SchedulingClass::Normal,
                },
                gpu: GpuPolicy {
                    allowed: false,
                    max_memory_bytes: 0,
                    allowed_devices: vec![],
                    min_compute_capability: (0, 0),
                },
                storage: StoragePolicy {
                    max_bytes: 1 << 30, // 1GB
                    io_limits: IoLimits {
                        read_bps: Some(10 << 20),  // 10MB/s
                        write_bps: Some(10 << 20), // 10MB/s
                        read_iops: Some(1000),
                        write_iops: Some(1000),
                    },
                    allowed_mounts: vec![],
                },
            },
            security: SecurityPolicy {
                allowed_syscalls: vec![0, 1, 2, 3], // read, write, open, close only
                allowed_devices: vec![
                    DeviceAccess {
                        major: 1,
                        minor: 3,
                        read: true,
                        write: true,
                        mknod: false,
                    },
                    DeviceAccess {
                        major: 1,
                        minor: 5,
                        read: true,
                        write: true,
                        mknod: false,
                    },
                ],
                security_context: None,
                capabilities: 0,
            },
            namespaces: NamespacePolicy {
                required_namespaces: 0x3F, // All namespaces
                user_mapping: UserMappingPolicy::Nobody,
                network_isolation: NetworkIsolation::None,
            },
            syscalls: SyscallPolicy {
                default_action: FilterAction::Deny,
                rules: vec![],
                process_creation: ProcessCreationPolicy {
                    max_children: 0,
                    allowed_clone_flags: 0,
                    require_namespaces: true,
                },
            },
            network: NetworkAccessPolicy {
                ingress: vec![],
                egress: vec![],
                rate_limits: RateLimits {
                    max_bandwidth_bps: 1 << 20, // 1MB/s
                    max_pps: 1000,
                    max_connections: 10,
                },
            },
            evolution: EvolutionPolicy {
                allow_evolution: false,
                max_mutation_rate: 0.0,
                allowed_mutations: vec![],
                fitness_requirements: FitnessRequirements {
                    min_fitness: 0.0,
                    improvement_rate: 0.0,
                    stagnation_limit: 0,
                },
            },
        };

        self.policies.push(restrictive);
        self.default_policy = PolicyId(1);
    }

    /// Get policy by ID
    pub fn get_policy(&self, id: PolicyId) -> Option<&AgentPolicy> {
        self.policies.iter().find(|p| p.id == id)
    }

    /// Get default policy
    pub fn get_default_policy(&self) -> &AgentPolicy {
        self.get_policy(self.default_policy)
            .expect("Default policy must exist")
    }

    /// Apply policy to get concrete configurations
    pub fn apply_policy(
        &self,
        policy_id: PolicyId,
        requested: &AgentRequest,
    ) -> KernelResult<AgentConfiguration> {
        let policy = self
            .get_policy(policy_id)
            .ok_or(KernelError::InvalidArgument)?;

        // Validate request against policy
        self.validate_request(policy, requested)?;

        // Generate configuration
        Ok(AgentConfiguration {
            resources: self.apply_resource_policy(&policy.resources, &requested.resources)?,
            namespaces: self.apply_namespace_policy(&policy.namespaces),
            cgroups: self.generate_cgroup_config(&policy.resources),
            syscalls: policy.syscalls.clone().into(),
            security: policy.security.clone(),
        })
    }

    /// Validate request against policy
    fn validate_request(&self, policy: &AgentPolicy, requested: &AgentRequest) -> KernelResult<()> {
        // Check memory limits
        if requested.resources.memory > policy.resources.memory.max_bytes {
            return Err(KernelError::ResourceLimitExceeded);
        }

        // Check CPU limits
        if requested.resources.cpu_percent > policy.resources.cpu.max_percent {
            return Err(KernelError::ResourceLimitExceeded);
        }

        // Check GPU access
        if requested.resources.gpu_memory > 0 && !policy.resources.gpu.allowed {
            return Err(KernelError::PermissionDenied);
        }

        Ok(())
    }

    /// Apply resource policy
    fn apply_resource_policy(
        &self,
        policy: &ResourcePolicy,
        requested: &RequestedResources,
    ) -> KernelResult<ResourceLimits> {
        Ok(ResourceLimits {
            memory_bytes: requested.memory.min(policy.memory.max_bytes),
            cpu_quota: requested.cpu_percent.min(policy.cpu.max_percent),
            gpu_memory_bytes: if policy.gpu.allowed {
                requested.gpu_memory.min(policy.gpu.max_memory_bytes)
            } else {
                0
            },
            max_fds: 1024,         // Standard limit
            network_bps: 10 << 20, // 10MB/s
        })
    }

    /// Apply namespace policy
    fn apply_namespace_policy(&self, policy: &NamespacePolicy) -> NamespaceSetup {
        NamespaceSetup {
            flags: policy.required_namespaces,
            uid_map: match policy.user_mapping {
                UserMappingPolicy::Nobody => vec![crate::namespace::IdMap {
                    inside_id: 0,
                    outside_id: 65534, // nobody
                    count: 1,
                }],
                UserMappingPolicy::Fixed { uid, gid } => vec![crate::namespace::IdMap {
                    inside_id: 0,
                    outside_id: uid,
                    count: 1,
                }],
                UserMappingPolicy::Dynamic => vec![crate::namespace::IdMap {
                    inside_id: 0,
                    outside_id: 1000, // Default
                    count: 1,
                }],
            },
            gid_map: vec![crate::namespace::IdMap {
                inside_id: 0,
                outside_id: 1000,
                count: 1,
            }],
            hostname: None,
            root_dir: None,
        }
    }

    /// Generate cgroup configuration from resource policy
    fn generate_cgroup_config(&self, resources: &ResourcePolicy) -> CgroupConfig {
        CgroupConfig {
            cpu: crate::cgroup::CpuConfig::from_percentage(resources.cpu.max_percent),
            memory: crate::cgroup::MemoryConfig {
                limit_bytes: resources.memory.max_bytes,
                soft_limit_bytes: Some(resources.memory.min_bytes),
                swap_limit_bytes: if resources.memory.allow_swap {
                    None
                } else {
                    Some(0)
                },
                kernel_memory: true,
            },
            io: Some(crate::cgroup::IoConfig {
                read_bps: resources.storage.io_limits.read_bps,
                write_bps: resources.storage.io_limits.write_bps,
                read_iops: resources.storage.io_limits.read_iops,
                write_iops: resources.storage.io_limits.write_iops,
                device_weights: vec![],
            }),
            max_pids: Some(1000),
        }
    }
}

/// Agent creation request
#[derive(Debug, Clone)]
pub struct AgentRequest {
    /// Requested resources
    pub resources: RequestedResources,
    /// Requested capabilities
    pub capabilities: Vec<String>,
    /// Network requirements
    pub network: NetworkRequirements,
}

#[derive(Debug, Clone, Copy)]
pub struct RequestedResources {
    pub memory: usize,
    pub cpu_percent: u32,
    pub gpu_memory: usize,
}

#[derive(Debug, Clone)]
pub struct NetworkRequirements {
    pub needs_internet: bool,
    pub exposed_ports: Vec<u16>,
}

/// Complete agent configuration after policy application
#[derive(Debug, Clone)]
pub struct AgentConfiguration {
    pub resources: ResourceLimits,
    pub namespaces: NamespaceSetup,
    pub cgroups: CgroupConfig,
    pub syscalls: SyscallFilter,
    pub security: SecurityPolicy,
}

impl From<SyscallPolicy> for SyscallFilter {
    fn from(policy: SyscallPolicy) -> Self {
        SyscallFilter {
            default_action: policy.default_action,
            rules: policy.rules,
            process_creation_policy: policy.process_creation,
            network_policy: NetworkPolicy {
                allowed_protocols: vec![6, 17], // TCP, UDP
                allowed_ports: vec![],
                blocked_ips: vec![],
            },
        }
    }
}
