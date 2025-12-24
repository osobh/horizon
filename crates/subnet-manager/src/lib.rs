//! WireGuard Subnet Manager
//!
//! Manages multi-dimensional subnet segmentation for heterogeneous node registration:
//! - Tenant isolation (per-organization subnets)
//! - Node type segregation (DataCenter/Workstation/Laptop/Edge)
//! - Geographic/regional subnets
//! - Resource pool isolation
//!
//! Features:
//! - CIDR-based subnet allocation
//! - Policy-based automatic node assignment
//! - Zero-downtime node migration
//! - Cross-subnet routing
//! - WireGuard configuration generation

pub mod allocator;
pub mod api;
pub mod error;
pub mod events;
pub mod integration;
pub mod migration;
pub mod models;
pub mod policy_engine;
pub mod service;
pub mod wireguard;

// Re-export core types
pub use error::{Error, Result};
pub use models::{
    AssignmentPolicy, CrossSubnetRoute, PolicyRule, RouteDirection, Subnet, SubnetAssignment,
    SubnetPurpose, SubnetStatus, SubnetTemplate,
};
pub use service::SubnetManager;
pub use wireguard::{
    ConfigChange, ConfigSyncService, InterfaceConfig, KeyPair, PeerConfig, SyncEvent, SyncStatus,
    WireGuardConfigGenerator, WireGuardKeys, WireGuardSubnetConfig,
    // Subnet-aware WireGuard
    InterfaceInfo, SubnetAwareWireGuard, SubnetPeer, SubnetWireGuardError,
    // Migration coordination
    MigrationCoordError, MigrationCoordStatus, MigrationCoordinator, ProbeConfig, ProbeResult,
};
pub use migration::{
    BulkMigrationPlan, Migration, MigrationConstraint, MigrationExecutor, MigrationHandle,
    MigrationOptions, MigrationPlan, MigrationPlanner, MigrationProgress, MigrationReason,
    MigrationStateMachine, MigrationStats, MigrationStatus, MigrationStep,
};
pub use api::{create_router, start_server, ApiServerConfig, AppState};
pub use events::{
    SubnetEventPublisher, SubnetMessage, EventTransport, InMemoryTransport,
    SUBNET_LIFECYCLE, SUBNET_TOPOLOGY, SUBNET_ASSIGNMENTS, SUBNET_ROUTES, SUBNET_WIREGUARD,
};
pub use integration::{
    SubnetMeshOrchestrator, NodeSubnetInfo, SubnetAffinity, NodeClassMapper,
    ClusterNodeInfo, MeshCallbacks, OrchestrationError,
};

/// Master address space configuration
pub mod address_space {
    use ipnet::Ipv4Net;
    use std::net::Ipv4Addr;

    /// Master address space: 10.0.0.0/8
    pub const MASTER_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 0, 0, 0), 8);

    /// Tenant isolation: 10.100.0.0/12 (4096 /20 subnets, 4094 hosts each)
    pub const TENANT_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 100, 0, 0), 12);
    pub const TENANT_SUBNET_PREFIX: u8 = 20;

    /// Node type segregation: 10.112.0.0/12
    pub const NODE_TYPE_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 112, 0, 0), 12);

    /// DataCenter nodes: 10.112.0.0/16
    pub const DATACENTER_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 112, 0, 0), 16);

    /// Workstation nodes: 10.113.0.0/16
    pub const WORKSTATION_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 113, 0, 0), 16);

    /// Laptop nodes: 10.114.0.0/16
    pub const LAPTOP_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 114, 0, 0), 16);

    /// Edge devices: 10.115.0.0/16
    pub const EDGE_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 115, 0, 0), 16);

    /// Geographic/regional: 10.128.0.0/12 (1024 /18 regions, 16K hosts each)
    pub const GEOGRAPHIC_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 128, 0, 0), 12);
    pub const GEOGRAPHIC_SUBNET_PREFIX: u8 = 18;

    /// Resource pool isolation: 10.144.0.0/12
    pub const RESOURCE_POOL_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 144, 0, 0), 12);
    pub const RESOURCE_POOL_SUBNET_PREFIX: u8 = 20;

    /// Reserved for legacy hpc-channels compatibility: 10.200.0.0/16
    pub const LEGACY_SPACE: Ipv4Net = Ipv4Net::new_assert(Ipv4Addr::new(10, 200, 0, 0), 16);
}
