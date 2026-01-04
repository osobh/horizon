//! WireGuard configuration and management
//!
//! Provides WireGuard configuration generation, key management, and
//! synchronization for subnet-based network isolation.
//!
//! # Components
//!
//! - **Backend**: Multiple WireGuard backend implementations (command, netlink, userspace)
//! - **Config**: Configuration structures and generation
//! - **Keys**: Key pair generation and management (using x25519-dalek)
//! - **Sync**: Configuration distribution to nodes
//! - **SubnetAware**: Multi-subnet WireGuard routing
//! - **MigrationCoord**: Zero-downtime migration coordination
//! - **Quality**: Connection quality tracking (RTT, jitter, packet loss)
//! - **Router**: Quality-aware route selection
//!
//! # Backend Selection
//!
//! The module supports multiple WireGuard backends:
//!
//! 1. **Command** (default) - Uses `wg` and `ip` commands, most portable
//! 2. **Netlink** (Linux only) - Direct kernel interface, fastest
//! 3. **Userspace** - boringtun, no kernel module needed
//!
//! ```ignore
//! use subnet_manager::wireguard::backend::AutoSelectBackend;
//!
//! let backend = AutoSelectBackend::new();
//! if let Some(wg) = backend.select() {
//!     println!("Using {} backend", wg.backend_type());
//! }
//! ```

pub mod backend;
mod config;
mod keys;
pub mod migration_coord;
pub mod quality;
pub mod router;
pub mod subnet_aware;
mod sync;

pub use backend::{AutoSelectBackend, BackendType, InterfaceStats, PeerStats, WireGuardBackend};
pub use config::{InterfaceConfig, PeerConfig, WireGuardConfigGenerator, WireGuardSubnetConfig};
pub use keys::{KeyPair, WireGuardKeys};
pub use migration_coord::{
    InMemoryNodeRegistry, MigrationCoordError, MigrationCoordStatus, MigrationCoordinator,
    NodeInfo, NodeRegistry, ProbeConfig, ProbeResult,
};
pub use quality::{ConnectionQuality, QualityMetrics, QualityRating};
pub use router::{
    PeerRoute, ProbeConfig as RouterProbeConfig, ProbeResult as RouterProbeResult,
    QualityAwareRouter, QualityProber, QualityThresholds, RouteStrategy, RouterError, RouterStats,
};
pub use subnet_aware::{InterfaceInfo, SubnetAwareWireGuard, SubnetPeer, SubnetWireGuardError};
pub use sync::{
    ConfigChange, ConfigSyncService, NodeConfigResponse, NodeHttpClient, NodePeerConfig,
    NodeWireGuardConfigRequest, SyncEvent, SyncResult, SyncStatus,
};
