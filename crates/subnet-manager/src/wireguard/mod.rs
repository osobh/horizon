//! WireGuard configuration and management
//!
//! Provides WireGuard configuration generation, key management, and
//! synchronization for subnet-based network isolation.
//!
//! # Components
//!
//! - **Config**: Configuration structures and generation
//! - **Keys**: Key pair generation and management
//! - **Sync**: Configuration distribution to nodes
//! - **SubnetAware**: Multi-subnet WireGuard routing
//! - **MigrationCoord**: Zero-downtime migration coordination
//! - **Quality**: Connection quality tracking (RTT, jitter, packet loss)

mod config;
mod keys;
pub mod migration_coord;
pub mod quality;
pub mod subnet_aware;
mod sync;

pub use config::{
    InterfaceConfig, PeerConfig, WireGuardConfigGenerator, WireGuardSubnetConfig,
};
pub use keys::{KeyPair, WireGuardKeys};
pub use migration_coord::{
    MigrationCoordError, MigrationCoordStatus, MigrationCoordinator, ProbeConfig, ProbeResult,
};
pub use subnet_aware::{
    InterfaceInfo, SubnetAwareWireGuard, SubnetPeer, SubnetWireGuardError,
};
pub use sync::{ConfigChange, ConfigSyncService, SyncEvent, SyncStatus};
pub use quality::{ConnectionQuality, QualityMetrics, QualityRating};
