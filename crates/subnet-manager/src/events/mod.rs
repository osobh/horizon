//! Event-driven subnet topology synchronization
//!
//! This module provides integration with hpc-channels for broadcasting
//! subnet-related events across the HPC platform.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      SubnetEventPublisher                        │
//! │  ┌───────────────┐    ┌──────────────┐    ┌─────────────────┐  │
//! │  │ subnet_created│───►│ SubnetMessage│───►│ EventTransport  │  │
//! │  │ node_assigned │    │ (serialize)  │    │ (hpc-channels)  │  │
//! │  │ route_created │    └──────────────┘    └─────────────────┘  │
//! │  └───────────────┘                                              │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Well-Known Channels                          │
//! │  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
//! │  │SUBNET_LIFECYCLE│  │SUBNET_ASSIGNMENTS│ │SUBNET_WIREGUARD  │  │
//! │  │SUBNET_TOPOLOGY │  │SUBNET_ROUTES     │ │SUBNET_POLICIES   │  │
//! │  └────────────────┘  └─────────────────┘  └──────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Channel Descriptions
//!
//! | Channel | Purpose |
//! |---------|---------|
//! | `SUBNET_LIFECYCLE` | Subnet creation, deletion, status changes |
//! | `SUBNET_TOPOLOGY` | Full topology snapshots for sync/reconciliation |
//! | `SUBNET_ASSIGNMENTS` | Node assignments, migrations |
//! | `SUBNET_ROUTES` | Cross-subnet route events |
//! | `SUBNET_WIREGUARD` | WireGuard config updates, key rotations |
//! | `SUBNET_POLICIES` | Policy creation, updates, deletion |
//!
//! # Usage
//!
//! ```rust,ignore
//! use subnet_manager::events::{SubnetEventPublisher, InMemoryTransport};
//! use std::sync::Arc;
//!
//! // Create publisher with transport
//! let transport = Arc::new(InMemoryTransport::new());
//! let publisher = SubnetEventPublisher::with_transport(transport);
//!
//! // Publish events
//! publisher.subnet_created(&subnet, None).await?;
//! publisher.node_assigned(&assignment).await?;
//!
//! // Subscribe locally
//! let mut rx = publisher.subscribe_local();
//! while let Some(msg) = rx.recv().await {
//!     println!("Event: {}", msg.description());
//! }
//! ```

pub mod channels;
pub mod messages;
pub mod publisher;

#[cfg(feature = "hpc-channels")]
pub mod hpc_bridge;

// Re-export main types
pub use channels::{
    all_subnet_channels, ChannelId, SUBNET_ASSIGNMENTS, SUBNET_LIFECYCLE, SUBNET_POLICIES,
    SUBNET_ROUTES, SUBNET_TOPOLOGY, SUBNET_WIREGUARD,
};

pub use messages::{
    // Assignment events
    BulkAssignmentEvent,
    // WireGuard events
    InterfaceCreatedEvent,
    InterfaceDeletedEvent,
    KeyRotatedEvent,
    // Migration events
    MigrationCompletedEvent,
    MigrationFailedEvent,
    MigrationStartedEvent,
    NodeAssignedEvent,
    NodeUnassignedEvent,
    PeerConfigUpdatedEvent,
    // Policy events
    PolicyCreatedEvent,
    PolicyDeletedEvent,
    PolicyUpdatedEvent,
    // Route events
    RouteCreatedEvent,
    RouteDeletedEvent,
    // Topology events
    RouteInfo,
    RouteStatusChangedEvent,
    // Lifecycle events
    SubnetCreatedEvent,
    SubnetDeletedEvent,
    SubnetInfo,
    // Main message type
    SubnetMessage,
    SubnetStatusChangedEvent,
    SubnetUpdatedEvent,
    TopologyDeltaEvent,
    TopologySnapshotEvent,
};

pub use publisher::{
    EventSubscription, EventTransport, InMemoryTransport, PublishError, PublisherStats,
    SubnetEventPublisher, SubscribeError,
};

// HPC-Channels integration exports
#[cfg(feature = "hpc-channels")]
pub use hpc_bridge::{
    create_shared_bridge, SharedSubnetHpcBridge, SubnetHpcBridge, SUBNET_MIGRATIONS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all expected types are exported
        let _: ChannelId = SUBNET_LIFECYCLE;
        let _: fn() -> SubnetEventPublisher = SubnetEventPublisher::new;
    }
}
