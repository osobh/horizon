//! Subnet event messages for hpc-channels broadcast
//!
//! These message types are published to well-known channels for
//! event-driven subnet topology synchronization across the platform.

use crate::models::{NodeType, RouteDirection, SubnetPurpose, SubnetStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::Ipv4Addr;
use uuid::Uuid;

/// Main subnet event message type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum SubnetMessage {
    // ========================================================================
    // Lifecycle Events (SUBNET_LIFECYCLE channel)
    // ========================================================================
    /// A new subnet was created
    SubnetCreated(SubnetCreatedEvent),

    /// Subnet status changed (e.g., Active → Draining)
    SubnetStatusChanged(SubnetStatusChangedEvent),

    /// Subnet was deleted
    SubnetDeleted(SubnetDeletedEvent),

    /// Subnet configuration was updated
    SubnetUpdated(SubnetUpdatedEvent),

    // ========================================================================
    // Assignment Events (SUBNET_ASSIGNMENTS channel)
    // ========================================================================
    /// A node was assigned to a subnet
    NodeAssigned(NodeAssignedEvent),

    /// A node was unassigned from a subnet
    NodeUnassigned(NodeUnassignedEvent),

    /// Bulk node assignment completed
    BulkAssignmentCompleted(BulkAssignmentEvent),

    // ========================================================================
    // Route Events (SUBNET_ROUTES channel)
    // ========================================================================
    /// A cross-subnet route was created
    RouteCreated(RouteCreatedEvent),

    /// A cross-subnet route was deleted
    RouteDeleted(RouteDeletedEvent),

    /// Route status changed
    RouteStatusChanged(RouteStatusChangedEvent),

    // ========================================================================
    // WireGuard Events (SUBNET_WIREGUARD channel)
    // ========================================================================
    /// Peer configuration was updated
    PeerConfigUpdated(PeerConfigUpdatedEvent),

    /// A WireGuard interface was created
    InterfaceCreated(InterfaceCreatedEvent),

    /// A WireGuard interface was deleted
    InterfaceDeleted(InterfaceDeletedEvent),

    /// Key rotation occurred
    KeyRotated(KeyRotatedEvent),

    // ========================================================================
    // Topology Events (SUBNET_TOPOLOGY channel)
    // ========================================================================
    /// Full topology snapshot (for initial sync or reconciliation)
    TopologySnapshot(TopologySnapshotEvent),

    /// Topology delta update
    TopologyDelta(TopologyDeltaEvent),

    // ========================================================================
    // Policy Events (SUBNET_POLICIES channel)
    // ========================================================================
    /// Policy was created
    PolicyCreated(PolicyCreatedEvent),

    /// Policy was updated
    PolicyUpdated(PolicyUpdatedEvent),

    /// Policy was deleted
    PolicyDeleted(PolicyDeletedEvent),

    // ========================================================================
    // Migration Events (SUBNET_ASSIGNMENTS channel)
    // ========================================================================
    /// Migration started
    MigrationStarted(MigrationStartedEvent),

    /// Migration completed successfully
    MigrationCompleted(MigrationCompletedEvent),

    /// Migration failed
    MigrationFailed(MigrationFailedEvent),
}

// ============================================================================
// Lifecycle Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetCreatedEvent {
    pub subnet_id: Uuid,
    pub name: String,
    pub cidr: String,
    pub purpose: SubnetPurpose,
    pub wg_interface: String,
    pub wg_listen_port: u16,
    pub created_at: DateTime<Utc>,
    pub created_by: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetStatusChangedEvent {
    pub subnet_id: Uuid,
    pub old_status: SubnetStatus,
    pub new_status: SubnetStatus,
    pub reason: Option<String>,
    pub changed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetDeletedEvent {
    pub subnet_id: Uuid,
    pub name: String,
    pub cidr: String,
    pub deleted_at: DateTime<Utc>,
    pub deleted_by: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetUpdatedEvent {
    pub subnet_id: Uuid,
    pub changes: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Assignment Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAssignedEvent {
    pub node_id: Uuid,
    pub subnet_id: Uuid,
    pub assigned_ip: Ipv4Addr,
    pub wg_public_key: String,
    pub policy_id: Option<Uuid>,
    pub assignment_method: String,
    pub assigned_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUnassignedEvent {
    pub node_id: Uuid,
    pub subnet_id: Uuid,
    pub released_ip: Ipv4Addr,
    pub reason: String,
    pub unassigned_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkAssignmentEvent {
    pub subnet_id: Uuid,
    pub nodes_assigned: u32,
    pub nodes_failed: u32,
    pub policy_id: Option<Uuid>,
    pub completed_at: DateTime<Utc>,
}

// ============================================================================
// Route Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteCreatedEvent {
    pub route_id: Uuid,
    pub source_subnet_id: Uuid,
    pub destination_subnet_id: Uuid,
    pub direction: RouteDirection,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDeletedEvent {
    pub route_id: Uuid,
    pub source_subnet_id: Uuid,
    pub destination_subnet_id: Uuid,
    pub deleted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteStatusChangedEvent {
    pub route_id: Uuid,
    pub old_status: String,
    pub new_status: String,
    pub changed_at: DateTime<Utc>,
}

// ============================================================================
// WireGuard Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfigUpdatedEvent {
    pub subnet_id: Uuid,
    pub node_id: Uuid,
    pub public_key: String,
    pub endpoint: Option<String>,
    pub allowed_ips: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceCreatedEvent {
    pub subnet_id: Uuid,
    pub interface_name: String,
    pub listen_port: u16,
    pub public_key: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceDeletedEvent {
    pub subnet_id: Uuid,
    pub interface_name: String,
    pub deleted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotatedEvent {
    pub subnet_id: Uuid,
    pub old_public_key: String,
    pub new_public_key: String,
    pub rotated_at: DateTime<Utc>,
}

// ============================================================================
// Topology Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySnapshotEvent {
    pub snapshot_id: Uuid,
    pub subnets: Vec<SubnetInfo>,
    pub routes: Vec<RouteInfo>,
    pub generated_at: DateTime<Utc>,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDeltaEvent {
    pub base_version: u64,
    pub new_version: u64,
    pub added_subnets: Vec<SubnetInfo>,
    pub removed_subnets: Vec<Uuid>,
    pub added_routes: Vec<RouteInfo>,
    pub removed_routes: Vec<Uuid>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetInfo {
    pub id: Uuid,
    pub name: String,
    pub cidr: String,
    pub purpose: SubnetPurpose,
    pub status: SubnetStatus,
    pub node_count: u32,
    pub wg_interface: String,
    pub wg_listen_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    pub id: Uuid,
    pub source_subnet_id: Uuid,
    pub destination_subnet_id: Uuid,
    pub direction: RouteDirection,
    pub status: String,
}

// ============================================================================
// Policy Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCreatedEvent {
    pub policy_id: Uuid,
    pub name: String,
    pub priority: i32,
    pub target_subnet_id: Uuid,
    pub rule_count: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyUpdatedEvent {
    pub policy_id: Uuid,
    pub changes: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDeletedEvent {
    pub policy_id: Uuid,
    pub name: String,
    pub deleted_at: DateTime<Utc>,
}

// ============================================================================
// Migration Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStartedEvent {
    pub migration_id: Uuid,
    pub node_id: Uuid,
    pub source_subnet_id: Uuid,
    pub target_subnet_id: Uuid,
    pub source_ip: Ipv4Addr,
    pub target_ip: Ipv4Addr,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCompletedEvent {
    pub migration_id: Uuid,
    pub node_id: Uuid,
    pub source_subnet_id: Uuid,
    pub target_subnet_id: Uuid,
    pub final_ip: Ipv4Addr,
    pub duration_ms: u64,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationFailedEvent {
    pub migration_id: Uuid,
    pub node_id: Uuid,
    pub source_subnet_id: Uuid,
    pub target_subnet_id: Uuid,
    pub error: String,
    pub rollback_successful: bool,
    pub failed_at: DateTime<Utc>,
}

// ============================================================================
// Helper implementations
// ============================================================================

impl SubnetMessage {
    /// Get the appropriate channel for this message type
    pub fn channel(&self) -> super::channels::ChannelId {
        use super::channels::*;

        match self {
            SubnetMessage::SubnetCreated(_)
            | SubnetMessage::SubnetStatusChanged(_)
            | SubnetMessage::SubnetDeleted(_)
            | SubnetMessage::SubnetUpdated(_) => SUBNET_LIFECYCLE,

            SubnetMessage::NodeAssigned(_)
            | SubnetMessage::NodeUnassigned(_)
            | SubnetMessage::BulkAssignmentCompleted(_)
            | SubnetMessage::MigrationStarted(_)
            | SubnetMessage::MigrationCompleted(_)
            | SubnetMessage::MigrationFailed(_) => SUBNET_ASSIGNMENTS,

            SubnetMessage::RouteCreated(_)
            | SubnetMessage::RouteDeleted(_)
            | SubnetMessage::RouteStatusChanged(_) => SUBNET_ROUTES,

            SubnetMessage::PeerConfigUpdated(_)
            | SubnetMessage::InterfaceCreated(_)
            | SubnetMessage::InterfaceDeleted(_)
            | SubnetMessage::KeyRotated(_) => SUBNET_WIREGUARD,

            SubnetMessage::TopologySnapshot(_) | SubnetMessage::TopologyDelta(_) => SUBNET_TOPOLOGY,

            SubnetMessage::PolicyCreated(_)
            | SubnetMessage::PolicyUpdated(_)
            | SubnetMessage::PolicyDeleted(_) => SUBNET_POLICIES,
        }
    }

    /// Get a short description of this event
    pub fn description(&self) -> &'static str {
        match self {
            SubnetMessage::SubnetCreated(_) => "subnet created",
            SubnetMessage::SubnetStatusChanged(_) => "subnet status changed",
            SubnetMessage::SubnetDeleted(_) => "subnet deleted",
            SubnetMessage::SubnetUpdated(_) => "subnet updated",
            SubnetMessage::NodeAssigned(_) => "node assigned",
            SubnetMessage::NodeUnassigned(_) => "node unassigned",
            SubnetMessage::BulkAssignmentCompleted(_) => "bulk assignment completed",
            SubnetMessage::RouteCreated(_) => "route created",
            SubnetMessage::RouteDeleted(_) => "route deleted",
            SubnetMessage::RouteStatusChanged(_) => "route status changed",
            SubnetMessage::PeerConfigUpdated(_) => "peer config updated",
            SubnetMessage::InterfaceCreated(_) => "interface created",
            SubnetMessage::InterfaceDeleted(_) => "interface deleted",
            SubnetMessage::KeyRotated(_) => "key rotated",
            SubnetMessage::TopologySnapshot(_) => "topology snapshot",
            SubnetMessage::TopologyDelta(_) => "topology delta",
            SubnetMessage::PolicyCreated(_) => "policy created",
            SubnetMessage::PolicyUpdated(_) => "policy updated",
            SubnetMessage::PolicyDeleted(_) => "policy deleted",
            SubnetMessage::MigrationStarted(_) => "migration started",
            SubnetMessage::MigrationCompleted(_) => "migration completed",
            SubnetMessage::MigrationFailed(_) => "migration failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let event = SubnetMessage::SubnetCreated(SubnetCreatedEvent {
            subnet_id: Uuid::new_v4(),
            name: "test-subnet".to_string(),
            cidr: "10.100.0.0/20".to_string(),
            purpose: SubnetPurpose::Tenant,
            wg_interface: "wg-test".to_string(),
            wg_listen_port: 51820,
            created_at: Utc::now(),
            created_by: None,
        });

        let json = serde_json::to_string(&event).unwrap();
        let parsed: SubnetMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            SubnetMessage::SubnetCreated(e) => {
                assert_eq!(e.name, "test-subnet");
                assert_eq!(e.cidr, "10.100.0.0/20");
            }
            _ => panic!("Expected SubnetCreated"),
        }
    }

    #[test]
    fn test_message_channel_routing() {
        use super::super::channels::*;

        let lifecycle_event = SubnetMessage::SubnetCreated(SubnetCreatedEvent {
            subnet_id: Uuid::new_v4(),
            name: "test".to_string(),
            cidr: "10.0.0.0/24".to_string(),
            purpose: SubnetPurpose::Tenant,
            wg_interface: "wg0".to_string(),
            wg_listen_port: 51820,
            created_at: Utc::now(),
            created_by: None,
        });

        assert_eq!(lifecycle_event.channel(), SUBNET_LIFECYCLE);

        let assignment_event = SubnetMessage::NodeAssigned(NodeAssignedEvent {
            node_id: Uuid::new_v4(),
            subnet_id: Uuid::new_v4(),
            assigned_ip: Ipv4Addr::new(10, 100, 0, 5),
            wg_public_key: "test-key".to_string(),
            policy_id: None,
            assignment_method: "policy".to_string(),
            assigned_at: Utc::now(),
        });

        assert_eq!(assignment_event.channel(), SUBNET_ASSIGNMENTS);
    }
}
