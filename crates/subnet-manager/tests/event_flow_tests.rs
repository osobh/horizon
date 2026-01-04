//! End-to-end event flow tests
//!
//! These tests verify that events properly flow through the event system,
//! from service operations through the publisher to subscribers.

use std::net::Ipv4Addr;
use std::str::FromStr;
use std::sync::Arc;

use ipnet::Ipv4Net;
use uuid::Uuid;

use chrono::Utc;
use subnet_manager::events::{
    InMemoryTransport, SubnetEventPublisher, SubnetMessage, SUBNET_ASSIGNMENTS, SUBNET_LIFECYCLE,
    SUBNET_POLICIES, SUBNET_ROUTES, SUBNET_TOPOLOGY, SUBNET_WIREGUARD,
};
use subnet_manager::migration::{Migration, MigrationReason};
use subnet_manager::models::{
    AssignmentPolicy, CrossSubnetRoute, NodeType, PolicyRule, RouteDirection, Subnet,
    SubnetAssignment, SubnetPurpose, SubnetStatus,
};
use subnet_manager::service::SubnetManager;

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_publisher() -> (SubnetEventPublisher, Arc<InMemoryTransport>) {
    let transport = Arc::new(InMemoryTransport::new());
    let publisher = SubnetEventPublisher::with_transport(transport.clone());
    (publisher, transport)
}

fn create_test_subnet(name: &str, cidr: &str) -> Subnet {
    Subnet::new(
        name,
        Ipv4Net::from_str(cidr).unwrap(),
        SubnetPurpose::Tenant,
        51820,
    )
}

fn create_test_assignment(subnet_id: Uuid, ip: Ipv4Addr) -> SubnetAssignment {
    SubnetAssignment {
        id: Uuid::new_v4(),
        node_id: Uuid::new_v4(),
        subnet_id,
        assigned_ip: ip,
        wg_public_key: "test-public-key".to_string(),
        assigned_at: Utc::now(),
        assignment_method: "test".to_string(),
        policy_id: None,
        is_migration_temp: false,
    }
}

fn create_test_route(source_id: Uuid, dest_id: Uuid) -> CrossSubnetRoute {
    CrossSubnetRoute::new(source_id, dest_id)
}

fn create_test_migration(node_id: Uuid, source_id: Uuid, target_id: Uuid) -> Migration {
    Migration::new(
        node_id,
        source_id,
        target_id,
        Ipv4Addr::new(10, 100, 0, 5),
        MigrationReason::PolicyChange,
    )
    .with_target_ip(Ipv4Addr::new(10, 101, 0, 5))
}

fn create_test_policy(name: &str, target_subnet_id: Uuid) -> AssignmentPolicy {
    AssignmentPolicy::new(name, target_subnet_id, 100)
        .with_rule(PolicyRule::node_type_equals(NodeType::DataCenter))
}

// ============================================================================
// Subnet Lifecycle Events
// ============================================================================

#[tokio::test]
async fn test_subnet_created_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet = create_test_subnet("test-subnet", "10.100.0.0/20");

    publisher.subnet_created(&subnet, None).await.unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_LIFECYCLE);

    match &messages[0].1 {
        SubnetMessage::SubnetCreated(event) => {
            assert_eq!(event.subnet_id, subnet.id);
            assert_eq!(event.name, "test-subnet");
            assert_eq!(event.cidr, "10.100.0.0/20");
            assert_eq!(event.purpose, SubnetPurpose::Tenant);
            assert_eq!(event.wg_listen_port, 51820);
        }
        _ => panic!("Expected SubnetCreated event"),
    }
}

#[tokio::test]
async fn test_subnet_status_changed_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet_id = Uuid::new_v4();

    publisher
        .subnet_status_changed(
            subnet_id,
            SubnetStatus::Active,
            SubnetStatus::Draining,
            Some("Maintenance window".to_string()),
        )
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_LIFECYCLE);

    match &messages[0].1 {
        SubnetMessage::SubnetStatusChanged(event) => {
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.old_status, SubnetStatus::Active);
            assert_eq!(event.new_status, SubnetStatus::Draining);
            assert_eq!(event.reason, Some("Maintenance window".to_string()));
        }
        _ => panic!("Expected SubnetStatusChanged event"),
    }
}

#[tokio::test]
async fn test_subnet_deleted_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet = create_test_subnet("delete-me", "10.100.16.0/20");

    publisher
        .subnet_deleted(&subnet, Some(Uuid::new_v4()))
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_LIFECYCLE);

    match &messages[0].1 {
        SubnetMessage::SubnetDeleted(event) => {
            assert_eq!(event.subnet_id, subnet.id);
            assert_eq!(event.name, "delete-me");
            assert!(event.deleted_by.is_some());
        }
        _ => panic!("Expected SubnetDeleted event"),
    }
}

// ============================================================================
// Node Assignment Events
// ============================================================================

#[tokio::test]
async fn test_node_assigned_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet_id = Uuid::new_v4();
    let assignment = create_test_assignment(subnet_id, Ipv4Addr::new(10, 100, 0, 5));

    publisher.node_assigned(&assignment).await.unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_ASSIGNMENTS);

    match &messages[0].1 {
        SubnetMessage::NodeAssigned(event) => {
            assert_eq!(event.node_id, assignment.node_id);
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.assigned_ip, Ipv4Addr::new(10, 100, 0, 5));
            assert_eq!(event.wg_public_key, "test-public-key");
        }
        _ => panic!("Expected NodeAssigned event"),
    }
}

#[tokio::test]
async fn test_node_unassigned_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let node_id = Uuid::new_v4();
    let subnet_id = Uuid::new_v4();
    let released_ip = Ipv4Addr::new(10, 100, 0, 10);

    publisher
        .node_unassigned(node_id, subnet_id, released_ip, "Node decommissioned")
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_ASSIGNMENTS);

    match &messages[0].1 {
        SubnetMessage::NodeUnassigned(event) => {
            assert_eq!(event.node_id, node_id);
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.released_ip, released_ip);
            assert_eq!(event.reason, "Node decommissioned");
        }
        _ => panic!("Expected NodeUnassigned event"),
    }
}

// ============================================================================
// Migration Events
// ============================================================================

#[tokio::test]
async fn test_migration_started_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let node_id = Uuid::new_v4();
    let source_id = Uuid::new_v4();
    let target_id = Uuid::new_v4();
    let migration = create_test_migration(node_id, source_id, target_id);

    publisher.migration_started(&migration).await.unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_ASSIGNMENTS);

    match &messages[0].1 {
        SubnetMessage::MigrationStarted(event) => {
            assert_eq!(event.migration_id, migration.id);
            assert_eq!(event.node_id, node_id);
            assert_eq!(event.source_subnet_id, source_id);
            assert_eq!(event.target_subnet_id, target_id);
            assert_eq!(event.source_ip, Ipv4Addr::new(10, 100, 0, 5));
            assert_eq!(event.target_ip, Ipv4Addr::new(10, 101, 0, 5));
        }
        _ => panic!("Expected MigrationStarted event"),
    }
}

#[tokio::test]
async fn test_migration_completed_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let migration = create_test_migration(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());

    publisher
        .migration_completed(&migration, 1500)
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);

    match &messages[0].1 {
        SubnetMessage::MigrationCompleted(event) => {
            assert_eq!(event.migration_id, migration.id);
            assert_eq!(event.duration_ms, 1500);
            assert_eq!(event.final_ip, Ipv4Addr::new(10, 101, 0, 5));
        }
        _ => panic!("Expected MigrationCompleted event"),
    }
}

#[tokio::test]
async fn test_migration_failed_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let migration = create_test_migration(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());

    publisher
        .migration_failed(&migration, "Connectivity check failed", true)
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);

    match &messages[0].1 {
        SubnetMessage::MigrationFailed(event) => {
            assert_eq!(event.migration_id, migration.id);
            assert_eq!(event.error, "Connectivity check failed");
            assert!(event.rollback_successful);
        }
        _ => panic!("Expected MigrationFailed event"),
    }
}

// ============================================================================
// Cross-Subnet Route Events
// ============================================================================

#[tokio::test]
async fn test_route_created_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let source_id = Uuid::new_v4();
    let dest_id = Uuid::new_v4();
    let route = create_test_route(source_id, dest_id);

    publisher.route_created(&route).await.unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_ROUTES);

    match &messages[0].1 {
        SubnetMessage::RouteCreated(event) => {
            assert_eq!(event.route_id, route.id);
            assert_eq!(event.source_subnet_id, source_id);
            assert_eq!(event.destination_subnet_id, dest_id);
            assert_eq!(event.direction, RouteDirection::Bidirectional);
        }
        _ => panic!("Expected RouteCreated event"),
    }
}

#[tokio::test]
async fn test_route_deleted_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let route_id = Uuid::new_v4();
    let source_id = Uuid::new_v4();
    let dest_id = Uuid::new_v4();

    publisher
        .route_deleted(route_id, source_id, dest_id)
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_ROUTES);

    match &messages[0].1 {
        SubnetMessage::RouteDeleted(event) => {
            assert_eq!(event.route_id, route_id);
            assert_eq!(event.source_subnet_id, source_id);
            assert_eq!(event.destination_subnet_id, dest_id);
        }
        _ => panic!("Expected RouteDeleted event"),
    }
}

// ============================================================================
// Topology Events
// ============================================================================

#[tokio::test]
async fn test_topology_snapshot_event_flow() {
    let (publisher, transport) = create_test_publisher();

    use subnet_manager::events::{RouteInfo, SubnetInfo};

    let subnets = vec![
        SubnetInfo {
            id: Uuid::new_v4(),
            name: "subnet-1".to_string(),
            cidr: "10.100.0.0/20".to_string(),
            purpose: SubnetPurpose::Tenant,
            status: SubnetStatus::Active,
            node_count: 10,
            wg_interface: "wg0".to_string(),
            wg_listen_port: 51820,
        },
        SubnetInfo {
            id: Uuid::new_v4(),
            name: "subnet-2".to_string(),
            cidr: "10.100.16.0/20".to_string(),
            purpose: SubnetPurpose::Tenant,
            status: SubnetStatus::Active,
            node_count: 5,
            wg_interface: "wg1".to_string(),
            wg_listen_port: 51821,
        },
    ];

    let routes = vec![RouteInfo {
        id: Uuid::new_v4(),
        source_subnet_id: subnets[0].id,
        destination_subnet_id: subnets[1].id,
        direction: RouteDirection::Bidirectional,
        status: "active".to_string(),
    }];

    publisher
        .topology_snapshot(subnets.clone(), routes.clone(), 1)
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_TOPOLOGY);

    match &messages[0].1 {
        SubnetMessage::TopologySnapshot(event) => {
            assert_eq!(event.subnets.len(), 2);
            assert_eq!(event.routes.len(), 1);
            assert_eq!(event.version, 1);
            assert_eq!(event.subnets[0].name, "subnet-1");
            assert_eq!(event.subnets[1].node_count, 5);
        }
        _ => panic!("Expected TopologySnapshot event"),
    }
}

// ============================================================================
// WireGuard Events
// ============================================================================

#[tokio::test]
async fn test_peer_config_updated_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet_id = Uuid::new_v4();
    let node_id = Uuid::new_v4();

    publisher
        .peer_config_updated(
            subnet_id,
            node_id,
            "base64-public-key",
            Some("192.168.1.100:51820".parse().unwrap()),
            vec!["10.100.0.5/32".to_string()],
        )
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_WIREGUARD);

    match &messages[0].1 {
        SubnetMessage::PeerConfigUpdated(event) => {
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.node_id, node_id);
            assert_eq!(event.public_key, "base64-public-key");
            assert_eq!(event.endpoint, Some("192.168.1.100:51820".to_string()));
            assert_eq!(event.allowed_ips, vec!["10.100.0.5/32"]);
        }
        _ => panic!("Expected PeerConfigUpdated event"),
    }
}

#[tokio::test]
async fn test_interface_created_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet_id = Uuid::new_v4();

    publisher
        .interface_created(subnet_id, "wg-tenant-acme", 51820, "interface-public-key")
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_WIREGUARD);

    match &messages[0].1 {
        SubnetMessage::InterfaceCreated(event) => {
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.interface_name, "wg-tenant-acme");
            assert_eq!(event.listen_port, 51820);
            assert_eq!(event.public_key, "interface-public-key");
        }
        _ => panic!("Expected InterfaceCreated event"),
    }
}

#[tokio::test]
async fn test_key_rotated_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let subnet_id = Uuid::new_v4();

    publisher
        .key_rotated(subnet_id, "old-key-base64", "new-key-base64")
        .await
        .unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_WIREGUARD);

    match &messages[0].1 {
        SubnetMessage::KeyRotated(event) => {
            assert_eq!(event.subnet_id, subnet_id);
            assert_eq!(event.old_public_key, "old-key-base64");
            assert_eq!(event.new_public_key, "new-key-base64");
        }
        _ => panic!("Expected KeyRotated event"),
    }
}

// ============================================================================
// Policy Events
// ============================================================================

#[tokio::test]
async fn test_policy_created_event_flow() {
    let (publisher, transport) = create_test_publisher();
    let target_subnet_id = Uuid::new_v4();
    let policy = create_test_policy("datacenter-policy", target_subnet_id);

    publisher.policy_created(&policy).await.unwrap();

    let messages = transport.messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].0, SUBNET_POLICIES);

    match &messages[0].1 {
        SubnetMessage::PolicyCreated(event) => {
            assert_eq!(event.policy_id, policy.id);
            assert_eq!(event.name, "datacenter-policy");
            assert_eq!(event.priority, 100);
            assert_eq!(event.target_subnet_id, target_subnet_id);
            assert_eq!(event.rule_count, 1);
        }
        _ => panic!("Expected PolicyCreated event"),
    }
}

// ============================================================================
// Local Subscriber Tests
// ============================================================================

#[tokio::test]
async fn test_local_subscriber_receives_all_events() {
    let (publisher, _transport) = create_test_publisher();
    let mut rx = publisher.subscribe_local();

    // Publish multiple event types
    let subnet = create_test_subnet("local-test", "10.100.0.0/20");
    publisher.subnet_created(&subnet, None).await.unwrap();

    let assignment = create_test_assignment(subnet.id, Ipv4Addr::new(10, 100, 0, 5));
    publisher.node_assigned(&assignment).await.unwrap();

    // Verify local subscriber received both
    let msg1 = rx.try_recv().unwrap();
    assert!(matches!(msg1, SubnetMessage::SubnetCreated(_)));

    let msg2 = rx.try_recv().unwrap();
    assert!(matches!(msg2, SubnetMessage::NodeAssigned(_)));
}

#[tokio::test]
async fn test_multiple_local_subscribers() {
    let (publisher, _transport) = create_test_publisher();
    let mut rx1 = publisher.subscribe_local();
    let mut rx2 = publisher.subscribe_local();

    let subnet = create_test_subnet("multi-sub-test", "10.100.0.0/20");
    publisher.subnet_created(&subnet, None).await.unwrap();

    // Both subscribers should receive the message
    let msg1 = rx1.try_recv().unwrap();
    let msg2 = rx2.try_recv().unwrap();

    assert!(matches!(msg1, SubnetMessage::SubnetCreated(_)));
    assert!(matches!(msg2, SubnetMessage::SubnetCreated(_)));

    assert_eq!(publisher.stats().subscribers, 2);
}

// ============================================================================
// Message Buffering Tests
// ============================================================================

#[tokio::test]
async fn test_message_buffering_on_disconnect() {
    let transport = Arc::new(InMemoryTransport::new());
    let publisher = SubnetEventPublisher::with_transport(transport.clone());

    // Disconnect transport
    transport.disconnect();

    let subnet = create_test_subnet("buffered-subnet", "10.100.0.0/20");
    let result = publisher.subnet_created(&subnet, None).await;

    // Should fail but buffer
    assert!(result.is_err());
    assert_eq!(publisher.stats().messages_queued, 1);
    assert_eq!(transport.messages().len(), 0);
}

#[tokio::test]
async fn test_buffer_flush_on_reconnect() {
    let transport = Arc::new(InMemoryTransport::new());
    let publisher = SubnetEventPublisher::with_transport(transport.clone());

    // Disconnect and buffer messages
    transport.disconnect();

    let subnet1 = create_test_subnet("buffer-1", "10.100.0.0/20");
    let subnet2 = create_test_subnet("buffer-2", "10.100.16.0/20");

    let _ = publisher.subnet_created(&subnet1, None).await;
    let _ = publisher.subnet_created(&subnet2, None).await;

    assert_eq!(publisher.stats().messages_queued, 2);

    // Reconnect and flush
    transport.connect();
    let flushed = publisher.flush_buffer().await.unwrap();

    assert_eq!(flushed, 2);
    assert_eq!(publisher.stats().messages_queued, 0);
    assert_eq!(transport.messages().len(), 2);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[tokio::test]
async fn test_complete_subnet_lifecycle_event_flow() {
    let (publisher, transport) = create_test_publisher();

    // 1. Create subnet
    let subnet = create_test_subnet("lifecycle-test", "10.100.0.0/20");
    publisher.subnet_created(&subnet, None).await.unwrap();

    // 2. Assign nodes
    let assignment1 = create_test_assignment(subnet.id, Ipv4Addr::new(10, 100, 0, 5));
    let assignment2 = create_test_assignment(subnet.id, Ipv4Addr::new(10, 100, 0, 6));
    publisher.node_assigned(&assignment1).await.unwrap();
    publisher.node_assigned(&assignment2).await.unwrap();

    // 3. Create route to another subnet
    let other_subnet_id = Uuid::new_v4();
    let route = create_test_route(subnet.id, other_subnet_id);
    publisher.route_created(&route).await.unwrap();

    // 4. Unassign a node
    publisher
        .node_unassigned(
            assignment1.node_id,
            subnet.id,
            assignment1.assigned_ip,
            "Node removed",
        )
        .await
        .unwrap();

    // 5. Change subnet status
    publisher
        .subnet_status_changed(
            subnet.id,
            SubnetStatus::Active,
            SubnetStatus::Draining,
            None,
        )
        .await
        .unwrap();

    // 6. Delete route
    publisher
        .route_deleted(route.id, subnet.id, other_subnet_id)
        .await
        .unwrap();

    // 7. Delete subnet
    publisher.subnet_deleted(&subnet, None).await.unwrap();

    // Verify all events were published
    let messages = transport.messages();
    assert_eq!(messages.len(), 8);

    // Verify event ordering and types
    assert!(matches!(&messages[0].1, SubnetMessage::SubnetCreated(_)));
    assert!(matches!(&messages[1].1, SubnetMessage::NodeAssigned(_)));
    assert!(matches!(&messages[2].1, SubnetMessage::NodeAssigned(_)));
    assert!(matches!(&messages[3].1, SubnetMessage::RouteCreated(_)));
    assert!(matches!(&messages[4].1, SubnetMessage::NodeUnassigned(_)));
    assert!(matches!(
        &messages[5].1,
        SubnetMessage::SubnetStatusChanged(_)
    ));
    assert!(matches!(&messages[6].1, SubnetMessage::RouteDeleted(_)));
    assert!(matches!(&messages[7].1, SubnetMessage::SubnetDeleted(_)));

    // Verify channel routing
    assert_eq!(messages[0].0, SUBNET_LIFECYCLE);
    assert_eq!(messages[1].0, SUBNET_ASSIGNMENTS);
    assert_eq!(messages[3].0, SUBNET_ROUTES);
}

#[tokio::test]
async fn test_complete_migration_event_flow() {
    let (publisher, transport) = create_test_publisher();

    let node_id = Uuid::new_v4();
    let source_subnet = create_test_subnet("source", "10.100.0.0/20");
    let target_subnet = create_test_subnet("target", "10.101.0.0/20");

    // 1. Subnets exist
    publisher
        .subnet_created(&source_subnet, None)
        .await
        .unwrap();
    publisher
        .subnet_created(&target_subnet, None)
        .await
        .unwrap();

    // 2. Node assigned to source
    let assignment = SubnetAssignment {
        id: Uuid::new_v4(),
        node_id,
        subnet_id: source_subnet.id,
        assigned_ip: Ipv4Addr::new(10, 100, 0, 5),
        wg_public_key: "node-public-key".to_string(),
        assigned_at: Utc::now(),
        assignment_method: "test".to_string(),
        policy_id: None,
        is_migration_temp: false,
    };
    publisher.node_assigned(&assignment).await.unwrap();

    // 3. Migration starts
    let migration = Migration::new(
        node_id,
        source_subnet.id,
        target_subnet.id,
        Ipv4Addr::new(10, 100, 0, 5),
        MigrationReason::PolicyChange,
    )
    .with_target_ip(Ipv4Addr::new(10, 101, 0, 5));

    publisher.migration_started(&migration).await.unwrap();

    // 4. Migration completes
    publisher
        .migration_completed(&migration, 2500)
        .await
        .unwrap();

    // 5. Node unassigned from source
    publisher
        .node_unassigned(
            node_id,
            source_subnet.id,
            Ipv4Addr::new(10, 100, 0, 5),
            "Migrated",
        )
        .await
        .unwrap();

    // 6. Node assigned to target
    let new_assignment = SubnetAssignment {
        id: Uuid::new_v4(),
        node_id,
        subnet_id: target_subnet.id,
        assigned_ip: Ipv4Addr::new(10, 101, 0, 5),
        wg_public_key: "node-public-key".to_string(),
        assigned_at: Utc::now(),
        assignment_method: "migration".to_string(),
        policy_id: None,
        is_migration_temp: false,
    };
    publisher.node_assigned(&new_assignment).await.unwrap();

    // Verify complete flow
    let messages = transport.messages();
    assert_eq!(messages.len(), 7);

    // Check migration events are on SUBNET_ASSIGNMENTS
    let migration_start = messages
        .iter()
        .find(|(_, m)| matches!(m, SubnetMessage::MigrationStarted(_)));
    assert!(migration_start.is_some());
    assert_eq!(migration_start.unwrap().0, SUBNET_ASSIGNMENTS);

    let migration_complete = messages
        .iter()
        .find(|(_, m)| matches!(m, SubnetMessage::MigrationCompleted(_)));
    assert!(migration_complete.is_some());
}

#[tokio::test]
async fn test_publisher_stats_accuracy() {
    let (publisher, transport) = create_test_publisher();

    // Subscribe locally
    let _rx = publisher.subscribe_local();

    // Publish 5 messages
    for i in 0..5 {
        let subnet = create_test_subnet(&format!("stats-test-{}", i), "10.100.0.0/20");
        publisher.subnet_created(&subnet, None).await.unwrap();
    }

    let stats = publisher.stats();
    assert_eq!(stats.messages_published, 5);
    assert_eq!(stats.messages_failed, 0);
    assert_eq!(stats.messages_queued, 0);
    assert_eq!(stats.subscribers, 1);
    assert_eq!(transport.messages().len(), 5);
}

#[tokio::test]
async fn test_event_channel_routing_correctness() {
    let (publisher, transport) = create_test_publisher();

    // Publish one event of each channel type
    let subnet = create_test_subnet("routing-test", "10.100.0.0/20");
    publisher.subnet_created(&subnet, None).await.unwrap(); // LIFECYCLE

    let assignment = create_test_assignment(subnet.id, Ipv4Addr::new(10, 100, 0, 5));
    publisher.node_assigned(&assignment).await.unwrap(); // ASSIGNMENTS

    let route = create_test_route(subnet.id, Uuid::new_v4());
    publisher.route_created(&route).await.unwrap(); // ROUTES

    publisher
        .interface_created(subnet.id, "wg0", 51820, "key")
        .await
        .unwrap(); // WIREGUARD

    use subnet_manager::events::SubnetInfo;
    publisher
        .topology_snapshot(
            vec![SubnetInfo {
                id: subnet.id,
                name: "test".to_string(),
                cidr: "10.100.0.0/20".to_string(),
                purpose: SubnetPurpose::Tenant,
                status: SubnetStatus::Active,
                node_count: 1,
                wg_interface: "wg0".to_string(),
                wg_listen_port: 51820,
            }],
            vec![],
            1,
        )
        .await
        .unwrap(); // TOPOLOGY

    let policy = create_test_policy("test-policy", subnet.id);
    publisher.policy_created(&policy).await.unwrap(); // POLICIES

    let messages = transport.messages();
    assert_eq!(messages.len(), 6);

    // Verify each message went to the correct channel
    assert_eq!(messages[0].0, SUBNET_LIFECYCLE);
    assert_eq!(messages[1].0, SUBNET_ASSIGNMENTS);
    assert_eq!(messages[2].0, SUBNET_ROUTES);
    assert_eq!(messages[3].0, SUBNET_WIREGUARD);
    assert_eq!(messages[4].0, SUBNET_TOPOLOGY);
    assert_eq!(messages[5].0, SUBNET_POLICIES);
}
