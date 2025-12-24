//! Integration tests for hpc-channels connectivity
//!
//! These tests verify that subnet events flow correctly through
//! the hpc-channels broadcast infrastructure.

#![cfg(feature = "hpc-channels")]

use std::sync::Arc;
use std::time::Duration;

use subnet_manager::events::{
    create_shared_bridge, SubnetEventPublisher, SubnetHpcBridge, SubnetMessage,
    SUBNET_ASSIGNMENTS, SUBNET_LIFECYCLE, SUBNET_MIGRATIONS,
};
use subnet_manager::models::{Subnet, SubnetPurpose};
use subnet_manager::policy_engine::PolicyEngine;
use subnet_manager::service::SubnetManager;

use ipnet::Ipv4Net;
use std::str::FromStr;
use uuid::Uuid;

/// Test that SubnetHpcBridge can be created and used as EventTransport
#[tokio::test]
async fn test_hpc_bridge_as_transport() {
    let bridge = Arc::new(SubnetHpcBridge::new());
    let publisher = SubnetEventPublisher::with_transport(bridge.clone());

    // Subscribe to lifecycle events
    let mut rx = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);

    // Create and publish a subnet event
    let subnet = Subnet::new(
        "test-subnet",
        Ipv4Net::from_str("10.100.0.0/20").unwrap(),
        SubnetPurpose::Tenant,
        51820,
    );

    publisher.subnet_created(&subnet, None).await.unwrap();

    // Verify we receive the event
    let received = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout waiting for event")
        .expect("channel closed");

    match received {
        SubnetMessage::SubnetCreated(e) => {
            assert_eq!(e.name, "test-subnet");
            assert_eq!(e.cidr, "10.100.0.0/20");
            assert_eq!(e.purpose, SubnetPurpose::Tenant);
        }
        _ => panic!("Expected SubnetCreated event, got {:?}", received),
    }
}

/// Test the with_hpc_channels() convenience constructor
#[tokio::test]
async fn test_publisher_with_hpc_channels() {
    let publisher = SubnetEventPublisher::with_hpc_channels();

    // Verify it works by publishing an event
    let subnet = Subnet::new(
        "hpc-test",
        Ipv4Net::from_str("10.101.0.0/20").unwrap(),
        SubnetPurpose::NodeType,
        51821,
    );

    let result = publisher.subnet_created(&subnet, Some(Uuid::new_v4())).await;
    assert!(result.is_ok());

    // Check stats
    let stats = publisher.stats();
    assert_eq!(stats.messages_published, 1);
}

/// Test that multiple subscribers receive the same event
#[tokio::test]
async fn test_broadcast_to_multiple_subscribers() {
    let bridge = create_shared_bridge();

    // Create multiple subscribers
    let mut rx1 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
    let mut rx2 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
    let mut rx3 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);

    // Publish event
    let subnet = Subnet::new(
        "broadcast-test",
        Ipv4Net::from_str("10.102.0.0/20").unwrap(),
        SubnetPurpose::Geographic,
        51822,
    );

    let publisher = SubnetEventPublisher::with_transport(bridge.clone());
    publisher.subnet_created(&subnet, None).await.unwrap();

    // All subscribers should receive the event
    let timeout = Duration::from_millis(100);

    let r1 = tokio::time::timeout(timeout, rx1.recv()).await;
    let r2 = tokio::time::timeout(timeout, rx2.recv()).await;
    let r3 = tokio::time::timeout(timeout, rx3.recv()).await;

    assert!(r1.is_ok() && r1.unwrap().is_ok());
    assert!(r2.is_ok() && r2.unwrap().is_ok());
    assert!(r3.is_ok() && r3.unwrap().is_ok());
}

/// Test channel routing - events go to correct channels
#[tokio::test]
async fn test_channel_routing() {
    let bridge = create_shared_bridge();

    // Subscribe to different channels
    let mut lifecycle_rx = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
    let mut assignments_rx = bridge.subscribe_broadcast(SUBNET_ASSIGNMENTS);

    let publisher = SubnetEventPublisher::with_transport(bridge.clone());

    // Publish a lifecycle event
    let subnet = Subnet::new(
        "routing-test",
        Ipv4Net::from_str("10.103.0.0/20").unwrap(),
        SubnetPurpose::ResourcePool,
        51823,
    );
    publisher.subnet_created(&subnet, None).await.unwrap();

    // Lifecycle should receive it
    let lifecycle_result =
        tokio::time::timeout(Duration::from_millis(50), lifecycle_rx.recv()).await;
    assert!(
        lifecycle_result.is_ok(),
        "Lifecycle channel should receive event"
    );

    // Assignments should NOT receive lifecycle events (different channel)
    let assignments_result =
        tokio::time::timeout(Duration::from_millis(50), assignments_rx.recv()).await;
    assert!(
        assignments_result.is_err(),
        "Assignments channel should timeout (no lifecycle events)"
    );
}

/// Test that receiver count is tracked correctly
#[test]
fn test_receiver_count_tracking() {
    let bridge = SubnetHpcBridge::new();

    assert_eq!(bridge.total_receiver_count(), 0);

    // Subscribe to lifecycle
    let _rx1 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
    assert_eq!(bridge.receiver_count(SUBNET_LIFECYCLE), 1);
    assert_eq!(bridge.total_receiver_count(), 1);

    // Subscribe to assignments
    let _rx2 = bridge.subscribe_broadcast(SUBNET_ASSIGNMENTS);
    assert_eq!(bridge.receiver_count(SUBNET_ASSIGNMENTS), 1);
    assert_eq!(bridge.total_receiver_count(), 2);

    // Multiple subscribers on same channel
    let _rx3 = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);
    assert_eq!(bridge.receiver_count(SUBNET_LIFECYCLE), 2);
    assert_eq!(bridge.total_receiver_count(), 3);
}

/// Test subscribe_all returns receivers for all channels
#[test]
fn test_subscribe_all_channels() {
    let bridge = SubnetHpcBridge::new();
    let receivers = bridge.subscribe_all();

    // Should have 7 channels
    assert_eq!(receivers.len(), 7);

    // All channels should have exactly one receiver now
    assert_eq!(bridge.total_receiver_count(), 7);
}

/// Test migration events flow to the correct channel
#[tokio::test]
async fn test_migration_events() {
    use subnet_manager::migration::{Migration, MigrationReason, MigrationStatus};
    use std::net::Ipv4Addr;
    use chrono::Utc;

    let bridge = create_shared_bridge();
    let mut rx = bridge.subscribe_broadcast(SUBNET_MIGRATIONS);

    let publisher = SubnetEventPublisher::with_transport(bridge.clone());

    // Create a migration using the constructor
    let mut migration = Migration::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        Ipv4Addr::new(10, 100, 0, 5),
        MigrationReason::PolicyChange,
    );
    migration.target_ip = Some(Ipv4Addr::new(10, 101, 0, 5));
    migration.started_at = Some(Utc::now());

    publisher.migration_started(&migration).await.unwrap();

    // Verify we receive the migration event
    let received = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .expect("timeout")
        .expect("recv");

    match received {
        SubnetMessage::MigrationStarted(e) => {
            assert_eq!(e.migration_id, migration.id);
            assert_eq!(e.node_id, migration.node_id);
        }
        _ => panic!("Expected MigrationStarted event"),
    }
}

/// Test SubnetMeshOrchestrator with_hpc_channels constructor
#[tokio::test]
async fn test_orchestrator_with_hpc_channels() {
    use subnet_manager::integration::SubnetMeshOrchestrator;

    let manager = Arc::new(SubnetManager::new());
    let policy_engine = Arc::new(PolicyEngine::new());

    // This should compile and work
    let orchestrator = SubnetMeshOrchestrator::with_hpc_channels(manager, policy_engine);

    // Verify it was created successfully
    assert!(orchestrator.all_assignments().is_empty());
}

/// Test that events published through publisher are received by bridge subscribers
#[tokio::test]
async fn test_publisher_to_bridge_flow() {
    // Create bridge and publisher
    let bridge = Arc::new(SubnetHpcBridge::new());
    let publisher = SubnetEventPublisher::with_transport(bridge.clone());

    // Subscribe before publishing
    let mut rx = bridge.subscribe_broadcast(SUBNET_LIFECYCLE);

    // Use local subscriber too
    let mut local_rx = publisher.subscribe_local();

    // Publish event
    let subnet = Subnet::new(
        "flow-test",
        Ipv4Net::from_str("10.104.0.0/20").unwrap(),
        SubnetPurpose::Tenant,
        51824,
    );
    publisher.subnet_created(&subnet, None).await.unwrap();

    // Both should receive the event
    let timeout = Duration::from_millis(100);

    let bridge_result = tokio::time::timeout(timeout, rx.recv()).await;
    let local_result = tokio::time::timeout(timeout, local_rx.recv()).await;

    assert!(bridge_result.is_ok(), "Bridge subscriber should receive");
    assert!(local_result.is_ok(), "Local subscriber should receive");
}
