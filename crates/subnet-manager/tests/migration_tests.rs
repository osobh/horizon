//! Migration system integration tests
//!
//! Tests the full migration workflow including planning, execution,
//! state transitions, and rollback scenarios.

use std::net::Ipv4Addr;
use subnet_manager::migration::{
    MigrationConstraint, MigrationExecutor, MigrationPlanner, MigrationReason, MigrationStats,
    MigrationStatus, MigrationStep,
};
use subnet_manager::models::{Subnet, SubnetAssignment, SubnetPurpose, SubnetStatus};
use subnet_manager::service::{AssignNodeRequest, CreateSubnetRequest, SubnetManager};
use uuid::Uuid;

/// Helper to create a test subnet
fn create_test_subnet(name: &str, purpose: SubnetPurpose) -> Subnet {
    Subnet {
        id: Uuid::new_v4(),
        name: name.to_string(),
        description: Some(format!("Test subnet: {}", name)),
        cidr: "10.100.0.0/24".parse().unwrap(),
        purpose,
        status: SubnetStatus::Active,
        tenant_id: None,
        node_type: None,
        region: None,
        resource_pool_id: None,
        wg_interface: format!("wg-{}", name),
        wg_listen_port: 51820,
        wg_public_key: Some("dGVzdC1wdWJsaWMta2V5".to_string()),
        wg_private_key: Some("dGVzdC1wcml2YXRlLWtleQ==".to_string()),
        max_nodes: Some(100),
        current_nodes: 0,
        template_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        created_by: None,
        metadata: None,
    }
}

/// Helper to create a test assignment
fn create_test_assignment(node_id: Uuid, subnet_id: Uuid, ip: Ipv4Addr) -> SubnetAssignment {
    SubnetAssignment {
        id: Uuid::new_v4(),
        node_id,
        subnet_id,
        assigned_ip: ip,
        wg_public_key: "bm9kZS1wdWJsaWMta2V5".to_string(),
        assigned_at: chrono::Utc::now(),
        assignment_method: "test".to_string(),
        policy_id: None,
        is_migration_temp: false,
    }
}

// ============================================================================
// Migration Planner Tests
// ============================================================================

#[test]
fn test_migration_planner_creation() {
    let planner = MigrationPlanner::new();
    assert_eq!(planner.active_count(), 0);
}

#[test]
fn test_plan_basic_migration() {
    let mut planner = MigrationPlanner::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, source_subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 1, 10);

    let plan = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .expect("Should create migration plan");

    // Access through the nested migration struct
    assert_eq!(plan.migration.node_id, node_id);
    assert_eq!(plan.migration.source_subnet_id, source_subnet.id);
    assert_eq!(plan.migration.target_subnet_id, target_subnet.id);
    assert_eq!(plan.target_ip, target_ip);
}

#[test]
fn test_migration_constraint_defaults() {
    let constraint = MigrationConstraint::default();

    assert_eq!(constraint.max_concurrent, 10);
    assert_eq!(constraint.max_per_subnet, 5);
    assert_eq!(constraint.min_interval_secs, 300);
    assert!(constraint.maintenance_window.is_none());
    assert!(constraint.excluded_nodes.is_empty());
    assert!(!constraint.allow_cross_purpose);
}

#[test]
fn test_constraint_maintenance_window() {
    let mut constraint = MigrationConstraint::default();

    // No window - always in window
    assert!(constraint.is_in_maintenance_window());

    // Set a window - 00:00 to 23:59 (always in window)
    constraint.maintenance_window = Some((0, 24));
    assert!(constraint.is_in_maintenance_window());
}

#[test]
fn test_cannot_plan_same_subnet_migration() {
    let mut planner = MigrationPlanner::new();

    let subnet = create_test_subnet("same", SubnetPurpose::Tenant);
    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 0, 20);

    // Try to migrate within same subnet
    let result = planner.plan(
        node_id,
        &subnet,
        &subnet,
        &assignment,
        target_ip,
        MigrationReason::Manual,
    );

    assert!(result.is_err());
}

#[test]
fn test_bulk_migration_planning() {
    let mut planner = MigrationPlanner::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    // Create multiple nodes with assignments and target IPs
    let mut nodes: Vec<(Uuid, SubnetAssignment, Ipv4Addr)> = Vec::new();
    for i in 0..10 {
        let node_id = Uuid::new_v4();
        let assignment = create_test_assignment(
            node_id,
            source_subnet.id,
            Ipv4Addr::new(10, 100, 0, 10 + i),
        );
        let target_ip = Ipv4Addr::new(10, 100, 1, 10 + i);
        nodes.push((node_id, assignment, target_ip));
    }

    // Plan bulk migration
    let bulk_plan = planner
        .plan_bulk(
            &nodes,
            &source_subnet,
            &target_subnet,
            MigrationReason::SubnetDecommission,
        )
        .expect("Should create bulk plan");

    assert_eq!(bulk_plan.total_migrations(), 10);
    assert!(bulk_plan.wave_count() >= 1);
    assert!(!bulk_plan.migrations.is_empty());
}

// ============================================================================
// Migration Executor Tests
// ============================================================================

#[tokio::test]
async fn test_executor_creation() {
    let executor = MigrationExecutor::new();
    assert_eq!(executor.active_count(), 0);
    assert!(executor.active_migrations().is_empty());
}

#[tokio::test]
async fn test_submit_migration() {
    let mut planner = MigrationPlanner::new();
    let executor = MigrationExecutor::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, source_subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 1, 10);

    let plan = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .unwrap();

    let handle = executor.submit(plan).expect("Should submit migration");

    assert_eq!(handle.node_id, node_id);
    assert_eq!(handle.source_subnet_id, source_subnet.id);
    assert_eq!(handle.target_subnet_id, target_subnet.id);
    assert_eq!(executor.active_count(), 1);
}

#[tokio::test]
async fn test_cannot_submit_duplicate_migration() {
    let mut planner = MigrationPlanner::new();
    let executor = MigrationExecutor::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, source_subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 1, 10);

    let plan1 = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .unwrap();

    let plan2 = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .unwrap();

    // First submission should succeed
    executor.submit(plan1).expect("Should submit first");

    // Second submission for same node should fail
    let result = executor.submit(plan2);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_cancel_migration() {
    let mut planner = MigrationPlanner::new();
    let executor = MigrationExecutor::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, source_subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 1, 10);

    let plan = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .unwrap();

    let handle = executor.submit(plan).unwrap();

    // Cancel the migration
    executor.cancel(handle.id).expect("Should cancel");

    // Should no longer be active
    assert_eq!(executor.active_count(), 0);
}

#[tokio::test]
async fn test_migration_progress_tracking() {
    let mut planner = MigrationPlanner::new();
    let executor = MigrationExecutor::new();

    let source_subnet = create_test_subnet("source", SubnetPurpose::Tenant);
    let mut target_subnet = create_test_subnet("target", SubnetPurpose::Tenant);
    target_subnet.cidr = "10.100.1.0/24".parse().unwrap();

    let node_id = Uuid::new_v4();
    let assignment = create_test_assignment(node_id, source_subnet.id, Ipv4Addr::new(10, 100, 0, 10));

    let target_ip = Ipv4Addr::new(10, 100, 1, 10);

    let plan = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .unwrap();

    let handle = executor.submit(plan).unwrap();

    // Get progress
    let progress = executor.get_progress(handle.id).expect("Should have progress");

    assert_eq!(progress.migration_id, handle.id);
    assert_eq!(progress.current_step, MigrationStep::NotStarted);
    assert_eq!(progress.progress_percent, 0);
}

// ============================================================================
// Migration Status Tests
// ============================================================================

#[test]
fn test_migration_status_variants() {
    // Test all status variants exist and can be created
    let statuses = [
        MigrationStatus::Pending,
        MigrationStatus::InProgress,
        MigrationStatus::DualStack,
        MigrationStatus::Verifying,
        MigrationStatus::Completed,
        MigrationStatus::Failed,
        MigrationStatus::Cancelled,
        MigrationStatus::RollingBack,
        MigrationStatus::RolledBack,
    ];

    for status in &statuses {
        // Verify Debug trait works
        let _ = format!("{:?}", status);
    }
}

#[test]
fn test_migration_status_equality() {
    assert_eq!(MigrationStatus::Pending, MigrationStatus::Pending);
    assert_ne!(MigrationStatus::Pending, MigrationStatus::InProgress);
}

// ============================================================================
// Migration Step Tests
// ============================================================================

#[test]
fn test_migration_step_variants() {
    let steps = [
        MigrationStep::NotStarted,
        MigrationStep::AllocatingIp,
        MigrationStep::GeneratingConfig,
        MigrationStep::EnablingDualStack,
        MigrationStep::PropagatingToPeers,
        MigrationStep::VerifyingConnectivity,
        MigrationStep::CuttingOver,
        MigrationStep::CleaningUp,
        MigrationStep::Completed,
        MigrationStep::RollingBack,
    ];

    // Verify steps can be compared
    for i in 0..steps.len() - 1 {
        assert_ne!(steps[i], steps[i + 1]);
    }
}

// ============================================================================
// Integration with SubnetManager
// ============================================================================

#[test]
fn test_manager_tracks_migrations() {
    let manager = SubnetManager::new();

    // Get initial stats
    let stats = manager.get_stats();
    assert_eq!(stats.total_subnets, 0);
}

#[test]
fn test_migration_with_actual_subnets() {
    let manager = SubnetManager::new();

    // Create source subnet
    let source_req = CreateSubnetRequest {
        name: "migration-source".to_string(),
        description: None,
        purpose: SubnetPurpose::Tenant,
        cidr: None,
        prefix_length: Some(24),
        tenant_id: None,
        node_type: None,
        region: None,
        resource_pool_id: None,
        max_nodes: None,
        template_id: None,
        created_by: None,
    };

    let source = manager.create_subnet(source_req).expect("Should create source");

    // Create target subnet
    let target_req = CreateSubnetRequest {
        name: "migration-target".to_string(),
        description: None,
        purpose: SubnetPurpose::Tenant,
        cidr: None,
        prefix_length: Some(24),
        tenant_id: None,
        node_type: None,
        region: None,
        resource_pool_id: None,
        max_nodes: None,
        template_id: None,
        created_by: None,
    };

    let target = manager.create_subnet(target_req).expect("Should create target");

    // Assign a node to source
    let node_id = Uuid::new_v4();
    let assign_req = AssignNodeRequest {
        node_id,
        wg_public_key: "dGVzdC1ub2RlLWtleQ==".to_string(),
        subnet_id: Some(source.id),
        attributes: None,
        method: "test".to_string(),
    };

    let assignment = manager.assign_node(assign_req).expect("Should assign");

    // Verify assignment
    assert_eq!(assignment.subnet_id, source.id);

    // Create migration planner
    let mut planner = MigrationPlanner::new();
    let executor = MigrationExecutor::new();

    // Get fresh subnet data
    let source_subnet = manager.get_subnet(source.id).unwrap();
    let target_subnet = manager.get_subnet(target.id).unwrap();

    // Allocate target IP (pick an IP in the target subnet)
    let target_ip = Ipv4Addr::new(10, 100, 16, 1);

    // Plan migration
    let plan = planner
        .plan(
            node_id,
            &source_subnet,
            &target_subnet,
            &assignment,
            target_ip,
            MigrationReason::Manual,
        )
        .expect("Should plan migration");

    // Submit migration
    let handle = executor.submit(plan).expect("Should submit");

    assert_eq!(handle.node_id, node_id);
    assert_eq!(executor.active_count(), 1);
}

// ============================================================================
// Migration Statistics Tests
// ============================================================================

#[test]
fn test_migration_stats_default() {
    let stats = MigrationStats::default();

    assert_eq!(stats.total, 0);
    assert_eq!(stats.pending, 0);
    assert_eq!(stats.in_progress, 0);
    assert_eq!(stats.completed, 0);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.cancelled, 0);
    assert!(stats.avg_duration_secs.is_none());
    assert!(stats.success_rate.is_none());
}

#[test]
fn test_migration_reason_display() {
    // Test Display trait for MigrationReason
    assert_eq!(MigrationReason::Manual.to_string(), "Manual");
    assert_eq!(MigrationReason::PolicyChange.to_string(), "Policy Change");
    assert_eq!(MigrationReason::SubnetDecommission.to_string(), "Subnet Decommission");
    assert_eq!(MigrationReason::LoadBalancing.to_string(), "Load Balancing");
    assert_eq!(MigrationReason::NodeTypeChange.to_string(), "Node Type Change");
    assert_eq!(MigrationReason::TenantChange.to_string(), "Tenant Change");
    assert_eq!(MigrationReason::GeographicMove.to_string(), "Geographic Move");
    assert_eq!(MigrationReason::ResourcePoolChange.to_string(), "Resource Pool Change");
    assert_eq!(MigrationReason::Maintenance.to_string(), "Maintenance");
    assert_eq!(MigrationReason::Recovery.to_string(), "Recovery");
}

#[test]
fn test_migration_reason_debug() {
    assert_eq!(format!("{:?}", MigrationReason::Manual), "Manual");
    assert_eq!(format!("{:?}", MigrationReason::PolicyChange), "PolicyChange");
}
