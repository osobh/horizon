//! Test to verify swarm_fault_tolerance module structure works correctly

use stratoswarm_evolution_engines::swarm_fault_tolerance::{
    AlertThresholds, CheckpointManager, CheckpointMetadata, CheckpointRecovery, CheckpointSnapshot,
    CheckpointStorage, CompressionAlgorithm, FailureDetectionAlgorithm, FaultDetector,
    HealthStatus, HybridRecovery, NodeHealth, NodeState, RecoveryEvent, RecoveryExecutor,
    RecoveryManager, RecoveryStatus, RedistributeRecovery, StorageType,
};

#[test]
fn test_fault_tolerance_module_imports() {
    // Test that we can create health status
    let status = HealthStatus::Healthy;
    assert!(matches!(status, HealthStatus::Healthy));

    // Test that enums work
    let _ = FailureDetectionAlgorithm::Heartbeat;
    let _ = RecoveryStatus::InProgress;
    let _ = CompressionAlgorithm::None;
    let _ = StorageType::Local;
}

#[test]
fn test_node_health_creation() {
    let health = NodeHealth {
        node_id: "test_node".to_string(),
        status: HealthStatus::Healthy,
        last_heartbeat: 1000,
        response_times: vec![10.0, 15.0],
        error_count: 0,
        cpu_utilization: 0.5,
        memory_utilization: 0.6,
        network_score: 0.9,
    };

    assert_eq!(health.node_id, "test_node");
    assert_eq!(health.status, HealthStatus::Healthy);
}

#[tokio::test]
async fn test_fault_detector_creation() {
    let config = Default::default();
    let detector = FaultDetector::new(config).await.unwrap();
    // Test detector can be created
    assert_eq!(detector.get_monitored_nodes().await.len(), 0);
}
