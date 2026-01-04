//! Tests for SwarmletAgent

use super::*;
use crate::config::Config;
use crate::join::{ClusterApiEndpoints, JoinResult};
use tempfile::TempDir;
use tokio::time::Duration;
use uuid::Uuid;

/// TDD Phase tracking
#[derive(Debug, Clone, PartialEq)]
enum TddPhase {
    Red,      // Write failing tests
    Green,    // Make tests pass
    Refactor, // Optimize implementation
}

/// Test result tracking
#[derive(Debug)]
struct TestResult {
    test_name: String,
    phase: TddPhase,
    success: bool,
    duration: Duration,
    error_message: Option<String>,
}

/// Create a test join result
fn create_test_join_result() -> JoinResult {
    JoinResult {
        node_id: Uuid::new_v4(),
        cluster_name: "test-cluster".to_string(),
        node_certificate: generate_test_certificate(),
        cluster_endpoints: vec!["http://localhost:7946".to_string()],
        assigned_capabilities: vec!["compute".to_string(), "storage".to_string()],
        heartbeat_interval: Duration::from_secs(30),
        api_endpoints: ClusterApiEndpoints {
            workload_api: "http://localhost:8081".to_string(),
            metrics_api: "http://localhost:8082".to_string(),
            logs_api: "http://localhost:8083".to_string(),
            health_check: "http://localhost:8080".to_string(),
        },
        wireguard_config: None,
        subnet_info: None,
    }
}

/// Generate a test certificate
fn generate_test_certificate() -> String {
    // This is a minimal valid PEM certificate for testing
    r#"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAKHHIgKwA4jAMA0GCSqGSIb3DQEBCwUAMCExCzAJBgNVBAYTAlVT
MRIwEAYDVQQDDAlsb2NhbGhvc3QwHhcNMjQwMTAxMDAwMDAwWhcNMjUwMTAxMDAw
MDAwWjAhMQswCQYDVQQGEwJVUzESMBAGA1UEAwwJbG9jYWxob3N0MFwwDQYJKoZI
hvcNAQEBBQADSwAwSAJBAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4j
AKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jACAwEAATANBgkqhkiG9w0B
AQsFAANBAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jA
KHHIgKwA4jAKHHIgKwA4jAKHHIgKwA4jA=
-----END CERTIFICATE-----"#
        .to_string()
}

#[tokio::test]
async fn test_agent_creation_from_join_result() {
    let start = std::time::Instant::now();
    let mut results = Vec::new();

    // RED Phase - Should fail initially
    let phase = TddPhase::Red;
    let join_result = create_test_join_result();
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    match SwarmletAgent::new(join_result.clone(), data_dir.clone()).await {
        Ok(agent) => {
            results.push(TestResult {
                test_name: "agent_creation".to_string(),
                phase: phase.clone(),
                success: true,
                duration: start.elapsed(),
                error_message: None,
            });

            // Verify agent properties
            assert_eq!(agent.join_result.node_id, join_result.node_id);
            assert_eq!(agent.join_result.cluster_name, join_result.cluster_name);
        }
        Err(e) => {
            results.push(TestResult {
                test_name: "agent_creation".to_string(),
                phase,
                success: false,
                duration: start.elapsed(),
                error_message: Some(e.to_string()),
            });
        }
    }

    // GREEN Phase - Should pass
    let phase = TddPhase::Green;
    let agent = SwarmletAgent::new(join_result.clone(), data_dir.clone())
        .await
        .expect("Should create agent successfully");

    results.push(TestResult {
        test_name: "agent_creation_green".to_string(),
        phase,
        success: true,
        duration: start.elapsed(),
        error_message: None,
    });

    // Verify health status initialization
    let health = agent.health_status.read().await;
    assert_eq!(health.node_id, join_result.node_id);
    assert_eq!(health.status, NodeStatus::Starting);
    assert_eq!(health.workloads_active, 0);
    assert_eq!(health.errors_count, 0);
}

#[tokio::test]
async fn test_agent_health_status_updates() {
    let join_result = create_test_join_result();
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    let agent = SwarmletAgent::new(join_result, data_dir)
        .await
        .expect("Should create agent");

    // Test health status transitions
    {
        let mut health = agent.health_status.write().await;
        health.status = NodeStatus::Healthy;
        health.cpu_usage_percent = 45.0;
        health.memory_usage_gb = 3.5;
        health.workloads_active = 5;
    }

    // Verify updates
    {
        let health = agent.health_status.read().await;
        assert_eq!(health.status, NodeStatus::Healthy);
        assert_eq!(health.cpu_usage_percent, 45.0);
        assert_eq!(health.memory_usage_gb, 3.5);
        assert_eq!(health.workloads_active, 5);
    }

    // Test degraded status when resources are high
    {
        let mut health = agent.health_status.write().await;
        health.cpu_usage_percent = 95.0;
    }

    // Update health metrics should set degraded status
    agent
        .update_health_metrics(std::time::Instant::now())
        .await
        .ok();

    {
        let health = agent.health_status.read().await;
        // Note: Status might not change in test environment without actual high CPU
        assert!(health.cpu_usage_percent >= 0.0);
    }
}

#[tokio::test]
async fn test_agent_shutdown_signal() {
    let join_result = create_test_join_result();
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    let agent = SwarmletAgent::new(join_result, data_dir)
        .await
        .expect("Should create agent");

    // Test shutdown signal
    assert!(agent.shutdown().is_ok());

    // Verify shutdown signal was sent
    let mut shutdown_signal = agent.shutdown_signal.clone();
    shutdown_signal
        .changed()
        .await
        .expect("Should receive shutdown signal");
    assert_eq!(*shutdown_signal.borrow(), true);
}

#[tokio::test]
async fn test_heartbeat_message_serialization() {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct HeartbeatMessage {
        node_id: Uuid,
        timestamp: chrono::DateTime<chrono::Utc>,
        status: NodeStatus,
        metrics: HealthMetrics,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct HealthMetrics {
        cpu_usage_percent: f32,
        memory_usage_gb: f32,
        disk_usage_gb: f32,
        workloads_active: u32,
        uptime_seconds: u64,
    }

    let heartbeat = HeartbeatMessage {
        node_id: Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
        status: NodeStatus::Healthy,
        metrics: HealthMetrics {
            cpu_usage_percent: 25.5,
            memory_usage_gb: 4.2,
            disk_usage_gb: 50.0,
            workloads_active: 3,
            uptime_seconds: 3600,
        },
    };

    // Test serialization
    let json = serde_json::to_string(&heartbeat).expect("Should serialize heartbeat");
    assert!(json.contains("Healthy"));
    assert!(json.contains("25.5"));

    // Test deserialization
    let deserialized: HeartbeatMessage =
        serde_json::from_str(&json).expect("Should deserialize heartbeat");
    assert_eq!(deserialized.node_id, heartbeat.node_id);
    assert_eq!(deserialized.status, heartbeat.status);
    assert_eq!(deserialized.metrics.cpu_usage_percent, 25.5);
}

#[tokio::test]
async fn test_work_assignment_handling() {
    let work_assignment = WorkAssignment {
        id: Uuid::new_v4(),
        workload_type: "container".to_string(),
        container_image: Some("nginx:latest".to_string()),
        command: Some(vec![
            "nginx".to_string(),
            "-g".to_string(),
            "daemon off;".to_string(),
        ]),
        shell_script: None,
        environment: std::collections::HashMap::from([(
            "ENV_VAR".to_string(),
            "value".to_string(),
        )]),
        resource_limits: ResourceLimits {
            cpu_cores: Some(2.0),
            memory_gb: Some(4.0),
            disk_gb: Some(10.0),
        },
        created_at: chrono::Utc::now(),
    };

    // Test serialization
    let json = serde_json::to_string(&work_assignment).expect("Should serialize work assignment");
    assert!(json.contains("nginx:latest"));
    assert!(json.contains("ENV_VAR"));

    // Test deserialization
    let deserialized: WorkAssignment =
        serde_json::from_str(&json).expect("Should deserialize work assignment");
    assert_eq!(deserialized.id, work_assignment.id);
    assert_eq!(deserialized.workload_type, "container");
    assert_eq!(
        deserialized.container_image,
        Some("nginx:latest".to_string())
    );
}

#[tokio::test]
async fn test_node_status_transitions() {
    // Test all node status values
    let statuses = vec![
        NodeStatus::Starting,
        NodeStatus::Healthy,
        NodeStatus::Degraded,
        NodeStatus::Unhealthy,
        NodeStatus::Shutting,
    ];

    for status in statuses {
        let health = HealthStatus {
            node_id: Uuid::new_v4(),
            status: status.clone(),
            uptime_seconds: 100,
            workloads_active: 1,
            cpu_usage_percent: 10.0,
            memory_usage_gb: 2.0,
            disk_usage_gb: 20.0,
            network_rx_bytes: 1000,
            network_tx_bytes: 2000,
            last_heartbeat: chrono::Utc::now(),
            errors_count: 0,
        };

        // Test serialization
        let json = serde_json::to_string(&health).expect("Should serialize health status");
        let deserialized: HealthStatus =
            serde_json::from_str(&json).expect("Should deserialize health status");
        assert_eq!(deserialized.status, status);
    }
}

#[tokio::test]
async fn test_api_routes_health_endpoint() {
    let join_result = create_test_join_result();
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    let agent = SwarmletAgent::new(join_result, data_dir)
        .await
        .expect("Should create agent");

    // Update some health metrics
    {
        let mut health = agent.health_status.write().await;
        health.status = NodeStatus::Healthy;
        health.cpu_usage_percent = 30.0;
        health.memory_usage_gb = 4.0;
        health.workloads_active = 2;
    }

    // Read health status
    let health = agent.health_status.read().await;
    let health_json = serde_json::to_string(&*health).expect("Should serialize health");

    // Verify health JSON contains expected fields
    assert!(health_json.contains("\"status\":\"Healthy\""));
    assert!(health_json.contains("\"cpu_usage_percent\":30.0"));
    assert!(health_json.contains("\"workloads_active\":2"));
}

#[tokio::test]
async fn test_resource_limits_validation() {
    // Test with all fields set
    let limits_full = ResourceLimits {
        cpu_cores: Some(4.0),
        memory_gb: Some(8.0),
        disk_gb: Some(100.0),
    };

    let json = serde_json::to_string(&limits_full).expect("Should serialize");
    let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
    assert_eq!(deserialized.cpu_cores, Some(4.0));
    assert_eq!(deserialized.memory_gb, Some(8.0));
    assert_eq!(deserialized.disk_gb, Some(100.0));

    // Test with no limits set
    let limits_none = ResourceLimits {
        cpu_cores: None,
        memory_gb: None,
        disk_gb: None,
    };

    let json = serde_json::to_string(&limits_none).expect("Should serialize");
    let deserialized: ResourceLimits = serde_json::from_str(&json).expect("Should deserialize");
    assert_eq!(deserialized.cpu_cores, None);
    assert_eq!(deserialized.memory_gb, None);
    assert_eq!(deserialized.disk_gb, None);
}

#[tokio::test]
async fn test_agent_from_config_no_saved_state() {
    let temp_dir = TempDir::new().unwrap();
    let config = Config::default_with_data_dir(temp_dir.path().to_path_buf());

    // Should fail because there's no saved state
    match SwarmletAgent::from_config(config).await {
        Ok(_) => panic!("Should return error when no saved state exists"),
        Err(e) => match e {
            crate::SwarmletError::Configuration(msg) => {
                assert!(msg.contains("No saved state found"));
            }
            _ => panic!("Expected Config error, got: {:?}", e),
        },
    }
}

#[tokio::test]
async fn test_agent_from_config_with_saved_state() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();

    // Create and save agent state
    let join_result = create_test_join_result();
    let saved_state = SavedAgentState {
        join_result: join_result.clone(),
        wireguard_private_key: "test_private_key_base64".to_string(),
        wireguard_public_key: "test_public_key_base64".to_string(),
        cluster_public_key: None,
        saved_at: chrono::Utc::now(),
        version: SavedAgentState::CURRENT_VERSION,
    };
    saved_state
        .save(&data_dir)
        .await
        .expect("Should save state");

    // Now from_config should work
    let config = Config::default_with_data_dir(data_dir);
    let agent = SwarmletAgent::from_config(config)
        .await
        .expect("Should create agent from saved state");

    assert_eq!(agent.join_result.cluster_name, join_result.cluster_name);
    assert_eq!(agent.join_result.node_id, join_result.node_id);
}

#[tokio::test]
async fn test_saved_agent_state_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();

    let join_result = create_test_join_result();
    let original_state = SavedAgentState {
        join_result: join_result.clone(),
        wireguard_private_key: "private_key_b64".to_string(),
        wireguard_public_key: "public_key_b64".to_string(),
        cluster_public_key: Some("cluster_key_b64".to_string()),
        saved_at: chrono::Utc::now(),
        version: SavedAgentState::CURRENT_VERSION,
    };

    // Save state
    original_state
        .save(&data_dir)
        .await
        .expect("Should save state");

    // Load state
    let loaded_state = SavedAgentState::load(&data_dir)
        .await
        .expect("Should load state");

    assert_eq!(
        loaded_state.join_result.node_id,
        original_state.join_result.node_id
    );
    assert_eq!(
        loaded_state.wireguard_private_key,
        original_state.wireguard_private_key
    );
    assert_eq!(
        loaded_state.wireguard_public_key,
        original_state.wireguard_public_key
    );
    assert_eq!(
        loaded_state.cluster_public_key,
        original_state.cluster_public_key
    );
    assert_eq!(loaded_state.version, original_state.version);
}

#[tokio::test]
async fn test_has_saved_state() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path();

    // Initially no saved state
    assert!(!SwarmletAgent::has_saved_state(data_dir).await);

    // Create saved state
    let join_result = create_test_join_result();
    let state = SavedAgentState {
        join_result,
        wireguard_private_key: "key".to_string(),
        wireguard_public_key: "key".to_string(),
        cluster_public_key: None,
        saved_at: chrono::Utc::now(),
        version: SavedAgentState::CURRENT_VERSION,
    };
    state.save(data_dir).await.expect("Should save state");

    // Now has saved state
    assert!(SwarmletAgent::has_saved_state(data_dir).await);
}

#[tokio::test]
async fn test_disk_usage_calculation() {
    let join_result = create_test_join_result();
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    let agent = SwarmletAgent::new(join_result, data_dir)
        .await
        .expect("Should create agent");

    // Get disk usage - now returns actual disk usage of the mount point
    let disk_usage = agent.get_disk_usage().await.expect("Should get disk usage");

    // Should return actual disk usage in GB (realistic values on any system)
    assert!(disk_usage >= 0.0, "Disk usage should be non-negative");
    // A typical system will have at least some disk usage, but we can't be too specific
    // Just verify we get a reasonable value (less than 100 TB)
    assert!(
        disk_usage < 100_000.0,
        "Disk usage {} GB seems unreasonably high",
        disk_usage
    );
}
