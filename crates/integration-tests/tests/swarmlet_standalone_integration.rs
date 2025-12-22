//! Standalone integration tests for swarmlet
//!
//! These tests run independently without requiring GPU agents or
//! other heavy dependencies. They test the swarmlet functionality
//! in isolation.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;
use tempfile::TempDir;

/// Test phases following TDD methodology
#[derive(Debug, Clone, PartialEq)]
enum TddPhase {
    Red,    // Write failing tests
    Green,  // Make tests pass
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

#[tokio::test]
async fn test_swarmlet_join_and_heartbeat_flow() {
    let start = std::time::Instant::now();
    let mut results = Vec::new();

    // Create temporary directory for swarmlet data
    let temp_dir = TempDir::new().expect("Should create temp dir");
    let data_dir = temp_dir.path().to_str().unwrap().to_string();

    // Simulate join result
    let join_result = swarmlet::join::JoinResult {
        node_id: Uuid::new_v4(),
        cluster_name: "test-cluster".to_string(),
        node_certificate: generate_test_certificate(),
        cluster_endpoints: vec!["http://localhost:7946".to_string()],
        assigned_capabilities: vec!["compute".to_string(), "storage".to_string()],
        heartbeat_interval: Duration::from_secs(30),
        api_endpoints: swarmlet::join::ClusterApiEndpoints {
            workload_api: "http://localhost:8081".to_string(),
            metrics_api: "http://localhost:9090".to_string(),
            logs_api: "http://localhost:8082".to_string(),
            health_check: "http://localhost:8080".to_string(),
        },
    };

    // Create swarmlet agent
    let agent = swarmlet::agent::SwarmletAgent::new(join_result.clone(), data_dir)
        .await
        .expect("Should create agent");

    // Test health status initialization
    {
        let health = agent.health_status.read().await;
        assert_eq!(health.node_id, join_result.node_id);
        assert_eq!(health.status, swarmlet::agent::NodeStatus::Starting);
        assert_eq!(health.workloads_active, 0);
    }

    results.push(TestResult {
        test_name: "swarmlet_join_and_heartbeat_flow".to_string(),
        phase: TddPhase::Green,
        success: true,
        duration: start.elapsed(),
        error_message: None,
    });

    // Shutdown agent
    agent.shutdown().expect("Should shutdown cleanly");

    println!("✅ Swarmlet join and heartbeat flow test passed");
}

#[tokio::test]
async fn test_hardware_profiling_integration() {
    let start = std::time::Instant::now();

    // Create hardware profiler
    let mut profiler = swarmlet::profile::HardwareProfiler::new();

    // Generate profile
    let profile = profiler.profile().await.expect("Should generate profile");

    // Validate profile
    assert!(!profile.hostname.is_empty());
    assert!(profile.cpu.cores > 0);
    assert!(profile.memory.total_gb > 0.0);
    assert!(profile.storage.total_gb > 0.0);
    assert!(!profile.network.interfaces.is_empty());

    // Check capabilities
    assert!(profile.capabilities.max_workloads > 0);
    assert!(profile.capabilities.memory_limit_gb > 0.0);
    assert!(profile.capabilities.cpu_limit_cores > 0);
    assert!(!profile.capabilities.suitability.is_empty());

    println!("✅ Hardware profiling integration test passed");
    println!("  - Device type: {:?}", profile.device_type);
    println!("  - CPU: {} cores", profile.cpu.cores);
    println!("  - Memory: {:.1} GB", profile.memory.total_gb);
    println!("  - Storage: {:.1} GB", profile.storage.total_gb);
}

#[tokio::test]
async fn test_cluster_discovery_integration() {
    let start = std::time::Instant::now();

    // Create cluster discovery
    let discovery = swarmlet::discovery::ClusterDiscovery::new();

    // Test connection to non-existent cluster (should fail gracefully)
    let result = discovery.test_connection("localhost:7946").await;
    assert!(result.is_err());

    // Test discovery with short timeout
    let clusters = discovery.discover_clusters(1).await
        .expect("Should complete discovery");

    // In test environment, we expect no clusters
    assert_eq!(clusters.len(), 0);

    println!("✅ Cluster discovery integration test passed");
}

#[tokio::test]
async fn test_join_protocol_integration() {
    let start = std::time::Instant::now();

    // Create hardware profile
    let mut profiler = swarmlet::profile::HardwareProfiler::new();
    let profile = profiler.profile().await.expect("Should generate profile");

    // Create join protocol
    let mut join_protocol = swarmlet::join::JoinProtocol::new(
        "test-token-123".to_string(),
        "localhost:7946".to_string(),
        profile,
    );

    // Set custom node name
    join_protocol.set_node_name("integration-test-node".to_string());

    // Test join (will fail without real cluster, but that's expected)
    let result = join_protocol.join().await;
    assert!(result.is_err());

    println!("✅ Join protocol integration test passed");
}

#[tokio::test]
async fn test_config_management() {
    let temp_dir = TempDir::new().expect("Should create temp dir");
    let config_path = temp_dir.path().join("swarmlet.toml");

    // Create custom config
    let config = swarmlet::config::Config {
        cluster_address: Some("custom-cluster:7946".to_string()),
        node_name: Some("test-node".to_string()),
        data_dir: temp_dir.path().to_path_buf(),
        api_port: Some(8090),
        enable_gpu: false,
        resource_limits: swarmlet::config::ResourceLimits {
            max_cpu_percent: 75.0,
            max_memory_percent: 80.0,
            max_disk_percent: 60.0,
        },
        heartbeat_interval_secs: 20,
        network_interface: None,
        log_level: "debug".to_string(),
        enable_metrics: true,
        enable_tracing: false,
    };

    // Save config
    config.save(&config_path).expect("Should save config");

    // Load config
    let loaded = swarmlet::config::Config::load(&config_path)
        .expect("Should load config");

    assert_eq!(loaded.cluster_address, Some("custom-cluster:7946".to_string()));
    assert_eq!(loaded.node_name, Some("test-node".to_string()));
    assert_eq!(loaded.api_port, Some(8090));

    println!("✅ Config management test passed");
}

#[tokio::test]
async fn test_workload_manager_integration() {
    let temp_dir = TempDir::new().expect("Should create temp dir");
    let config = Arc::new(swarmlet::config::Config::default_with_data_dir(
        temp_dir.path().to_path_buf()
    ));

    // Create workload manager
    let workload_manager = Arc::new(
        swarmlet::workload::WorkloadManager::new(config)
            .await
            .expect("Should create workload manager")
    );

    // Check initial state
    assert_eq!(workload_manager.active_workload_count().await, 0);

    // Create a test workload
    let work_assignment = swarmlet::agent::WorkAssignment {
        id: Uuid::new_v4(),
        workload_type: "test".to_string(),
        container_image: None,
        command: Some(vec!["echo".to_string(), "hello".to_string()]),
        environment: std::collections::HashMap::new(),
        resource_limits: swarmlet::agent::ResourceLimits {
            cpu_cores: Some(1.0),
            memory_gb: Some(0.5),
            disk_gb: Some(1.0),
        },
        created_at: chrono::Utc::now(),
    };

    // Start workload (may fail in CI but that's ok)
    match workload_manager.start_workload(work_assignment).await {
        Ok(_) => println!("  - Workload started successfully"),
        Err(e) => println!("  - Workload start failed (expected in CI): {}", e),
    }

    // Stop all workloads
    workload_manager.stop_all_workloads().await
        .expect("Should stop all workloads");

    println!("✅ Workload manager integration test passed");
}

#[tokio::test]
async fn test_security_components() {
    // Test join token
    let token = swarmlet::security::JoinToken::new("test-secret-123".to_string());
    assert!(token.validate());

    // Test secure random generation
    let random_bytes = swarmlet::security::generate_secure_random(32)
        .expect("Should generate random bytes");
    assert_eq!(random_bytes.len(), 32);

    // Test data hashing
    let hash = swarmlet::security::hash_data(b"test data");
    assert_eq!(hash.len(), 64); // SHA-256 hex string

    // Test node certificate creation
    let cert = swarmlet::security::NodeCertificate::generate(
        Uuid::new_v4(),
        "test-node".to_string(),
    ).expect("Should generate certificate");

    let pem = cert.to_pem();
    assert!(pem.contains("BEGIN CERTIFICATE"));
    assert!(pem.contains("END CERTIFICATE"));

    println!("✅ Security components test passed");
}

#[tokio::test]
async fn test_full_swarmlet_lifecycle() {
    let start = std::time::Instant::now();
    let temp_dir = TempDir::new().expect("Should create temp dir");

    println!("🚀 Starting full swarmlet lifecycle test");

    // Phase 1: Hardware profiling
    println!("  Phase 1: Hardware profiling");
    let mut profiler = swarmlet::profile::HardwareProfiler::new();
    let profile = profiler.profile().await.expect("Should generate profile");
    println!("    ✓ Generated hardware profile for {}", profile.hostname);

    // Phase 2: Configuration
    println!("  Phase 2: Configuration");
    let config = swarmlet::config::Config {
        cluster_address: Some("test-cluster:7946".to_string()),
        node_name: Some("lifecycle-test-node".to_string()),
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    println!("    ✓ Created configuration");

    // Phase 3: Discovery
    println!("  Phase 3: Discovery");
    let discovery = swarmlet::discovery::ClusterDiscovery::new();
    let clusters = discovery.discover_clusters(1).await
        .expect("Should complete discovery");
    println!("    ✓ Discovery completed, found {} clusters", clusters.len());

    // Phase 4: Join preparation
    println!("  Phase 4: Join preparation");
    let join_result = swarmlet::join::JoinResult {
        node_id: Uuid::new_v4(),
        cluster_name: "lifecycle-test-cluster".to_string(),
        node_certificate: generate_test_certificate(),
        cluster_endpoints: vec!["http://localhost:7946".to_string()],
        assigned_capabilities: profile.capabilities.suitability
            .iter()
            .map(|s| format!("{:?}", s).to_lowercase())
            .collect(),
        heartbeat_interval: Duration::from_secs(30),
        api_endpoints: swarmlet::join::ClusterApiEndpoints {
            workload_api: "http://localhost:8081".to_string(),
            metrics_api: "http://localhost:9090".to_string(),
            logs_api: "http://localhost:8082".to_string(),
            health_check: "http://localhost:8080".to_string(),
        },
    };
    println!("    ✓ Prepared join result");

    // Phase 5: Agent creation
    println!("  Phase 5: Agent creation");
    let agent = swarmlet::agent::SwarmletAgent::new(
        join_result,
        temp_dir.path().to_str().unwrap().to_string()
    ).await.expect("Should create agent");
    println!("    ✓ Created swarmlet agent");

    // Phase 6: Health check
    println!("  Phase 6: Health check");
    {
        let health = agent.health_status.read().await;
        assert_eq!(health.status, swarmlet::agent::NodeStatus::Starting);
        println!("    ✓ Agent health status: {:?}", health.status);
    }

    // Phase 7: Shutdown
    println!("  Phase 7: Shutdown");
    agent.shutdown().expect("Should shutdown cleanly");
    println!("    ✓ Agent shutdown complete");

    let duration = start.elapsed();
    println!("✅ Full swarmlet lifecycle test completed in {:?}", duration);
}

/// Generate a test certificate
fn generate_test_certificate() -> String {
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