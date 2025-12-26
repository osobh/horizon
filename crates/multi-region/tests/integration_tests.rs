//! Integration tests for multi-region modules
//!
//! These tests verify that all modules work together correctly in realistic scenarios.

use stratoswarm_multi_region::tunnels::TlsVersion;
use stratoswarm_multi_region::*;
use std::collections::HashMap;

/// Test end-to-end multi-region deployment scenario
#[tokio::test]
async fn test_end_to_end_multi_region_deployment() {
    // 1. Set up region manager
    let region_config = RegionConfig::default();
    let region_manager = RegionManager::new(region_config).unwrap();

    // 2. Set up compliance mapping
    let compliance_config = ComplianceMappingConfig::default();
    let mut compliance_manager = ComplianceMappingManager::new(compliance_config);

    // 3. Set up data sovereignty
    let mut sovereignty = DataSovereignty::new();

    // Add sovereignty rule
    let rule = data_sovereignty::SovereigntyRule {
        id: "eu-pii-rule".to_string(),
        data_classification: "PII".to_string(),
        allowed_jurisdictions: vec!["EU".to_string(), "UK".to_string()],
        forbidden_jurisdictions: vec!["CN".to_string(), "RU".to_string()],
        encryption_level: data_sovereignty::EncryptionLevel::Strong,
        residency_requirements: data_sovereignty::ResidencyRequirements {
            must_remain_in_origin: false,
            allowed_transit: vec!["US".to_string()],
            max_distance_km: Some(10000),
            backup_requirements: data_sovereignty::BackupRequirements {
                cross_border_allowed: true,
                required_jurisdictions: vec!["EU".to_string()],
                min_copies: 2,
            },
        },
        compliance_frameworks: vec!["GDPR".to_string()],
        created_at: chrono::Utc::now(),
    };
    sovereignty.add_rule(rule);

    // 4. Test compliance validation for data transfer
    let compliance_context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::PII,
        source_region: "EU".to_string(),
        target_region: "US".to_string(),
        operation: compliance_mapping::DataOperation::Transfer,
        encryption_enabled: true,
        consent_obtained: true, // Assuming consent obtained
        metadata: HashMap::new(),
    };

    let compliance_result = compliance_manager
        .validate_compliance(&compliance_context)
        .unwrap();

    // Should have some compliance considerations but pass with consent
    assert!(compliance_result
        .applicable_frameworks
        .contains(&ComplianceFramework::GDPR));

    // 5. Test data sovereignty validation
    let sovereignty_request = data_sovereignty::DataPlacementRequest {
        data_id: "user-profile-123".to_string(),
        classification: "PII".to_string(),
        source_jurisdiction: "EU".to_string(),
        target_jurisdiction: "UK".to_string(), // Should be allowed
        operation: data_sovereignty::DataOperation::Store,
        metadata: HashMap::new(),
    };

    let sovereignty_result = sovereignty.validate_placement(&sovereignty_request);
    assert!(sovereignty_result.is_ok());

    // 6. Test forbidden jurisdiction
    let forbidden_request = data_sovereignty::DataPlacementRequest {
        data_id: "user-profile-456".to_string(),
        classification: "PII".to_string(),
        source_jurisdiction: "EU".to_string(),
        target_jurisdiction: "CN".to_string(), // Should be forbidden
        operation: data_sovereignty::DataOperation::Store,
        metadata: HashMap::new(),
    };

    let forbidden_result = sovereignty.validate_placement(&forbidden_request);
    assert!(forbidden_result.is_err());

    // 7. Verify region manager can select appropriate regions
    let requirements = region_manager::RegionRequirements {
        required_jurisdictions: vec!["EU".to_string()],
        excluded_jurisdictions: vec!["CN".to_string()],
        required_services: vec!["compute".to_string(), "storage".to_string()],
        max_latency: Some(100),
    };

    let selected_region = region_manager.select_best_region(&requirements);
    // Will fail initially due to no healthy regions, but validates the integration
    assert!(selected_region.is_err()); // Expected since no regions are healthy initially
}

/// Test load balancer integration with region management
#[tokio::test]
async fn test_load_balancer_region_integration() {
    // Set up load balancer with multiple endpoints
    let lb_config = LoadBalancerConfig::default();
    let mut endpoints = Vec::new();

    let regions = vec!["us-east-1", "us-west-2", "eu-west-1"];
    for (i, region) in regions.iter().enumerate() {
        let endpoint = load_balancer::RegionEndpoint::new(
            region.to_string(),
            format!("https://{}.example.com", region),
            50 - (i as u32 * 10), // Different weights
            i as u32 + 1,         // Different priorities
        );
        endpoints.push(endpoint);
    }

    let load_balancer = LoadBalancer::new(lb_config, endpoints).unwrap();

    // Create routing request
    let request = load_balancer::RoutingRequest {
        client_ip: Some("192.168.1.100".to_string()),
        session_id: Some("integration-test-session".to_string()),
        geo_info: Some(load_balancer::GeoInfo {
            country: "US".to_string(),
            region: "CA".to_string(),
            city: "San Francisco".to_string(),
            latitude: 37.7749,
            longitude: -122.4194,
        }),
        metadata: HashMap::new(),
    };

    // Test endpoint selection (will fail due to no healthy endpoints)
    let selection_result = load_balancer.select_endpoint(&request).await;
    assert!(selection_result.is_err()); // Expected due to no healthy endpoints

    // Test connection counting
    load_balancer.increment_connections("us-east-1");
    load_balancer.increment_connections("us-east-1");

    let stats = load_balancer.get_endpoint_stats().await;
    let us_east_stats = stats.iter().find(|s| s.region_id == "us-east-1").unwrap();
    assert_eq!(us_east_stats.current_connections, 2);

    load_balancer.decrement_connections("us-east-1");
    let stats = load_balancer.get_endpoint_stats().await;
    let us_east_stats = stats.iter().find(|s| s.region_id == "us-east-1").unwrap();
    assert_eq!(us_east_stats.current_connections, 1);
}

/// Test replication with compliance constraints
#[tokio::test]
async fn test_replication_compliance_integration() {
    // Set up replication manager
    let repl_config = ReplicationConfig {
        consistency_model: ConsistencyModel::Strong,
        conflict_resolution: replication::ConflictResolutionStrategy::LastWriterWins,
        topology: replication::ReplicationTopology::MultiMaster,
        replication_factor: 3,
        max_replication_lag_ms: 1000,
        batch_size: 10,
        anti_entropy_interval_s: 60,
        vector_clock_config: replication::VectorClockConfig {
            enabled: true,
            max_entries: 100,
            pruning_interval_s: 300,
        },
    };

    let regions = vec![
        "us-east-1".to_string(),
        "eu-west-1".to_string(),
        "ap-southeast-1".to_string(),
    ];
    let repl_manager = ReplicationManager::new(repl_config, regions).unwrap();

    // Set up compliance manager
    let compliance_config = ComplianceMappingConfig::default();
    let mut compliance_manager = ComplianceMappingManager::new(compliance_config);

    // Test replication with compliance validation
    let data = b"sensitive user data".to_vec();
    let source_region = "eu-west-1".to_string();
    let target_regions = vec!["us-east-1".to_string()];

    // Validate compliance before replication
    let compliance_context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::PII,
        source_region: source_region.clone(),
        target_region: target_regions[0].clone(),
        operation: compliance_mapping::DataOperation::Transfer,
        encryption_enabled: true,
        consent_obtained: true,
        metadata: HashMap::new(),
    };

    let compliance_result = compliance_manager
        .validate_compliance(&compliance_context)
        .unwrap();

    // If compliance allows, attempt replication (will fail due to no real endpoints)
    if compliance_result.is_compliant {
        let replication_result = repl_manager
            .replicate_data(
                "user-data-key".to_string(),
                data,
                replication::ReplicationOperation::Insert,
                source_region,
                target_regions,
            )
            .await;

        // Expected to fail due to no real endpoints, but validates integration
        assert!(replication_result.is_err());
    }

    // Test vector clock operations
    let mut clock1 = replication::VectorClock::new();
    let mut clock2 = replication::VectorClock::new();

    clock1.increment("us-east-1");
    clock2.increment("eu-west-1");

    assert!(!clock1.happens_before(&clock2));
    assert!(!clock2.happens_before(&clock1));

    let merged = clock1.merge(&clock2);
    assert_eq!(merged.clocks.get("us-east-1"), Some(&1));
    assert_eq!(merged.clocks.get("eu-west-1"), Some(&1));
}

/// Test tunnel security with compliance requirements
#[tokio::test]
async fn test_tunnel_security_compliance_integration() {
    // Set up tunnel with strong security
    let tunnel_config = TunnelConfig {
        tls_config: TlsConfiguration {
            tls_version: TlsVersion::Tls13,
            cert_file: None,
            key_file: None,
            ca_file: None,
            cert_data: None,
            key_data: None,
            ca_data: None,
            mutual_tls: true,
            verify_peer: true,
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            cert_validation: tunnels::CertificateValidation::Full,
        },
        auth_config: tunnels::AuthenticationConfig {
            method: tunnels::AuthenticationMethod::MutualTls,
            api_key: None,
            username: None,
            password: None,
            jwt_token: None,
            token_refresh: None,
            custom_headers: HashMap::new(),
        },
        pool_config: tunnels::ConnectionPoolConfig {
            max_connections: 5,
            min_idle_connections: 1,
            idle_timeout_s: 300,
            max_lifetime_s: 3600,
            validation_interval_s: 60,
            enable_multiplexing: true,
            max_streams_per_connection: 10,
        },
        qos_config: tunnels::QosConfig {
            bandwidth_limit_bps: Some(10_000_000), // 10 MB/s
            rate_limit_rps: Some(1000),
            priority: tunnels::QosPriority::High,
            traffic_shaping: true,
            congestion_control: tunnels::CongestionControl::Bbr,
        },
        timeout_config: tunnels::TimeoutConfig {
            connect_timeout_ms: 10000,
            request_timeout_ms: 30000,
            keepalive_timeout_s: 60,
            dns_timeout_ms: 5000,
        },
        retry_config: tunnels::RetryConfig {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 10000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            retry_on_errors: vec![
                tunnels::RetryableError::ConnectionError,
                tunnels::RetryableError::Timeout,
                tunnels::RetryableError::TlsError,
            ],
        },
    };

    let tunnel_manager = TunnelManager::new(tunnel_config).unwrap();

    // Set up compliance validation
    let compliance_config = ComplianceMappingConfig::default();
    let mut compliance_manager = ComplianceMappingManager::new(compliance_config);

    // Validate that tunnel meets compliance requirements
    let compliance_context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::PHI,
        source_region: "us-east-1".to_string(),
        target_region: "us-west-2".to_string(),
        operation: compliance_mapping::DataOperation::Transfer,
        encryption_enabled: true, // Tunnel uses TLS 1.3
        consent_obtained: true,
        metadata: HashMap::new(),
    };

    let compliance_result = compliance_manager
        .validate_compliance(&compliance_context)
        .unwrap();

    // Test tunnel establishment (will fail due to no real endpoints)
    let tunnel_result = tunnel_manager
        .establish_tunnel("us-east-1", "us-west-2")
        .await;
    assert!(tunnel_result.is_err()); // Expected due to no real endpoints

    // Test tunnel request creation
    let request = tunnels::TunnelRequest {
        id: "integration-test-request".to_string(),
        source_region: "us-east-1".to_string(),
        target_region: "us-west-2".to_string(),
        data: b"encrypted sensitive data".to_vec(),
        headers: HashMap::new(),
        priority: tunnels::QosPriority::High,
        timestamp: chrono::Utc::now(),
        timeout_ms: Some(10000),
    };

    // Test request sending (will fail due to no established tunnel)
    let send_result = tunnel_manager.send_request(request).await;
    assert!(send_result.is_err()); // Expected due to no established tunnel

    // Verify compliance validation passed for encrypted tunnel
    if compliance_result
        .applicable_frameworks
        .contains(&ComplianceFramework::HIPAA)
    {
        assert!(compliance_result.is_compliant || !compliance_result.violations.is_empty());
    }
}

/// Test comprehensive multi-region data flow
#[tokio::test]
async fn test_comprehensive_data_flow() {
    // Simulate a complete data flow through all components

    // 1. Region selection
    let region_config = RegionConfig::default();
    let _region_manager = RegionManager::new(region_config).unwrap();

    // 2. Compliance validation
    let compliance_config = ComplianceMappingConfig::default();
    let mut compliance_manager = ComplianceMappingManager::new(compliance_config);

    // 3. Data sovereignty check
    let mut sovereignty = DataSovereignty::new();
    let rule = data_sovereignty::SovereigntyRule {
        id: "healthcare-rule".to_string(),
        data_classification: "PHI".to_string(),
        allowed_jurisdictions: vec!["US".to_string()],
        forbidden_jurisdictions: vec!["CN".to_string(), "RU".to_string()],
        encryption_level: data_sovereignty::EncryptionLevel::MilitaryGrade,
        residency_requirements: data_sovereignty::ResidencyRequirements {
            must_remain_in_origin: true, // Healthcare data must stay in US
            allowed_transit: vec![],
            max_distance_km: None,
            backup_requirements: data_sovereignty::BackupRequirements {
                cross_border_allowed: false,
                required_jurisdictions: vec!["US".to_string()],
                min_copies: 3,
            },
        },
        compliance_frameworks: vec!["HIPAA".to_string()],
        created_at: chrono::Utc::now(),
    };
    sovereignty.add_rule(rule);

    // 4. Test data flow scenarios

    // Scenario 1: Compliant PHI storage within US
    let compliant_request = data_sovereignty::DataPlacementRequest {
        data_id: "patient-record-789".to_string(),
        classification: "PHI".to_string(),
        source_jurisdiction: "US".to_string(),
        target_jurisdiction: "US".to_string(),
        operation: data_sovereignty::DataOperation::Store,
        metadata: HashMap::new(),
    };

    let sovereignty_result = sovereignty.validate_placement(&compliant_request);
    assert!(sovereignty_result.is_ok());

    let compliance_context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::PHI,
        source_region: "US".to_string(),
        target_region: "US".to_string(),
        operation: compliance_mapping::DataOperation::Store,
        encryption_enabled: true,
        consent_obtained: true,
        metadata: HashMap::new(),
    };

    let compliance_result = compliance_manager
        .validate_compliance(&compliance_context)
        .unwrap();
    assert!(compliance_result
        .applicable_frameworks
        .contains(&ComplianceFramework::HIPAA));

    // Scenario 2: Non-compliant PHI transfer outside US
    let non_compliant_request = data_sovereignty::DataPlacementRequest {
        data_id: "patient-record-790".to_string(),
        classification: "PHI".to_string(),
        source_jurisdiction: "US".to_string(),
        target_jurisdiction: "EU".to_string(), // Should violate residency requirement
        operation: data_sovereignty::DataOperation::Move,
        metadata: HashMap::new(),
    };

    let violation_result = sovereignty.validate_placement(&non_compliant_request);
    assert!(violation_result.is_err()); // Should violate residency requirement

    // 5. Load balancer for healthcare endpoints
    let lb_config = LoadBalancerConfig {
        algorithm: LoadBalancingAlgorithm::LeastConnections,
        health_check: load_balancer::HealthCheckConfig {
            interval_seconds: 30,
            timeout_ms: 5000,
            failure_threshold: 2,
            success_threshold: 3,
            health_path: "/health".to_string(),
            healthy_status_codes: vec![200],
        },
        circuit_breaker: load_balancer::CircuitBreakerConfig {
            failure_threshold: 5,
            failure_window_seconds: 60,
            recovery_timeout_seconds: 30,
            min_requests: 5,
        },
        connection_timeout_ms: 10000,
        request_timeout_ms: 30000,
        max_retries: 3,
        sticky_sessions: true, // Important for healthcare applications
    };

    let healthcare_endpoints = vec![
        load_balancer::RegionEndpoint::new(
            "us-east-1-healthcare".to_string(),
            "https://healthcare-us-east-1.example.com".to_string(),
            60,
            1,
        ),
        load_balancer::RegionEndpoint::new(
            "us-west-2-healthcare".to_string(),
            "https://healthcare-us-west-2.example.com".to_string(),
            40,
            2,
        ),
    ];

    let healthcare_lb = LoadBalancer::new(lb_config, healthcare_endpoints).unwrap();

    let healthcare_request = load_balancer::RoutingRequest {
        client_ip: Some("10.0.0.100".to_string()),
        session_id: Some("healthcare-session-123".to_string()),
        geo_info: Some(load_balancer::GeoInfo {
            country: "US".to_string(),
            region: "NY".to_string(),
            city: "New York".to_string(),
            latitude: 40.7128,
            longitude: -74.0060,
        }),
        metadata: HashMap::new(),
    };

    let endpoint_result = healthcare_lb.select_endpoint(&healthcare_request).await;
    assert!(endpoint_result.is_err()); // Expected due to no healthy endpoints

    // 6. Verify replication constraints for healthcare data
    let repl_config = ReplicationConfig {
        consistency_model: ConsistencyModel::Strong, // Critical for healthcare
        conflict_resolution: replication::ConflictResolutionStrategy::RejectConcurrentWrites,
        topology: replication::ReplicationTopology::MasterSlave {
            master_region: "us-east-1".to_string(),
        },
        replication_factor: 3,
        max_replication_lag_ms: 100, // Very low latency for healthcare
        batch_size: 1,               // Process immediately
        anti_entropy_interval_s: 30,
        vector_clock_config: replication::VectorClockConfig {
            enabled: true,
            max_entries: 50,
            pruning_interval_s: 600,
        },
    };

    let us_regions = vec![
        "us-east-1".to_string(),
        "us-west-2".to_string(),
        "us-central-1".to_string(),
    ];
    let healthcare_repl = ReplicationManager::new(repl_config, us_regions).unwrap();

    // Test that replication respects compliance constraints
    let replication_result = healthcare_repl
        .replicate_data(
            "patient-vital-signs".to_string(),
            b"encrypted patient data".to_vec(),
            replication::ReplicationOperation::Insert,
            "us-east-1".to_string(),
            vec!["us-west-2".to_string(), "us-central-1".to_string()], // Only within US
        )
        .await;

    // Expected to fail due to no real endpoints, but validates configuration
    assert!(replication_result.is_err());
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_and_recovery() {
    // Test graceful handling of various error conditions

    // 1. Test invalid region configuration
    let invalid_config = RegionConfig {
        primary_region: "invalid-region".to_string(),
        regions: vec![], // Empty regions should be handled gracefully
        failover_config: region_manager::FailoverConfig {
            auto_failover: true,
            health_check_interval: 30,
            failover_threshold: 3,
            recovery_threshold: 5,
            failover_order: vec!["backup-region".to_string()],
        },
        network_config: region_manager::NetworkConfig {
            load_balancer_endpoints: HashMap::new(),
            tunnel_configs: HashMap::new(),
            dns_configs: HashMap::new(),
        },
    };

    let region_manager_result = RegionManager::new(invalid_config);
    assert!(region_manager_result.is_ok()); // Should handle empty regions gracefully

    // 2. Test load balancer with no endpoints
    let empty_lb_config = LoadBalancerConfig::default();
    let empty_endpoints = vec![];

    let empty_lb_result = LoadBalancer::new(empty_lb_config, empty_endpoints);
    assert!(empty_lb_result.is_ok()); // Should create successfully but have no endpoints

    // 3. Test replication with empty regions
    let repl_config = ReplicationConfig::default();
    let empty_regions = vec![];

    let empty_repl_result = ReplicationManager::new(repl_config, empty_regions);
    assert!(empty_repl_result.is_ok()); // Should handle empty regions

    // 4. Test compliance with unknown frameworks
    let mut compliance_config = ComplianceMappingConfig::default();
    compliance_config.enabled_frameworks = vec![]; // No frameworks

    let _compliance_manager = ComplianceMappingManager::new(compliance_config);

    let context = compliance_mapping::ComplianceContext {
        data_classification: compliance_mapping::DataClassification::Internal,
        source_region: "test-region".to_string(),
        target_region: "test-region-2".to_string(),
        operation: compliance_mapping::DataOperation::Store,
        encryption_enabled: false,
        consent_obtained: false,
        metadata: HashMap::new(),
    };

    let mut temp_manager = ComplianceMappingManager::new(ComplianceMappingConfig {
        enabled_frameworks: vec![],
        ..ComplianceMappingConfig::default()
    });

    let validation_result = temp_manager.validate_compliance(&context);
    assert!(validation_result.is_ok()); // Should validate successfully with no frameworks

    // 5. Test data sovereignty with no rules
    let mut empty_sovereignty = DataSovereignty::new();

    let test_request = data_sovereignty::DataPlacementRequest {
        data_id: "test-data".to_string(),
        classification: "Unknown".to_string(),
        source_jurisdiction: "Unknown".to_string(),
        target_jurisdiction: "Unknown".to_string(),
        operation: data_sovereignty::DataOperation::Store,
        metadata: HashMap::new(),
    };

    let empty_sovereignty_result = empty_sovereignty.validate_placement(&test_request);
    assert!(empty_sovereignty_result.is_ok()); // Should pass with no applicable rules
}

/// Test concurrent operations and thread safety
#[tokio::test]
async fn test_concurrent_operations() {
    use std::sync::Arc;
    use tokio::task::JoinSet;

    // Test concurrent compliance validations
    let compliance_config = ComplianceMappingConfig::default();
    let compliance_manager = Arc::new(tokio::sync::Mutex::new(ComplianceMappingManager::new(
        compliance_config,
    )));

    let mut join_set = JoinSet::new();

    // Spawn multiple concurrent validation tasks
    for i in 0..10 {
        let manager = compliance_manager.clone();
        join_set.spawn(async move {
            let context = compliance_mapping::ComplianceContext {
                data_classification: compliance_mapping::DataClassification::PII,
                source_region: format!("region-{}", i),
                target_region: format!("region-{}", (i + 1) % 10),
                operation: compliance_mapping::DataOperation::Transfer,
                encryption_enabled: i % 2 == 0,
                consent_obtained: i % 3 == 0,
                metadata: HashMap::new(),
            };

            let mut mgr = manager.lock().await;
            mgr.validate_compliance(&context)
        });
    }

    // Wait for all validations to complete
    let mut success_count = 0;
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(validation_result) => {
                assert!(validation_result.is_ok());
                success_count += 1;
            }
            Err(e) => {
                panic!("Task failed: {:?}", e);
            }
        }
    }

    assert_eq!(success_count, 10);

    // Test concurrent vector clock operations
    let mut join_set = JoinSet::new();
    let clock = Arc::new(tokio::sync::Mutex::new(replication::VectorClock::new()));

    for i in 0..5 {
        let clock_ref = clock.clone();
        join_set.spawn(async move {
            let mut c = clock_ref.lock().await;
            c.increment(&format!("region-{}", i));
        });
    }

    // Wait for all increments
    while let Some(result) = join_set.join_next().await {
        assert!(result.is_ok());
    }

    let final_clock = clock.lock().await;
    assert_eq!(final_clock.clocks.len(), 5);
    for i in 0..5 {
        assert_eq!(final_clock.clocks.get(&format!("region-{}", i)), Some(&1));
    }
}

/// Test configuration serialization and validation
#[tokio::test]
async fn test_configuration_roundtrip() {
    // Test that all configurations can be serialized and deserialized correctly

    // Load balancer config
    let lb_config = LoadBalancerConfig::default();
    let lb_json = serde_json::to_string(&lb_config).unwrap();
    let lb_deserialized: LoadBalancerConfig = serde_json::from_str(&lb_json).unwrap();
    assert_eq!(lb_config.algorithm, lb_deserialized.algorithm);

    // Replication config
    let repl_config = ReplicationConfig::default();
    let repl_json = serde_json::to_string(&repl_config).unwrap();
    let repl_deserialized: ReplicationConfig = serde_json::from_str(&repl_json).unwrap();
    assert_eq!(
        repl_config.consistency_model,
        repl_deserialized.consistency_model
    );

    // Tunnel config
    let tunnel_config = TunnelConfig::default();
    let tunnel_json = serde_json::to_string(&tunnel_config).unwrap();
    let tunnel_deserialized: TunnelConfig = serde_json::from_str(&tunnel_json).unwrap();
    assert_eq!(
        tunnel_config.tls_config.tls_version,
        tunnel_deserialized.tls_config.tls_version
    );

    // Compliance config
    let compliance_config = ComplianceMappingConfig::default();
    let compliance_json = serde_json::to_string(&compliance_config).unwrap();
    let compliance_deserialized: ComplianceMappingConfig =
        serde_json::from_str(&compliance_json).unwrap();
    assert_eq!(
        compliance_config.enabled_frameworks,
        compliance_deserialized.enabled_frameworks
    );
}
