//! Comprehensive tests for distributed coordinator

use crate::checkpoint::CheckpointId;
use crate::coordinator::*;
use crate::error::{FaultToleranceError, HealthStatus};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[tokio::test]
async fn test_node_id_uniqueness() {
    let mut node_ids = Vec::new();
    for _ in 0..1000 {
        node_ids.push(NodeId::new());
    }

    // All node IDs should be unique
    for (i, id1) in node_ids.iter().enumerate() {
        for (j, id2) in node_ids.iter().enumerate() {
            if i != j {
                assert_ne!(id1, id2, "Node IDs should be unique");
            }
        }
    }
}

#[tokio::test]
async fn test_node_info_creation() {
    let node_id = NodeId::new();
    let address: SocketAddr = "192.168.1.100:8080".parse().unwrap();

    let node_info = NodeInfo {
        id: node_id.clone(),
        address,
        role: NodeRole::Leader,
        status: NodeStatus::Active,
        last_heartbeat: 123456789,
        capabilities: vec!["checkpoint".to_string(), "recovery".to_string()],
    };

    assert_eq!(node_info.id, node_id);
    assert_eq!(node_info.address, address);
    assert_eq!(node_info.role, NodeRole::Leader);
    assert_eq!(node_info.status, NodeStatus::Active);
    assert_eq!(node_info.last_heartbeat, 123456789);
    assert_eq!(node_info.capabilities.len(), 2);
}

#[tokio::test]
async fn test_node_role_variants() {
    let roles = vec![NodeRole::Leader, NodeRole::Follower, NodeRole::Observer];

    for role in roles {
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: NodeRole = serde_json::from_str(&serialized)?;
        assert_eq!(role, deserialized);
    }
}

#[tokio::test]
async fn test_node_status_variants() {
    let statuses = vec![
        NodeStatus::Active,
        NodeStatus::Degraded,
        NodeStatus::Failed,
        NodeStatus::Recovering,
    ];

    for status in statuses {
        let serialized = serde_json::to_string(&status)?;
        let deserialized: NodeStatus = serde_json::from_str(&serialized)?;
        assert_eq!(status, deserialized);
    }
}

#[tokio::test]
async fn test_coordination_message_serialization() {
    let node_id = NodeId::new();
    let checkpoint_id = CheckpointId::new();

    let messages = vec![
        CoordinationMessage::Heartbeat {
            node_id: node_id.clone(),
            timestamp: 123456789,
            status: NodeStatus::Active,
        },
        CoordinationMessage::CheckpointRequest {
            initiator: node_id.clone(),
            checkpoint_id: checkpoint_id.clone(),
            priority: Priority::Critical,
        },
        CoordinationMessage::CheckpointComplete {
            node_id: node_id.clone(),
            checkpoint_id: checkpoint_id.clone(),
            success: true,
        },
        CoordinationMessage::RecoveryRequest {
            initiator: node_id.clone(),
            checkpoint_id: checkpoint_id.clone(),
            target_nodes: vec![node_id.clone()],
        },
        CoordinationMessage::RecoveryProgress {
            node_id: node_id.clone(),
            checkpoint_id,
            progress_percent: 50,
        },
        CoordinationMessage::LeaderElection {
            candidate: node_id.clone(),
            term: 5,
        },
        CoordinationMessage::HealthCheck {
            requester: node_id.clone(),
        },
        CoordinationMessage::HealthResponse {
            node_id,
            health: HealthStatus::Healthy,
            metrics: HashMap::new(),
        },
    ];

    for message in messages {
        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: CoordinationMessage = serde_json::from_str(&serialized).unwrap();

        // Verify message type is preserved
        match (&message, &deserialized) {
            (CoordinationMessage::Heartbeat { .. }, CoordinationMessage::Heartbeat { .. }) => {}
            (
                CoordinationMessage::CheckpointRequest { .. },
                CoordinationMessage::CheckpointRequest { .. },
            ) => {}
            (
                CoordinationMessage::CheckpointComplete { .. },
                CoordinationMessage::CheckpointComplete { .. },
            ) => {}
            (
                CoordinationMessage::RecoveryRequest { .. },
                CoordinationMessage::RecoveryRequest { .. },
            ) => {}
            (
                CoordinationMessage::RecoveryProgress { .. },
                CoordinationMessage::RecoveryProgress { .. },
            ) => {}
            (
                CoordinationMessage::LeaderElection { .. },
                CoordinationMessage::LeaderElection { .. },
            ) => {}
            (CoordinationMessage::HealthCheck { .. }, CoordinationMessage::HealthCheck { .. }) => {}
            (
                CoordinationMessage::HealthResponse { .. },
                CoordinationMessage::HealthResponse { .. },
            ) => {}
            _ => panic!("Message type not preserved during serialization"),
        }
    }
}

#[tokio::test]
async fn test_priority_ordering() {
    let priorities = vec![
        Priority::Critical,
        Priority::High,
        Priority::Normal,
        Priority::Low,
    ];

    for priority in priorities {
        let serialized = serde_json::to_string(&priority)?;
        let deserialized: Priority = serde_json::from_str(&serialized)?;
        assert_eq!(priority, deserialized);
    }
}

#[tokio::test]
async fn test_distributed_coordinator_initialization() {
    let coordinator = DistributedCoordinator::new().unwrap();

    assert!(!coordinator.is_leader);
    assert_eq!(coordinator.election_term, 0);
    assert_eq!(coordinator.known_nodes.len(), 0);
    assert_eq!(coordinator.node_info.role, NodeRole::Follower);
    assert_eq!(coordinator.node_info.status, NodeStatus::Active);
    assert!(!coordinator.node_info.capabilities.is_empty());
}

#[tokio::test]
async fn test_coordinator_with_custom_address() {
    let addresses = vec![
        "0.0.0.0:8080",
        "192.168.1.100:9000",
        "10.0.0.1:8081",
        "[::1]:8080", // IPv6
    ];

    for addr_str in addresses {
        if let Ok(addr) = addr_str.parse::<SocketAddr>() {
            let coordinator = DistributedCoordinator::with_address(addr)?;
            assert_eq!(coordinator.node_info.address, addr);
        }
    }
}

#[tokio::test]
async fn test_join_cluster_multiple_bootstrap_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    let bootstrap_nodes = vec![
        "127.0.0.1:8081".parse().unwrap(),
        "127.0.0.1:8082".parse()?,
        "127.0.0.1:8083".parse()?,
    ];

    let result = coordinator.join_cluster(bootstrap_nodes.clone()).await;
    assert!(result.is_ok());
    assert_eq!(coordinator.known_nodes.len(), bootstrap_nodes.len());

    // Verify all bootstrap nodes are added
    for &addr in &bootstrap_nodes {
        let found = coordinator
            .known_nodes
            .values()
            .any(|node| node.address == addr);
        assert!(found, "Bootstrap node {} not found", addr);
    }
}

#[tokio::test]
async fn test_join_empty_cluster() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    let result = coordinator.join_cluster(vec![]).await;
    assert!(result.is_ok());
    assert_eq!(coordinator.known_nodes.len(), 0);
}

#[tokio::test]
async fn test_heartbeat_with_no_known_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let initial_heartbeat = coordinator.node_info.last_heartbeat;

    let result = coordinator.send_heartbeat().await;
    assert!(result.is_ok());

    // Heartbeat timestamp should be updated even with no nodes
    assert!(coordinator.node_info.last_heartbeat > initial_heartbeat);
}

#[tokio::test]
async fn test_heartbeat_with_mixed_node_statuses() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Add nodes with different statuses
    let statuses = vec![
        NodeStatus::Active,
        NodeStatus::Degraded,
        NodeStatus::Failed,
        NodeStatus::Recovering,
    ];

    for (i, status) in statuses.iter().enumerate() {
        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:808{}", i).parse().unwrap(),
            role: NodeRole::Follower,
            status: status.clone(),
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    let result = coordinator.send_heartbeat().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_leader_election_single_node() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    let won = coordinator.initiate_leader_election().await.unwrap();
    assert!(won);
    assert!(coordinator.is_leader);
    assert_eq!(coordinator.node_info.role, NodeRole::Leader);
    assert_eq!(coordinator.election_term, 1);
}

#[tokio::test]
async fn test_leader_election_with_failed_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Add a mix of active and failed nodes
    for i in 0..6 {
        let status = if i < 3 {
            NodeStatus::Active
        } else {
            NodeStatus::Failed
        };
        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:808{}", i).parse().unwrap(),
            role: NodeRole::Follower,
            status,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    let won = coordinator.initiate_leader_election().await.unwrap();
    assert!(won); // Should win as mock votes for all active nodes
    assert!(coordinator.is_leader);
}

#[tokio::test]
async fn test_coordinate_checkpoint_permission_check() {
    let coordinator = DistributedCoordinator::new().unwrap();
    let checkpoint_id = CheckpointId::new();

    // Non-leader should not be able to coordinate
    let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        FaultToleranceError::CoordinationError(msg) => {
            assert!(msg.contains("leader"));
        }
        _ => panic!("Expected coordination error"),
    }
}

#[tokio::test]
async fn test_coordinate_checkpoint_with_active_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;

    // Add active nodes
    for i in 0..3 {
        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:808{}", i).parse()?,
            role: NodeRole::Follower,
            status: NodeStatus::Active,
            last_heartbeat: 0,
            capabilities: vec!["checkpoint".to_string()],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    let checkpoint_id = CheckpointId::new();
    let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_coordinate_checkpoint_skip_inactive_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;

    // Add nodes with mixed status - inactive should be skipped
    let statuses = vec![NodeStatus::Active, NodeStatus::Failed, NodeStatus::Degraded];

    for (i, status) in statuses.iter().enumerate() {
        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:808{}", i).parse()?,
            role: NodeRole::Follower,
            status: status.clone(),
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    let checkpoint_id = CheckpointId::new();
    let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_coordinate_recovery_permission_check() {
    let coordinator = DistributedCoordinator::new().unwrap();
    let checkpoint_id = CheckpointId::new();
    let target_nodes = vec![NodeId::new()];

    // Non-leader should not be able to coordinate recovery
    let result = coordinator
        .coordinate_recovery(checkpoint_id, target_nodes)
        .await;
    assert!(result.is_err());

    match result.unwrap_err() {
        FaultToleranceError::CoordinationError(msg) => {
            assert!(msg.contains("leader"));
        }
        _ => panic!("Expected coordination error"),
    }
}

#[tokio::test]
async fn test_coordinate_recovery_with_mixed_target_nodes() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;

    // Add target nodes with different statuses
    let target_node_active = NodeId::new();
    let target_node_failed = NodeId::new();
    let target_node_unknown = NodeId::new(); // Not in known_nodes

    let active_node = NodeInfo {
        id: target_node_active.clone(),
        address: "127.0.0.1:8081".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec!["recovery".to_string()],
    };

    let failed_node = NodeInfo {
        id: target_node_failed.clone(),
        address: "127.0.0.1:8082".parse().unwrap(),
        role: NodeRole::Follower,
        status: NodeStatus::Failed, // Should be skipped
        last_heartbeat: 0,
        capabilities: vec!["recovery".to_string()],
    };

    coordinator
        .known_nodes
        .insert(target_node_active.clone(), active_node);
    coordinator
        .known_nodes
        .insert(target_node_failed.clone(), failed_node);

    let checkpoint_id = CheckpointId::new();
    let target_nodes = vec![target_node_active, target_node_failed, target_node_unknown];

    let result = coordinator
        .coordinate_recovery(checkpoint_id, target_nodes)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_system_health_calculations() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Test different health scenarios
    let scenarios = vec![
        (vec![], HealthStatus::Healthy), // No nodes - single node system
        (
            vec![
                NodeStatus::Active,
                NodeStatus::Active,
                NodeStatus::Active,
                NodeStatus::Active,
            ],
            HealthStatus::Healthy,
        ), // 100% healthy
        (
            vec![
                NodeStatus::Active,
                NodeStatus::Active,
                NodeStatus::Active,
                NodeStatus::Failed,
            ],
            HealthStatus::Healthy,
        ), // 75% healthy
        (
            vec![
                NodeStatus::Active,
                NodeStatus::Active,
                NodeStatus::Failed,
                NodeStatus::Failed,
            ],
            HealthStatus::Degraded,
        ), // 50% healthy
        (
            vec![
                NodeStatus::Active,
                NodeStatus::Failed,
                NodeStatus::Failed,
                NodeStatus::Failed,
            ],
            HealthStatus::Failed,
        ), // 25% healthy
        (
            vec![
                NodeStatus::Failed,
                NodeStatus::Failed,
                NodeStatus::Failed,
                NodeStatus::Failed,
            ],
            HealthStatus::Failed,
        ), // 0% healthy
    ];

    for (statuses, expected_health) in scenarios {
        coordinator.known_nodes.clear();

        for (i, status) in statuses.iter().enumerate() {
            let node = NodeInfo {
                id: NodeId::new(),
                address: format!("127.0.0.1:808{}", i).parse().unwrap(),
                role: NodeRole::Follower,
                status: status.clone(),
                last_heartbeat: 0,
                capabilities: vec![],
            };
            coordinator.known_nodes.insert(node.id.clone(), node);
        }

        let health = coordinator.system_health().await;
        assert_eq!(
            health, expected_health,
            "Failed for statuses: {:?}",
            statuses
        );
    }
}

#[tokio::test]
async fn test_cluster_status_comprehensive() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;
    coordinator.election_term = 5;

    // Add nodes with different statuses
    for i in 0..5 {
        let status = match i {
            0..=2 => NodeStatus::Active,
            3 => NodeStatus::Degraded,
            _ => NodeStatus::Failed,
        };

        let node = NodeInfo {
            id: NodeId::new(),
            address: format!("127.0.0.1:808{}", i).parse().unwrap(),
            role: NodeRole::Follower,
            status,
            last_heartbeat: 0,
            capabilities: vec![],
        };
        coordinator.known_nodes.insert(node.id.clone(), node);
    }

    let status = coordinator.cluster_status();

    // Verify all expected fields are present
    assert!(status.contains_key("total_nodes"));
    assert!(status.contains_key("active_nodes"));
    assert!(status.contains_key("is_leader"));
    assert!(status.contains_key("election_term"));

    // Verify values
    if let Some(serde_json::Value::Number(total)) = status.get("total_nodes") {
        assert_eq!(total.as_u64().unwrap(), 6); // 5 nodes + self
    }

    if let Some(serde_json::Value::Number(active)) = status.get("active_nodes") {
        assert_eq!(active.as_u64().unwrap(), 4); // 3 active nodes + self
    }

    if let Some(serde_json::Value::Bool(is_leader)) = status.get("is_leader") {
        assert!(is_leader);
    }

    if let Some(serde_json::Value::Number(term)) = status.get("election_term") {
        assert_eq!(term.as_u64().unwrap(), 5);
    }
}

#[tokio::test]
async fn test_handle_heartbeat_update_existing_node() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let node_id = NodeId::new();

    // Add node with initial status
    let initial_node = NodeInfo {
        id: node_id.clone(),
        address: "127.0.0.1:8081".parse()?,
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 100,
        capabilities: vec![],
    };
    coordinator
        .known_nodes
        .insert(node_id.clone(), initial_node);

    // Send heartbeat with updated status
    let heartbeat = CoordinationMessage::Heartbeat {
        node_id: node_id.clone(),
        timestamp: 200,
        status: NodeStatus::Degraded,
    };

    let result = coordinator.handle_message(heartbeat).await;
    assert!(result.is_ok());

    // Verify node was updated
    let updated_node = coordinator.known_nodes.get(&node_id).unwrap();
    assert_eq!(updated_node.status, NodeStatus::Degraded);
    assert_eq!(updated_node.last_heartbeat, 200);
}

#[tokio::test]
async fn test_handle_heartbeat_unknown_node() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let unknown_node_id = NodeId::new();

    let heartbeat = CoordinationMessage::Heartbeat {
        node_id: unknown_node_id.clone(),
        timestamp: 200,
        status: NodeStatus::Active,
    };

    let result = coordinator.handle_message(heartbeat).await;
    assert!(result.is_ok());

    // Unknown node should not be added to known_nodes
    assert!(!coordinator.known_nodes.contains_key(&unknown_node_id));
}

#[tokio::test]
async fn test_handle_leader_election_term_logic() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;
    coordinator.election_term = 5;
    coordinator.node_info.role = NodeRole::Leader;

    // Test with equal term (should not change leadership)
    let equal_term_election = CoordinationMessage::LeaderElection {
        candidate: NodeId::new(),
        term: 5,
    };

    let result = coordinator.handle_message(equal_term_election).await;
    assert!(result.is_ok());
    assert!(coordinator.is_leader); // Should remain leader
    assert_eq!(coordinator.election_term, 5);

    // Test with newer term (should step down)
    let newer_term_election = CoordinationMessage::LeaderElection {
        candidate: NodeId::new(),
        term: 10,
    };

    let result = coordinator.handle_message(newer_term_election).await;
    assert!(result.is_ok());
    assert!(!coordinator.is_leader); // Should step down
    assert_eq!(coordinator.election_term, 10);
    assert_eq!(coordinator.node_info.role, NodeRole::Follower);
}

#[tokio::test]
async fn test_handle_checkpoint_messages() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let checkpoint_id = CheckpointId::new();

    let messages = vec![
        CoordinationMessage::CheckpointRequest {
            initiator: NodeId::new(),
            checkpoint_id: checkpoint_id.clone(),
            priority: Priority::High,
        },
        CoordinationMessage::CheckpointComplete {
            node_id: NodeId::new(),
            checkpoint_id: checkpoint_id.clone(),
            success: true,
        },
        CoordinationMessage::CheckpointComplete {
            node_id: NodeId::new(),
            checkpoint_id,
            success: false, // Test failure case
        },
    ];

    for message in messages {
        let result = coordinator.handle_message(message).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_handle_recovery_messages() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let checkpoint_id = CheckpointId::new();
    let node_id = NodeId::new();

    let messages = vec![
        CoordinationMessage::RecoveryRequest {
            initiator: NodeId::new(),
            checkpoint_id: checkpoint_id.clone(),
            target_nodes: vec![node_id.clone()],
        },
        CoordinationMessage::RecoveryProgress {
            node_id: node_id.clone(),
            checkpoint_id: checkpoint_id.clone(),
            progress_percent: 0,
        },
        CoordinationMessage::RecoveryProgress {
            node_id: node_id.clone(),
            checkpoint_id: checkpoint_id.clone(),
            progress_percent: 50,
        },
        CoordinationMessage::RecoveryProgress {
            node_id,
            checkpoint_id,
            progress_percent: 100,
        },
    ];

    for message in messages {
        let result = coordinator.handle_message(message).await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_handle_health_messages() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let node_id = NodeId::new();

    let health_check = CoordinationMessage::HealthCheck {
        requester: node_id.clone(),
    };

    let result = coordinator.handle_message(health_check).await;
    assert!(result.is_ok());

    let mut metrics = HashMap::new();
    metrics.insert("cpu_usage".to_string(), 75.5);
    metrics.insert("memory_usage".to_string(), 60.2);

    let health_response = CoordinationMessage::HealthResponse {
        node_id,
        health: HealthStatus::Degraded,
        metrics,
    };

    let result = coordinator.handle_message(health_response).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_concurrent_message_handling() {
    let coordinator = std::sync::Arc::new(tokio::sync::Mutex::new(
        DistributedCoordinator::new().unwrap(),
    ));

    let mut handles = Vec::new();

    // Send multiple messages concurrently
    for i in 0..10 {
        let coordinator_clone = coordinator.clone();
        let handle = tokio::spawn(async move {
            let message = CoordinationMessage::Heartbeat {
                node_id: NodeId::new(),
                timestamp: i * 1000,
                status: NodeStatus::Active,
            };

            let mut coord = coordinator_clone.lock().await;
            coord.handle_message(message).await
        });
        handles.push(handle);
    }

    // Wait for all messages to be processed
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_coordinator_network_delay_simulation() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;

    // Add a node
    let node = NodeInfo {
        id: NodeId::new(),
        address: "127.0.0.1:8081".parse()?,
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec![],
    };
    coordinator.known_nodes.insert(node.id.clone(), node);

    // Test checkpoint coordination with simulated network delay
    let checkpoint_id = CheckpointId::new();
    let start = std::time::Instant::now();
    let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(elapsed >= Duration::from_millis(100)); // Should include simulated delays
}

#[tokio::test]
async fn test_coordinator_edge_cases() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    // Test with node having same ID as coordinator
    let self_node = NodeInfo {
        id: coordinator.node_id.clone(),
        address: "127.0.0.1:8081".parse()?,
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec![],
    };
    coordinator
        .known_nodes
        .insert(coordinator.node_id.clone(), self_node);

    // Heartbeat should exclude self
    let result = coordinator.send_heartbeat().await;
    assert!(result.is_ok());

    // System health should still work correctly
    let health = coordinator.system_health().await;
    assert!(matches!(
        health,
        HealthStatus::Healthy | HealthStatus::Degraded
    ));
}

#[tokio::test]
async fn test_coordinator_address_validation() {
    let invalid_addresses = vec![
        "not_an_address",
        "192.168.1.300:8080", // Invalid IP
        "192.168.1.1:99999",  // Invalid port
    ];

    for addr_str in invalid_addresses {
        if let Ok(addr) = addr_str.parse::<SocketAddr>() {
            // If it parses successfully, test with it
            let result = DistributedCoordinator::with_address(addr);
            assert!(result.is_ok());
        }
        // If it doesn't parse, that's expected for invalid addresses
    }
}

#[tokio::test]
async fn test_coordinator_capabilities_management() {
    let coordinator = DistributedCoordinator::new().unwrap();

    assert!(!coordinator.node_info.capabilities.is_empty());
    assert!(coordinator
        .node_info
        .capabilities
        .contains(&"checkpoint".to_string()));
    assert!(coordinator
        .node_info
        .capabilities
        .contains(&"recovery".to_string()));
}

#[tokio::test]
async fn test_coordinate_checkpoint_with_priorities() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    coordinator.is_leader = true;

    // Test coordination doesn't depend on priority (mock implementation)
    let checkpoint_id = CheckpointId::new();
    let result = coordinator.coordinate_checkpoint(checkpoint_id).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_election_term_progression() {
    let mut coordinator = DistributedCoordinator::new().unwrap();

    assert_eq!(coordinator.election_term, 0);

    // Multiple elections should increment term
    for expected_term in 1..=5 {
        let won = coordinator.initiate_leader_election().await?;
        assert!(won);
        assert_eq!(coordinator.election_term, expected_term);
    }
}

#[tokio::test]
async fn test_node_status_transitions() {
    let mut coordinator = DistributedCoordinator::new().unwrap();
    let node_id = NodeId::new();

    // Add node with initial status
    let node = NodeInfo {
        id: node_id.clone(),
        address: "127.0.0.1:8081".parse()?,
        role: NodeRole::Follower,
        status: NodeStatus::Active,
        last_heartbeat: 0,
        capabilities: vec![],
    };
    coordinator.known_nodes.insert(node_id.clone(), node);

    // Test status transitions through heartbeat messages
    let transitions = vec![
        NodeStatus::Active,
        NodeStatus::Degraded,
        NodeStatus::Failed,
        NodeStatus::Recovering,
        NodeStatus::Active,
    ];

    for (i, status) in transitions.iter().enumerate() {
        let heartbeat = CoordinationMessage::Heartbeat {
            node_id: node_id.clone(),
            timestamp: (i + 1) as u64 * 1000,
            status: status.clone(),
        };

        let result = coordinator.handle_message(heartbeat).await;
        assert!(result.is_ok());

        let updated_node = coordinator.known_nodes.get(&node_id).unwrap();
        assert_eq!(updated_node.status, *status);
        assert_eq!(updated_node.last_heartbeat, (i + 1) as u64 * 1000);
    }
}
