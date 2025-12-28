//! Integration tests for cluster-mesh

use stratoswarm_cluster_mesh::distribution::{JobRequirements, LocalityPreference};
use stratoswarm_cluster_mesh::*;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

/// Create a test cluster node
fn create_test_node(name: &str, class: NodeClass) -> ClusterNode {
    ClusterNode {
        id: Uuid::new_v4(),
        hostname: name.to_string(),
        class,
        hardware: HardwareProfile {
            cpu_model: "Test CPU".to_string(),
            cpu_cores: 16,
            memory_gb: 32.0,
            storage_gb: 1000.0,
            gpus: vec![],
        },
        network: NetworkCharacteristics {
            bandwidth_mbps: 1000.0,
            latency_ms: 1.0,
            jitter_ms: 0.1,
            packet_loss: 0.0,
            nat_type: discovery::NatType::None,
        },
        status: NodeStatus::Online,
        capabilities: NodeCapabilities {
            cpu_cores: 16,
            memory_gb: 32.0,
            gpu_count: 0,
            gpu_memory_gb: None,
            storage_gb: 1000.0,
            network_bandwidth_mbps: 1000.0,
            supports_gpu_direct: false,
            battery_powered: false,
            thermal_constraints: None,
        },
        last_heartbeat: chrono::Utc::now(),
        wg_public_key: None,
        subnet_info: None,
    }
}

#[tokio::test]
async fn test_cluster_mesh_lifecycle() {
    // Create cluster mesh
    let mesh = ClusterMesh::new().await.unwrap();

    // Add nodes of different types
    let datacenter_node = create_test_node(
        "dc-1",
        NodeClass::DataCenter {
            gpus: vec![],
            bandwidth: classification::Bandwidth::TenGigabit(10.0),
        },
    );

    let workstation_node = create_test_node(
        "ws-1",
        NodeClass::Workstation {
            gpu: None,
            schedule: classification::Schedule::AlwaysOn,
        },
    );

    let laptop_node = create_test_node(
        "laptop-1",
        NodeClass::Laptop {
            battery: false,
            mobility: classification::MobilityPattern::Stationary,
        },
    );

    // Add nodes
    mesh.add_node(datacenter_node.clone()).await.unwrap();
    mesh.add_node(workstation_node.clone()).await.unwrap();
    mesh.add_node(laptop_node.clone()).await.unwrap();

    // Verify nodes were added
    let nodes = mesh.get_nodes().await;
    assert_eq!(nodes.len(), 3);

    // Test node classification query
    let dc_nodes = mesh
        .get_nodes_by_class(NodeClass::DataCenter {
            gpus: vec![],
            bandwidth: classification::Bandwidth::TenGigabit(10.0),
        })
        .await;
    assert_eq!(dc_nodes.len(), 1);

    // Test statistics
    let stats = mesh.get_statistics().await;
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.datacenter_nodes, 1);
    assert_eq!(stats.workstation_nodes, 1);
    assert_eq!(stats.laptop_nodes, 1);
    assert_eq!(stats.online_nodes, 3);

    // Remove a node
    mesh.remove_node(laptop_node.id).await.unwrap();

    // Verify removal
    let nodes = mesh.get_nodes().await;
    assert_eq!(nodes.len(), 2);
}

#[tokio::test]
async fn test_job_scheduling() {
    let mesh = ClusterMesh::new().await.unwrap();

    // Add a capable node
    let node = create_test_node(
        "worker-1",
        NodeClass::Workstation {
            gpu: None,
            schedule: classification::Schedule::AlwaysOn,
        },
    );
    mesh.add_node(node.clone()).await.unwrap();

    // Create a job
    let job = Job {
        id: Uuid::new_v4(),
        name: "Test Job".to_string(),
        requirements: JobRequirements {
            cpu_cores: Some(8),
            memory_gb: Some(16.0),
            gpu_count: None,
            gpu_memory_gb: None,
            storage_gb: Some(100.0),
            network_bandwidth_mbps: Some(100.0),
            requires_gpu_direct: false,
            node_affinity: None,
            anti_affinity: None,
            locality_preference: distribution::LocalityPreference::None,
            max_latency_ms: None,
            battery_safe: true,
        },
        priority: JobPriority::Normal,
        submitted_at: chrono::Utc::now(),
    };

    // Schedule the job
    let expected_id = job.id;
    let job_id = mesh.schedule_job(job).await.unwrap();
    assert_eq!(job_id, expected_id);
}

#[tokio::test]
async fn test_node_offline_handling() {
    let mesh = ClusterMesh::new().await.unwrap();

    // Add online and offline nodes
    let mut online_node = create_test_node(
        "online-1",
        NodeClass::Workstation {
            gpu: None,
            schedule: classification::Schedule::AlwaysOn,
        },
    );

    let mut offline_node = online_node.clone();
    offline_node.id = Uuid::new_v4();
    offline_node.hostname = "offline-1".to_string();
    offline_node.status = NodeStatus::Offline;

    mesh.add_node(online_node.clone()).await.unwrap();
    mesh.add_node(offline_node.clone()).await.unwrap();

    // Create a job
    let job = Job {
        id: Uuid::new_v4(),
        name: "Test Job".to_string(),
        requirements: JobRequirements {
            cpu_cores: Some(4),
            memory_gb: Some(8.0),
            gpu_count: None,
            gpu_memory_gb: None,
            storage_gb: None,
            network_bandwidth_mbps: None,
            requires_gpu_direct: false,
            node_affinity: None,
            anti_affinity: None,
            locality_preference: distribution::LocalityPreference::None,
            max_latency_ms: None,
            battery_safe: true,
        },
        priority: JobPriority::Normal,
        submitted_at: chrono::Utc::now(),
    };

    // Should schedule on online node
    let expected_id = job.id;
    let job_id = mesh.schedule_job(job).await.unwrap();
    assert_eq!(job_id, expected_id);

    // Check statistics
    let stats = mesh.get_statistics().await;
    assert_eq!(stats.total_nodes, 2);
    assert_eq!(stats.online_nodes, 1);
}

#[tokio::test]
async fn test_heterogeneous_cluster() {
    // Test managing a cluster with diverse hardware
    let mesh = ClusterMesh::new().await.unwrap();

    // Add various node types
    let nodes = vec![
        create_test_node(
            "dc-gpu-1",
            NodeClass::DataCenter {
                gpus: vec![discovery::GpuInfo {
                    index: 0,
                    name: "A100".to_string(),
                    memory_mb: 40 * 1024,
                    compute_capability: (8, 0),
                    pci_bus_id: "0:1:0.0".to_string(),
                }],
                bandwidth: classification::Bandwidth::HundredGigabit(100.0),
            },
        ),
        create_test_node(
            "edge-pi-1",
            NodeClass::Edge {
                device_type: classification::EdgeType::RaspberryPi,
                power_budget: classification::Watts(15.0),
            },
        ),
        create_test_node(
            "laptop-mobile-1",
            NodeClass::Laptop {
                battery: true,
                mobility: classification::MobilityPattern::Frequent,
            },
        ),
    ];

    for node in &nodes {
        mesh.add_node(node.clone()).await.unwrap();
    }

    // Test statistics
    let stats = mesh.get_statistics().await;
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.datacenter_nodes, 1);
    assert_eq!(stats.edge_nodes, 1);
    assert_eq!(stats.laptop_nodes, 1);

    // Test GPU job scheduling - should fail as we don't have GPU nodes with proper capabilities
    let gpu_job = Job {
        id: Uuid::new_v4(),
        name: "GPU Job".to_string(),
        requirements: JobRequirements {
            cpu_cores: Some(4),
            memory_gb: Some(8.0),
            gpu_count: Some(1),
            gpu_memory_gb: Some(20.0),
            storage_gb: None,
            network_bandwidth_mbps: None,
            requires_gpu_direct: true,
            node_affinity: None,
            anti_affinity: None,
            locality_preference: distribution::LocalityPreference::None,
            max_latency_ms: None,
            battery_safe: false,
        },
        priority: JobPriority::High,
        submitted_at: chrono::Utc::now(),
    };

    // Should fail - no suitable node
    let result = mesh.schedule_job(gpu_job).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_operations() {
    use tokio::task::JoinSet;

    let mesh = ClusterMesh::new().await.unwrap();

    // Add initial nodes
    for i in 0..5 {
        let node = create_test_node(
            &format!("node-{}", i),
            NodeClass::Workstation {
                gpu: None,
                schedule: classification::Schedule::AlwaysOn,
            },
        );
        mesh.add_node(node).await.unwrap();
    }

    // Concurrent operations
    let mut tasks = JoinSet::new();

    // Task 1: Add more nodes
    let mesh_clone = mesh.clone();
    tasks.spawn(async move {
        for i in 5..10 {
            let node = create_test_node(
                &format!("node-{}", i),
                NodeClass::Workstation {
                    gpu: None,
                    schedule: classification::Schedule::AlwaysOn,
                },
            );
            mesh_clone.add_node(node).await.unwrap();
            sleep(Duration::from_millis(10)).await;
        }
    });

    // Task 2: Schedule jobs
    let mesh_clone = mesh.clone();
    tasks.spawn(async move {
        for i in 0..5 {
            let job = Job {
                id: Uuid::new_v4(),
                name: format!("Job {}", i),
                requirements: JobRequirements {
                    cpu_cores: Some(2),
                    memory_gb: Some(4.0),
                    gpu_count: None,
                    gpu_memory_gb: None,
                    storage_gb: None,
                    network_bandwidth_mbps: None,
                    requires_gpu_direct: false,
                    node_affinity: None,
                    anti_affinity: None,
                    locality_preference: distribution::LocalityPreference::None,
                    max_latency_ms: None,
                    battery_safe: true,
                },
                priority: JobPriority::Normal,
                submitted_at: chrono::Utc::now(),
            };
            mesh_clone.schedule_job(job).await.unwrap();
            sleep(Duration::from_millis(10)).await;
        }
    });

    // Task 3: Query statistics
    let mesh_clone = mesh.clone();
    tasks.spawn(async move {
        for _ in 0..10 {
            let stats = mesh_clone.get_statistics().await;
            assert!(stats.total_nodes >= 5);
            sleep(Duration::from_millis(5)).await;
        }
    });

    // Wait for all tasks
    while let Some(result) = tasks.join_next().await {
        result.unwrap();
    }

    // Final verification
    let stats = mesh.get_statistics().await;
    assert_eq!(stats.total_nodes, 10);
}

// Re-export modules for testing
mod classification {
    pub use stratoswarm_cluster_mesh::classification::*;
}

mod discovery {
    pub use stratoswarm_cluster_mesh::discovery::*;
}

mod distribution {
    pub use stratoswarm_cluster_mesh::distribution::*;
}
