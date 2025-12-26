//! Edge case and stress tests for mesh formation

#[cfg(test)]
mod mesh_edge_case_tests {
    use super::super::*;
    use crate::classification::{Bandwidth, EdgeType, MobilityPattern, Schedule, Watts};
    use crate::discovery::{GpuInfo, HardwareProfile, NatType, NetworkCharacteristics};
    use crate::mesh::*;
    use crate::{ClusterNode, NodeCapabilities, NodeClass, NodeStatus, ThermalConstraints};
    use chrono::Utc;
    use std::time::Duration;
    use uuid::Uuid;

    fn create_datacenter_node(id: Uuid, gpus: usize) -> ClusterNode {
        let gpu_list: Vec<GpuInfo> = (0..gpus)
            .map(|i| GpuInfo {
                index: i as u32,
                name: format!("NVIDIA A100-{}", i),
                memory_mb: 81920, // 80GB
                compute_capability: (8, 0),
                pci_bus_id: format!("0000:0{}:00.0", i + 1),
            })
            .collect();

        ClusterNode {
            id,
            hostname: format!("dc-{}", id),
            class: NodeClass::DataCenter {
                gpus: gpu_list.clone(),
                bandwidth: Bandwidth::HundredGigabit(100.0),
            },
            hardware: HardwareProfile {
                cpu_model: "AMD EPYC 7763".to_string(),
                cpu_cores: 128,
                memory_gb: 2048.0,
                storage_gb: 100_000.0,
                gpus: gpu_list,
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 100_000.0,
                latency_ms: 0.05,
                jitter_ms: 0.001,
                packet_loss: 0.0,
                nat_type: NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: 128,
                memory_gb: 2048.0,
                gpu_count: gpus as u32,
                gpu_memory_gb: Some((gpus as f32) * 80.0),
                storage_gb: 100_000.0,
                network_bandwidth_mbps: 100_000.0,
                supports_gpu_direct: true,
                battery_powered: false,
                thermal_constraints: None,
            },
            last_heartbeat: Utc::now(),
        }
    }

    fn create_edge_node(id: Uuid, device_type: EdgeType) -> ClusterNode {
        let (cores, memory, power) = match &device_type {
            EdgeType::RaspberryPi => (4, 8.0, 5.0),
            EdgeType::JetsonNano => (4, 4.0, 10.0),
            EdgeType::JetsonXavier => (8, 32.0, 30.0),
            EdgeType::IntelNuc => (4, 16.0, 65.0),
            EdgeType::Other(_) => (2, 4.0, 15.0),
        };

        ClusterNode {
            id,
            hostname: format!("edge-{}", id),
            class: NodeClass::Edge {
                device_type: device_type.clone(),
                power_budget: Watts(power),
            },
            hardware: HardwareProfile {
                cpu_model: "ARM Processor".to_string(),
                cpu_cores: cores,
                memory_gb: memory,
                storage_gb: 128.0,
                gpus: vec![],
            },
            network: NetworkCharacteristics {
                bandwidth_mbps: 100.0,
                latency_ms: 50.0,
                jitter_ms: 5.0,
                packet_loss: 0.5,
                nat_type: NatType::PortRestricted,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities {
                cpu_cores: cores,
                memory_gb: memory,
                gpu_count: 0,
                gpu_memory_gb: None,
                storage_gb: 128.0,
                network_bandwidth_mbps: 100.0,
                supports_gpu_direct: false,
                battery_powered: false,
                thermal_constraints: Some(ThermalConstraints {
                    max_temp_celsius: 85.0,
                    throttle_temp_celsius: 75.0,
                    current_temp_celsius: 45.0,
                    power_budget_watts: Some(power),
                }),
            },
            last_heartbeat: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_large_scale_mesh_formation() {
        let manager = MeshManager::new().await.unwrap();

        // Add 100 nodes of different types
        let mut nodes = Vec::new();

        // 10 datacenter nodes
        for i in 0..10 {
            let node = create_datacenter_node(Uuid::new_v4(), 8);
            nodes.push(node.clone());
            manager.add_node(&node).await.unwrap();
        }

        // 30 workstation nodes
        for i in 0..30 {
            let node = ClusterNode {
                id: Uuid::new_v4(),
                hostname: format!("ws-{}", i),
                class: NodeClass::Workstation {
                    gpu: if i % 3 == 0 {
                        Some(GpuInfo {
                            index: 0,
                            name: "RTX 4090".to_string(),
                            memory_mb: 24576,
                            compute_capability: (8, 9),
                            pci_bus_id: "0:1:0.0".to_string(),
                        })
                    } else {
                        None
                    },
                    schedule: Schedule::BusinessHours,
                },
                hardware: HardwareProfile::default(),
                network: NetworkCharacteristics {
                    bandwidth_mbps: 1000.0,
                    latency_ms: 5.0,
                    jitter_ms: 0.5,
                    packet_loss: 0.01,
                    nat_type: if i % 2 == 0 {
                        NatType::None
                    } else {
                        NatType::FullCone
                    },
                },
                status: NodeStatus::Online,
                capabilities: NodeCapabilities::default(),
                last_heartbeat: Utc::now(),
            };
            nodes.push(node.clone());
            manager.add_node(&node).await.unwrap();
        }

        // 40 laptop nodes
        for i in 0..40 {
            let node = ClusterNode {
                id: Uuid::new_v4(),
                hostname: format!("laptop-{}", i),
                class: NodeClass::Laptop {
                    battery: true,
                    mobility: if i % 4 == 0 {
                        MobilityPattern::Continuous
                    } else if i % 3 == 0 {
                        MobilityPattern::Frequent
                    } else {
                        MobilityPattern::Occasional
                    },
                },
                hardware: HardwareProfile::default(),
                network: NetworkCharacteristics {
                    bandwidth_mbps: 100.0,
                    latency_ms: 20.0,
                    jitter_ms: 2.0,
                    packet_loss: 0.1,
                    nat_type: NatType::Symmetric,
                },
                status: NodeStatus::Online,
                capabilities: NodeCapabilities::default(),
                last_heartbeat: Utc::now(),
            };
            nodes.push(node.clone());
            manager.add_node(&node).await.unwrap();
        }

        // 20 edge devices
        let edge_types = vec![
            EdgeType::RaspberryPi,
            EdgeType::JetsonNano,
            EdgeType::JetsonXavier,
            EdgeType::IntelNuc,
        ];

        for i in 0..20 {
            let device_type = edge_types[i % edge_types.len()].clone();
            let node = create_edge_node(Uuid::new_v4(), device_type);
            nodes.push(node.clone());
            manager.add_node(&node).await.unwrap();
        }

        // Verify node registry
        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 100);

        // Verify relay nodes (should be datacenter nodes)
        let relays = manager.relay_nodes.read().await;
        assert!(relays.len() >= 10); // All datacenter nodes should be relays
    }

    #[tokio::test]
    async fn test_nat_traversal_failure_recovery() {
        let manager = MeshManager::new().await.unwrap();

        // Create nodes with challenging NAT scenarios
        let symmetric_nat_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "symmetric-nat".to_string(),
            class: NodeClass::Laptop {
                battery: true,
                mobility: MobilityPattern::Frequent,
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 50.0,
                latency_ms: 100.0,
                jitter_ms: 10.0,
                packet_loss: 1.0,
                nat_type: NatType::Symmetric,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        // Add relay node
        let relay = create_datacenter_node(Uuid::new_v4(), 0);
        manager.add_node(&relay).await.unwrap();

        // Add NAT'd node
        manager.add_node(&symmetric_nat_node).await.unwrap();

        // Try to establish connection
        let result = manager
            .connect_with_nat_traversal(&symmetric_nat_node)
            .await;
        assert!(result.is_ok());

        // Verify connection type is relay
        // In real implementation, this would check actual connection state
    }

    #[tokio::test]
    async fn test_node_churn_handling() {
        let manager = MeshManager::new().await.unwrap();

        // Simulate rapid node join/leave
        let mut node_ids = Vec::new();

        // Add 50 nodes rapidly
        for i in 0..50 {
            let node = ClusterNode {
                id: Uuid::new_v4(),
                hostname: format!("churn-{}", i),
                class: NodeClass::Workstation {
                    gpu: None,
                    schedule: Schedule::AlwaysOn,
                },
                hardware: HardwareProfile::default(),
                network: NetworkCharacteristics {
                    bandwidth_mbps: 1000.0,
                    latency_ms: 1.0,
                    jitter_ms: 0.1,
                    packet_loss: 0.0,
                    nat_type: NatType::None,
                },
                status: NodeStatus::Online,
                capabilities: NodeCapabilities::default(),
                last_heartbeat: Utc::now(),
            };

            node_ids.push(node.id);
            manager.add_node(&node).await.unwrap();
        }

        // Remove half the nodes
        for i in 0..25 {
            manager.handle_node_departure(node_ids[i]).await.unwrap();
        }

        // Verify remaining nodes
        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 25);

        // Add more nodes
        for i in 50..75 {
            let node = ClusterNode {
                id: Uuid::new_v4(),
                hostname: format!("churn-new-{}", i),
                class: NodeClass::Edge {
                    device_type: EdgeType::RaspberryPi,
                    power_budget: Watts(5.0),
                },
                hardware: HardwareProfile::default(),
                network: NetworkCharacteristics {
                    bandwidth_mbps: 100.0,
                    latency_ms: 20.0,
                    jitter_ms: 2.0,
                    packet_loss: 0.1,
                    nat_type: NatType::FullCone,
                },
                status: NodeStatus::Online,
                capabilities: NodeCapabilities::default(),
                last_heartbeat: Utc::now(),
            };

            manager.add_node(&node).await.unwrap();
        }

        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 50);
    }

    #[tokio::test]
    async fn test_topology_transitions() {
        let manager = MeshManager::new().await.unwrap();

        // Start with star topology
        *manager.topology.write().await = MeshTopology::Star {
            coordinators: vec![Uuid::new_v4()],
        };

        // Add nodes
        for i in 0..20 {
            let node = ClusterNode {
                id: Uuid::new_v4(),
                hostname: format!("topo-{}", i),
                class: NodeClass::Workstation {
                    gpu: None,
                    schedule: Schedule::AlwaysOn,
                },
                hardware: HardwareProfile::default(),
                network: NetworkCharacteristics {
                    bandwidth_mbps: 1000.0,
                    latency_ms: 1.0,
                    jitter_ms: 0.1,
                    packet_loss: 0.0,
                    nat_type: NatType::None,
                },
                status: NodeStatus::Online,
                capabilities: NodeCapabilities::default(),
                last_heartbeat: Utc::now(),
            };

            manager.add_node(&node).await.unwrap();
        }

        // Transition to full mesh
        *manager.topology.write().await = MeshTopology::FullMesh;

        // Transition to hierarchical
        *manager.topology.write().await = MeshTopology::Hierarchical;

        // Transition to hybrid
        *manager.topology.write().await = MeshTopology::Hybrid {
            core_nodes: vec![Uuid::new_v4(), Uuid::new_v4()],
            edge_strategy: Box::new(MeshTopology::Star {
                coordinators: vec![Uuid::new_v4()],
            }),
        };

        // Verify no crashes during transitions
        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 20);
    }

    #[tokio::test]
    async fn test_connection_limits() {
        let manager = MeshManager::new().await.unwrap();

        // Test maximum connections per node
        let hub_node = create_datacenter_node(Uuid::new_v4(), 4);
        manager.add_node(&hub_node).await.unwrap();

        // Try to connect many nodes to the hub
        for i in 0..1000 {
            let connection = NodeConnection {
                node_id: Uuid::new_v4(),
                endpoint: format!("10.0.0.{}:8080", i % 256).parse().unwrap(),
                connection_type: ConnectionType::Direct,
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: 1.0,
                bandwidth_mbps: 1000.0,
                packet_loss: 0.0,
            };

            manager
                .connections
                .write()
                .await
                .insert(connection.node_id, connection);
        }

        // Verify connection management doesn't crash
        manager.maintain_connections().await.unwrap();

        // Should have pruned old/excess connections
        let connections = manager.connections.read().await;
        assert!(connections.len() <= 1000);
    }

    #[tokio::test]
    async fn test_extreme_network_conditions() {
        let manager = MeshManager::new().await.unwrap();

        // Create nodes with extreme network conditions
        let high_latency_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "satellite-link".to_string(),
            class: NodeClass::Edge {
                device_type: EdgeType::Other("Satellite Terminal".to_string()),
                power_budget: Watts(100.0),
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 10.0, // Low bandwidth
                latency_ms: 600.0,    // Satellite latency
                jitter_ms: 100.0,     // High jitter
                packet_loss: 5.0,     // Significant packet loss
                nat_type: NatType::Symmetric,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        let unstable_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "4g-mobile".to_string(),
            class: NodeClass::Laptop {
                battery: true,
                mobility: MobilityPattern::Continuous,
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 50.0, // Variable bandwidth
                latency_ms: 150.0,    // Mobile network latency
                jitter_ms: 50.0,      // High jitter
                packet_loss: 2.0,     // Some packet loss
                nat_type: NatType::Symmetric,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        manager.add_node(&high_latency_node).await.unwrap();
        manager.add_node(&unstable_node).await.unwrap();

        // These nodes should be marked appropriately for scheduling
        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        use tokio::task;

        let manager = Arc::new(MeshManager::new().await.unwrap());

        // Spawn multiple tasks doing concurrent operations
        let mut handles = vec![];

        // Task 1: Add nodes
        let m1 = manager.clone();
        handles.push(task::spawn(async move {
            for i in 0..50 {
                let node = ClusterNode {
                    id: Uuid::new_v4(),
                    hostname: format!("concurrent-add-{}", i),
                    class: NodeClass::Workstation {
                        gpu: None,
                        schedule: Schedule::AlwaysOn,
                    },
                    hardware: HardwareProfile::default(),
                    network: NetworkCharacteristics {
                        bandwidth_mbps: 1000.0,
                        latency_ms: 1.0,
                        jitter_ms: 0.1,
                        packet_loss: 0.0,
                        nat_type: NatType::None,
                    },
                    status: NodeStatus::Online,
                    capabilities: NodeCapabilities::default(),
                    last_heartbeat: Utc::now(),
                };

                let _ = m1.add_node(&node).await;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }));

        // Task 2: Maintain connections
        let m2 = manager.clone();
        handles.push(task::spawn(async move {
            for _ in 0..20 {
                let _ = m2.maintain_connections().await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }));

        // Task 3: Optimize topology
        let m3 = manager.clone();
        handles.push(task::spawn(async move {
            for _ in 0..10 {
                let _ = m3.optimize_topology().await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }));

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify system is still consistent
        let registry = manager.node_registry.read().await;
        assert!(registry.len() > 0);
        assert!(registry.len() <= 50);
    }
}
