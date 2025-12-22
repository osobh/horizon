//! Additional tests for mesh module

#[cfg(test)]
mod mesh_coverage_tests {
    use super::super::*;
    use crate::classification::{Bandwidth, EdgeType, MobilityPattern, Schedule, Watts};
    use crate::discovery::{HardwareProfile, NatType, NetworkCharacteristics};
    use crate::{ClusterNode, NodeCapabilities, NodeClass, NodeStatus};
    use chrono::Utc;
    use uuid::Uuid;

    fn create_test_node_with_nat(id: Uuid, nat_type: NatType) -> ClusterNode {
        ClusterNode {
            id,
            hostname: format!("node-{}", id),
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
                nat_type,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_endpoint_initialization() {
        let manager = MeshManager::new().await.unwrap();

        // Initialize endpoint
        manager.initialize_endpoint().await.unwrap();

        // Check endpoint is set
        let endpoint = manager.endpoint.read().await;
        assert!(endpoint.is_some());
    }

    #[tokio::test]
    async fn test_topology_management() {
        let manager = MeshManager::new().await.unwrap();

        // Test different topology types
        *manager.topology.write().await = MeshTopology::FullMesh;
        let topology = manager.topology.read().await;
        assert!(matches!(*topology, MeshTopology::FullMesh));

        // Test star topology
        let coordinators = vec![Uuid::new_v4(), Uuid::new_v4()];
        *manager.topology.write().await = MeshTopology::Star {
            coordinators: coordinators.clone(),
        };
        let topology = manager.topology.read().await;
        match &*topology {
            MeshTopology::Star { coordinators: c } => assert_eq!(c, &coordinators),
            _ => panic!("Expected star topology"),
        }
    }

    #[tokio::test]
    async fn test_node_types_mesh_formation() {
        let manager = MeshManager::new().await.unwrap();

        // Add different node types
        let dc_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "dc-1".to_string(),
            class: NodeClass::DataCenter {
                gpus: vec![],
                bandwidth: Bandwidth::TenGigabit(10.0),
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 10000.0,
                latency_ms: 0.1,
                jitter_ms: 0.01,
                packet_loss: 0.0,
                nat_type: NatType::None,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        let laptop_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "laptop-1".to_string(),
            class: NodeClass::Laptop {
                battery: true,
                mobility: MobilityPattern::Frequent,
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 100.0,
                latency_ms: 10.0,
                jitter_ms: 1.0,
                packet_loss: 0.1,
                nat_type: NatType::Symmetric,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        let edge_node = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "edge-1".to_string(),
            class: NodeClass::Edge {
                device_type: EdgeType::RaspberryPi,
                power_budget: Watts(15.0),
            },
            hardware: HardwareProfile::default(),
            network: NetworkCharacteristics {
                bandwidth_mbps: 50.0,
                latency_ms: 20.0,
                jitter_ms: 2.0,
                packet_loss: 0.5,
                nat_type: NatType::PortRestricted,
            },
            status: NodeStatus::Online,
            capabilities: NodeCapabilities::default(),
            last_heartbeat: Utc::now(),
        };

        // Add nodes
        manager.add_node(&dc_node).await.unwrap();
        manager.add_node(&laptop_node).await.unwrap();
        manager.add_node(&edge_node).await.unwrap();

        // Check registry
        let registry = manager.node_registry.read().await;
        assert_eq!(registry.len(), 3);
        assert!(registry.contains_key(&dc_node.id));
        assert!(registry.contains_key(&laptop_node.id));
        assert!(registry.contains_key(&edge_node.id));

        // Check relay nodes (only datacenter should be relay)
        let relays = manager.relay_nodes.read().await;
        assert_eq!(relays.len(), 1);
        assert!(relays.contains(&dc_node.id));
    }

    #[tokio::test]
    async fn test_nat_traversal_scenarios() {
        let manager = MeshManager::new().await.unwrap();

        // Test different NAT types
        let nat_types = vec![
            NatType::None,
            NatType::FullCone,
            NatType::RestrictedCone,
            NatType::PortRestricted,
            NatType::Symmetric,
        ];

        for nat_type in nat_types {
            let node = create_test_node_with_nat(Uuid::new_v4(), nat_type.clone());
            manager.add_node(&node).await.unwrap();

            // Verify node was added with correct NAT type
            let registry = manager.node_registry.read().await;
            let endpoint = registry.get(&node.id).unwrap();
            assert_eq!(endpoint.nat_type, nat_type);
        }
    }

    #[tokio::test]
    async fn test_connection_management() {
        let manager = MeshManager::new().await.unwrap();

        // Add a connection
        let node_id = Uuid::new_v4();
        let connection = NodeConnection {
            node_id,
            endpoint: "127.0.0.1:8080".parse().unwrap(),
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
            .insert(node_id, connection);

        // Test connection types
        let conn_types = vec![
            ConnectionType::Direct,
            ConnectionType::NatTraversal,
            ConnectionType::Relay(Uuid::new_v4()),
            ConnectionType::Tunnel,
        ];

        for conn_type in conn_types {
            let node_id = Uuid::new_v4();
            let mut connection = NodeConnection {
                node_id,
                endpoint: "127.0.0.1:8081".parse().unwrap(),
                connection_type: conn_type.clone(),
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: 5.0,
                bandwidth_mbps: 500.0,
                packet_loss: 0.1,
            };

            manager
                .connections
                .write()
                .await
                .insert(node_id, connection.clone());

            // Verify connection
            let connections = manager.connections.read().await;
            let stored_conn = connections.get(&node_id).unwrap();
            assert_eq!(stored_conn.connection_type, conn_type);
        }
    }

    #[tokio::test]
    async fn test_node_departure_handling() {
        let manager = MeshManager::new().await.unwrap();

        // Create relay node
        let relay_id = Uuid::new_v4();
        manager.relay_nodes.write().await.insert(relay_id);

        // Add connections that depend on the relay
        let dependent_nodes = vec![Uuid::new_v4(), Uuid::new_v4()];
        for node_id in &dependent_nodes {
            let connection = NodeConnection {
                node_id: *node_id,
                endpoint: "127.0.0.1:9000".parse().unwrap(),
                connection_type: ConnectionType::Relay(relay_id),
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: 10.0,
                bandwidth_mbps: 100.0,
                packet_loss: 0.2,
            };
            manager
                .connections
                .write()
                .await
                .insert(*node_id, connection);
        }

        // Handle relay departure
        manager.handle_node_departure(relay_id).await.unwrap();

        // In a real implementation, connections would be reestablished
        // For now, we just verify the method doesn't panic
    }

    #[tokio::test]
    async fn test_mesh_topology_establishment() {
        let manager = MeshManager::new().await.unwrap();

        // Test hierarchical topology
        *manager.topology.write().await = MeshTopology::Hierarchical;

        let workstation = ClusterNode {
            id: Uuid::new_v4(),
            hostname: "ws-1".to_string(),
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

        // This will call establish_hierarchical_connections
        manager.establish_connections(&workstation).await.unwrap();

        // Test star topology
        let coordinator_id = Uuid::new_v4();
        *manager.topology.write().await = MeshTopology::Star {
            coordinators: vec![coordinator_id],
        };

        // Add coordinator to registry
        let coordinator_endpoint = NodeEndpoint {
            node_id: coordinator_id,
            public_addr: Some("1.2.3.4:5000".parse().unwrap()),
            private_addr: "192.168.1.1:5000".parse().unwrap(),
            nat_type: NatType::None,
            last_updated: Utc::now(),
        };
        manager
            .node_registry
            .write()
            .await
            .insert(coordinator_id, coordinator_endpoint);

        // Test connection establishment
        manager.establish_connections(&workstation).await.unwrap();
    }

    #[tokio::test]
    async fn test_connection_maintenance() {
        let manager = MeshManager::new().await.unwrap();

        // Add stale connection
        let stale_node_id = Uuid::new_v4();
        let stale_connection = NodeConnection {
            node_id: stale_node_id,
            endpoint: "127.0.0.1:7000".parse().unwrap(),
            connection_type: ConnectionType::Direct,
            established_at: Utc::now() - chrono::Duration::minutes(10),
            last_heartbeat: Utc::now() - chrono::Duration::minutes(5), // Old heartbeat
            latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            packet_loss: 0.0,
        };

        // Add fresh connection
        let fresh_node_id = Uuid::new_v4();
        let fresh_connection = NodeConnection {
            node_id: fresh_node_id,
            endpoint: "127.0.0.1:7001".parse().unwrap(),
            connection_type: ConnectionType::Direct,
            established_at: Utc::now(),
            last_heartbeat: Utc::now(), // Recent heartbeat
            latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            packet_loss: 0.0,
        };

        manager
            .connections
            .write()
            .await
            .insert(stale_node_id, stale_connection);
        manager
            .connections
            .write()
            .await
            .insert(fresh_node_id, fresh_connection);

        // Run maintenance
        manager.maintain_connections().await.unwrap();

        // Check that stale connection was removed
        let connections = manager.connections.read().await;
        assert!(!connections.contains_key(&stale_node_id));
        assert!(connections.contains_key(&fresh_node_id));
    }

    #[tokio::test]
    async fn test_topology_optimization() {
        let manager = MeshManager::new().await.unwrap();

        // Add connections with various metrics
        for i in 0..5 {
            let node_id = Uuid::new_v4();
            let connection = NodeConnection {
                node_id,
                endpoint: format!("127.0.0.1:{}", 8000 + i).parse().unwrap(),
                connection_type: ConnectionType::Direct,
                established_at: Utc::now(),
                last_heartbeat: Utc::now(),
                latency_ms: (i + 1) as f32 * 2.0,
                bandwidth_mbps: 1000.0 / (i + 1) as f32,
                packet_loss: i as f32 * 0.01,
            };
            manager
                .connections
                .write()
                .await
                .insert(node_id, connection);
        }

        // Run optimization (this just logs metrics currently)
        manager.optimize_topology().await.unwrap();
    }
}

// Add coordinator to registry
use super::NodeEndpoint;
