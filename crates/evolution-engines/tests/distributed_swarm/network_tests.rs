//! Network topology, message passing, and communication reliability tests

use super::{create_distributed_config, create_mock_agent};
use exorust_evolution_engines::{
    error::EvolutionEngineResult,
    swarm_distributed::DistributedSwarmEngine,
};
use std::time::Duration;

#[tokio::test]
async fn test_network_topology_adaptation() -> EvolutionEngineResult<()> {
    // Test dynamic network topology changes
    let mut swarm_nodes = vec![];

    // Create 4 nodes for topology testing
    for i in 1..=4 {
        let mut config = create_distributed_config(&format!("topo_node_{}", i), 8070 + i as u16);

        // Initial star topology - all connect to node 1
        if i > 1 {
            config
                .network_config
                .bootstrap_peers
                .push("127.0.0.1:8071".to_string());
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Wait for initial topology formation
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check initial star topology
    let center_peer_count = swarm_nodes[0].get_peer_count().await;
    assert!(center_peer_count >= 3, "Center node should have 3 peers");

    for i in 1..4 {
        let peer_count = swarm_nodes[i].get_peer_count().await;
        assert!(peer_count >= 1, "Leaf nodes should have at least 1 peer");
    }

    // Trigger topology adaptation
    for node in &mut swarm_nodes {
        node.adapt_topology().await?;
    }

    // Wait for adaptation
    tokio::time::sleep(Duration::from_millis(300)).await;

    // After adaptation, nodes should have more balanced connections
    let total_connections: usize = swarm_nodes
        .iter()
        .map(|node| futures::executor::block_on(node.get_peer_count()))
        .sum();

    // Should have increased connectivity (each node connected to more peers)
    assert!(
        total_connections >= 8,
        "Should have more connections after adaptation"
    );

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_message_passing_reliability() -> EvolutionEngineResult<()> {
    // Test reliable message passing between nodes
    let mut swarm_nodes = vec![];

    // Create 3 nodes for message passing
    for i in 1..=3 {
        let mut config = create_distributed_config(&format!("msg_node_{}", i), 8090 + i as u16);

        if i > 1 {
            config
                .network_config
                .bootstrap_peers
                .push("127.0.0.1:8091".to_string());
        }

        let node = DistributedSwarmEngine::new(config).await?;
        swarm_nodes.push(node);
    }

    // Start nodes
    for node in &mut swarm_nodes {
        node.start().await?;
    }

    // Wait for network formation
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Test direct message sending
    let test_messages = vec![
        "Hello from node 1",
        "Data sync request",
        "Evolution update",
        "Checkpoint notification",
    ];

    // Send messages from node 1 to others
    for (_i, message) in test_messages.iter().enumerate() {
        swarm_nodes[0]
            .send_message_to_node("msg_node_2", message.to_string())
            .await?;
        swarm_nodes[0]
            .send_message_to_node("msg_node_3", message.to_string())
            .await?;

        // Small delay between messages
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Wait for message delivery
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify message receipt
    let node2_messages = swarm_nodes[1].get_received_messages().await;
    let node3_messages = swarm_nodes[2].get_received_messages().await;

    assert_eq!(
        node2_messages.len(),
        test_messages.len(),
        "Node 2 should receive all messages"
    );
    assert_eq!(
        node3_messages.len(),
        test_messages.len(),
        "Node 3 should receive all messages"
    );

    // Test broadcast functionality
    swarm_nodes[1]
        .broadcast_message("Broadcast test message".to_string())
        .await?;

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Both other nodes should receive broadcast
    let node1_broadcasts = swarm_nodes[0].get_received_broadcasts().await;
    let node3_broadcasts = swarm_nodes[2].get_received_broadcasts().await;

    assert!(
        node1_broadcasts.len() >= 1,
        "Node 1 should receive broadcast"
    );
    assert!(
        node3_broadcasts.len() >= 1,
        "Node 3 should receive broadcast"
    );

    // Cleanup
    for node in &mut swarm_nodes {
        node.stop().await?;
    }

    Ok(())
}