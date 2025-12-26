//! Simple integration tests for ExoRust components

use stratoswarm_memory::{GpuMemoryAllocator, MemoryManager, MemoryPool};
use stratoswarm_net::{MemoryNetwork, Message, MessageType, Network};
use stratoswarm_storage::{MemoryStorage, Storage};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Test basic memory and storage integration
#[tokio::test]
async fn test_memory_storage_integration() {
    // Initialize memory management
    let memory_allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 10) // 10MB
            .expect("Failed to create memory allocator"),
    );

    // Initialize storage
    let storage = Arc::new(MemoryStorage::new(1024 * 1024 * 5)); // 5MB storage

    // Allocate memory
    let handle1 = memory_allocator
        .allocate(1024 * 1024)
        .await
        .expect("Failed to allocate 1MB");

    // Store data
    let data = b"Test data for integration";
    storage
        .store("test_key", data)
        .await
        .expect("Failed to store data");

    // Retrieve and verify
    let retrieved = storage
        .retrieve("test_key")
        .await
        .expect("Failed to retrieve data");
    assert_eq!(retrieved, data);

    // Clean up
    memory_allocator
        .deallocate(handle1)
        .await
        .expect("Failed to deallocate");
    storage
        .delete("test_key")
        .await
        .expect("Failed to delete data");
}

/// Test network and storage integration
#[tokio::test]
async fn test_network_storage_integration() {
    let network = Arc::new(MemoryNetwork::new("test-node".to_string()));
    let storage = Arc::new(MemoryStorage::new(1024 * 1024)); // 1MB

    // Store some data
    let important_data = b"Important message content";
    storage
        .store("msg/001", important_data)
        .await
        .expect("Failed to store message data");

    // Send network message about the stored data
    let msg = Message::new(
        MessageType::KnowledgeSync,
        b"msg/001".to_vec(),
    );

    network
        .send("peer-node", msg.clone())
        .await
        .expect("Failed to send message");

    // Verify network stats
    let stats = network.stats().await.expect("Failed to get stats");
    assert_eq!(stats.messages_sent, 1);
    assert_eq!(stats.bytes_sent, b"msg/001".len() as u64);

    // Simulate receiving the message
    network
        .simulate_receive("peer-node".to_string(), msg.clone())
        .await
        .expect("Failed to simulate receive");

    // Receive and process
    let (from, received_msg) = network.receive().await.expect("Failed to receive message");
    assert_eq!(from, "peer-node");
    assert_eq!(received_msg.msg_type, MessageType::KnowledgeSync);

    // Retrieve the referenced data
    let key = String::from_utf8(received_msg.payload).expect("Invalid UTF-8 in payload");
    let data = storage
        .retrieve(&key)
        .await
        .expect("Failed to retrieve referenced data");
    assert_eq!(data, important_data);
}

/// Test memory pool with multiple allocations
#[tokio::test]
async fn test_memory_pool_integration() {
    let allocator = Arc::new(
        GpuMemoryAllocator::new(1024 * 1024 * 20) // 20MB
            .expect("Failed to create allocator"),
    );

    let pool = Arc::new(MemoryPool::new(1024 * 1024, 5, allocator.clone())); // 1MB blocks

    // Acquire multiple blocks
    let mut handles = Vec::new();
    for i in 0..3 {
        let handle = pool
            .acquire()
            .await
            .expect(&format!("Failed to acquire block {i}"));
        assert_eq!(handle.size, 1024 * 1024);
        handles.push(handle);
    }

    // Check pool stats
    let stats = pool.stats().expect("Failed to get pool stats");
    assert_eq!(stats.available_blocks, 0); // All 3 blocks in use
    assert_eq!(stats.utilization_percent, 100.0);

    // Release one block
    let released = handles.pop().unwrap();
    pool.release(released).expect("Failed to release block");

    // Check updated stats
    let stats = pool.stats().expect("Failed to get pool stats");
    assert_eq!(stats.available_blocks, 1);
    assert!((stats.utilization_percent - 66.66667).abs() < 0.01);

    // Clean up remaining blocks
    for handle in handles {
        pool.release(handle).expect("Failed to release block");
    }
}

/// Test storage capacity limits
#[tokio::test]
async fn test_storage_capacity_integration() {
    let storage = MemoryStorage::new(1024); // 1KB capacity

    // Store data that fits
    let small_data = vec![0u8; 400];
    storage
        .store("small", &small_data)
        .await
        .expect("Should store small data");

    // Store more data that still fits
    let medium_data = vec![1u8; 500];
    storage
        .store("medium", &medium_data)
        .await
        .expect("Should store medium data");

    // Try to store data that exceeds capacity
    let large_data = vec![2u8; 200];
    let result = storage.store("large", &large_data).await;
    assert!(
        result.is_err(),
        "Should fail to store data exceeding capacity"
    );

    // Verify storage stats
    let stats = storage.stats().await.expect("Failed to get stats");
    assert_eq!(stats.total_bytes, 1024);
    assert_eq!(
        stats.used_bytes,
        (400 + 500 + "small".len() + "medium".len()) as u64
    );
    assert_eq!(stats.total_files, 2);
}

/// Test concurrent network operations
#[tokio::test]
async fn test_concurrent_network_operations() {
    let network = Arc::new(MemoryNetwork::new("concurrent-test".to_string()));

    // Spawn multiple tasks sending messages
    let mut handles = Vec::new();

    for i in 0..5 {
        let net = network.clone();
        let handle = tokio::spawn(async move {
            let msg = Message::new(
                MessageType::ResourceRequest,
                format!("Request {i}").into_bytes(),
            );
            net.send(&format!("node-{i}"), msg).await
        });
        handles.push(handle);
    }

    // Wait for all sends to complete
    for handle in handles {
        handle.await.expect("Task failed").expect("Send failed");
    }

    // Verify all messages were sent
    let stats = network.stats().await.expect("Failed to get stats");
    assert_eq!(stats.messages_sent, 5);
}

/// Test message routing simulation
#[tokio::test]
async fn test_message_routing() {
    // Create a simple network of 3 nodes
    let node_a = MemoryNetwork::new("node-a".to_string());
    let node_b = MemoryNetwork::new("node-b".to_string());
    let node_c = MemoryNetwork::new("node-c".to_string());

    // Node A sends to Node B
    let msg_ab = Message::new(
        MessageType::KnowledgeSync,
        b"knowledge-from-a".to_vec(),
    );

    node_a
        .send("node-b", msg_ab.clone())
        .await
        .expect("Failed to send from A to B");

    // Simulate B receiving from A
    node_b
        .simulate_receive("node-a".to_string(), msg_ab.clone())
        .await
        .expect("Failed to simulate receive at B");

    // B forwards to C
    let msg_bc = Message::new(
        MessageType::KnowledgeSync,
        b"knowledge-from-a-via-b".to_vec(),
    );

    node_b
        .send("node-c", msg_bc.clone())
        .await
        .expect("Failed to send from B to C");

    // Simulate C receiving from B
    node_c
        .simulate_receive("node-b".to_string(), msg_bc.clone())
        .await
        .expect("Failed to simulate receive at C");

    // Verify message flow
    let (from_a, _msg_at_b) = node_b
        .receive()
        .await
        .expect("B should have message from A");
    assert_eq!(from_a, "node-a");

    let (from_b, _msg_at_c) = node_c
        .receive()
        .await
        .expect("C should have message from B");
    assert_eq!(from_b, "node-b");

    // Verify stats
    let stats_a = node_a.stats().await.expect("Failed to get A stats");
    let stats_b = node_b.stats().await.expect("Failed to get B stats");
    let stats_c = node_c.stats().await.expect("Failed to get C stats");

    assert_eq!(stats_a.messages_sent, 1);
    assert_eq!(stats_b.messages_sent, 1);
    assert_eq!(stats_b.messages_received, 1);
    assert_eq!(stats_c.messages_received, 1);
}
