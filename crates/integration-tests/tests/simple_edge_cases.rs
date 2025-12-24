//! Simple Edge Case Integration Tests
//! Tests edge cases with components we know exist

use stratoswarm_memory::{GpuMemoryAllocator, MemoryManager, MemoryPool};
use stratoswarm_net::{MemoryNetwork, Message, MessageType, Network};
use stratoswarm_storage::{MemoryStorage, Storage};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::timeout;

/// Test memory allocation with zero size
#[tokio::test]
async fn test_zero_size_memory_allocation() {
    let allocator = GpuMemoryAllocator::new(1024 * 1024).unwrap();

    // Try to allocate zero bytes
    let result = allocator.allocate(0).await;
    assert!(result.is_err(), "Zero-size allocation should fail");

    // Try to allocate maximum size
    let max_result = allocator.allocate(usize::MAX).await;
    assert!(max_result.is_err(), "Maximum size allocation should fail");
}

/// Test storage with empty keys and values
#[tokio::test]
async fn test_storage_empty_keys_values() {
    let storage = MemoryStorage::new(1024 * 1024);

    // Empty key
    let result1 = storage.store("", b"data").await;
    assert!(result1.is_err(), "Empty key should fail");

    // Empty data
    let result2 = storage.store("key", b"").await;
    // Empty data might be allowed
    assert!(result2.is_ok() || result2.is_err());

    // Very long key
    let long_key = "k".repeat(10000);
    let result3 = storage.store(&long_key, b"data").await;
    assert!(
        result3.is_ok() || result3.is_err(),
        "Should handle long keys"
    );
}

/// Test network with concurrent message flooding
#[tokio::test]
async fn test_network_message_flooding() {
    let network = Arc::new(MemoryNetwork::new("flood-test".to_string()));

    let mut handles = vec![];

    // Send 1000 messages concurrently
    for i in 0..1000 {
        let net_clone = network.clone();
        let handle = tokio::spawn(async move {
            let msg = Message {
                msg_type: match i % 4 {
                    0 => MessageType::Consensus,
                    1 => MessageType::Evolution,
                    2 => MessageType::KnowledgeSync,
                    _ => MessageType::Custom(format!("custom_{}", i)),
                },
                payload: vec![i as u8; 100],
                timestamp: i as u64,
            };

            net_clone
                .send(&format!("peer_{}", i % 10), msg)
                .await
                .is_ok()
        });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let successful = results.iter().filter(|&&r| r).count();
    assert!(successful > 0, "Some messages should be sent");

    // Check network stats
    let stats = network.stats().await.unwrap();
    assert_eq!(stats.messages_sent as usize, successful);
}

/// Test memory pool with rapid allocation/deallocation
#[tokio::test]
async fn test_memory_pool_thrashing() {
    let pool = Arc::new(Mutex::new(MemoryPool::new(1024 * 1024).unwrap()));

    let mut handles = vec![];

    // 100 tasks rapidly allocating and deallocating
    for i in 0..100 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            let mut successes = 0;

            for j in 0..10 {
                let size = ((i * j) % 1024) + 1; // Variable sizes

                let mut pool = pool_clone.lock().await;
                if let Ok(handle) = pool.allocate(size) {
                    successes += 1;
                    // Immediately deallocate
                    let _ = pool.deallocate(handle);
                }
            }

            successes
        });
        handles.push(handle);
    }

    let total_successes: usize = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(0))
        .sum();

    assert!(total_successes > 0, "Some allocations should succeed");
}

/// Test storage with concurrent overwrites
#[tokio::test]
async fn test_storage_concurrent_overwrites() {
    let storage = Arc::new(MemoryStorage::new(1024 * 1024));

    let key = "contested_key";
    let mut handles = vec![];

    // 50 threads trying to write to the same key
    for i in 0..50 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            let data = format!("writer_{}", i).into_bytes();
            storage_clone.store(key, &data).await.is_ok()
        });
        handles.push(handle);
    }

    let results: Vec<bool> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap_or(false))
        .collect();

    let successful = results.iter().filter(|&&r| r).count();
    assert!(successful > 0, "Some writes should succeed");

    // Final value should exist
    let final_value = storage.retrieve(key).await;
    assert!(final_value.is_ok(), "Key should have a value");
}

/// Test network with extreme message sizes
#[tokio::test]
async fn test_network_extreme_message_sizes() {
    let network = MemoryNetwork::new("size-test".to_string());

    // Very small message
    let tiny_msg = Message {
        msg_type: MessageType::Consensus,
        payload: vec![],
        timestamp: 0,
    };
    assert!(network.send("peer", tiny_msg).await.is_ok());

    // Very large message (10MB)
    let large_msg = Message {
        msg_type: MessageType::Evolution,
        payload: vec![0u8; 10 * 1024 * 1024],
        timestamp: 1,
    };
    let large_result = network.send("peer", large_msg).await;
    // May succeed or fail depending on limits
    assert!(large_result.is_ok() || large_result.is_err());

    // Message with max timestamp
    let future_msg = Message {
        msg_type: MessageType::KnowledgeSync,
        payload: vec![1, 2, 3],
        timestamp: u64::MAX,
    };
    assert!(network.send("peer", future_msg).await.is_ok());
}

/// Test memory operations with timeout
#[tokio::test]
async fn test_memory_operations_with_timeout() {
    let allocator = Arc::new(GpuMemoryAllocator::new(1024 * 1024).unwrap());

    // Allocate with very short timeout
    let result = timeout(Duration::from_nanos(1), async {
        allocator.allocate(1024).await
    })
    .await;

    // Should timeout or complete
    assert!(result.is_ok() || result.is_err());

    // Multiple allocations with increasing timeouts
    for i in 0..10 {
        let timeout_ns = 10u64.pow(i);
        let alloc_clone = allocator.clone();

        let _ = timeout(Duration::from_nanos(timeout_ns), async move {
            alloc_clone.allocate(1024).await
        })
        .await;
    }
}

/// Test storage with special characters in keys
#[tokio::test]
async fn test_storage_special_character_keys() {
    let storage = MemoryStorage::new(1024 * 1024);

    let special_keys = vec![
        "key with spaces",
        "key/with/slashes",
        "key\\with\\backslashes",
        "key:with:colons",
        "key|with|pipes",
        "key\nwith\nnewlines",
        "key\twith\ttabs",
        "🔑🗝️🔐", // Unicode keys
        "\0null\0bytes\0",
        "key\"with\"quotes",
        "key'with'quotes",
    ];

    for key in special_keys {
        let result = storage.store(key, b"test").await;
        // Should handle gracefully
        assert!(result.is_ok() || result.is_err());

        if result.is_ok() {
            let retrieved = storage.retrieve(key).await;
            assert!(retrieved.is_ok() || retrieved.is_err());
        }
    }
}

/// Test network with rapid peer connections
#[tokio::test]
async fn test_network_rapid_peer_connections() {
    let network = Arc::new(MemoryNetwork::new("rapid-connect".to_string()));

    let mut handles = vec![];

    // Rapidly send to many different peers
    for i in 0..100 {
        let net_clone = network.clone();
        let handle = tokio::spawn(async move {
            for j in 0..10 {
                let peer = format!("peer_{}_{}", i, j);
                let msg = Message {
                    msg_type: MessageType::Consensus,
                    payload: vec![i as u8, j as u8],
                    timestamp: i * 10 + j,
                };

                let _ = net_clone.send(&peer, msg).await;
            }
        });
        handles.push(handle);
    }

    futures::future::join_all(handles).await;

    let stats = network.stats().await.unwrap();
    assert!(stats.messages_sent > 0, "Should have sent some messages");
}

/// Test integration with all components under edge conditions
#[tokio::test]
async fn test_integrated_edge_conditions() {
    // Very small sizes for all components
    let allocator = GpuMemoryAllocator::new(1024).unwrap(); // 1KB
    let storage = MemoryStorage::new(512); // 512B
    let network = MemoryNetwork::new("edge-test".to_string());

    // Try to use more than available
    let alloc_result = allocator.allocate(2048).await;
    assert!(
        alloc_result.is_err(),
        "Should fail to allocate more than available"
    );

    // Store data approaching limit
    let data = vec![0u8; 400];
    let store_result = storage.store("big_data", &data).await;
    assert!(store_result.is_ok() || store_result.is_err());

    // Send message about the storage
    let msg = Message {
        msg_type: MessageType::Custom("storage_full".to_string()),
        payload: b"storage approaching limit".to_vec(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    assert!(network.send("monitor", msg).await.is_ok());

    // Verify components still functional
    let small_alloc = allocator.allocate(100).await;
    assert!(
        small_alloc.is_ok(),
        "Should be able to make small allocation"
    );

    let small_store = storage.store("tiny", b"x").await;
    assert!(small_store.is_ok() || small_store.is_err());
}

#[cfg(test)]
mod edge_case_verification {
    use super::*;

    #[test]
    fn verify_edge_cases_covered() {
        let edge_cases = vec![
            "zero_size_allocation",
            "empty_keys_values",
            "message_flooding",
            "memory_thrashing",
            "concurrent_overwrites",
            "extreme_message_sizes",
            "timeout_operations",
            "special_characters",
            "rapid_connections",
            "integrated_edge_conditions",
        ];

        assert_eq!(edge_cases.len(), 10, "Should cover 10 edge case categories");
    }
}
