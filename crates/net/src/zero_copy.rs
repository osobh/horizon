//! Zero-copy transport implementation

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{Message, Network, NetworkError, NetworkStats};

/// Zero-copy transport for GPU memory operations
pub struct ZeroCopyTransport {
    buffer_pools: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    stats: Arc<Mutex<NetworkStats>>,
    message_queue: Arc<Mutex<Vec<(String, Message)>>>,
}

impl ZeroCopyTransport {
    /// Create new zero-copy transport
    pub fn new() -> Self {
        Self {
            buffer_pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                messages_sent: 0,
                messages_received: 0,
                average_latency_us: 0.1,  // Very low latency for zero-copy
                throughput_mbps: 40000.0, // High throughput simulation for GPU memory
            })),
            message_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Allocate shared buffer for zero-copy operations
    pub fn allocate_shared_buffer(&self, buffer_id: &str, size: usize) -> Result<(), NetworkError> {
        let mut pools = self.buffer_pools.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire buffer pool lock: {e}"
            )))
        })?;

        pools.insert(buffer_id.to_string(), vec![0u8; size]);
        Ok(())
    }
}

impl Default for ZeroCopyTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Network for ZeroCopyTransport {
    async fn send(&self, endpoint: &str, message: Message) -> Result<(), NetworkError> {
        // Update stats
        {
            let mut stats = self.stats.lock().map_err(|e| {
                NetworkError::Io(std::io::Error::other(format!(
                    "Failed to acquire stats lock: {e}"
                )))
            })?;

            let message_size = message.payload.len() as u64;
            stats.bytes_sent += message_size;
            stats.messages_sent += 1;
        }

        // Store message in queue for testing purposes
        {
            let mut queue = self.message_queue.lock().map_err(|e| {
                NetworkError::Io(std::io::Error::other(format!(
                    "Failed to acquire message queue lock: {e}"
                )))
            })?;

            queue.push((endpoint.to_string(), message));
        }

        // In a real implementation, this would use GPU memory directly
        Ok(())
    }

    async fn receive(&self) -> Result<(String, Message), NetworkError> {
        let mut queue = self.message_queue.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire message queue lock: {e}"
            )))
        })?;

        if let Some((endpoint, message)) = queue.pop() {
            // Update receive stats
            let mut stats = self.stats.lock().map_err(|e| {
                NetworkError::Io(std::io::Error::other(format!(
                    "Failed to acquire stats lock: {e}"
                )))
            })?;

            stats.bytes_received += message.payload.len() as u64;
            stats.messages_received += 1;

            Ok((endpoint, message))
        } else {
            Err(NetworkError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "No messages available in zero-copy transport",
            )))
        }
    }

    async fn stats(&self) -> Result<NetworkStats, NetworkError> {
        let stats = self.stats.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire stats lock: {e}"
            )))
        })?;
        Ok(stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageType;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_message(msg_type: MessageType, payload: &[u8]) -> Message {
        Message {
            id: rand::random(),
            msg_type,
            payload: payload.to_vec(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    #[tokio::test]
    async fn test_zero_copy_transport_creation() {
        let transport = ZeroCopyTransport::new();
        let stats = transport.stats().await.expect("Failed to get stats");

        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.average_latency_us, 0.1); // Very low latency
        assert_eq!(stats.throughput_mbps, 40000.0); // High throughput
    }

    #[tokio::test]
    async fn test_zero_copy_send() {
        let transport = ZeroCopyTransport::new();
        let message = create_test_message(MessageType::AgentSpawn, b"zero_copy_data");

        transport
            .send("gpu_node", message)
            .await
            .expect("Failed to send message");

        let stats = transport.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.bytes_sent, b"zero_copy_data".len() as u64);
    }

    #[tokio::test]
    async fn test_buffer_allocation() {
        let transport = ZeroCopyTransport::new();

        transport
            .allocate_shared_buffer("buffer_1", 1024)
            .expect("Failed to allocate buffer");
        transport
            .allocate_shared_buffer("buffer_2", 2048)
            .expect("Failed to allocate buffer");

        // Verify buffers exist in internal state
        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools.len(), 2);
        assert!(pools.contains_key("buffer_1"));
        assert!(pools.contains_key("buffer_2"));
        assert_eq!(pools["buffer_1"].len(), 1024);
        assert_eq!(pools["buffer_2"].len(), 2048);
    }

    #[tokio::test]
    async fn test_default_creation() {
        let transport = ZeroCopyTransport::default();
        let stats = transport.stats().await.expect("Failed to get stats");

        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.throughput_mbps, 40000.0);
    }

    #[tokio::test]
    async fn test_high_throughput_simulation() {
        let transport = ZeroCopyTransport::new();
        let large_payload = vec![0u8; 1024 * 1024]; // 1MB

        for i in 0..10 {
            let message = create_test_message(MessageType::KnowledgeSync, &large_payload);
            transport
                .send(&format!("node_{i}"), message)
                .await
                .expect("Failed to send large message");
        }

        let stats = transport.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 10);
        assert_eq!(stats.bytes_sent, 10 * 1024 * 1024); // 10MB total
        assert_eq!(stats.throughput_mbps, 40000.0); // Still high throughput
    }

    #[tokio::test]
    async fn test_send_and_receive() {
        let transport = ZeroCopyTransport::new();

        // Send a message
        let message = create_test_message(MessageType::AgentSpawn, b"test_data");
        transport
            .send("test_endpoint", message.clone())
            .await
            .unwrap();

        // Receive the message
        let (endpoint, received) = transport.receive().await.unwrap();
        assert_eq!(endpoint, "test_endpoint");
        assert_eq!(received.msg_type, message.msg_type);
        assert_eq!(received.payload, message.payload);

        // Check stats were updated
        let stats = transport.stats().await.unwrap();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_sent, b"test_data".len() as u64);
        assert_eq!(stats.bytes_received, b"test_data".len() as u64);
    }

    #[tokio::test]
    async fn test_receive_empty_queue() {
        let transport = ZeroCopyTransport::new();

        // Receive should fail when queue is empty
        let result = transport.receive().await;
        assert!(
            matches!(result, Err(NetworkError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock)
        );
    }

    #[tokio::test]
    async fn test_buffer_operations() {
        let transport = ZeroCopyTransport::new();

        // Test allocating multiple buffers
        transport
            .allocate_shared_buffer("gpu_buffer_1", 4096)
            .expect("Failed to allocate buffer 1");
        transport
            .allocate_shared_buffer("gpu_buffer_2", 8192)
            .expect("Failed to allocate buffer 2");
        transport
            .allocate_shared_buffer("gpu_buffer_3", 16384)
            .expect("Failed to allocate buffer 3");

        // Verify internal buffer state
        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools.len(), 3);
        assert_eq!(pools["gpu_buffer_1"].len(), 4096);
        assert_eq!(pools["gpu_buffer_2"].len(), 8192);
        assert_eq!(pools["gpu_buffer_3"].len(), 16384);
    }

    #[tokio::test]
    async fn test_buffer_overwrite() {
        let transport = ZeroCopyTransport::new();

        // Allocate buffer
        transport
            .allocate_shared_buffer("overwrite_buffer", 1024)
            .expect("Failed to allocate initial buffer");

        // Verify initial allocation
        {
            let pools = transport.buffer_pools.lock().unwrap();
            assert_eq!(pools["overwrite_buffer"].len(), 1024);
        }

        // Overwrite with different size
        transport
            .allocate_shared_buffer("overwrite_buffer", 2048)
            .expect("Failed to overwrite buffer");

        // Verify overwrite
        {
            let pools = transport.buffer_pools.lock().unwrap();
            assert_eq!(pools["overwrite_buffer"].len(), 2048);
        }
    }

    #[tokio::test]
    async fn test_zero_size_buffer() {
        let transport = ZeroCopyTransport::new();

        // Allocate zero-size buffer
        transport
            .allocate_shared_buffer("zero_buffer", 0)
            .expect("Failed to allocate zero-size buffer");

        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools["zero_buffer"].len(), 0);
    }

    #[tokio::test]
    async fn test_empty_message_transport() {
        let transport = ZeroCopyTransport::new();
        let empty_message = create_test_message(MessageType::AgentSpawn, &[]);

        transport
            .send("empty_endpoint", empty_message)
            .await
            .expect("Failed to send empty message");

        let stats = transport.stats().await.expect("Failed to get stats");
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_sent, 1);
    }

    #[tokio::test]
    async fn test_stats_accumulation() {
        let transport = ZeroCopyTransport::new();

        // Send messages of various sizes
        let small_msg = create_test_message(MessageType::AgentSpawn, &vec![1u8; 100]);
        let medium_msg = create_test_message(MessageType::ResourceRequest, &vec![2u8; 1000]);
        let large_msg = create_test_message(MessageType::KnowledgeSync, &vec![3u8; 10000]);

        transport
            .send("endpoint1", small_msg)
            .await
            .expect("Failed to send small");
        transport
            .send("endpoint2", medium_msg)
            .await
            .expect("Failed to send medium");
        transport
            .send("endpoint3", large_msg)
            .await
            .expect("Failed to send large");

        let stats = transport.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 3);
        assert_eq!(stats.bytes_sent, 100 + 1000 + 10000); // Total bytes
        assert_eq!(stats.average_latency_us, 0.1);
        assert_eq!(stats.throughput_mbps, 40000.0);
    }

    #[tokio::test]
    async fn test_concurrent_buffer_allocation() {
        let transport = Arc::new(ZeroCopyTransport::new());

        // Create multiple tasks allocating buffers concurrently
        let mut handles = Vec::new();
        for i in 0..10 {
            let transport_clone = transport.clone();
            let handle = tokio::spawn(async move {
                let buffer_name = format!("concurrent_buffer_{i}");
                let buffer_size = (i + 1) * 1024;
                transport_clone.allocate_shared_buffer(&buffer_name, buffer_size)
            });
            handles.push(handle);
        }

        // Wait for all allocations to complete
        for handle in handles {
            handle
                .await
                .expect("Task failed")
                .expect("Buffer allocation failed");
        }

        // Verify all buffers were allocated
        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools.len(), 10);

        for i in 0..10 {
            let buffer_name = format!("concurrent_buffer_{i}");
            let expected_size = (i + 1) * 1024;
            assert_eq!(pools[&buffer_name].len(), expected_size);
        }
    }

    #[tokio::test]
    async fn test_concurrent_send_operations() {
        let transport = Arc::new(ZeroCopyTransport::new());

        // Send multiple messages concurrently
        let mut handles = Vec::new();
        for i in 0..20 {
            let transport_clone = transport.clone();
            let handle = tokio::spawn(async move {
                let payload = vec![(i % 256) as u8; i * 100];
                let message = create_test_message(MessageType::KnowledgeSync, &payload);
                transport_clone
                    .send(&format!("concurrent_endpoint_{i}"), message)
                    .await
            });
            handles.push(handle);
        }

        // Wait for all sends to complete
        for handle in handles {
            handle.await.expect("Task failed").expect("Send failed");
        }

        let stats = transport.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 20);

        // Calculate expected total bytes: sum of (i * 100) for i from 0 to 19
        let expected_bytes: u64 = (0..20).map(|i| (i * 100) as u64).sum();
        assert_eq!(stats.bytes_sent, expected_bytes);
    }

    #[tokio::test]
    async fn test_mutex_poisoning_allocate_buffer() {
        use crate::test_helpers::tests::PoisonedZeroCopyTransport;

        let poisoned = PoisonedZeroCopyTransport::new_buffer_pools_poisoned();
        let transport = ZeroCopyTransport {
            buffer_pools: poisoned.buffer_pools,
            stats: poisoned.stats,
            message_queue: poisoned.message_queue,
        };

        let result = transport.allocate_shared_buffer("test_buffer", 1024);
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire buffer pool lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_send() {
        use crate::test_helpers::tests::PoisonedZeroCopyTransport;

        let poisoned = PoisonedZeroCopyTransport::new_stats_poisoned();
        let transport = ZeroCopyTransport {
            buffer_pools: poisoned.buffer_pools,
            stats: poisoned.stats,
            message_queue: poisoned.message_queue,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = transport.send("endpoint", message).await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::PoisonedZeroCopyTransport;

        let poisoned = PoisonedZeroCopyTransport::new_stats_poisoned();
        let transport = ZeroCopyTransport {
            buffer_pools: poisoned.buffer_pools,
            stats: poisoned.stats,
            message_queue: poisoned.message_queue,
        };

        let result = transport.stats().await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_send_message_queue() {
        use crate::test_helpers::tests::PoisonedZeroCopyTransport;

        let poisoned = PoisonedZeroCopyTransport::new_message_queue_poisoned();
        let transport = ZeroCopyTransport {
            buffer_pools: poisoned.buffer_pools,
            stats: poisoned.stats,
            message_queue: poisoned.message_queue,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = transport.send("endpoint", message).await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e
                    .to_string()
                    .contains("Failed to acquire message queue lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_receive() {
        use crate::test_helpers::tests::PoisonedZeroCopyTransport;

        let poisoned = PoisonedZeroCopyTransport::new_message_queue_poisoned();
        let transport = ZeroCopyTransport {
            buffer_pools: poisoned.buffer_pools,
            stats: poisoned.stats,
            message_queue: poisoned.message_queue,
        };

        let result = transport.receive().await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e
                    .to_string()
                    .contains("Failed to acquire message queue lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_receive_empty_queue_duplicate() {
        let transport = ZeroCopyTransport::new();

        // Try to receive when queue is empty
        let result = transport.receive().await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Expected error when queue is empty
            }
            _ => panic!("Expected WouldBlock error when queue is empty"),
        }
    }

    #[tokio::test]
    async fn test_large_buffer_allocation() {
        let transport = ZeroCopyTransport::new();

        // Allocate very large buffer (100MB)
        transport
            .allocate_shared_buffer("huge_buffer", 100 * 1024 * 1024)
            .expect("Failed to allocate huge buffer");

        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools["huge_buffer"].len(), 100 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_special_characters_in_buffer_names() {
        let transport = ZeroCopyTransport::new();

        // Test buffer names with special characters
        let special_names = vec![
            "buffer-with-dashes",
            "buffer_with_underscores",
            "buffer.with.dots",
            "buffer:with:colons",
            "buffer/with/slashes",
            "buffer@with@ats",
            "buffer#with#hashes",
            "buffer with spaces",
            "buffer\twith\ttabs",
            "缓冲区",     // Chinese characters
            "バッファ",   // Japanese characters
            "буфер",      // Russian characters
            "🚀buffer🚀", // Emojis
        ];

        for (i, name) in special_names.iter().enumerate() {
            transport
                .allocate_shared_buffer(name, (i + 1) * 100)
                .expect(&format!("Failed to allocate buffer with name: {}", name));
        }

        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools.len(), special_names.len());

        for (i, name) in special_names.iter().enumerate() {
            assert!(pools.contains_key(*name));
            assert_eq!(pools[*name].len(), (i + 1) * 100);
        }
    }

    #[tokio::test]
    async fn test_message_queue_ordering() {
        let transport = ZeroCopyTransport::new();

        // Send multiple messages
        for i in 0..5 {
            let msg =
                create_test_message(MessageType::AgentSpawn, format!("message_{}", i).as_bytes());
            transport
                .send(&format!("endpoint_{}", i), msg)
                .await
                .expect("Failed to send");
        }

        // Receive messages - ZeroCopyTransport uses pop() so it's LIFO order
        for i in (0..5).rev() {
            let (endpoint, msg) = transport.receive().await.expect("Failed to receive");
            assert_eq!(endpoint, format!("endpoint_{}", i));
            assert_eq!(msg.payload, format!("message_{}", i).as_bytes());
        }

        // Queue should be empty now
        assert!(transport.receive().await.is_err());
    }

    #[tokio::test]
    async fn test_stats_precision() {
        let transport = ZeroCopyTransport::new();

        // Test that stats maintain precision
        let initial_stats = transport.stats().await.unwrap();
        assert_eq!(initial_stats.average_latency_us, 0.1);
        assert_eq!(initial_stats.throughput_mbps, 40000.0);

        // Send many small messages
        for _ in 0..1000 {
            let msg = create_test_message(MessageType::AgentSpawn, b"x");
            transport.send("endpoint", msg).await.unwrap();
        }

        let final_stats = transport.stats().await.unwrap();
        assert_eq!(final_stats.messages_sent, 1000);
        assert_eq!(final_stats.bytes_sent, 1000);
        // Latency and throughput should remain constant (simulated values)
        assert_eq!(final_stats.average_latency_us, 0.1);
        assert_eq!(final_stats.throughput_mbps, 40000.0);
    }

    #[tokio::test]
    async fn test_buffer_name_collisions() {
        let transport = ZeroCopyTransport::new();

        // Allocate initial buffer
        transport
            .allocate_shared_buffer("collision_test", 1024)
            .expect("Failed to allocate initial buffer");

        // Store initial buffer content
        {
            let mut pools = transport.buffer_pools.lock().unwrap();
            // Modify buffer content to verify overwrite
            pools.get_mut("collision_test").unwrap()[0] = 42;
        }

        // Overwrite with same name but different size
        transport
            .allocate_shared_buffer("collision_test", 2048)
            .expect("Failed to overwrite buffer");

        // Verify buffer was completely replaced
        let pools = transport.buffer_pools.lock().unwrap();
        assert_eq!(pools["collision_test"].len(), 2048);
        assert_eq!(pools["collision_test"][0], 0); // Should be fresh allocation
    }

    #[tokio::test]
    async fn test_endpoint_name_variations() {
        let transport = ZeroCopyTransport::new();

        let endpoint_names = vec![
            "",  // Empty endpoint
            "a", // Single character
            "A", // Capital letter
            "1", // Number
            "_", // Underscore
            ".", // Dot
            "localhost",
            "127.0.0.1",
            "::1", // IPv6 loopback
            "[2001:db8::1]",
            "node-1.cluster.local",
            "user@host",
            "http://example.com",
            "very-long-endpoint-name-that-exceeds-typical-length-limits-and-tests-boundary-conditions",
        ];

        for endpoint in &endpoint_names {
            let msg = create_test_message(MessageType::AgentSpawn, b"test");
            transport
                .send(endpoint, msg)
                .await
                .expect(&format!("Failed to send to endpoint: {}", endpoint));
        }

        let stats = transport.stats().await.unwrap();
        assert_eq!(stats.messages_sent, endpoint_names.len() as u64);
    }

    #[tokio::test]
    async fn test_mixed_message_types_ordering() {
        let transport = ZeroCopyTransport::new();

        let message_types = vec![
            (MessageType::AgentSpawn, "spawn"),
            (MessageType::AgentTerminate, "terminate"),
            (MessageType::ResourceRequest, "request"),
            (MessageType::ResourceResponse, "response"),
            (MessageType::KnowledgeSync, "sync"),
        ];

        // Send messages with different types
        for (i, (msg_type, data)) in message_types.iter().enumerate() {
            let msg = create_test_message(msg_type.clone(), data.as_bytes());
            transport
                .send(&format!("endpoint_{}", i), msg)
                .await
                .unwrap();
        }

        // Verify order and types are preserved (LIFO order due to pop())
        for (i, (expected_type, expected_data)) in message_types.iter().enumerate().rev() {
            let (endpoint, msg) = transport.receive().await.unwrap();
            assert_eq!(endpoint, format!("endpoint_{}", i));
            assert_eq!(msg.msg_type, *expected_type);
            assert_eq!(msg.payload, expected_data.as_bytes());
        }
    }

    #[tokio::test]
    async fn test_transport_stress_test() {
        let transport = Arc::new(ZeroCopyTransport::new());
        let num_threads = 10;
        let messages_per_thread = 100;

        let mut handles = vec![];

        // Spawn multiple threads sending messages concurrently
        for thread_id in 0..num_threads {
            let transport_clone = transport.clone();
            let handle = tokio::spawn(async move {
                for msg_id in 0..messages_per_thread {
                    let payload = format!("thread_{}_msg_{}", thread_id, msg_id);
                    let msg = create_test_message(MessageType::KnowledgeSync, payload.as_bytes());
                    transport_clone
                        .send(&format!("endpoint_{}_{}", thread_id, msg_id), msg)
                        .await
                        .expect("Failed to send in stress test");
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.await.expect("Thread failed");
        }

        let stats = transport.stats().await.unwrap();
        assert_eq!(
            stats.messages_sent,
            (num_threads * messages_per_thread) as u64
        );

        // Verify we can receive all messages
        let mut received_count = 0;
        while transport.receive().await.is_ok() {
            received_count += 1;
        }
        assert_eq!(received_count, num_threads * messages_per_thread);
    }
}
