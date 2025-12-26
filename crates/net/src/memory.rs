//! In-memory network implementation for development and testing

use std::sync::{Arc, Mutex};

use crate::{Message, Network, NetworkError, NetworkStats};

/// In-memory network implementation using channels
pub struct MemoryNetwork {
    endpoint: String,
    messages: Arc<Mutex<Vec<(String, Message)>>>, // (from_endpoint, message)
    sent_messages: Arc<Mutex<Vec<(String, Message)>>>, // (to_endpoint, message)
    stats: Arc<Mutex<NetworkStats>>,
}

impl MemoryNetwork {
    /// Create new in-memory network
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            messages: Arc::new(Mutex::new(Vec::new())),
            sent_messages: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                messages_sent: 0,
                messages_received: 0,
                average_latency_us: 0.0,
                throughput_mbps: 1000.0, // Mock high throughput for in-memory
            })),
        }
    }

    /// Get endpoint name
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Simulate receiving a message from another endpoint
    pub async fn simulate_receive(
        &self,
        from_endpoint: String,
        message: Message,
    ) -> Result<(), NetworkError> {
        let mut messages = self.messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire lock: {e}"
            )))
        })?;

        let mut stats = self.stats.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire stats lock: {e}"
            )))
        })?;

        let message_size = message.payload.len() as u64;
        stats.bytes_received += message_size;
        stats.messages_received += 1;

        messages.push((from_endpoint, message));
        Ok(())
    }

    /// Get all sent messages for testing
    pub fn get_sent_messages(&self) -> Result<Vec<(String, Message)>, NetworkError> {
        let sent = self.sent_messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire sent messages lock: {e}"
            )))
        })?;
        Ok(sent.clone())
    }

    /// Clear all messages (useful for testing)
    pub fn clear_messages(&self) -> Result<(), NetworkError> {
        let mut messages = self.messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire messages lock: {e}"
            )))
        })?;
        let mut sent = self.sent_messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire sent messages lock: {e}"
            )))
        })?;
        messages.clear();
        sent.clear();
        Ok(())
    }
}

#[async_trait::async_trait]
impl Network for MemoryNetwork {
    async fn send(&self, endpoint: &str, message: Message) -> Result<(), NetworkError> {
        let mut sent = self.sent_messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire sent messages lock: {e}"
            )))
        })?;

        let mut stats = self.stats.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire stats lock: {e}"
            )))
        })?;

        let message_size = message.payload.len() as u64;
        stats.bytes_sent += message_size;
        stats.messages_sent += 1;

        sent.push((endpoint.to_string(), message));
        Ok(())
    }

    async fn receive(&self) -> Result<(String, Message), NetworkError> {
        let mut messages = self.messages.lock().map_err(|e| {
            NetworkError::Io(std::io::Error::other(format!(
                "Failed to acquire messages lock: {e}"
            )))
        })?;

        if messages.is_empty() {
            return Err(NetworkError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "No messages available",
            )));
        }

        Ok(messages.remove(0))
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

    fn create_test_message(msg_type: MessageType, payload: &[u8]) -> Message {
        Message {
            id: 42, // Fixed ID for tests
            msg_type,
            payload: payload.to_vec(),
            timestamp: 1234567890, // Fixed timestamp for tests
        }
    }

    #[tokio::test]
    async fn test_memory_network_creation() {
        let network = MemoryNetwork::new("test_node".to_string());
        assert_eq!(network.endpoint(), "test_node");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.bytes_received, 0);
        assert_eq!(stats.messages_received, 0);
    }

    #[tokio::test]
    async fn test_send_message() {
        let network = MemoryNetwork::new("sender".to_string());
        let message = create_test_message(MessageType::AgentSpawn, b"spawn_data");

        network
            .send("receiver", message.clone())
            .await
            .expect("Failed to send message");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.bytes_sent, b"spawn_data".len() as u64);

        let sent_messages = network
            .get_sent_messages()
            .expect("Failed to get sent messages");
        assert_eq!(sent_messages.len(), 1);
        assert_eq!(sent_messages[0].0, "receiver");
        assert_eq!(sent_messages[0].1.payload, b"spawn_data");
    }

    #[tokio::test]
    async fn test_receive_message() {
        let network = MemoryNetwork::new("receiver".to_string());
        let message = create_test_message(MessageType::ResourceRequest, b"request_data");

        // No messages initially
        let result = network.receive().await;
        assert!(result.is_err());

        // Simulate receiving a message
        network
            .simulate_receive("sender".to_string(), message.clone())
            .await
            .expect("Failed to simulate receive");

        let (from, received_msg) = network.receive().await.expect("Failed to receive message");
        assert_eq!(from, "sender");
        assert_eq!(received_msg.payload, b"request_data");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_received, b"request_data".len() as u64);
    }

    #[tokio::test]
    async fn test_multiple_messages() {
        let network = MemoryNetwork::new("test_node".to_string());

        let msg1 = create_test_message(MessageType::AgentSpawn, b"data1");
        let msg2 = create_test_message(MessageType::AgentTerminate, b"data2");
        let msg3 = create_test_message(MessageType::KnowledgeSync, b"data3");

        // Send messages
        network
            .send("node1", msg1.clone())
            .await
            .expect("Failed to send msg1");
        network
            .send("node2", msg2.clone())
            .await
            .expect("Failed to send msg2");
        network
            .send("node3", msg3.clone())
            .await
            .expect("Failed to send msg3");

        // Receive messages
        network
            .simulate_receive("node1".to_string(), msg1.clone())
            .await
            .expect("Failed to simulate receive 1");
        network
            .simulate_receive("node2".to_string(), msg2.clone())
            .await
            .expect("Failed to simulate receive 2");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 3);
        assert_eq!(stats.messages_received, 2);

        let sent_messages = network
            .get_sent_messages()
            .expect("Failed to get sent messages");
        assert_eq!(sent_messages.len(), 3);
    }

    #[tokio::test]
    async fn test_message_types() {
        let network = MemoryNetwork::new("test_node".to_string());

        let message_types = vec![
            MessageType::AgentSpawn,
            MessageType::AgentTerminate,
            MessageType::ResourceRequest,
            MessageType::ResourceResponse,
            MessageType::KnowledgeSync,
        ];

        for (i, msg_type) in message_types.iter().enumerate() {
            let message = create_test_message(msg_type.clone(), format!("data{i}").as_bytes());
            network
                .send(&format!("node{i}"), message.clone())
                .await
                .expect("Failed to send message");
        }

        let sent_messages = network
            .get_sent_messages()
            .expect("Failed to get sent messages");
        assert_eq!(sent_messages.len(), 5);

        // Verify each message type was sent
        for (i, (endpoint, msg)) in sent_messages.iter().enumerate() {
            assert_eq!(*endpoint, format!("node{i}"));
            assert_eq!(msg.payload, format!("data{i}").as_bytes());
        }
    }

    #[tokio::test]
    async fn test_clear_messages() {
        let network = MemoryNetwork::new("test_node".to_string());

        let message = create_test_message(MessageType::AgentSpawn, b"test_data");
        network
            .send("endpoint", message.clone())
            .await
            .expect("Failed to send");
        network
            .simulate_receive("sender".to_string(), message)
            .await
            .expect("Failed to simulate receive");

        // Verify messages exist
        let sent = network
            .get_sent_messages()
            .expect("Failed to get sent messages");
        assert_eq!(sent.len(), 1);

        let (_, _) = network.receive().await.expect("Failed to receive");

        // Clear and verify
        network.clear_messages().expect("Failed to clear messages");
        let sent_after = network
            .get_sent_messages()
            .expect("Failed to get sent messages after clear");
        assert_eq!(sent_after.len(), 0);

        let receive_result = network.receive().await;
        assert!(receive_result.is_err());
    }

    #[tokio::test]
    async fn test_network_stats_accumulation() {
        let network = MemoryNetwork::new("stats_test".to_string());

        let large_payload = vec![0u8; 1024];
        let small_payload = vec![1u8; 100];

        let large_msg = create_test_message(MessageType::KnowledgeSync, &large_payload);
        let small_msg = create_test_message(MessageType::ResourceRequest, &small_payload);

        network
            .send("node1", large_msg.clone())
            .await
            .expect("Failed to send large message");
        network
            .send("node2", small_msg.clone())
            .await
            .expect("Failed to send small message");

        network
            .simulate_receive("node3".to_string(), large_msg)
            .await
            .expect("Failed to simulate receive large");
        network
            .simulate_receive("node4".to_string(), small_msg)
            .await
            .expect("Failed to simulate receive small");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.messages_sent, 2);
        assert_eq!(stats.messages_received, 2);
        assert_eq!(stats.bytes_sent, 1024 + 100);
        assert_eq!(stats.bytes_received, 1024 + 100);
        assert_eq!(stats.throughput_mbps, 1000.0); // Mock value
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let network = Arc::new(MemoryNetwork::new("concurrent_test".to_string()));
        let network_clone1 = network.clone();
        let network_clone2 = network.clone();

        let send_task = tokio::spawn(async move {
            for i in 0..10 {
                let message =
                    create_test_message(MessageType::AgentSpawn, format!("data{i}").as_bytes());
                network_clone1
                    .send(&format!("node{i}"), message)
                    .await
                    .expect("Failed to send in concurrent test");
            }
        });

        let receive_task = tokio::spawn(async move {
            for i in 0..5 {
                let message = create_test_message(
                    MessageType::ResourceResponse,
                    format!("response{i}").as_bytes(),
                );
                network_clone2
                    .simulate_receive(format!("sender{i}"), message)
                    .await
                    .expect("Failed to simulate receive in concurrent test");
            }
        });

        let (send_result, receive_result) = tokio::join!(send_task, receive_task);
        send_result.expect("Send task failed");
        receive_result.expect("Receive task failed");

        let final_stats = network.stats().await.expect("Failed to get final stats");
        assert_eq!(final_stats.messages_sent, 10);
        assert_eq!(final_stats.messages_received, 5);
    }

    #[tokio::test]
    async fn test_empty_receive_queue() {
        let network = MemoryNetwork::new("empty_test".to_string());

        // Try to receive from empty queue
        let result = network.receive().await;
        assert!(
            matches!(result, Err(NetworkError::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock)
        );
    }

    #[tokio::test]
    async fn test_network_edge_cases() {
        let network = MemoryNetwork::new("edge_test".to_string());

        // Test with empty payload
        let empty_msg = create_test_message(MessageType::AgentSpawn, &[]);
        network
            .send("endpoint", empty_msg.clone())
            .await
            .expect("Failed to send empty message");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.messages_sent, 1);

        // Test receiving empty message
        network
            .simulate_receive("sender".to_string(), empty_msg)
            .await
            .expect("Failed to simulate empty receive");
        let (from, received) = network
            .receive()
            .await
            .expect("Failed to receive empty message");
        assert_eq!(from, "sender");
        assert!(received.payload.is_empty());

        let stats_after = network.stats().await.expect("Failed to get stats after");
        assert_eq!(stats_after.bytes_received, 0);
        assert_eq!(stats_after.messages_received, 1);
    }

    #[tokio::test]
    async fn test_large_messages() {
        let network = MemoryNetwork::new("large_test".to_string());

        // Create a large message (1MB)
        let large_payload = vec![0u8; 1024 * 1024];
        let large_msg = create_test_message(MessageType::KnowledgeSync, &large_payload);

        network
            .send("large_endpoint", large_msg.clone())
            .await
            .expect("Failed to send large message");

        let stats = network.stats().await.expect("Failed to get stats");
        assert_eq!(stats.bytes_sent, 1024 * 1024);
        assert_eq!(stats.messages_sent, 1);

        // Test receiving large message
        network
            .simulate_receive("large_sender".to_string(), large_msg)
            .await
            .expect("Failed to simulate large receive");
        let (from, received) = network
            .receive()
            .await
            .expect("Failed to receive large message");
        assert_eq!(from, "large_sender");
        assert_eq!(received.payload.len(), 1024 * 1024);

        let stats_after = network.stats().await.expect("Failed to get stats after");
        assert_eq!(stats_after.bytes_received, 1024 * 1024);
        assert_eq!(stats_after.messages_received, 1);
    }

    #[tokio::test]
    async fn test_message_ordering() {
        let network = MemoryNetwork::new("ordering_test".to_string());

        let msg1 = create_test_message(MessageType::AgentSpawn, b"first");
        let msg2 = create_test_message(MessageType::AgentTerminate, b"second");
        let msg3 = create_test_message(MessageType::ResourceRequest, b"third");

        // Send messages in order
        network
            .simulate_receive("sender1".to_string(), msg1.clone())
            .await
            .expect("Failed to send msg1");
        network
            .simulate_receive("sender2".to_string(), msg2.clone())
            .await
            .expect("Failed to send msg2");
        network
            .simulate_receive("sender3".to_string(), msg3.clone())
            .await
            .expect("Failed to send msg3");

        // Receive messages - should be in FIFO order
        let (from1, received1) = network.receive().await.expect("Failed to receive msg1");
        assert_eq!(from1, "sender1");
        assert_eq!(received1.payload, b"first");

        let (from2, received2) = network.receive().await.expect("Failed to receive msg2");
        assert_eq!(from2, "sender2");
        assert_eq!(received2.payload, b"second");

        let (from3, received3) = network.receive().await.expect("Failed to receive msg3");
        assert_eq!(from3, "sender3");
        assert_eq!(received3.payload, b"third");

        // Queue should now be empty
        let result = network.receive().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clear_mixed_operations() {
        let network = MemoryNetwork::new("clear_mixed_test".to_string());

        // Send some messages
        let msg1 = create_test_message(MessageType::AgentSpawn, b"send1");
        let msg2 = create_test_message(MessageType::AgentSpawn, b"send2");
        network
            .send("endpoint1", msg1)
            .await
            .expect("Failed to send msg1");
        network
            .send("endpoint2", msg2)
            .await
            .expect("Failed to send msg2");

        // Receive some messages
        let msg3 = create_test_message(MessageType::ResourceResponse, b"receive1");
        let msg4 = create_test_message(MessageType::ResourceResponse, b"receive2");
        network
            .simulate_receive("sender1".to_string(), msg3)
            .await
            .expect("Failed to simulate receive msg3");
        network
            .simulate_receive("sender2".to_string(), msg4)
            .await
            .expect("Failed to simulate receive msg4");

        // Verify messages exist
        let sent_msgs = network
            .get_sent_messages()
            .expect("Failed to get sent messages");
        assert_eq!(sent_msgs.len(), 2);

        let (_, _) = network.receive().await.expect("Failed to receive first");
        let (_, _) = network.receive().await.expect("Failed to receive second");

        // Clear everything
        network.clear_messages().expect("Failed to clear messages");

        // Verify everything is cleared
        let sent_after_clear = network
            .get_sent_messages()
            .expect("Failed to get sent messages after clear");
        assert_eq!(sent_after_clear.len(), 0);

        let receive_result = network.receive().await;
        assert!(receive_result.is_err());
    }

    #[tokio::test]
    async fn test_mutex_poisoning_send_messages() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_sent_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = network.send("endpoint", message).await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e
                    .to_string()
                    .contains("Failed to acquire sent messages lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_send_stats() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_stats_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: Arc::new(Mutex::new(Vec::new())),
            sent_messages: Arc::new(Mutex::new(Vec::new())),
            stats: poisoned.stats,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = network.send("endpoint", message).await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_receive() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_messages_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let result = network.receive().await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire messages lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_stats() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_stats_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let result = network.stats().await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_simulate_receive_messages() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_messages_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = network
            .simulate_receive("sender".to_string(), message)
            .await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_simulate_receive_stats() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_stats_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: Arc::new(Mutex::new(Vec::new())),
            sent_messages: Arc::new(Mutex::new(Vec::new())),
            stats: poisoned.stats,
        };

        let message = create_test_message(MessageType::AgentSpawn, b"test");
        let result = network
            .simulate_receive("sender".to_string(), message)
            .await;
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire stats lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_get_sent_messages() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_sent_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let result = network.get_sent_messages();
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e
                    .to_string()
                    .contains("Failed to acquire sent messages lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_clear_messages() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_messages_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: poisoned.messages,
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let result = network.clear_messages();
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e.to_string().contains("Failed to acquire messages lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }

    #[tokio::test]
    async fn test_mutex_poisoning_clear_sent() {
        use crate::test_helpers::tests::PoisonedMemoryNetwork;

        let poisoned = PoisonedMemoryNetwork::new_sent_poisoned();
        let network = MemoryNetwork {
            endpoint: "test".to_string(),
            messages: Arc::new(Mutex::new(Vec::new())),
            sent_messages: poisoned.sent_messages,
            stats: poisoned.stats,
        };

        let result = network.clear_messages();
        assert!(result.is_err());

        match result {
            Err(NetworkError::Io(e)) => {
                assert!(e
                    .to_string()
                    .contains("Failed to acquire sent messages lock"));
            }
            _ => panic!("Expected IO error with lock failure"),
        }
    }
}
