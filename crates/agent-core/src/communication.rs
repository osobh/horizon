//! Inter-agent communication infrastructure

use crate::agent::AgentId;
use crate::error::{AgentError, AgentResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

/// Global message bus instance
static MESSAGE_BUS: Lazy<Arc<MessageBus>> = Lazy::new(|| Arc::new(MessageBus::new()));

/// Message ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

impl MessageId {
    /// Create new message ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// Message types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Request message
    Request,
    /// Response message
    Response,
    /// Notification
    Notification,
    /// Broadcast
    Broadcast,
    /// Error
    Error,
}

/// Message priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Urgent priority
    Urgent = 3,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Inter-agent message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message ID
    pub id: MessageId,
    /// Message type
    pub msg_type: MessageType,
    /// Sender agent ID
    pub sender: AgentId,
    /// Recipient agent ID (None for broadcasts)
    pub recipient: Option<AgentId>,
    /// Message subject
    pub subject: String,
    /// Message body
    pub body: serde_json::Value,
    /// Priority
    pub priority: MessagePriority,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// In reply to message ID
    pub in_reply_to: Option<MessageId>,
    /// Message expiry time
    pub expires_at: Option<DateTime<Utc>>,
}

impl Message {
    /// Create a new message
    pub fn new(
        sender: AgentId,
        recipient: Option<AgentId>,
        subject: String,
        body: serde_json::Value,
    ) -> Self {
        Self {
            id: MessageId::new(),
            msg_type: MessageType::Request,
            sender,
            recipient,
            subject,
            body,
            priority: MessagePriority::Normal,
            timestamp: Utc::now(),
            in_reply_to: None,
            expires_at: None,
        }
    }

    /// Create a broadcast message
    pub fn broadcast(sender: AgentId, subject: String, body: serde_json::Value) -> Self {
        let mut msg = Self::new(sender, None, subject, body);
        msg.msg_type = MessageType::Broadcast;
        msg
    }

    /// Create a response message
    pub fn response_to(original: &Message, sender: AgentId, body: serde_json::Value) -> Self {
        Self {
            id: MessageId::new(),
            msg_type: MessageType::Response,
            sender,
            recipient: Some(original.sender),
            subject: format!("Re: {}", original.subject),
            body,
            priority: original.priority,
            timestamp: Utc::now(),
            in_reply_to: Some(original.id),
            expires_at: None,
        }
    }

    /// Check if message has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }
}

/// Agent communication channel
pub struct AgentChannel {
    /// Agent ID
    agent_id: AgentId,
    /// Receive channel
    receiver: Arc<RwLock<mpsc::Receiver<Message>>>,
    /// Send handle to message bus
    sender: mpsc::Sender<Message>,
}

impl AgentChannel {
    /// Create new agent channel
    fn new(
        agent_id: AgentId,
        receiver: mpsc::Receiver<Message>,
        sender: mpsc::Sender<Message>,
    ) -> Self {
        Self {
            agent_id,
            receiver: Arc::new(RwLock::new(receiver)),
            sender,
        }
    }

    /// Send a message
    pub async fn send(&self, mut message: Message) -> AgentResult<()> {
        // Ensure sender is set correctly
        message.sender = self.agent_id;

        self.sender
            .send(message)
            .await
            .map_err(|_| AgentError::CommunicationFailure {
                message: "Failed to send message".to_string(),
            })
    }

    /// Receive next message
    pub async fn receive(&self) -> Option<Message> {
        let mut receiver = self.receiver.write().await;
        receiver.recv().await
    }

    /// Try to receive without blocking
    pub async fn try_receive(&self) -> Option<Message> {
        let mut receiver = self.receiver.write().await;
        receiver.try_recv().ok()
    }
}

/// Message bus for inter-agent communication
pub struct MessageBus {
    /// Agent channels
    channels: DashMap<AgentId, mpsc::Sender<Message>>,
    /// Broadcast channel
    broadcast_sender: mpsc::Sender<Message>,
    /// Message history (optional)
    history: Arc<RwLock<Vec<Message>>>,
    /// Maximum history size
    max_history: usize,
}

impl MessageBus {
    /// Create new message bus
    pub fn new() -> Self {
        let (broadcast_sender, _) = mpsc::channel(1000);

        Self {
            channels: DashMap::new(),
            broadcast_sender,
            history: Arc::new(RwLock::new(Vec::new())),
            max_history: 10000,
        }
    }

    /// Register an agent
    pub async fn register(&self, agent_id: AgentId) -> AgentResult<AgentChannel> {
        // Check if already registered
        if self.channels.contains_key(&agent_id) {
            return Err(AgentError::AgentAlreadyExists {
                id: agent_id.to_string(),
            });
        }

        // Create channel
        let (tx, rx) = mpsc::channel(100);
        self.channels.insert(agent_id, tx.clone());

        // Create agent channel
        let channel = AgentChannel::new(agent_id, rx, self.broadcast_sender.clone());

        Ok(channel)
    }

    /// Unregister an agent
    pub async fn unregister(&self, agent_id: AgentId) -> AgentResult<()> {
        self.channels.remove(&agent_id);
        Ok(())
    }

    /// Route a message
    pub async fn route(&self, message: Message) -> AgentResult<()> {
        // Store in history
        self.store_message(&message).await;

        // Check if expired
        if message.is_expired() {
            return Ok(());
        }

        match &message.recipient {
            Some(recipient) => {
                // Direct message
                if let Some(channel) = self.channels.get(recipient) {
                    channel.send(message.clone()).await.map_err(|_| {
                        AgentError::CommunicationFailure {
                            message: format!("Failed to deliver message to agent {recipient}"),
                        }
                    })?;
                } else {
                    return Err(AgentError::AgentNotFound {
                        id: recipient.to_string(),
                    });
                }
            }
            None => {
                // Broadcast message
                for channel in self.channels.iter() {
                    // Skip sender
                    if *channel.key() != message.sender {
                        let _ = channel.value().send(message.clone()).await;
                    }
                }
            }
        }

        Ok(())
    }

    /// Store message in history
    async fn store_message(&self, message: &Message) {
        let mut history = self.history.write().await;
        history.push(message.clone());

        // Trim history if needed
        if history.len() > self.max_history {
            let drain_count = history.len() - self.max_history;
            history.drain(0..drain_count);
        }
    }

    /// Get message history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<Message> {
        let history = self.history.read().await;

        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.clone(),
        }
    }

    /// Get active agents
    pub fn active_agents(&self) -> Vec<AgentId> {
        self.channels.iter().map(|entry| *entry.key()).collect()
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize global message bus
pub async fn init_message_bus() -> AgentResult<()> {
    // Message bus is already initialized via Lazy
    Ok(())
}

/// Get global message bus instance
pub fn message_bus() -> Arc<MessageBus> {
    MESSAGE_BUS.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let sender = AgentId::new();
        let recipient = AgentId::new();

        let msg = Message::new(
            sender,
            Some(recipient),
            "test".to_string(),
            serde_json::json!({"data": "test"}),
        );

        assert_eq!(msg.sender, sender);
        assert_eq!(msg.recipient, Some(recipient));
        assert_eq!(msg.subject, "test");
        assert_eq!(msg.msg_type, MessageType::Request);
    }

    #[test]
    fn test_broadcast_message() {
        let sender = AgentId::new();

        let msg = Message::broadcast(
            sender,
            "announcement".to_string(),
            serde_json::json!({"info": "broadcast"}),
        );

        assert!(msg.recipient.is_none());
        assert_eq!(msg.msg_type, MessageType::Broadcast);
    }

    #[test]
    fn test_response_message() {
        let sender1 = AgentId::new();
        let sender2 = AgentId::new();

        let original = Message::new(
            sender1,
            Some(sender2),
            "request".to_string(),
            serde_json::json!({}),
        );

        let response =
            Message::response_to(&original, sender2, serde_json::json!({"status": "ok"}));

        assert_eq!(response.recipient, Some(sender1));
        assert_eq!(response.in_reply_to, Some(original.id));
        assert_eq!(response.subject, "Re: request");
    }

    #[tokio::test]
    async fn test_message_bus_registration() {
        let bus = MessageBus::new();
        let agent_id = AgentId::new();

        let channel = bus.register(agent_id).await.unwrap();
        assert_eq!(channel.agent_id, agent_id);

        // Try to register again
        let result = bus.register(agent_id).await;
        assert!(matches!(result, Err(AgentError::AgentAlreadyExists { .. })));

        // Unregister
        assert!(bus.unregister(agent_id).await.is_ok());
    }

    #[tokio::test]
    async fn test_direct_messaging() {
        let bus = MessageBus::new();

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let channel1 = bus.register(agent1).await.unwrap();
        let channel2 = bus.register(agent2).await.unwrap();

        // Send message from agent1 to agent2
        let msg = Message::new(
            agent1,
            Some(agent2),
            "hello".to_string(),
            serde_json::json!({"greeting": "hi"}),
        );

        bus.route(msg.clone()).await.unwrap();

        // Agent2 should receive the message
        let received = channel2.receive().await;
        assert!(received.is_some());

        let received_msg = received.unwrap();
        assert_eq!(received_msg.id, msg.id);
        assert_eq!(received_msg.sender, agent1);

        // Agent1 should not receive it
        assert!(channel1.try_receive().await.is_none());
    }

    #[tokio::test]
    async fn test_broadcast_messaging() {
        let bus = MessageBus::new();

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();

        let _channel1 = bus.register(agent1).await.unwrap();
        let channel2 = bus.register(agent2).await.unwrap();
        let channel3 = bus.register(agent3).await.unwrap();

        // Broadcast from agent1
        let msg = Message::broadcast(
            agent1,
            "announcement".to_string(),
            serde_json::json!({"info": "broadcast"}),
        );

        bus.route(msg).await.unwrap();

        // Both agent2 and agent3 should receive it
        assert!(channel2.receive().await.is_some());
        assert!(channel3.receive().await.is_some());
    }

    #[tokio::test]
    async fn test_message_history() {
        let bus = MessageBus::new();

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        bus.register(agent1).await.unwrap();
        bus.register(agent2).await.unwrap();

        // Send some messages
        for i in 0..5 {
            let msg = Message::new(
                agent1,
                Some(agent2),
                format!("msg_{i}"),
                serde_json::json!({}),
            );
            let _ = bus.route(msg).await; // Messages to unregistered agents will fail
        }

        // Check history
        let history = bus.get_history(Some(3)).await;
        assert_eq!(history.len(), 3);

        let all_history = bus.get_history(None).await;
        assert_eq!(all_history.len(), 5);
    }

    #[tokio::test]
    async fn test_active_agents() {
        let bus = MessageBus::new();

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        bus.register(agent1).await.unwrap();
        bus.register(agent2).await.unwrap();

        let active = bus.active_agents();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&agent1));
        assert!(active.contains(&agent2));

        bus.unregister(agent1).await.unwrap();

        let active = bus.active_agents();
        assert_eq!(active.len(), 1);
        assert!(!active.contains(&agent1));
    }

    #[test]
    fn test_message_creation() {
        let sender = AgentId::new();
        let recipient = AgentId::new();
        let msg_type = "test_type".to_string();
        let payload = serde_json::json!({"key": "value"});

        let msg = Message::new(sender, Some(recipient), msg_type.clone(), payload.clone());

        assert_eq!(msg.sender, sender);
        assert_eq!(msg.recipient, Some(recipient));
        assert_eq!(msg.msg_type, msg_type);
        assert_eq!(msg.payload, payload);
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_message_broadcast() {
        let sender = AgentId::new();
        let msg = Message::broadcast(
            sender,
            "broadcast_type".to_string(),
            serde_json::json!({"broadcast": true}),
        );

        assert_eq!(msg.sender, sender);
        assert!(msg.recipient.is_none());
        assert_eq!(msg.msg_type, "broadcast_type");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::new(
            AgentId::new(),
            Some(AgentId::new()),
            "test".to_string(),
            serde_json::json!({"data": [1, 2, 3]}),
        );

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();

        assert_eq!(msg.id, deserialized.id);
        assert_eq!(msg.sender, deserialized.sender);
        assert_eq!(msg.recipient, deserialized.recipient);
        assert_eq!(msg.msg_type, deserialized.msg_type);
        assert_eq!(msg.payload, deserialized.payload);
        assert_eq!(msg.timestamp, deserialized.timestamp);
    }

    #[test]
    fn test_message_type_constants() {
        assert_eq!(MessageType::GoalAssigned, "goal_assigned");
        assert_eq!(MessageType::GoalCompleted, "goal_completed");
        assert_eq!(MessageType::GoalFailed, "goal_failed");
        assert_eq!(MessageType::StatusUpdate, "status_update");
        assert_eq!(MessageType::ResourceRequest, "resource_request");
        assert_eq!(MessageType::ResourceResponse, "resource_response");
        assert_eq!(MessageType::Heartbeat, "heartbeat");
    }

    #[tokio::test]
    async fn test_agent_channel_try_receive() {
        let (sender, receiver) = mpsc::channel(10);
        let channel = AgentChannel { receiver };

        // No messages yet
        assert!(channel.try_receive().await.is_none());

        // Send a message
        let msg = Message::broadcast(AgentId::new(), "test".to_string(), serde_json::json!({}));
        sender.send(msg.clone()).await.unwrap();

        // Now should receive
        let received = channel.try_receive().await;
        assert!(received.is_some());
        assert_eq!(received.unwrap().id, msg.id);
    }

    #[tokio::test]
    async fn test_agent_channel_receive_timeout() {
        let (_sender, receiver) = mpsc::channel::<Message>(10);
        let channel = AgentChannel { receiver };

        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(100);
        let result = channel.receive_timeout(timeout).await;

        assert!(result.is_none());
        assert!(start.elapsed() >= timeout);
        assert!(start.elapsed() < timeout * 2); // Should not wait too long
    }

    #[tokio::test]
    async fn test_message_bus_double_registration() {
        let bus = MessageBus::new();
        let agent_id = AgentId::new();

        // First registration should succeed
        assert!(bus.register(agent_id).await.is_ok());

        // Second registration should fail
        let result = bus.register(agent_id).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::AgentAlreadyExists { .. }
        ));
    }

    #[tokio::test]
    async fn test_message_bus_unregister_nonexistent() {
        let bus = MessageBus::new();
        let agent_id = AgentId::new();

        // Unregistering non-existent agent should fail
        let result = bus.unregister(agent_id).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::AgentNotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_message_bus_route_to_nonexistent() {
        let bus = MessageBus::new();
        let sender = AgentId::new();
        let recipient = AgentId::new();

        bus.register(sender).await.unwrap();

        let msg = Message::new(
            sender,
            Some(recipient),
            "test".to_string(),
            serde_json::json!({}),
        );

        // Should fail to route to non-existent recipient
        let result = bus.route(msg).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentError::AgentNotFound { .. }
        ));
    }

    #[tokio::test]
    async fn test_message_bus_history_limit() {
        let mut bus = MessageBus::new();
        bus.max_history = 5; // Set small limit for testing

        let agent = AgentId::new();
        bus.register(agent).await.unwrap();

        // Send more messages than history limit
        for i in 0..10 {
            let msg = Message::broadcast(agent, format!("msg_{i}"), serde_json::json!({}));
            let _ = bus.route(msg).await;
        }

        // History should be limited
        let history = bus.get_history(None).await;
        assert_eq!(history.len(), 5);

        // Should have the last 5 messages
        assert_eq!(history[0].msg_type, "msg_5");
        assert_eq!(history[4].msg_type, "msg_9");
    }

    #[tokio::test]
    async fn test_concurrent_message_routing() {
        let bus = Arc::new(MessageBus::new());
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        let channel1 = bus.register(agent1).await.unwrap();
        let channel2 = bus.register(agent2).await.unwrap();

        let mut handles = vec![];

        // Send many messages concurrently
        for i in 0..100 {
            let bus_clone = bus.clone();
            let sender = if i % 2 == 0 { agent1 } else { agent2 };
            let recipient = if i % 2 == 0 { agent2 } else { agent1 };

            let handle = tokio::spawn(async move {
                let msg = Message::new(
                    sender,
                    Some(recipient),
                    format!("msg_{i}"),
                    serde_json::json!({"index": i}),
                );
                bus_clone.route(msg).await
            });
            handles.push(handle);
        }

        // Wait for all sends
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Both agents should receive 50 messages each
        let mut count1 = 0;
        let mut count2 = 0;

        while channel1.try_receive().await.is_some() {
            count1 += 1;
        }
        while channel2.try_receive().await.is_some() {
            count2 += 1;
        }

        assert_eq!(count1, 50);
        assert_eq!(count2, 50);
    }

    #[tokio::test]
    async fn test_broadcast_excludes_sender() {
        let bus = MessageBus::new();

        let agent1 = AgentId::new();
        let agent2 = AgentId::new();
        let agent3 = AgentId::new();

        let channel1 = bus.register(agent1).await.unwrap();
        let channel2 = bus.register(agent2).await.unwrap();
        let channel3 = bus.register(agent3).await.unwrap();

        // Agent1 broadcasts
        let msg = Message::broadcast(agent1, "announcement".to_string(), serde_json::json!({}));
        bus.route(msg).await.unwrap();

        // Agent1 should NOT receive its own broadcast
        assert!(channel1.try_receive().await.is_none());

        // Others should receive it
        assert!(channel2.try_receive().await.is_some());
        assert!(channel3.try_receive().await.is_some());
    }

    #[test]
    fn test_message_id_uniqueness() {
        use std::collections::HashSet;
        let mut ids = HashSet::new();

        for _ in 0..1000 {
            let msg = Message::new(
                AgentId::new(),
                None,
                "test".to_string(),
                serde_json::json!({}),
            );
            assert!(ids.insert(msg.id));
        }
    }

    #[test]
    fn test_message_timestamp() {
        let before = chrono::Utc::now().timestamp_millis() as u64;
        let msg = Message::new(
            AgentId::new(),
            None,
            "test".to_string(),
            serde_json::json!({}),
        );
        let after = chrono::Utc::now().timestamp_millis() as u64;

        assert!(msg.timestamp >= before);
        assert!(msg.timestamp <= after);
    }

    #[tokio::test]
    async fn test_channel_capacity() {
        let bus = MessageBus::new();
        let agent1 = AgentId::new();
        let agent2 = AgentId::new();

        bus.register(agent1).await.unwrap();
        let channel2 = bus.register(agent2).await.unwrap();

        // Send many messages quickly
        for i in 0..100 {
            let msg = Message::new(
                agent1,
                Some(agent2),
                format!("msg_{i}"),
                serde_json::json!({}),
            );
            bus.route(msg).await.unwrap();
        }

        // Should be able to receive all
        let mut count = 0;
        while channel2.try_receive().await.is_some() {
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_message_memory_efficiency() {
        use std::mem::size_of;

        // Ensure Message is reasonably sized
        assert!(size_of::<Message>() < 256); // Should be compact
        assert!(size_of::<MessageId>() == 16); // UUID size
    }

    #[tokio::test]
    async fn test_message_bus_clear_history() {
        let bus = MessageBus::new();
        let agent = AgentId::new();
        bus.register(agent).await.unwrap();

        // Add some messages
        for i in 0..5 {
            let msg = Message::broadcast(agent, format!("msg_{i}"), serde_json::json!({}));
            let _ = bus.route(msg).await;
        }

        assert_eq!(bus.get_history(None).await.len(), 5);

        // Clear history
        bus.clear_history().await;
        assert_eq!(bus.get_history(None).await.len(), 0);
    }

    #[test]
    fn test_message_payload_types() {
        let agent = AgentId::new();

        // Test various payload types
        let payloads = vec![
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!(42),
            serde_json::json!(3.14),
            serde_json::json!("string"),
            serde_json::json!([1, 2, 3]),
            serde_json::json!({"nested": {"key": "value"}}),
        ];

        for payload in payloads {
            let msg = Message::new(agent, None, "test".to_string(), payload.clone());
            assert_eq!(msg.payload, payload);
        }
    }
}
