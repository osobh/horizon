//! Network protocol definitions

use serde::{Deserialize, Serialize};

/// Message types for inter-node communication
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    AgentSpawn,
    AgentTerminate,
    ResourceRequest,
    ResourceResponse,
    KnowledgeSync,
    Data,
    Control,
    Handshake,
    Heartbeat,
    Error,
    Close,
}

/// Network message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: u64,
    pub msg_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

impl Message {
    /// Create a new message
    pub fn new(msg_type: MessageType, payload: Vec<u8>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            id: rand::random(),
            msg_type,
            payload,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Get message size in bytes
    pub fn size(&self) -> usize {
        // Approximate size: id (8) + type (1) + timestamp (8) + payload length (8) + payload
        8 + 1 + 8 + 8 + self.payload.len()
    }

    /// Serialize message to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize message from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

/// Protocol-specific message builders
pub mod builders {
    use super::*;

    /// Build an agent spawn message
    pub fn agent_spawn(agent_id: &str, config: &[u8]) -> Message {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(agent_id.len() as u32).to_le_bytes());
        payload.extend_from_slice(agent_id.as_bytes());
        payload.extend_from_slice(config);

        Message::new(MessageType::AgentSpawn, payload)
    }

    /// Build an agent terminate message
    pub fn agent_terminate(agent_id: &str, reason: &str) -> Message {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(agent_id.len() as u32).to_le_bytes());
        payload.extend_from_slice(agent_id.as_bytes());
        payload.extend_from_slice(&(reason.len() as u32).to_le_bytes());
        payload.extend_from_slice(reason.as_bytes());

        Message::new(MessageType::AgentTerminate, payload)
    }

    /// Build a resource request message
    pub fn resource_request(resource_type: &str, amount: u64) -> Message {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(resource_type.len() as u32).to_le_bytes());
        payload.extend_from_slice(resource_type.as_bytes());
        payload.extend_from_slice(&amount.to_le_bytes());

        Message::new(MessageType::ResourceRequest, payload)
    }

    /// Build a resource response message
    pub fn resource_response(granted: bool, details: &[u8]) -> Message {
        let mut payload = Vec::new();
        payload.push(if granted { 1 } else { 0 });
        payload.extend_from_slice(details);

        Message::new(MessageType::ResourceResponse, payload)
    }

    /// Build a knowledge sync message
    pub fn knowledge_sync(node_count: u32, edge_count: u32, data: &[u8]) -> Message {
        let mut payload = Vec::new();
        payload.extend_from_slice(&node_count.to_le_bytes());
        payload.extend_from_slice(&edge_count.to_le_bytes());
        payload.extend_from_slice(data);

        Message::new(MessageType::KnowledgeSync, payload)
    }
}

/// Protocol parsers for message payloads
pub mod parsers {

    /// Parse agent spawn message
    pub fn parse_agent_spawn(payload: &[u8]) -> Option<(String, Vec<u8>)> {
        if payload.len() < 4 {
            return None;
        }

        let agent_id_len = u32::from_le_bytes(payload[0..4].try_into().ok()?) as usize;
        if payload.len() < 4 + agent_id_len {
            return None;
        }

        let agent_id = String::from_utf8(payload[4..4 + agent_id_len].to_vec()).ok()?;
        let config = payload[4 + agent_id_len..].to_vec();

        Some((agent_id, config))
    }

    /// Parse agent terminate message
    pub fn parse_agent_terminate(payload: &[u8]) -> Option<(String, String)> {
        if payload.len() < 4 {
            return None;
        }

        let agent_id_len = u32::from_le_bytes(payload[0..4].try_into().ok()?) as usize;
        if payload.len() < 4 + agent_id_len + 4 {
            return None;
        }

        let agent_id = String::from_utf8(payload[4..4 + agent_id_len].to_vec()).ok()?;

        let reason_start = 4 + agent_id_len;
        let reason_len =
            u32::from_le_bytes(payload[reason_start..reason_start + 4].try_into().ok()?) as usize;

        if payload.len() < reason_start + 4 + reason_len {
            return None;
        }

        let reason =
            String::from_utf8(payload[reason_start + 4..reason_start + 4 + reason_len].to_vec())
                .ok()?;

        Some((agent_id, reason))
    }

    /// Parse resource request message
    pub fn parse_resource_request(payload: &[u8]) -> Option<(String, u64)> {
        if payload.len() < 4 {
            return None;
        }

        let type_len = u32::from_le_bytes(payload[0..4].try_into().ok()?) as usize;
        if payload.len() < 4 + type_len + 8 {
            return None;
        }

        let resource_type = String::from_utf8(payload[4..4 + type_len].to_vec()).ok()?;
        let amount = u64::from_le_bytes(payload[4 + type_len..4 + type_len + 8].try_into().ok()?);

        Some((resource_type, amount))
    }

    /// Parse resource response message
    pub fn parse_resource_response(payload: &[u8]) -> Option<(bool, Vec<u8>)> {
        if payload.is_empty() {
            return None;
        }

        let granted = payload[0] != 0;
        let details = payload[1..].to_vec();

        Some((granted, details))
    }

    /// Parse knowledge sync message
    pub fn parse_knowledge_sync(payload: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
        if payload.len() < 8 {
            return None;
        }

        let node_count = u32::from_le_bytes(payload[0..4].try_into().ok()?);
        let edge_count = u32::from_le_bytes(payload[4..8].try_into().ok()?);
        let data = payload[8..].to_vec();

        Some((node_count, edge_count, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new(MessageType::AgentSpawn, vec![1, 2, 3]);

        assert_eq!(msg.msg_type, MessageType::AgentSpawn);
        assert_eq!(msg.payload, vec![1, 2, 3]);
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_message_size() {
        let msg = Message::new(MessageType::ResourceRequest, vec![0; 100]);
        assert_eq!(msg.size(), 8 + 1 + 8 + 8 + 100); // 125 bytes
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::new(MessageType::KnowledgeSync, vec![42; 10]);

        let bytes = msg.to_bytes().unwrap();
        let deserialized = Message::from_bytes(&bytes).unwrap();

        assert_eq!(msg.msg_type, deserialized.msg_type);
        assert_eq!(msg.payload, deserialized.payload);
        assert_eq!(msg.timestamp, deserialized.timestamp);
    }

    #[test]
    fn test_agent_spawn_builder_and_parser() {
        let msg = builders::agent_spawn("agent_123", b"config_data");

        assert_eq!(msg.msg_type, MessageType::AgentSpawn);

        let (agent_id, config) = parsers::parse_agent_spawn(&msg.payload).unwrap();
        assert_eq!(agent_id, "agent_123");
        assert_eq!(config, b"config_data");
    }

    #[test]
    fn test_agent_terminate_builder_and_parser() {
        let msg = builders::agent_terminate("agent_456", "out of memory");

        assert_eq!(msg.msg_type, MessageType::AgentTerminate);

        let (agent_id, reason) = parsers::parse_agent_terminate(&msg.payload).unwrap();
        assert_eq!(agent_id, "agent_456");
        assert_eq!(reason, "out of memory");
    }

    #[test]
    fn test_resource_request_builder_and_parser() {
        let msg = builders::resource_request("gpu_memory", 1024 * 1024 * 1024);

        assert_eq!(msg.msg_type, MessageType::ResourceRequest);

        let (resource_type, amount) = parsers::parse_resource_request(&msg.payload).unwrap();
        assert_eq!(resource_type, "gpu_memory");
        assert_eq!(amount, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_resource_response_builder_and_parser() {
        let msg = builders::resource_response(true, b"allocated on device 0");

        assert_eq!(msg.msg_type, MessageType::ResourceResponse);

        let (granted, details) = parsers::parse_resource_response(&msg.payload).unwrap();
        assert!(granted);
        assert_eq!(details, b"allocated on device 0");
    }

    #[test]
    fn test_knowledge_sync_builder_and_parser() {
        let msg = builders::knowledge_sync(1000, 5000, b"graph_data");

        assert_eq!(msg.msg_type, MessageType::KnowledgeSync);

        let (nodes, edges, data) = parsers::parse_knowledge_sync(&msg.payload).unwrap();
        assert_eq!(nodes, 1000);
        assert_eq!(edges, 5000);
        assert_eq!(data, b"graph_data");
    }

    #[test]
    fn test_parser_edge_cases() {
        // Empty payloads
        assert!(parsers::parse_agent_spawn(&[]).is_none());
        assert!(parsers::parse_agent_terminate(&[]).is_none());
        assert!(parsers::parse_resource_request(&[]).is_none());
        assert!(parsers::parse_resource_response(&[]).is_none());
        assert!(parsers::parse_knowledge_sync(&[]).is_none());

        // Truncated payloads
        assert!(parsers::parse_agent_spawn(&[1, 2, 3]).is_none());
        assert!(parsers::parse_knowledge_sync(&[1, 2, 3, 4, 5, 6, 7]).is_none());
    }

    #[test]
    fn test_empty_strings_in_messages() {
        let msg = builders::agent_spawn("", b"");
        let (agent_id, config) = parsers::parse_agent_spawn(&msg.payload).unwrap();
        assert_eq!(agent_id, "");
        assert_eq!(config, b"");

        let msg = builders::agent_terminate("", "");
        let (agent_id, reason) = parsers::parse_agent_terminate(&msg.payload).unwrap();
        assert_eq!(agent_id, "");
        assert_eq!(reason, "");
    }

    #[test]
    fn test_large_payloads() {
        let large_config = vec![0u8; 1024 * 1024]; // 1MB
        let msg = builders::agent_spawn("large_agent", &large_config);

        let (agent_id, config) = parsers::parse_agent_spawn(&msg.payload).unwrap();
        assert_eq!(agent_id, "large_agent");
        assert_eq!(config.len(), 1024 * 1024);
    }

    #[test]
    fn test_parse_agent_spawn_exact_length_mismatch() {
        // Test where payload indicates agent_id is longer than available data
        let mut payload = vec![];
        payload.extend_from_slice(&100u32.to_le_bytes()); // agent_id length = 100
        payload.extend_from_slice(b"short_id"); // but only 8 bytes
        assert!(parsers::parse_agent_spawn(&payload).is_none());
    }

    #[test]
    fn test_parse_agent_terminate_missing_reason_data() {
        // Test with agent_id but missing reason length
        let mut payload = vec![];
        payload.extend_from_slice(&5u32.to_le_bytes()); // agent_id length
        payload.extend_from_slice(b"agent"); // agent_id
        payload.extend_from_slice(&10u32.to_le_bytes()); // reason length = 10
        payload.extend_from_slice(b"short"); // but only 5 bytes for reason
        assert!(parsers::parse_agent_terminate(&payload).is_none());
    }

    #[test]
    fn test_parse_resource_response_truncated() {
        // Test with empty payload - should fail
        let payload = vec![];
        assert!(parsers::parse_resource_response(&payload).is_none());
    }

    #[test]
    fn test_message_type_equality() {
        assert_eq!(MessageType::AgentSpawn, MessageType::AgentSpawn);
        assert_ne!(MessageType::AgentSpawn, MessageType::AgentTerminate);
        assert_ne!(MessageType::ResourceRequest, MessageType::ResourceResponse);
    }

    #[test]
    fn test_message_with_zero_timestamp() {
        let mut msg = Message::new(MessageType::KnowledgeSync, vec![]);
        msg.timestamp = 0;
        assert_eq!(msg.timestamp, 0);

        let bytes = msg.to_bytes().unwrap();
        let deserialized = Message::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.timestamp, 0);
    }

    #[test]
    fn test_message_with_max_timestamp() {
        let mut msg = Message::new(MessageType::AgentSpawn, vec![1, 2, 3]);
        msg.timestamp = u64::MAX;

        let bytes = msg.to_bytes().unwrap();
        let deserialized = Message::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.timestamp, u64::MAX);
    }

    #[test]
    fn test_parse_invalid_utf8_strings() {
        // Invalid UTF-8 in agent_id
        let mut payload = vec![];
        payload.extend_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]); // Invalid UTF-8
        payload.extend_from_slice(b"config");

        assert!(parsers::parse_agent_spawn(&payload).is_none());
    }

    #[test]
    fn test_resource_request_zero_amount() {
        let msg = builders::resource_request("cpu_cores", 0);
        let (resource_type, amount) = parsers::parse_resource_request(&msg.payload).unwrap();
        assert_eq!(resource_type, "cpu_cores");
        assert_eq!(amount, 0);
    }

    #[test]
    fn test_resource_request_max_amount() {
        let msg = builders::resource_request("memory_bytes", u64::MAX);
        let (resource_type, amount) = parsers::parse_resource_request(&msg.payload).unwrap();
        assert_eq!(resource_type, "memory_bytes");
        assert_eq!(amount, u64::MAX);
    }

    #[test]
    fn test_knowledge_sync_zero_counts() {
        let msg = builders::knowledge_sync(0, 0, b"empty_graph");
        let (nodes, edges, data) = parsers::parse_knowledge_sync(&msg.payload).unwrap();
        assert_eq!(nodes, 0);
        assert_eq!(edges, 0);
        assert_eq!(data, b"empty_graph");
    }

    #[test]
    fn test_knowledge_sync_max_counts() {
        let msg = builders::knowledge_sync(u32::MAX, u32::MAX, b"huge_graph");
        let (nodes, edges, data) = parsers::parse_knowledge_sync(&msg.payload).unwrap();
        assert_eq!(nodes, u32::MAX);
        assert_eq!(edges, u32::MAX);
        assert_eq!(data, b"huge_graph");
    }

    #[test]
    fn test_resource_response_false_with_details() {
        let msg = builders::resource_response(false, b"insufficient memory");
        let (granted, details) = parsers::parse_resource_response(&msg.payload).unwrap();
        assert!(!granted);
        assert_eq!(details, b"insufficient memory");
    }

    #[test]
    fn test_message_serialization_error_handling() {
        let msg = Message::new(MessageType::AgentSpawn, vec![0; 1000]);

        // Deserialize from truncated bytes should fail
        let bytes = msg.to_bytes().unwrap();
        let truncated = &bytes[..bytes.len() / 2];
        assert!(Message::from_bytes(truncated).is_err());
    }

    #[test]
    fn test_builders_with_unicode_strings() {
        let msg = builders::agent_spawn("代理-エージェント-🤖", b"config");
        let (agent_id, _config) = parsers::parse_agent_spawn(&msg.payload).unwrap();
        assert_eq!(agent_id, "代理-エージェント-🤖");

        let msg = builders::agent_terminate("测试代理", "内存不足");
        let (agent_id, reason) = parsers::parse_agent_terminate(&msg.payload).unwrap();
        assert_eq!(agent_id, "测试代理");
        assert_eq!(reason, "内存不足");
    }

    #[test]
    fn test_message_size_calculation_edge_cases() {
        let msg = Message::new(MessageType::KnowledgeSync, vec![]);
        assert_eq!(msg.size(), 25); // 8 + 1 + 8 + 8 + 0

        let msg = Message::new(MessageType::ResourceRequest, vec![0; 1000]);
        assert_eq!(msg.size(), 1025); // 8 + 1 + 8 + 8 + 1000
    }

    #[test]
    fn test_parser_boundary_conditions() {
        // Test exact boundary conditions for parsers
        let mut payload = vec![];
        payload.extend_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(b"test");
        payload.extend_from_slice(&8u64.to_le_bytes());

        // This should parse successfully
        let result = parsers::parse_resource_request(&payload);
        assert!(result.is_some());
        let (resource_type, amount) = result.unwrap();
        assert_eq!(resource_type, "test");
        assert_eq!(amount, 8);

        // Remove one byte and it should fail
        payload.pop();
        assert!(parsers::parse_resource_request(&payload).is_none());
    }

    #[test]
    fn test_message_clone_and_debug() {
        let msg = Message {
            id: 42,
            msg_type: MessageType::KnowledgeSync,
            payload: vec![1, 2, 3, 4, 5],
            timestamp: 123456789,
        };

        let cloned = msg.clone();
        assert_eq!(cloned.msg_type, msg.msg_type);
        assert_eq!(cloned.payload, msg.payload);
        assert_eq!(cloned.timestamp, msg.timestamp);

        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("KnowledgeSync"));
        assert!(debug_str.contains("123456789"));
    }

    #[test]
    fn test_message_type_serialization() {
        let types = vec![
            MessageType::AgentSpawn,
            MessageType::AgentTerminate,
            MessageType::ResourceRequest,
            MessageType::ResourceResponse,
            MessageType::KnowledgeSync,
        ];

        for msg_type in types {
            let json = serde_json::to_string(&msg_type).unwrap();
            let deserialized: MessageType = serde_json::from_str(&json).unwrap();
            assert_eq!(msg_type, deserialized);
        }
    }

    #[test]
    fn test_concurrent_message_creation() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        for i in 0..10 {
            let messages_clone = messages.clone();
            let handle = thread::spawn(move || {
                let msg = Message::new(MessageType::AgentSpawn, vec![i as u8; 10]);
                messages_clone.lock().unwrap().push((i, msg));
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let mut messages = messages.lock().unwrap();
        assert_eq!(messages.len(), 10);

        // Sort by index to ensure consistent ordering
        messages.sort_by_key(|(i, _)| *i);

        // All messages should have different payloads based on thread index
        for (idx, (i, msg)) in messages.iter().enumerate() {
            assert_eq!(*i, idx);
            assert_eq!(msg.payload.len(), 10);
            assert_eq!(msg.payload[0], idx as u8);
        }
    }

    #[test]
    fn test_parse_agent_terminate_no_reason_length() {
        // Test missing reason length bytes
        let mut payload = vec![];
        payload.extend_from_slice(&5u32.to_le_bytes());
        payload.extend_from_slice(b"agent");
        // No reason length bytes
        assert!(parsers::parse_agent_terminate(&payload).is_none());
    }

    #[test]
    fn test_parse_resource_request_partial_amount() {
        // Test with incomplete amount bytes
        let mut payload = vec![];
        payload.extend_from_slice(&3u32.to_le_bytes());
        payload.extend_from_slice(b"cpu");
        payload.extend_from_slice(&[1, 2, 3, 4]); // Only 4 bytes instead of 8
        assert!(parsers::parse_resource_request(&payload).is_none());
    }
}
