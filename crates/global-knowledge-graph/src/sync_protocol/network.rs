//! Network communication for synchronization protocol

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

use crate::sync_protocol::types::KnowledgeOperation;
use crate::sync_protocol::consensus::{ConsensusProposal, ConsensusVote};

/// Message types for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMessage {
    pub message_id: Uuid,
    pub sender: String,
    pub recipient: String,
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Heartbeat,
    OperationBroadcast(KnowledgeOperation),
    ConsensusProposal(ConsensusProposal),
    ConsensusVote(ConsensusVote),
    SyncRequest,
    SyncResponse,
    Acknowledgment,
}

/// Message queue for handling network messages
pub struct MessageQueue {
    incoming: tokio::sync::RwLock<VecDeque<SyncMessage>>,
    outgoing: tokio::sync::RwLock<VecDeque<SyncMessage>>,
    max_queue_size: usize,
}

impl MessageQueue {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            incoming: tokio::sync::RwLock::new(VecDeque::new()),
            outgoing: tokio::sync::RwLock::new(VecDeque::new()),
            max_queue_size,
        }
    }

    pub async fn enqueue_incoming(&self, message: SyncMessage) -> Result<(), String> {
        let mut queue = self.incoming.write().await;
        if queue.len() >= self.max_queue_size {
            return Err("Incoming queue full".to_string());
        }
        queue.push_back(message);
        Ok(())
    }

    pub async fn dequeue_incoming(&self) -> Option<SyncMessage> {
        let mut queue = self.incoming.write().await;
        queue.pop_front()
    }

    pub async fn enqueue_outgoing(&self, message: SyncMessage) -> Result<(), String> {
        let mut queue = self.outgoing.write().await;
        if queue.len() >= self.max_queue_size {
            return Err("Outgoing queue full".to_string());
        }
        queue.push_back(message);
        Ok(())
    }

    pub async fn dequeue_outgoing(&self) -> Option<SyncMessage> {
        let mut queue = self.outgoing.write().await;
        queue.pop_front()
    }

    pub async fn incoming_size(&self) -> usize {
        self.incoming.read().await.len()
    }

    pub async fn outgoing_size(&self) -> usize {
        self.outgoing.read().await.len()
    }
}