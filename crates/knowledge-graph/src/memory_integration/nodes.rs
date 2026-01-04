//! Memory node types for knowledge graph integration

use crate::semantic::EmbeddingVector;
use serde::{Deserialize, Serialize};
use stratoswarm_agent_core::{memory::MemoryType, AgentId};

/// Memory node in knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Agent ID that owns this memory
    pub agent_id: AgentId,
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory content
    pub content: serde_json::Value,
    /// Memory metadata
    pub metadata: MemoryMetadata,
    /// Semantic embedding
    pub embedding: Option<EmbeddingVector>,
}

/// Memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Memory importance score
    pub importance: f64,
    /// Access frequency
    pub access_count: u64,
    /// Last access timestamp
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata fields
    pub custom_fields: std::collections::HashMap<String, serde_json::Value>,
}
