//! Memory synchronization functionality

use super::{config::MemoryIntegrationConfig, nodes::*, stats::MemoryIntegrationStats};
use crate::error::{KnowledgeGraphError, KnowledgeGraphResult};
use crate::graph::{Edge, EdgeType, KnowledgeGraph, Node, NodeType};
use crate::semantic::{EmbeddingVector, SemanticSearchEngine};
use chrono::{DateTime, Utc};
use exorust_agent_core::{memory::*, Agent, AgentId};
use std::collections::HashMap;

/// Pending memory update
#[derive(Debug, Clone)]
pub struct PendingMemoryUpdate {
    /// Agent ID
    pub agent_id: AgentId,
    /// Memory ID
    pub memory_id: String,
    /// Update type
    pub update_type: MemoryUpdateType,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Memory update types
#[derive(Debug, Clone)]
pub enum MemoryUpdateType {
    /// Memory created
    Created,
    /// Memory updated
    Updated,
    /// Memory deleted
    Deleted,
    /// Memory accessed
    Accessed,
}

/// Memory synchronization implementation
pub struct MemorySync {
    /// Configuration
    config: MemoryIntegrationConfig,
    /// Semantic search engine for memory indexing
    semantic_engine: Option<SemanticSearchEngine>,
    /// Last sync time
    last_sync: Option<DateTime<Utc>>,
    /// Pending memory updates
    pending_updates: Vec<PendingMemoryUpdate>,
}

impl MemorySync {
    /// Create new memory sync
    pub fn new(
        config: MemoryIntegrationConfig,
        semantic_engine: Option<SemanticSearchEngine>,
    ) -> Self {
        Self {
            config,
            semantic_engine,
            last_sync: None,
            pending_updates: Vec::new(),
        }
    }

    /// Sync agent memory to knowledge graph
    pub async fn sync_agent_memory(
        &mut self,
        agent: &Agent,
        graph: &mut KnowledgeGraph,
        stats: &mut MemoryIntegrationStats,
    ) -> KnowledgeGraphResult<usize> {
        let start_time = std::time::Instant::now();
        let agent_id = agent.id();
        let mut synced_count = 0;

        // Get agent memories through memory system
        let memory = agent.memory();
        let working_memories = memory.search(MemoryType::Working, |_| true).await;
        let episodic_memories = memory.search(MemoryType::Episodic, |_| true).await;
        let semantic_memories = memory.search(MemoryType::Semantic, |_| true).await;
        let procedural_memories = memory.search(MemoryType::Procedural, |_| true).await;

        // Sync each memory type
        synced_count += self
            .sync_memory_entries(
                agent_id,
                &working_memories,
                MemoryType::Working,
                graph,
                stats,
            )
            .await?;

        synced_count += self
            .sync_memory_entries(
                agent_id,
                &episodic_memories,
                MemoryType::Episodic,
                graph,
                stats,
            )
            .await?;

        synced_count += self
            .sync_memory_entries(
                agent_id,
                &semantic_memories,
                MemoryType::Semantic,
                graph,
                stats,
            )
            .await?;

        synced_count += self
            .sync_memory_entries(
                agent_id,
                &procedural_memories,
                MemoryType::Procedural,
                graph,
                stats,
            )
            .await?;

        // Create agent-memory relationships
        self.create_memory_relationships(agent_id, graph).await?;

        // Update statistics
        let duration = start_time.elapsed().as_millis() as f64;
        stats.total_synced += synced_count;
        stats.last_sync = Some(Utc::now());
        stats.avg_sync_time_ms =
            (stats.avg_sync_time_ms * (stats.total_synced - synced_count) as f64 + duration)
                / stats.total_synced as f64;

        self.last_sync = Some(Utc::now());

        Ok(synced_count)
    }

    /// Sync memory entries of a specific type
    async fn sync_memory_entries(
        &mut self,
        agent_id: AgentId,
        memories: &[MemoryEntry],
        memory_type: MemoryType,
        graph: &mut KnowledgeGraph,
        stats: &mut MemoryIntegrationStats,
    ) -> KnowledgeGraphResult<usize> {
        let mut synced = 0;

        for memory in memories.iter().take(self.config.max_sync_batch) {
            match self
                .sync_single_memory(agent_id, memory, memory_type.clone(), graph)
                .await
            {
                Ok(_) => {
                    synced += 1;
                    *stats
                        .by_type
                        .entry(format!("{:?}", memory_type))
                        .or_insert(0) += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to sync memory {:?}: {}", memory.id, e);
                    stats.sync_errors += 1;
                }
            }
        }

        Ok(synced)
    }

    /// Sync a single memory entry
    async fn sync_single_memory(
        &mut self,
        agent_id: AgentId,
        memory: &MemoryEntry,
        memory_type: MemoryType,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<String> {
        // Create memory node
        let memory_node_id = format!("memory_{}_{:?}", agent_id.0, memory.id);

        // Check if memory node already exists
        if graph.get_node(&memory_node_id).is_ok() {
            // Update existing node
            return self
                .update_existing_memory(memory_node_id, memory, graph)
                .await;
        }

        // Create new memory node
        let mut properties = HashMap::new();
        properties.insert(
            "agent_id".to_string(),
            serde_json::Value::String(agent_id.0.to_string()),
        );
        properties.insert(
            "memory_id".to_string(),
            serde_json::Value::String(memory.id.0.to_string()),
        );
        properties.insert(
            "memory_type".to_string(),
            serde_json::Value::String(format!("{:?}", memory.memory_type)),
        );
        properties.insert(
            "key".to_string(),
            serde_json::Value::String(memory.key.clone()),
        );
        properties.insert("value".to_string(), memory.value.clone());
        properties.insert(
            "importance".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(memory.importance as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
        properties.insert(
            "access_count".to_string(),
            serde_json::Value::Number(memory.access_count.into()),
        );

        let mut memory_node = Node::new(NodeType::Memory, properties);
        memory_node.id = memory_node_id.clone();

        // Add semantic embedding if enabled
        if let Some(semantic_engine) = &mut self.semantic_engine {
            let key_text = &memory.key;
            let value_text = match &memory.value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(obj) => obj
                    .values()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => memory.value.to_string(),
            };

            let combined_text = format!("{} {}", key_text, value_text);

            if let Ok(embedding) = semantic_engine.text_to_embedding(&combined_text).await {
                memory_node.set_embedding(embedding);
            }
        }

        graph.add_node(memory_node)?;

        Ok(memory_node_id)
    }

    /// Update existing memory node
    async fn update_existing_memory(
        &mut self,
        memory_node_id: String,
        memory: &MemoryEntry,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<String> {
        let memory_node = graph.get_node_mut(&memory_node_id)?;

        // Update properties
        memory_node.update_property(
            "key".to_string(),
            serde_json::Value::String(memory.key.clone()),
        );
        memory_node.update_property("value".to_string(), memory.value.clone());
        memory_node.update_property(
            "importance".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(memory.importance as f64)
                    .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        );
        memory_node.update_property(
            "access_count".to_string(),
            serde_json::Value::Number(memory.access_count.into()),
        );

        // Update embedding if semantic indexing is enabled
        if let Some(semantic_engine) = &mut self.semantic_engine {
            let key_text = &memory.key;
            let value_text = match &memory.value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(obj) => obj
                    .values()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => memory.value.to_string(),
            };

            let combined_text = format!("{} {}", key_text, value_text);

            if let Ok(embedding) = semantic_engine.text_to_embedding(&combined_text).await {
                memory_node.set_embedding(embedding);
            }
        }

        Ok(memory_node_id)
    }

    /// Create relationships between agent and memories
    async fn create_memory_relationships(
        &self,
        agent_id: AgentId,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<()> {
        let agent_node_id = format!("agent_{}", agent_id.0);

        // Find all memory nodes for this agent and collect edges to add
        let mut edges_to_add = Vec::new();
        {
            let memory_nodes = graph.get_nodes_by_type(&NodeType::Memory);

            for memory_node in memory_nodes {
                if let Some(node_agent_id) = memory_node.get_property("agent_id") {
                    if let Some(id_str) = node_agent_id.as_str() {
                        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                            if AgentId(uuid) == agent_id {
                                // Create "has memory" relationship
                                let edge = Edge::new(
                                    agent_node_id.clone(),
                                    memory_node.id.clone(),
                                    EdgeType::Contains,
                                    1.0,
                                );

                                // Only add if both nodes exist and edge doesn't exist already
                                if graph.get_node(&agent_node_id).is_ok() {
                                    // Check if edge already exists
                                    let edge_exists = {
                                        let existing_edges =
                                            graph.get_outgoing_edges(&agent_node_id);
                                        existing_edges.iter().any(|e| {
                                            e.target_id == memory_node.id
                                                && e.edge_type == EdgeType::Contains
                                        })
                                    };

                                    if !edge_exists {
                                        edges_to_add.push(edge);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add collected edges
        for edge in edges_to_add {
            graph.add_edge(edge)?;
        }

        Ok(())
    }

    /// Record memory update
    pub fn record_memory_update(
        &mut self,
        agent_id: AgentId,
        memory_id: String,
        update_type: MemoryUpdateType,
    ) {
        let update = PendingMemoryUpdate {
            agent_id,
            memory_id,
            update_type,
            timestamp: Utc::now(),
        };
        self.pending_updates.push(update);
    }

    /// Process pending updates
    pub async fn process_pending_updates(
        &mut self,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<usize> {
        let updates_to_process = std::mem::take(&mut self.pending_updates);
        let mut processed = 0;

        for update in updates_to_process {
            match update.update_type {
                MemoryUpdateType::Deleted => {
                    let memory_node_id =
                        format!("memory_{}_{}", update.agent_id.0, update.memory_id);
                    if graph.get_node(&memory_node_id).is_ok() {
                        graph.remove_node(&memory_node_id)?;
                        processed += 1;
                    }
                }
                _ => {
                    // Other update types would need agent memory access
                    // For now, just mark as processed
                    processed += 1;
                }
            }
        }

        Ok(processed)
    }

    /// Should auto-sync based on interval
    pub fn should_auto_sync(&self) -> bool {
        if !self.config.auto_sync {
            return false;
        }

        if let Some(last_sync) = self.last_sync {
            let elapsed = Utc::now() - last_sync;
            elapsed.num_seconds() as u64 >= self.config.sync_interval_seconds
        } else {
            true // First sync
        }
    }
}
