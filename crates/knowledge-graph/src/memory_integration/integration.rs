//! Main memory integration functionality

use super::{
    config::MemoryIntegrationConfig,
    nodes::*,
    stats::MemoryIntegrationStats,
    sync::{MemorySync, MemoryUpdateType},
};
use crate::error::{KnowledgeGraphError, KnowledgeGraphResult};
use crate::graph::{KnowledgeGraph, Node, NodeType};
use crate::semantic::SemanticSearchEngine;
use chrono::{DateTime, Utc};
use stratoswarm_agent_core::{memory::*, Agent, AgentId};
use std::collections::HashMap;

/// Agent memory integration system
pub struct MemoryIntegration {
    /// Memory synchronization component
    sync: MemorySync,
    /// Memory sync statistics
    stats: MemoryIntegrationStats,
}

impl MemoryIntegration {
    /// Create a new memory integration system
    pub async fn new(config: MemoryIntegrationConfig) -> KnowledgeGraphResult<Self> {
        let semantic_engine = if config.semantic_indexing {
            Some(SemanticSearchEngine::new(
                crate::semantic::EmbeddingConfig::default(),
            ))
        } else {
            None
        };

        let sync = MemorySync::new(config, semantic_engine);

        Ok(Self {
            sync,
            stats: MemoryIntegrationStats {
                total_synced: 0,
                by_type: HashMap::new(),
                last_sync: None,
                sync_errors: 0,
                avg_sync_time_ms: 0.0,
            },
        })
    }

    /// Sync agent memory to knowledge graph
    pub async fn sync_agent_memory(
        &mut self,
        agent: &Agent,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<usize> {
        self.sync
            .sync_agent_memory(agent, graph, &mut self.stats)
            .await
    }

    /// Retrieve agent memories from knowledge graph
    pub async fn retrieve_agent_memories(
        &mut self,
        agent_id: AgentId,
        memory_type: Option<MemoryType>,
        graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<Vec<MemoryEntry>> {
        let mut memories = Vec::new();
        let memory_nodes = graph.get_nodes_by_type(&NodeType::Memory);

        for memory_node in memory_nodes {
            // Check if memory belongs to the agent
            if let Some(node_agent_id) = memory_node.get_property("agent_id") {
                if let Some(id_str) = node_agent_id.as_str() {
                    if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
                        if AgentId(uuid) != agent_id {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }

            // Check memory type filter
            if let Some(ref filter_type) = memory_type {
                if let Some(node_type) = memory_node.get_property("memory_type") {
                    if let Some(type_str) = node_type.as_str() {
                        let node_memory_type = match type_str {
                            "Working" => MemoryType::Working,
                            "Episodic" => MemoryType::Episodic,
                            "Semantic" => MemoryType::Semantic,
                            "Procedural" => MemoryType::Procedural,
                            _ => continue,
                        };
                        if &node_memory_type != filter_type {
                            continue;
                        }
                    }
                }
            }

            // Convert node to memory entry
            if let Ok(memory_entry) = self.node_to_memory_entry(memory_node) {
                memories.push(memory_entry);
            }
        }

        Ok(memories)
    }

    /// Convert knowledge graph node to memory entry
    fn node_to_memory_entry(&self, node: &Node) -> KnowledgeGraphResult<MemoryEntry> {
        let memory_id_str = node
            .get_property("memory_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| KnowledgeGraphError::Other("Missing memory_id".to_string()))?;

        let memory_id = uuid::Uuid::parse_str(memory_id_str)
            .map_err(|_| KnowledgeGraphError::Other("Invalid memory_id format".to_string()))?;

        let memory_type_str = node
            .get_property("memory_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| KnowledgeGraphError::Other("Missing memory_type".to_string()))?;

        let memory_type = match memory_type_str {
            "Working" => MemoryType::Working,
            "Episodic" => MemoryType::Episodic,
            "Semantic" => MemoryType::Semantic,
            "Procedural" => MemoryType::Procedural,
            _ => {
                return Err(KnowledgeGraphError::Other(
                    "Invalid memory_type".to_string(),
                ))
            }
        };

        let key = node
            .get_property("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| KnowledgeGraphError::Other("Missing key".to_string()))?
            .to_string();

        let value = node
            .get_property("value")
            .ok_or_else(|| KnowledgeGraphError::Other("Missing value".to_string()))?
            .clone();

        let importance = node
            .get_property("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let access_count = node
            .get_property("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Ok(MemoryEntry {
            id: MemoryId(memory_id),
            memory_type,
            key,
            value,
            metadata: HashMap::new(),
            created_at: node.created_at,
            last_accessed: node.updated_at,
            access_count,
            importance,
            ttl: None,
        })
    }

    /// Search memories using semantic similarity
    pub async fn search_memories(
        &mut self,
        agent_id: AgentId,
        query: &str,
        memory_type: Option<MemoryType>,
        top_k: usize,
        graph: &KnowledgeGraph,
    ) -> KnowledgeGraphResult<Vec<(MemoryEntry, f64)>> {
        // For now, fallback to simple text matching since we don't have semantic engine access
        let memories = self
            .retrieve_agent_memories(agent_id, memory_type, graph)
            .await?;
        let mut results = Vec::new();

        for memory in memories {
            let combined_text =
                format!("{} {}", memory.key, memory.value.to_string()).to_lowercase();
            let query_lower = query.to_lowercase();

            if combined_text.contains(&query_lower) {
                // Simple similarity based on query term frequency
                let similarity = query_lower
                    .split_whitespace()
                    .map(|term| {
                        if combined_text.contains(term) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / query_lower.split_whitespace().count() as f64;

                results.push((memory, similarity));
            }
        }

        // Sort by similarity and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);

        Ok(results)
    }

    /// Apply memory retention policy
    pub async fn apply_retention_policy(
        &self,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<usize> {
        let mut removed_count = 0;
        let now = Utc::now();
        let memory_nodes: Vec<_> = graph
            .get_nodes_by_type(&NodeType::Memory)
            .into_iter()
            .cloned()
            .collect();
        let mut nodes_to_remove = Vec::new();

        for memory_node in &memory_nodes {
            let should_remove = if let Some(memory_type_str) = memory_node
                .get_property("memory_type")
                .and_then(|v| v.as_str())
            {
                match memory_type_str {
                    "Working" => {
                        let age = now - memory_node.created_at;
                        age.num_hours() > 24 // Default working memory hours
                    }
                    "Episodic" => {
                        let age = now - memory_node.created_at;
                        age.num_days() > 30 // Default episodic memory days
                    }
                    "Semantic" => false,   // Permanent by default
                    "Procedural" => false, // Permanent by default
                    _ => false,
                }
            } else {
                false
            };

            if should_remove {
                nodes_to_remove.push(memory_node.id.clone());
            }
        }

        // Remove nodes after iteration
        for node_id in nodes_to_remove {
            graph.remove_node(&node_id)?;
            removed_count += 1;
        }

        Ok(removed_count)
    }

    /// Get integration statistics
    pub fn stats(&self) -> &MemoryIntegrationStats {
        &self.stats
    }

    /// Record memory update
    pub fn record_memory_update(
        &mut self,
        agent_id: AgentId,
        memory_id: String,
        update_type: MemoryUpdateType,
    ) {
        self.sync
            .record_memory_update(agent_id, memory_id, update_type);
    }

    /// Process pending updates
    pub async fn process_pending_updates(
        &mut self,
        graph: &mut KnowledgeGraph,
    ) -> KnowledgeGraphResult<usize> {
        self.sync.process_pending_updates(graph).await
    }

    /// Should auto-sync based on interval
    pub fn should_auto_sync(&self) -> bool {
        self.sync.should_auto_sync()
    }
}
