//! Agent memory integration for knowledge graphs
//!
//! This module provides functionality to integrate agent memory systems
//! with knowledge graphs, including automatic synchronization, semantic
//! indexing, and retention policies.

mod config;
mod integration;
mod nodes;
mod stats;
mod sync;

pub use config::{MemoryIntegrationConfig, MemoryRetentionPolicy};
pub use integration::MemoryIntegration;
pub use nodes::{MemoryMetadata, MemoryNode};
pub use stats::MemoryIntegrationStats;
pub use sync::{MemorySync, MemoryUpdateType, PendingMemoryUpdate};

// Re-export MemoryType from agent-core
pub use exorust_agent_core::memory::MemoryType;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraph, KnowledgeGraphConfig, NodeType};
    use exorust_agent_core::{memory::*, Agent, AgentConfig, AgentId};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_memory_integration_creation() {
        let config = MemoryIntegrationConfig::default();
        let integration = MemoryIntegration::new(config).await;
        assert!(integration.is_ok());
    }

    #[tokio::test]
    async fn test_sync_agent_memory() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        let agent_config = AgentConfig {
            name: "test_agent".to_string(),
            agent_type: "test".to_string(),
            max_memory: 1000,
            max_gpu_memory: 500,
            priority: 1,
            metadata: serde_json::Value::Null,
        };
        let agent = Agent::new(agent_config).unwrap();

        // Add some memories to the agent
        let memory_entry = MemoryEntry::new(
            MemoryType::Working,
            "test_memory".to_string(),
            serde_json::Value::String("Test memory content".to_string()),
        );

        agent.memory().store(memory_entry).await.unwrap();

        let config = MemoryIntegrationConfig {
            semantic_indexing: false,
            ..Default::default()
        };
        let mut integration = MemoryIntegration::new(config).await.unwrap();

        let synced_count = integration
            .sync_agent_memory(&agent, &mut graph)
            .await
            .unwrap();
        assert!(synced_count > 0);

        // Verify memory node was created
        let memory_nodes = graph.get_nodes_by_type(&NodeType::Memory);
        assert!(!memory_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_agent_memories() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        let test_uuid = uuid::Uuid::new_v4();
        let agent_id = AgentId(test_uuid);

        // Manually create a memory node
        let mut properties = HashMap::new();
        properties.insert(
            "agent_id".to_string(),
            serde_json::Value::String(test_uuid.to_string()),
        );
        properties.insert(
            "memory_id".to_string(),
            serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
        );
        properties.insert(
            "memory_type".to_string(),
            serde_json::Value::String("Working".to_string()),
        );
        properties.insert(
            "key".to_string(),
            serde_json::Value::String("test_key".to_string()),
        );
        properties.insert(
            "value".to_string(),
            serde_json::Value::String("Test content".to_string()),
        );
        properties.insert(
            "importance".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
        );
        properties.insert(
            "access_count".to_string(),
            serde_json::Value::Number(2.into()),
        );

        let memory_node = crate::graph::Node::new(NodeType::Memory, properties);
        graph.add_node(memory_node).unwrap();

        let config = MemoryIntegrationConfig {
            semantic_indexing: false,
            ..Default::default()
        };
        let mut integration = MemoryIntegration::new(config).await.unwrap();

        let memories = integration
            .retrieve_agent_memories(agent_id, Some(MemoryType::Working), &graph)
            .await
            .unwrap();

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].key, "test_key");
        assert_eq!(memories[0].importance, 0.7);
        assert_eq!(memories[0].access_count, 2);
    }

    #[tokio::test]
    async fn test_memory_search() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        let test_uuid = uuid::Uuid::new_v4();
        let agent_id = AgentId(test_uuid);

        // Create memory nodes with different content
        for (i, content) in [
            "artificial intelligence",
            "machine learning",
            "data science",
        ]
        .iter()
        .enumerate()
        {
            let mut properties = HashMap::new();
            properties.insert(
                "agent_id".to_string(),
                serde_json::Value::String(test_uuid.to_string()),
            );
            properties.insert(
                "memory_id".to_string(),
                serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
            );
            properties.insert(
                "memory_type".to_string(),
                serde_json::Value::String("Semantic".to_string()),
            );
            properties.insert(
                "key".to_string(),
                serde_json::Value::String(format!("key_{i}")),
            );
            properties.insert(
                "value".to_string(),
                serde_json::Value::String(content.to_string()),
            );
            properties.insert(
                "importance".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()),
            );
            properties.insert(
                "access_count".to_string(),
                serde_json::Value::Number(1.into()),
            );

            let memory_node = crate::graph::Node::new(NodeType::Memory, properties);
            graph.add_node(memory_node).unwrap();
        }

        let config = MemoryIntegrationConfig {
            semantic_indexing: false, // Use simple text matching
            ..Default::default()
        };
        let mut integration = MemoryIntegration::new(config).await.unwrap();

        let results = integration
            .search_memories(agent_id, "machine", Some(MemoryType::Semantic), 10, &graph)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].0.value.to_string().contains("machine"));
        assert!(results[0].1 > 0.0); // Should have positive similarity score
    }

    #[tokio::test]
    async fn test_retention_policy() {
        let graph_config = KnowledgeGraphConfig {
            gpu_enabled: false,
            ..Default::default()
        };
        let mut graph = KnowledgeGraph::new(graph_config).await?;

        // Create an old working memory node (should be removed)
        let mut properties = HashMap::new();
        properties.insert(
            "agent_id".to_string(),
            serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
        );
        properties.insert(
            "memory_id".to_string(),
            serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
        );
        properties.insert(
            "memory_type".to_string(),
            serde_json::Value::String("Working".to_string()),
        );
        properties.insert(
            "key".to_string(),
            serde_json::Value::String("old_memory".to_string()),
        );
        properties.insert(
            "value".to_string(),
            serde_json::Value::String("Old content".to_string()),
        );

        let mut old_memory_node = crate::graph::Node::new(NodeType::Memory, properties);
        // Make it old (more than 24 hours)
        old_memory_node.created_at = chrono::Utc::now() - chrono::Duration::hours(25);
        graph.add_node(old_memory_node).unwrap();

        let config = MemoryIntegrationConfig::default();
        let integration = MemoryIntegration::new(config).await.unwrap();

        let removed_count = integration
            .apply_retention_policy(&mut graph)
            .await
            .unwrap();
        assert_eq!(removed_count, 1);

        // Verify the node was removed
        let memory_nodes = graph.get_nodes_by_type(&NodeType::Memory);
        assert!(memory_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = MemoryIntegrationConfig::default();
        assert!(config.auto_sync);
        assert_eq!(config.sync_interval_seconds, 300);
        assert_eq!(config.max_sync_batch, 100);
        assert!(config.semantic_indexing);

        let retention = config.retention_policy;
        assert_eq!(retention.working_memory_hours, 24);
        assert_eq!(retention.episodic_memory_days, 30);
        assert!(retention.semantic_memory_days.is_none());
        assert!(retention.procedural_memory_days.is_none());
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = MemoryIntegrationConfig::default();
        let integration = MemoryIntegration::new(config).await.unwrap();

        let stats = integration.stats();
        assert_eq!(stats.total_synced, 0);
        assert_eq!(stats.sync_errors, 0);
        assert!(stats.last_sync.is_none());
        assert_eq!(stats.avg_sync_time_ms, 0.0);
    }
}
