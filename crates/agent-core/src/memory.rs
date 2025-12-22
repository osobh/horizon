//! Agent memory management
//!
//! This module provides a comprehensive memory system for agents with support for
//! different memory types (working, episodic, semantic, procedural), advanced
//! search capabilities, eviction policies, and memory statistics.

mod entry;
mod search;
mod store;
mod system;
mod types;

// Re-export public types and functions
pub use entry::MemoryEntry;
pub use search::{MemoryQuery, SearchFilters, SearchResult, SortOrder};
pub use store::MemoryStore;
pub use system::AgentMemory;
pub use types::{MemoryId, MemoryStats, MemoryType};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentId;
    use uuid::Uuid;

    #[test]
    fn test_memory_entry() {
        let entry = MemoryEntry::new(
            MemoryType::Working,
            "test".to_string(),
            serde_json::json!({"data": "value"}),
        );

        assert_eq!(entry.memory_type, MemoryType::Working);
        assert_eq!(entry.key, "test");
        assert_eq!(entry.access_count, 0);
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_memory_entry_expiration() {
        let mut entry = MemoryEntry::new(
            MemoryType::Working,
            "test".to_string(),
            serde_json::json!({}),
        );

        entry.ttl = Some(std::time::Duration::from_secs(0));
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(entry.is_expired());
    }

    #[test]
    fn test_memory_score() {
        let mut entry = MemoryEntry::new(
            MemoryType::Semantic,
            "test".to_string(),
            serde_json::json!({}),
        );

        entry.set_importance(1.0);
        entry.access_count = 10;

        let score = entry.memory_score();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[tokio::test]
    async fn test_agent_memory_creation() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        assert_eq!(memory.agent_id(), agent_id);
        assert_eq!(memory.total_entries().await, 0);
    }

    #[tokio::test]
    async fn test_memory_store_retrieve() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Store
        let id = memory
            .store_memory(
                MemoryType::Working,
                "key1".to_string(),
                serde_json::json!({"value": 42}),
            )
            .await
            .unwrap();

        assert_ne!(id.0, Uuid::nil());

        // Retrieve
        let value = memory.retrieve(MemoryType::Working, "key1").await;
        assert!(value.is_some());
        assert_eq!(value.unwrap(), serde_json::json!({"value": 42}));

        // Non-existent key
        let value = memory.retrieve(MemoryType::Working, "nonexistent").await;
        assert!(value.is_none());
    }

    #[tokio::test]
    async fn test_memory_types() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Store in different memory types
        memory
            .store_episode(
                "Test episode".to_string(),
                serde_json::json!({"context": "test"}),
            )
            .await
            .unwrap();

        memory
            .store_fact(
                "rust".to_string(),
                serde_json::json!({"type": "programming_language"}),
            )
            .await
            .unwrap();

        memory
            .store_procedure(
                "compile".to_string(),
                serde_json::json!({"steps": ["parse", "analyze", "codegen"]}),
            )
            .await
            .unwrap();

        let stats = memory.stats().await;
        assert_eq!(stats.total_entries, 3);
    }

    #[tokio::test]
    async fn test_memory_search() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Store multiple entries
        for i in 0..5 {
            memory
                .store_memory(
                    MemoryType::Semantic,
                    format!("fact_{i}"),
                    serde_json::json!({"index": i}),
                )
                .await
                .unwrap();
        }

        // Search
        let results = memory
            .search(MemoryType::Semantic, |entry| entry.key.starts_with("fact_"))
            .await;

        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Initial stats
        let stats = memory.stats().await;
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_stores, 0);
        assert_eq!(stats.total_retrieves, 0);

        // Store and retrieve
        memory
            .store_memory(
                MemoryType::Working,
                "test".to_string(),
                serde_json::json!({}),
            )
            .await
            .unwrap();

        memory.retrieve(MemoryType::Working, "test").await;

        let stats = memory.stats().await;
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.total_stores, 1);
        assert_eq!(stats.total_retrieves, 1);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_memory_id_creation() {
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        assert_ne!(id1, id2);
        assert_ne!(id1.0, Uuid::nil());
    }

    #[test]
    fn test_memory_id_default() {
        let id = MemoryId::default();
        assert_ne!(id.0, Uuid::nil());
    }

    #[test]
    fn test_memory_entry_record_access() {
        let mut entry = MemoryEntry::new(
            MemoryType::Episodic,
            "event".to_string(),
            serde_json::json!({"event": "test"}),
        );

        let initial_time = entry.last_accessed;
        let initial_count = entry.access_count;

        std::thread::sleep(std::time::Duration::from_millis(10));
        entry.record_access();

        assert!(entry.last_accessed > initial_time);
        assert_eq!(entry.access_count, initial_count + 1);
    }

    #[test]
    fn test_memory_entry_with_metadata() {
        let mut entry = MemoryEntry::new(
            MemoryType::Procedural,
            "skill".to_string(),
            serde_json::json!({"action": "compute"}),
        );

        entry.add_metadata("version".to_string(), serde_json::json!(1));
        entry.add_metadata("author".to_string(), serde_json::json!("agent"));

        assert_eq!(entry.metadata.len(), 2);
        assert_eq!(entry.get_metadata("version"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn test_memory_entry_importance() {
        let mut entry = MemoryEntry::new(
            MemoryType::Semantic,
            "fact".to_string(),
            serde_json::json!({}),
        );

        // Default importance
        assert_eq!(entry.importance, 0.5);

        // Update importance
        entry.set_importance(0.9);
        assert_eq!(entry.importance, 0.9);
    }

    #[test]
    fn test_memory_entry_ttl() {
        let mut entry = MemoryEntry::new(
            MemoryType::Working,
            "temp".to_string(),
            serde_json::json!({}),
        );

        // No TTL by default
        assert!(entry.ttl.is_none());
        assert!(!entry.is_expired());

        // Set TTL
        entry.set_ttl(std::time::Duration::from_secs(3600));
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_memory_score_calculation() {
        let mut entry = MemoryEntry::new(
            MemoryType::Semantic,
            "knowledge".to_string(),
            serde_json::json!({}),
        );

        // Fresh entry with no accesses
        entry.set_importance(0.5);
        entry.access_count = 0;
        let score1 = entry.memory_score();

        // Frequently accessed entry
        entry.access_count = 100;
        let score2 = entry.memory_score();
        assert!(score2 > score1);

        // High importance entry
        entry.set_importance(1.0);
        let score3 = entry.memory_score();
        assert!(score3 > score2);
    }

    #[test]
    fn test_memory_type_variants() {
        let types = vec![
            MemoryType::Working,
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ];

        for mem_type in types {
            match mem_type {
                MemoryType::Working => assert_eq!(format!("{:?}", mem_type), "Working"),
                MemoryType::Episodic => assert_eq!(format!("{:?}", mem_type), "Episodic"),
                MemoryType::Semantic => assert_eq!(format!("{:?}", mem_type), "Semantic"),
                MemoryType::Procedural => assert_eq!(format!("{:?}", mem_type), "Procedural"),
            }
        }
    }

    #[tokio::test]
    async fn test_memory_store_eviction() {
        let store = MemoryStore::new(MemoryType::Working, 2); // Small limit for testing

        // Fill the store
        let entry1 = MemoryEntry::new(
            MemoryType::Working,
            "key1".to_string(),
            serde_json::json!({"data": 1}),
        );
        let entry2 = MemoryEntry::new(
            MemoryType::Working,
            "key2".to_string(),
            serde_json::json!({"data": 2}),
        );

        store.store(entry1).await.unwrap();
        store.store(entry2).await.unwrap();

        assert_eq!(store.entry_count(), 2);

        // Adding third entry should trigger eviction
        let entry3 = MemoryEntry::new(
            MemoryType::Working,
            "key3".to_string(),
            serde_json::json!({"data": 3}),
        );

        store.store(entry3).await.unwrap();
        assert_eq!(store.entry_count(), 2); // One should have been evicted
    }

    #[tokio::test]
    async fn test_memory_store_expired_retrieval() {
        let store = MemoryStore::new(MemoryType::Working, 10);

        let mut entry = MemoryEntry::new(
            MemoryType::Working,
            "expired".to_string(),
            serde_json::json!({"temp": true}),
        );
        entry.set_ttl(std::time::Duration::from_millis(1));

        store.store(entry).await.unwrap();

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Should not retrieve expired entry
        let result = store.retrieve("expired").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_memory_search_with_predicate() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Store entries with different patterns
        for i in 0..10 {
            let key = if i % 2 == 0 {
                format!("even_{i}")
            } else {
                format!("odd_{i}")
            };

            memory
                .store_memory(MemoryType::Semantic, key, serde_json::json!({"value": i}))
                .await
                .unwrap();
        }

        // Search for even entries
        let even_results = memory
            .search(MemoryType::Semantic, |entry| entry.key.starts_with("even_"))
            .await;
        assert_eq!(even_results.len(), 5);

        // Search for odd entries
        let odd_results = memory
            .search(MemoryType::Semantic, |entry| entry.key.starts_with("odd_"))
            .await;
        assert_eq!(odd_results.len(), 5);
    }

    #[tokio::test]
    async fn test_memory_cleanup() {
        let agent_id = AgentId::new();
        let memory = AgentMemory::new(agent_id, 1024 * 1024).unwrap();

        // Store some entries with TTL that will expire
        for i in 0..5 {
            let mut entry = MemoryEntry::new(
                MemoryType::Working,
                format!("temp_{i}"),
                serde_json::json!({"index": i}),
            );
            entry.set_ttl(std::time::Duration::from_millis(1));

            memory.store(entry).await.unwrap();
        }

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Cleanup should remove expired entries
        let removed = memory.cleanup().await.unwrap();
        assert_eq!(removed, 5);
    }

    #[test]
    fn test_memory_type_serialization() {
        let memory_types = vec![
            MemoryType::Working,
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ];

        for memory_type in memory_types {
            let json = serde_json::to_string(&memory_type).unwrap();
            let deserialized: MemoryType = serde_json::from_str(&json).unwrap();
            assert_eq!(memory_type, deserialized);
        }
    }

    #[test]
    fn test_memory_entry_serialization() {
        let entry = MemoryEntry::new(
            MemoryType::Semantic,
            "serialization-test".to_string(),
            serde_json::json!({"test": true, "value": 42}),
        );

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.memory_type, deserialized.memory_type);
        assert_eq!(entry.key, deserialized.key);
        assert_eq!(entry.value, deserialized.value);
        assert_eq!(entry.created_at, deserialized.created_at);
        assert_eq!(entry.access_count, deserialized.access_count);
    }

    #[test]
    fn test_memory_entry_age_calculation() {
        let entry = MemoryEntry::new(
            MemoryType::Working,
            "age-test".to_string(),
            serde_json::json!({}),
        );

        let age = entry.age_seconds();
        assert!(age >= 0);

        let idle = entry.idle_seconds();
        assert!(idle >= 0);
    }

    #[test]
    fn test_memory_search_filters() {
        let entry = MemoryEntry::new(
            MemoryType::Semantic,
            "test-key".to_string(),
            serde_json::json!({"content": "test value"}),
        );

        assert!(SearchFilters::key_contains("test")(&entry));
        assert!(SearchFilters::key_starts_with("test")(&entry));
        assert!(SearchFilters::importance_above(0.0)(&entry));
        assert!(SearchFilters::access_count_above(0)(&entry));
        assert!(SearchFilters::newer_than_seconds(3600)(&entry));
        assert!(SearchFilters::accessed_within_seconds(3600)(&entry));
        assert!(SearchFilters::value_contains_string("test")(&entry));
    }

    #[test]
    fn test_memory_query() {
        let query = MemoryQuery::new()
            .with_text("test".to_string())
            .with_min_importance(0.5)
            .with_max_age_seconds(3600)
            .with_limit(10)
            .with_sort(SortOrder::CreatedDesc);

        let entry = MemoryEntry::new(
            MemoryType::Working,
            "test-query".to_string(),
            serde_json::json!({"data": "test"}),
        );

        assert!(query.matches(&entry));

        let entries = vec![entry];
        let sorted = query.sort_entries(entries);
        assert_eq!(sorted.len(), 1);
    }
}
