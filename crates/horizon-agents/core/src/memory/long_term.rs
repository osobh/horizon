use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalEntry {
    pub id: Uuid,
    pub key: String,
    pub value: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl HistoricalEntry {
    pub fn new(key: String, value: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            key,
            value,
            timestamp: chrono::Utc::now(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    pub fn with_metadata(mut self, k: String, v: String) -> Self {
        self.metadata.insert(k, v);
        self
    }
}

pub struct LongTermMemory {
    entries: Vec<HistoricalEntry>,
    max_entries: usize,
}

impl LongTermMemory {
    pub fn new(max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max entries must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            entries: Vec::new(),
            max_entries,
        })
    }

    pub fn append(&mut self, entry: HistoricalEntry) -> Result<()> {
        if self.entries.len() >= self.max_entries {
            // Remove oldest entry
            self.entries.remove(0);
        }

        self.entries.push(entry);
        Ok(())
    }

    pub fn query_by_key(&self, key: &str) -> Vec<&HistoricalEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.key == key)
            .collect()
    }

    pub fn query_by_tag(&self, tag: &str) -> Vec<&HistoricalEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.tags.contains(&tag.to_string()))
            .collect()
    }

    pub fn query_recent(&self, limit: usize) -> Vec<&HistoricalEntry> {
        let start = if self.entries.len() > limit {
            self.entries.len() - limit
        } else {
            0
        };

        self.entries[start..].iter().rev().collect()
    }

    pub fn query_all(&self) -> Vec<&HistoricalEntry> {
        self.entries.iter().collect()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_entry_creation() {
        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        assert_eq!(entry.key, "key1");
        assert_eq!(entry.value, "value1");
        assert!(entry.tags.is_empty());
        assert!(entry.metadata.is_empty());
    }

    #[test]
    fn test_historical_entry_with_tag() {
        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string())
            .with_tag("tag1".to_string())
            .with_tag("tag2".to_string());

        assert_eq!(entry.tags.len(), 2);
        assert!(entry.tags.contains(&"tag1".to_string()));
    }

    #[test]
    fn test_historical_entry_with_metadata() {
        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string())
            .with_metadata("meta1".to_string(), "val1".to_string());

        assert_eq!(entry.metadata.len(), 1);
        assert_eq!(entry.metadata.get("meta1").unwrap(), "val1");
    }

    #[test]
    fn test_long_term_memory_creation() {
        let memory = LongTermMemory::new(100);
        assert!(memory.is_ok());

        let memory = memory.unwrap();
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_long_term_memory_zero_capacity() {
        let memory = LongTermMemory::new(0);
        assert!(memory.is_err());
    }

    #[test]
    fn test_long_term_memory_append() {
        let mut memory = LongTermMemory::new(100).unwrap();

        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        memory.append(entry).unwrap();

        assert_eq!(memory.size(), 1);
    }

    #[test]
    fn test_long_term_memory_append_with_eviction() {
        let mut memory = LongTermMemory::new(2).unwrap();

        let entry1 = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        let entry2 = HistoricalEntry::new("key2".to_string(), "value2".to_string());
        let entry3 = HistoricalEntry::new("key3".to_string(), "value3".to_string());

        memory.append(entry1).unwrap();
        memory.append(entry2).unwrap();
        memory.append(entry3).unwrap();

        assert_eq!(memory.size(), 2);
        let all = memory.query_all();
        assert_eq!(all[0].key, "key2");
        assert_eq!(all[1].key, "key3");
    }

    #[test]
    fn test_long_term_memory_query_by_key() {
        let mut memory = LongTermMemory::new(100).unwrap();

        let entry1 = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        let entry2 = HistoricalEntry::new("key1".to_string(), "value2".to_string());
        let entry3 = HistoricalEntry::new("key2".to_string(), "value3".to_string());

        memory.append(entry1).unwrap();
        memory.append(entry2).unwrap();
        memory.append(entry3).unwrap();

        let results = memory.query_by_key("key1");
        assert_eq!(results.len(), 2);

        let results = memory.query_by_key("key2");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_long_term_memory_query_by_tag() {
        let mut memory = LongTermMemory::new(100).unwrap();

        let entry1 = HistoricalEntry::new("key1".to_string(), "value1".to_string())
            .with_tag("important".to_string());
        let entry2 = HistoricalEntry::new("key2".to_string(), "value2".to_string())
            .with_tag("important".to_string());
        let entry3 = HistoricalEntry::new("key3".to_string(), "value3".to_string())
            .with_tag("normal".to_string());

        memory.append(entry1).unwrap();
        memory.append(entry2).unwrap();
        memory.append(entry3).unwrap();

        let results = memory.query_by_tag("important");
        assert_eq!(results.len(), 2);

        let results = memory.query_by_tag("normal");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_long_term_memory_query_recent() {
        let mut memory = LongTermMemory::new(100).unwrap();

        for i in 0..5 {
            let entry = HistoricalEntry::new(format!("key{}", i), format!("value{}", i));
            memory.append(entry).unwrap();
        }

        let recent = memory.query_recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].key, "key4");
        assert_eq!(recent[1].key, "key3");
        assert_eq!(recent[2].key, "key2");
    }

    #[test]
    fn test_long_term_memory_query_recent_more_than_exists() {
        let mut memory = LongTermMemory::new(100).unwrap();

        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        memory.append(entry).unwrap();

        let recent = memory.query_recent(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_long_term_memory_query_all() {
        let mut memory = LongTermMemory::new(100).unwrap();

        for i in 0..3 {
            let entry = HistoricalEntry::new(format!("key{}", i), format!("value{}", i));
            memory.append(entry).unwrap();
        }

        let all = memory.query_all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_long_term_memory_clear() {
        let mut memory = LongTermMemory::new(100).unwrap();

        let entry = HistoricalEntry::new("key1".to_string(), "value1".to_string());
        memory.append(entry).unwrap();

        assert_eq!(memory.size(), 1);
        memory.clear();
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_long_term_memory_size_tracking() {
        let mut memory = LongTermMemory::new(100).unwrap();

        assert_eq!(memory.size(), 0);

        for i in 0..5 {
            let entry = HistoricalEntry::new(format!("key{}", i), format!("value{}", i));
            memory.append(entry).unwrap();
            assert_eq!(memory.size(), i + 1);
        }
    }
}
