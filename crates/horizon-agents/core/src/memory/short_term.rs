use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::error::{AgentError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub key: String,
    pub value: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    pub fn new(key: String, value: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            key,
            value,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, k: String, v: String) -> Self {
        self.metadata.insert(k, v);
        self
    }
}

pub struct ShortTermMemory {
    entries: HashMap<String, MemoryEntry>,
    max_entries: usize,
}

impl ShortTermMemory {
    pub fn new(max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(AgentError::InvalidConfiguration(
                "Max entries must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            entries: HashMap::new(),
            max_entries,
        })
    }

    pub fn store(&mut self, key: String, value: String) -> Result<()> {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            // Evict oldest entry
            self.evict_oldest()?;
        }

        let entry = MemoryEntry::new(key.clone(), value);
        self.entries.insert(key, entry);
        Ok(())
    }

    pub fn retrieve(&self, key: &str) -> Result<String> {
        self.entries
            .get(key)
            .map(|entry| entry.value.clone())
            .ok_or_else(|| AgentError::MemoryError(format!("Key not found: {}", key)))
    }

    pub fn delete(&mut self, key: &str) -> Result<()> {
        self.entries
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| AgentError::MemoryError(format!("Key not found: {}", key)))
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

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    fn evict_oldest(&mut self) -> Result<()> {
        let oldest_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(key, _)| key.clone())
            .ok_or_else(|| AgentError::MemoryError("No entries to evict".to_string()))?;

        self.entries.remove(&oldest_key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new("key1".to_string(), "value1".to_string());
        assert_eq!(entry.key, "key1");
        assert_eq!(entry.value, "value1");
        assert!(entry.metadata.is_empty());
    }

    #[test]
    fn test_memory_entry_with_metadata() {
        let entry = MemoryEntry::new("key1".to_string(), "value1".to_string())
            .with_metadata("meta1".to_string(), "metaval1".to_string());

        assert_eq!(entry.metadata.len(), 1);
        assert_eq!(entry.metadata.get("meta1").unwrap(), "metaval1");
    }

    #[test]
    fn test_short_term_memory_creation() {
        let memory = ShortTermMemory::new(10);
        assert!(memory.is_ok());

        let memory = memory.unwrap();
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_short_term_memory_zero_capacity() {
        let memory = ShortTermMemory::new(0);
        assert!(memory.is_err());
    }

    #[test]
    fn test_short_term_memory_store_and_retrieve() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        let value = memory.retrieve("key1").unwrap();
        assert_eq!(value, "value1");
    }

    #[test]
    fn test_short_term_memory_retrieve_nonexistent() {
        let memory = ShortTermMemory::new(10).unwrap();
        let result = memory.retrieve("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_short_term_memory_update() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        memory
            .store("key1".to_string(), "value2".to_string())
            .unwrap();

        let value = memory.retrieve("key1").unwrap();
        assert_eq!(value, "value2");
        assert_eq!(memory.size(), 1);
    }

    #[test]
    fn test_short_term_memory_delete() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        assert!(memory.contains("key1"));

        memory.delete("key1").unwrap();
        assert!(!memory.contains("key1"));
    }

    #[test]
    fn test_short_term_memory_delete_nonexistent() {
        let mut memory = ShortTermMemory::new(10).unwrap();
        let result = memory.delete("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_short_term_memory_clear() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        memory
            .store("key2".to_string(), "value2".to_string())
            .unwrap();
        assert_eq!(memory.size(), 2);

        memory.clear();
        assert_eq!(memory.size(), 0);
        assert!(memory.is_empty());
    }

    #[test]
    fn test_short_term_memory_eviction() {
        let mut memory = ShortTermMemory::new(2).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        memory
            .store("key2".to_string(), "value2".to_string())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        // This should evict key1 (oldest)
        memory
            .store("key3".to_string(), "value3".to_string())
            .unwrap();

        assert_eq!(memory.size(), 2);
        assert!(!memory.contains("key1"));
        assert!(memory.contains("key2"));
        assert!(memory.contains("key3"));
    }

    #[test]
    fn test_short_term_memory_keys() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        memory
            .store("key2".to_string(), "value2".to_string())
            .unwrap();

        let keys = memory.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
    }

    #[test]
    fn test_short_term_memory_contains() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        assert!(!memory.contains("key1"));
        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        assert!(memory.contains("key1"));
    }

    #[test]
    fn test_short_term_memory_size_tracking() {
        let mut memory = ShortTermMemory::new(10).unwrap();

        assert_eq!(memory.size(), 0);
        memory
            .store("key1".to_string(), "value1".to_string())
            .unwrap();
        assert_eq!(memory.size(), 1);
        memory
            .store("key2".to_string(), "value2".to_string())
            .unwrap();
        assert_eq!(memory.size(), 2);
        memory.delete("key1").unwrap();
        assert_eq!(memory.size(), 1);
    }
}
