//! Memory search and query functionality

use super::entry::MemoryEntry;
use super::types::MemoryType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Memory search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Memory types to search
    pub memory_types: Vec<MemoryType>,
    /// Text to search for in keys and values
    pub text: Option<String>,
    /// Importance threshold (minimum)
    pub min_importance: Option<f32>,
    /// Maximum age in seconds
    pub max_age_seconds: Option<i64>,
    /// Minimum access count
    pub min_access_count: Option<u64>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Sort order
    pub sort_by: Option<SortOrder>,
}

/// Sort order for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    /// By creation time (newest first)
    CreatedDesc,
    /// By creation time (oldest first)
    CreatedAsc,
    /// By last access time (most recent first)
    AccessedDesc,
    /// By last access time (least recent first)
    AccessedAsc,
    /// By importance (highest first)
    ImportanceDesc,
    /// By importance (lowest first)
    ImportanceAsc,
    /// By access count (highest first)
    AccessCountDesc,
    /// By access count (lowest first)
    AccessCountAsc,
    /// By memory score (highest first)
    ScoreDesc,
    /// By memory score (lowest first)
    ScoreAsc,
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            memory_types: vec![
                MemoryType::Working,
                MemoryType::Episodic,
                MemoryType::Semantic,
                MemoryType::Procedural,
            ],
            text: None,
            min_importance: None,
            max_age_seconds: None,
            min_access_count: None,
            limit: Some(100),
            sort_by: Some(SortOrder::CreatedDesc),
        }
    }
}

impl MemoryQuery {
    /// Create new query for all memory types
    pub fn new() -> Self {
        Self::default()
    }

    /// Create query for specific memory types
    pub fn for_types(memory_types: Vec<MemoryType>) -> Self {
        Self {
            memory_types,
            ..Default::default()
        }
    }

    /// Create query for single memory type
    pub fn for_type(memory_type: MemoryType) -> Self {
        Self {
            memory_types: vec![memory_type],
            ..Default::default()
        }
    }

    /// Set text search term
    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }

    /// Set minimum importance
    pub fn with_min_importance(mut self, importance: f32) -> Self {
        self.min_importance = Some(importance);
        self
    }

    /// Set maximum age in seconds
    pub fn with_max_age_seconds(mut self, seconds: i64) -> Self {
        self.max_age_seconds = Some(seconds);
        self
    }

    /// Set minimum access count
    pub fn with_min_access_count(mut self, count: u64) -> Self {
        self.min_access_count = Some(count);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set sort order
    pub fn with_sort(mut self, sort_by: SortOrder) -> Self {
        self.sort_by = Some(sort_by);
        self
    }

    /// Check if entry matches this query
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        // Check memory type
        if !self.memory_types.contains(&entry.memory_type) {
            return false;
        }

        // Check text search
        if let Some(ref text) = self.text {
            let search_text = text.to_lowercase();
            let key_matches = entry.key.to_lowercase().contains(&search_text);
            let value_matches = entry
                .value
                .to_string()
                .to_lowercase()
                .contains(&search_text);

            if !key_matches && !value_matches {
                return false;
            }
        }

        // Check importance threshold
        if let Some(min_importance) = self.min_importance {
            if entry.importance < min_importance {
                return false;
            }
        }

        // Check age
        if let Some(max_age) = self.max_age_seconds {
            let age = entry.age_seconds();
            if age > max_age {
                return false;
            }
        }

        // Check access count
        if let Some(min_access) = self.min_access_count {
            if entry.access_count < min_access {
                return false;
            }
        }

        true
    }

    /// Sort entries according to sort order
    pub fn sort_entries(&self, mut entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        if let Some(ref sort_order) = self.sort_by {
            match sort_order {
                SortOrder::CreatedDesc => {
                    entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                }
                SortOrder::CreatedAsc => {
                    entries.sort_by(|a, b| a.created_at.cmp(&b.created_at));
                }
                SortOrder::AccessedDesc => {
                    entries.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
                }
                SortOrder::AccessedAsc => {
                    entries.sort_by(|a, b| a.last_accessed.cmp(&b.last_accessed));
                }
                SortOrder::ImportanceDesc => {
                    entries.sort_by(|a, b| {
                        b.importance
                            .partial_cmp(&a.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                SortOrder::ImportanceAsc => {
                    entries.sort_by(|a, b| {
                        a.importance
                            .partial_cmp(&b.importance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                SortOrder::AccessCountDesc => {
                    entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
                }
                SortOrder::AccessCountAsc => {
                    entries.sort_by(|a, b| a.access_count.cmp(&b.access_count));
                }
                SortOrder::ScoreDesc => {
                    entries.sort_by(|a, b| {
                        b.memory_score()
                            .partial_cmp(&a.memory_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                SortOrder::ScoreAsc => {
                    entries.sort_by(|a, b| {
                        a.memory_score()
                            .partial_cmp(&b.memory_score())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        // Apply limit
        if let Some(limit) = self.limit {
            entries.truncate(limit);
        }

        entries
    }
}

/// Memory search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Matching entries
    pub entries: Vec<MemoryEntry>,
    /// Total matches before limit was applied
    pub total_matches: usize,
    /// Query that was executed
    pub query: MemoryQuery,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

impl SearchResult {
    /// Create new search result
    pub fn new(
        entries: Vec<MemoryEntry>,
        total_matches: usize,
        query: MemoryQuery,
        execution_time_ms: u64,
    ) -> Self {
        Self {
            entries,
            total_matches,
            query,
            execution_time_ms,
        }
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get number of returned entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if more results are available
    pub fn has_more(&self) -> bool {
        self.total_matches > self.entries.len()
    }
}

/// Memory search filters
pub struct SearchFilters;

impl SearchFilters {
    /// Filter by key pattern
    pub fn key_contains(pattern: &str) -> impl Fn(&MemoryEntry) -> bool + '_ {
        move |entry: &MemoryEntry| entry.key.contains(pattern)
    }

    /// Filter by key starting with prefix
    pub fn key_starts_with(prefix: &str) -> impl Fn(&MemoryEntry) -> bool + '_ {
        move |entry: &MemoryEntry| entry.key.starts_with(prefix)
    }

    /// Filter by importance threshold
    pub fn importance_above(threshold: f32) -> impl Fn(&MemoryEntry) -> bool {
        move |entry: &MemoryEntry| entry.importance >= threshold
    }

    /// Filter by access count threshold
    pub fn access_count_above(threshold: u64) -> impl Fn(&MemoryEntry) -> bool {
        move |entry: &MemoryEntry| entry.access_count >= threshold
    }

    /// Filter by age (entries newer than specified seconds)
    pub fn newer_than_seconds(seconds: i64) -> impl Fn(&MemoryEntry) -> bool {
        move |entry: &MemoryEntry| entry.age_seconds() < seconds
    }

    /// Filter by idle time (entries accessed more recently than specified seconds)
    pub fn accessed_within_seconds(seconds: i64) -> impl Fn(&MemoryEntry) -> bool {
        move |entry: &MemoryEntry| entry.idle_seconds() < seconds
    }

    /// Filter by value containing specific data
    pub fn value_contains_string(search: &str) -> impl Fn(&MemoryEntry) -> bool + '_ {
        move |entry: &MemoryEntry| {
            entry
                .value
                .to_string()
                .to_lowercase()
                .contains(&search.to_lowercase())
        }
    }

    /// Filter by metadata key existence
    pub fn has_metadata_key(key: &str) -> impl Fn(&MemoryEntry) -> bool + '_ {
        move |entry: &MemoryEntry| entry.metadata.contains_key(key)
    }

    /// Filter by date range
    pub fn created_between(
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> impl Fn(&MemoryEntry) -> bool {
        move |entry: &MemoryEntry| entry.created_at >= start && entry.created_at <= end
    }
}
