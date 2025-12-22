//! Entity access tracking functionality
//!
//! This module handles tracking when entities are accessed to support
//! usage-based pruning strategies.

use super::types::EntityAccess;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

/// Access tracking manager
pub struct AccessTracker {
    /// Entity access tracking
    tracking: HashMap<String, EntityAccess>,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new() -> Self {
        Self {
            tracking: HashMap::new(),
        }
    }

    /// Record entity access
    pub fn record_access(&mut self, entity_id: String) {
        let now = Utc::now();

        let access = self
            .tracking
            .entry(entity_id.clone())
            .or_insert_with(|| EntityAccess {
                entity_id,
                last_accessed: now,
                access_count: 0,
                recent_accesses: Vec::new(),
            });

        access.last_accessed = now;
        access.access_count += 1;
        access.recent_accesses.push(now);

        // Keep only recent accesses (last 24 hours)
        let cutoff = now - Duration::hours(24);
        access
            .recent_accesses
            .retain(|&timestamp| timestamp > cutoff);
    }

    /// Get access information for an entity
    pub fn get_access(&self, entity_id: &str) -> Option<&EntityAccess> {
        self.tracking.get(entity_id)
    }

    /// Get all tracked accesses
    pub fn get_all_accesses(&self) -> &HashMap<String, EntityAccess> {
        &self.tracking
    }

    /// Clear all access tracking
    pub fn clear(&mut self) {
        self.tracking.clear();
    }

    /// Get entities with insufficient access count in time window
    pub fn get_underaccessed_entities(
        &self,
        min_access_count: u32,
        time_window_hours: i64,
    ) -> Vec<String> {
        let cutoff = Utc::now() - Duration::hours(time_window_hours);
        let mut underaccessed = Vec::new();

        for (entity_id, access) in &self.tracking {
            // Count recent accesses
            let recent_count = access
                .recent_accesses
                .iter()
                .filter(|&&timestamp| timestamp > cutoff)
                .count() as u32;

            if recent_count < min_access_count {
                underaccessed.push(entity_id.clone());
            }
        }

        underaccessed
    }

    /// Clean up old access records
    pub fn cleanup_old_records(&mut self, max_age_hours: i64) {
        let cutoff = Utc::now() - Duration::hours(max_age_hours);

        self.tracking
            .retain(|_, access| access.last_accessed > cutoff);
    }

    /// Get number of tracked entities
    pub fn len(&self) -> usize {
        self.tracking.len()
    }

    /// Check if tracker is empty
    pub fn is_empty(&self) -> bool {
        self.tracking.is_empty()
    }
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tracking() {
        let mut tracker = AccessTracker::new();
        let entity_id = "test_entity".to_string();

        assert_eq!(tracker.len(), 0);

        tracker.record_access(entity_id.clone());
        tracker.record_access(entity_id.clone());

        assert_eq!(tracker.len(), 1);

        let access = tracker.get_access(&entity_id).unwrap();
        assert_eq!(access.access_count, 2);
        assert_eq!(access.recent_accesses.len(), 2);
    }

    #[test]
    fn test_underaccessed_entities() {
        let mut tracker = AccessTracker::new();

        // Entity with sufficient access
        let good_entity = "good_entity".to_string();
        for _ in 0..5 {
            tracker.record_access(good_entity.clone());
        }

        // Entity with insufficient access
        let bad_entity = "bad_entity".to_string();
        tracker.record_access(bad_entity.clone());

        let underaccessed = tracker.get_underaccessed_entities(3, 24);

        assert_eq!(underaccessed.len(), 1);
        assert_eq!(underaccessed[0], bad_entity);
    }

    #[test]
    fn test_clear_tracking() {
        let mut tracker = AccessTracker::new();

        tracker.record_access("entity1".to_string());
        tracker.record_access("entity2".to_string());
        assert_eq!(tracker.len(), 2);

        tracker.clear();
        assert_eq!(tracker.len(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_cleanup_old_records() {
        let mut tracker = AccessTracker::new();
        let entity_id = "test_entity".to_string();

        // Add access entry
        tracker.record_access(entity_id.clone());

        // Manually set old timestamp
        if let Some(access) = tracker.tracking.get_mut(&entity_id) {
            access.last_accessed = Utc::now() - Duration::hours(48);
        }

        assert_eq!(tracker.len(), 1);

        // Cleanup entries older than 24 hours
        tracker.cleanup_old_records(24);

        assert_eq!(tracker.len(), 0);
    }
}
