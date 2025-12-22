//! Core types for global synchronization protocol

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Knowledge operation for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeOperation {
    pub id: Uuid,
    pub operation_type: OperationType,
    pub data: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub cluster_id: String,
    pub vector_clock: VectorClock,
    pub priority: OperationPriority,
    pub dependencies: Vec<Uuid>,
    pub metadata: HashMap<String, String>,
}

/// Type of knowledge operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Create,
    Update,
    Delete,
    Merge,
    Query,
    Replicate,
    Checkpoint,
    Rollback,
}

/// Vector clock for distributed consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}

/// Operation priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Operation evidence for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationEvidence {
    pub operation_id: Uuid,
    pub signatures: Vec<String>,
    pub witnesses: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub hash: String,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    pub fn increment(&mut self, cluster_id: &str) {
        *self.clocks.entry(cluster_id.to_string()).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (cluster_id, &clock) in &other.clocks {
            let entry = self.clocks.entry(cluster_id.clone()).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    pub fn happens_before(&self, other: &VectorClock) -> bool {
        for (cluster_id, &clock) in &self.clocks {
            if let Some(&other_clock) = other.clocks.get(cluster_id) {
                if clock > other_clock {
                    return false;
                }
            } else if clock > 0 {
                return false;
            }
        }
        true
    }

    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

impl Default for VectorClock {
    fn default() -> Self {
        Self::new()
    }
}