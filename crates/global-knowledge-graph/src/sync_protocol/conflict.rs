//! Conflict detection and resolution for distributed synchronization

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sync_protocol::types::KnowledgeOperation;

/// Represents a conflict between operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictingOperation {
    pub operation1: KnowledgeOperation,
    pub operation2: KnowledgeOperation,
    pub conflict_type: ConflictType,
    pub severity: ConflictSeverity,
}

/// Types of conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    WriteWrite,
    WriteDelete,
    DeleteDelete,
    CausalityViolation,
    SchemaConflict,
}

/// Severity levels for conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Resolution for a conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub conflict_id: Uuid,
    pub strategy: ResolutionStrategy,
    pub winning_operation: Option<KnowledgeOperation>,
    pub merged_operation: Option<KnowledgeOperation>,
}

/// Strategies for resolving conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    LastWriterWins,
    FirstWriterWins,
    Merge,
    Manual,
    Custom(String),
}

/// Trait for conflict resolution
#[async_trait::async_trait]
pub trait ConflictResolver: Send + Sync {
    async fn detect_conflicts(&self, operations: &[KnowledgeOperation]) -> Vec<ConflictingOperation>;
    async fn resolve(&self, conflict: ConflictingOperation) -> Resolution;
    async fn apply_resolution(&self, resolution: Resolution);
}

/// Default conflict resolver implementation
pub struct DefaultConflictResolver {
    resolution_history: tokio::sync::RwLock<Vec<Resolution>>,
}

impl DefaultConflictResolver {
    pub fn new() -> Self {
        Self {
            resolution_history: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl ConflictResolver for DefaultConflictResolver {
    async fn detect_conflicts(&self, operations: &[KnowledgeOperation]) -> Vec<ConflictingOperation> {
        let mut conflicts = Vec::new();
        
        // Check for conflicts between all pairs of operations
        for i in 0..operations.len() {
            for j in i + 1..operations.len() {
                let op1 = &operations[i];
                let op2 = &operations[j];
                
                // Check if operations are concurrent
                if op1.vector_clock.concurrent_with(&op2.vector_clock) {
                    // Detect conflict type based on operation types
                    let conflict_type = match (&op1.operation_type, &op2.operation_type) {
                        (crate::sync_protocol::types::OperationType::Update, 
                         crate::sync_protocol::types::OperationType::Update) => ConflictType::WriteWrite,
                        (crate::sync_protocol::types::OperationType::Update, 
                         crate::sync_protocol::types::OperationType::Delete) |
                        (crate::sync_protocol::types::OperationType::Delete, 
                         crate::sync_protocol::types::OperationType::Update) => ConflictType::WriteDelete,
                        (crate::sync_protocol::types::OperationType::Delete, 
                         crate::sync_protocol::types::OperationType::Delete) => ConflictType::DeleteDelete,
                        _ => continue,
                    };
                    
                    conflicts.push(ConflictingOperation {
                        operation1: op1.clone(),
                        operation2: op2.clone(),
                        conflict_type,
                        severity: ConflictSeverity::Medium,
                    });
                }
            }
        }
        
        conflicts
    }

    async fn resolve(&self, conflict: ConflictingOperation) -> Resolution {
        // Default: Last Writer Wins based on timestamp
        let winning_operation = if conflict.operation1.timestamp > conflict.operation2.timestamp {
            Some(conflict.operation1)
        } else {
            Some(conflict.operation2)
        };
        
        let resolution = Resolution {
            conflict_id: Uuid::new_v4(),
            strategy: ResolutionStrategy::LastWriterWins,
            winning_operation,
            merged_operation: None,
        };
        
        // Store resolution in history
        let mut history = self.resolution_history.write().await;
        history.push(resolution.clone());
        
        resolution
    }

    async fn apply_resolution(&self, _resolution: Resolution) {
        // In production, this would apply the resolution to the knowledge graph
    }
}