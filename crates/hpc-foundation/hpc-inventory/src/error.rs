//! Inventory error types

use thiserror::Error;

/// Inventory operation errors
#[derive(Debug, Error)]
pub enum InventoryError {
    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Duplicate node ID
    #[error("Node with ID {0} already exists")]
    DuplicateNode(String),

    /// Duplicate node address
    #[error("Node with address {0} already exists")]
    DuplicateAddress(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Database error (when postgres feature is enabled)
    #[cfg(feature = "postgres")]
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
}

/// Result type for inventory operations
pub type Result<T> = std::result::Result<T, InventoryError>;
