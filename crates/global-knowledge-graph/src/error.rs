//! Global knowledge graph error types

use thiserror::Error;

/// Global knowledge graph error types
#[derive(Debug, Error)]
pub enum GlobalKnowledgeGraphError {
    /// Replication failed
    #[error("Replication failed: {region} - {reason}")]
    ReplicationFailed { region: String, reason: String },

    /// Query execution failed
    #[error("Query execution failed: {query_type} - {reason}")]
    QueryExecutionFailed { query_type: String, reason: String },

    /// Compliance violation detected
    #[error("Compliance violation: {regulation} in {region} - {details}")]
    ComplianceViolation {
        regulation: String,
        region: String,
        details: String,
    },

    /// Consistency conflict
    #[error("Consistency conflict between regions: {region1} and {region2} - {conflict_type}")]
    ConsistencyConflict {
        region1: String,
        region2: String,
        conflict_type: String,
    },

    /// Region unavailable
    #[error("Region unavailable: {region} - {reason}")]
    RegionUnavailable { region: String, reason: String },

    /// Cache synchronization failed
    #[error("Cache synchronization failed: {cache_layer} - {reason}")]
    CacheSyncFailed { cache_layer: String, reason: String },

    /// Data sovereignty violation
    #[error(
        "Data sovereignty violation: data from {origin_region} cannot be stored in {target_region}"
    )]
    DataSovereigntyViolation {
        origin_region: String,
        target_region: String,
    },

    /// Graph operation failed
    #[error("Graph operation failed: {operation} - {reason}")]
    GraphOperationFailed { operation: String, reason: String },

    /// Node not found
    #[error("Node not found: {node_id} in region {region}")]
    NodeNotFound { node_id: String, region: String },

    /// Edge not found
    #[error("Edge not found: {edge_id} between {source_node} and {target_node}")]
    EdgeNotFound {
        edge_id: String,
        source_node: String,
        target_node: String,
    },

    /// Query timeout
    #[error("Query timeout exceeded: {elapsed_ms}ms > {timeout_ms}ms")]
    QueryTimeout { elapsed_ms: u64, timeout_ms: u64 },

    /// Serialization error
    #[error("Serialization error: {context} - {details}")]
    SerializationError { context: String, details: String },

    /// Configuration error
    #[error("Configuration error: {parameter} - {reason}")]
    ConfigurationError { parameter: String, reason: String },

    /// Network error
    #[error("Network error: {endpoint} - {details}")]
    NetworkError { endpoint: String, details: String },

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Bincode error
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Other error
    #[error("Other error: {0}")]
    Other(String),
}

/// Global knowledge graph result type
pub type GlobalKnowledgeGraphResult<T> = Result<T, GlobalKnowledgeGraphError>;
