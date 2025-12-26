use crate::cache_layer::*;
use crate::compliance_handler::*;
use crate::consistency_manager::*;
use crate::error::*;
use crate::graph_manager::*;
use crate::query_engine::*;
use crate::region_router::*;
use crate::replication::*;
use crate::*;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

#[tokio::test]
async fn test_all_error_types() {
    let errors = vec![
        GlobalKnowledgeGraphError::ReplicationFailed {
            region: "us-east-1".to_string(),
            reason: "Network timeout".to_string(),
        },
        GlobalKnowledgeGraphError::QueryExecutionFailed {
            query_type: "NodeQuery".to_string(),
            reason: "Syntax error".to_string(),
        },
        GlobalKnowledgeGraphError::ComplianceViolation {
            regulation: "GDPR".to_string(),
            region: "eu-west-1".to_string(),
            details: "Personal data in wrong region".to_string(),
        },
        GlobalKnowledgeGraphError::ConsistencyConflict {
            region1: "us-east-1".to_string(),
            region2: "eu-west-1".to_string(),
            conflict_type: "Version conflict".to_string(),
        },
        GlobalKnowledgeGraphError::RegionUnavailable {
            region: "ap-southeast-1".to_string(),
            reason: "Maintenance mode".to_string(),
        },
    ];

    for error in errors {
        assert!(!error.to_string().is_empty());
    }
}

#[tokio::test]
async fn test_error_serialization() {
    let error = GlobalKnowledgeGraphError::NodeNotFound {
        node_id: "node_123".to_string(),
        region: "us-east-1".to_string(),
    };

    let error_string = error.to_string();
    assert!(error_string.contains("node_123"));
    assert!(error_string.contains("us-east-1"));
}

#[tokio::test]
async fn test_cache_sync_errors() {
    let error = GlobalKnowledgeGraphError::CacheSyncFailed {
        cache_layer: "L1".to_string(),
        reason: "Connection refused".to_string(),
    };

    assert!(error.to_string().contains("Cache synchronization failed"));
}

#[tokio::test]
async fn test_data_sovereignty_errors() {
    let error = GlobalKnowledgeGraphError::DataSovereigntyViolation {
        origin_region: "eu-west-1".to_string(),
        target_region: "us-east-1".to_string(),
    };

    assert!(error.to_string().contains("Data sovereignty violation"));
}

#[tokio::test]
async fn test_timeout_errors() {
    let error = GlobalKnowledgeGraphError::QueryTimeout {
        elapsed_ms: 5000,
        timeout_ms: 3000,
    };

    assert!(error.to_string().contains("timeout exceeded"));
}

#[tokio::test]
async fn test_configuration_errors() {
    let error = GlobalKnowledgeGraphError::ConfigurationError {
        parameter: "max_nodes_per_region".to_string(),
        reason: "Value must be positive".to_string(),
    };

    assert!(error.to_string().contains("Configuration error"));
}

#[tokio::test]
async fn test_network_errors() {
    let error = GlobalKnowledgeGraphError::NetworkError {
        endpoint: "https://api.region.com".to_string(),
        details: "Connection timeout".to_string(),
    };

    assert!(error.to_string().contains("Network error"));
}

#[tokio::test]
async fn test_graph_operation_errors() {
    let error = GlobalKnowledgeGraphError::GraphOperationFailed {
        operation: "add_edge".to_string(),
        reason: "Target node not found".to_string(),
    };

    assert!(error.to_string().contains("Graph operation failed"));
}

#[tokio::test]
async fn test_edge_not_found_errors() {
    let error = GlobalKnowledgeGraphError::EdgeNotFound {
        edge_id: "edge_123".to_string(),
        source_node: "node_1".to_string(),
        target_node: "node_2".to_string(),
    };

    assert!(error.to_string().contains("Edge not found"));
}

#[tokio::test]
async fn test_serialization_errors() {
    let error = GlobalKnowledgeGraphError::SerializationError {
        context: "Node serialization".to_string(),
        details: "Invalid UTF-8 sequence".to_string(),
    };

    assert!(error.to_string().contains("Serialization error"));
}
