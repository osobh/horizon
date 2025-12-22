//! Error types for the knowledge graph system

use std::fmt;

/// Result type for knowledge graph operations
pub type KnowledgeGraphResult<T> = Result<T, KnowledgeGraphError>;

/// Knowledge graph error types
#[derive(Debug)]
pub enum KnowledgeGraphError {
    /// Node not found
    NodeNotFound {
        /// Node ID
        node_id: String,
    },
    /// Edge not found
    EdgeNotFound {
        /// Edge ID
        edge_id: String,
    },
    /// Invalid query syntax
    InvalidQuery {
        /// Query description
        query: String,
        /// Error message
        message: String,
    },
    /// GPU operation failed
    GpuError {
        /// Error message
        message: String,
    },
    /// Semantic search error
    SemanticError {
        /// Error message
        message: String,
    },
    /// Pattern discovery error
    PatternError {
        /// Error message
        message: String,
    },
    /// Memory integration error
    MemoryError {
        /// Error message
        message: String,
    },
    /// Evolution tracking error
    EvolutionError {
        /// Error message
        message: String,
    },
    /// Pruning error
    PruningError {
        /// Error message
        message: String,
    },
    /// Storage error
    StorageError {
        /// Error message
        message: String,
    },
    /// Serialization error
    SerializationError {
        /// Error message
        message: String,
    },
    /// Compression error
    Compression {
        /// Error message
        message: String,
    },
    /// Other error
    Other(String),
}

impl fmt::Display for KnowledgeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound { node_id } => {
                write!(f, "Node not found: {node_id}")
            }
            Self::EdgeNotFound { edge_id } => {
                write!(f, "Edge not found: {edge_id}")
            }
            Self::InvalidQuery { query, message } => {
                write!(f, "Invalid query '{query}': {message}")
            }
            Self::GpuError { message } => {
                write!(f, "GPU error: {message}")
            }
            Self::SemanticError { message } => {
                write!(f, "Semantic search error: {message}")
            }
            Self::PatternError { message } => {
                write!(f, "Pattern discovery error: {message}")
            }
            Self::MemoryError { message } => {
                write!(f, "Memory integration error: {message}")
            }
            Self::EvolutionError { message } => {
                write!(f, "Evolution tracking error: {message}")
            }
            Self::PruningError { message } => {
                write!(f, "Pruning error: {message}")
            }
            Self::StorageError { message } => {
                write!(f, "Storage error: {message}")
            }
            Self::SerializationError { message } => {
                write!(f, "Serialization error: {message}")
            }
            Self::Compression { message } => {
                write!(f, "Compression error: {message}")
            }
            Self::Other(msg) => write!(f, "Knowledge graph error: {msg}"),
        }
    }
}

impl std::error::Error for KnowledgeGraphError {}

impl From<anyhow::Error> for KnowledgeGraphError {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<serde_json::Error> for KnowledgeGraphError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError {
            message: err.to_string(),
        }
    }
}

impl From<exorust_cuda::error::CudaError> for KnowledgeGraphError {
    fn from(err: exorust_cuda::error::CudaError) -> Self {
        Self::GpuError {
            message: err.to_string(),
        }
    }
}

impl From<exorust_storage::error::StorageError> for KnowledgeGraphError {
    fn from(err: exorust_storage::error::StorageError) -> Self {
        Self::StorageError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = KnowledgeGraphError::NodeNotFound {
            node_id: "node123".to_string(),
        };
        assert_eq!(error.to_string(), "Node not found: node123");

        let error = KnowledgeGraphError::InvalidQuery {
            query: "SELECT * FROM".to_string(),
            message: "Incomplete query".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid query 'SELECT * FROM': Incomplete query"
        );
    }

    #[test]
    fn test_error_conversion() {
        let json_error =
            serde_json::Error::syntax(serde_json::error::ErrorCode::EofWhileParsingValue, 0, 0);
        let kg_error: KnowledgeGraphError = json_error.into();

        match kg_error {
            KnowledgeGraphError::SerializationError { .. } => (),
            _ => panic!("Expected SerializationError"),
        }
    }

    #[test]
    fn test_all_error_variants_display() {
        let errors = vec![
            KnowledgeGraphError::NodeNotFound {
                node_id: "test-node".to_string(),
            },
            KnowledgeGraphError::EdgeNotFound {
                edge_id: "test-edge".to_string(),
            },
            KnowledgeGraphError::InvalidQuery {
                query: "MATCH (n)".to_string(),
                message: "Missing WHERE clause".to_string(),
            },
            KnowledgeGraphError::GpuError {
                message: "Out of memory".to_string(),
            },
            KnowledgeGraphError::SemanticError {
                message: "Embedding dimension mismatch".to_string(),
            },
            KnowledgeGraphError::PatternError {
                message: "Pattern too complex".to_string(),
            },
            KnowledgeGraphError::MemoryError {
                message: "Memory sync failed".to_string(),
            },
            KnowledgeGraphError::EvolutionError {
                message: "Invalid lineage".to_string(),
            },
            KnowledgeGraphError::PruningError {
                message: "Cannot prune protected node".to_string(),
            },
            KnowledgeGraphError::StorageError {
                message: "Disk full".to_string(),
            },
            KnowledgeGraphError::SerializationError {
                message: "Invalid JSON".to_string(),
            },
            KnowledgeGraphError::Other("Unknown error".to_string()),
        ];

        // Verify all errors can be displayed
        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_error_result_type() {
        fn returns_result() -> KnowledgeGraphResult<String> {
            Ok("success".to_string())
        }

        fn returns_error() -> KnowledgeGraphResult<String> {
            Err(KnowledgeGraphError::Other("failed".to_string()))
        }

        assert!(returns_result().is_ok());
        assert!(returns_error().is_err());
    }

    #[test]
    fn test_anyhow_conversion() {
        let anyhow_error = anyhow::anyhow!("Test error");
        let kg_error: KnowledgeGraphError = anyhow_error.into();

        match kg_error {
            KnowledgeGraphError::Other(msg) => {
                assert_eq!(msg, "Test error");
            }
            _ => panic!("Expected Other error"),
        }
    }

    #[test]
    fn test_cuda_error_conversion() {
        use exorust_cuda::error::CudaError;

        let cuda_error = CudaError::InitializationError {
            message: "CUDA not available".to_string(),
        };
        let kg_error: KnowledgeGraphError = cuda_error.into();

        match kg_error {
            KnowledgeGraphError::GpuError { message } => {
                assert!(message.contains("CUDA"));
            }
            _ => panic!("Expected GpuError"),
        }
    }

    #[test]
    fn test_storage_error_conversion() {
        use exorust_storage::error::StorageError;

        let storage_error = StorageError::IoError {
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
        };
        let kg_error: KnowledgeGraphError = storage_error.into();

        match kg_error {
            KnowledgeGraphError::StorageError { message } => {
                assert!(message.contains("IoError"));
            }
            _ => panic!("Expected StorageError"),
        }
    }

    #[test]
    fn test_error_source_chain() {
        use std::error::Error;

        let error = KnowledgeGraphError::NodeNotFound {
            node_id: "test".to_string(),
        };

        // KnowledgeGraphError implements std::error::Error
        assert!(error.source().is_none()); // No underlying cause
    }

    #[test]
    fn test_error_pattern_matching() {
        let error = KnowledgeGraphError::InvalidQuery {
            query: "test".to_string(),
            message: "error".to_string(),
        };

        match &error {
            KnowledgeGraphError::InvalidQuery { query, message } => {
                assert_eq!(query, "test");
                assert_eq!(message, "error");
            }
            _ => panic!("Pattern match failed"),
        }
    }

    #[test]
    fn test_error_equality() {
        // Note: KnowledgeGraphError doesn't implement PartialEq, but we can test string representation
        let error1 = KnowledgeGraphError::NodeNotFound {
            node_id: "node1".to_string(),
        };
        let error2 = KnowledgeGraphError::NodeNotFound {
            node_id: "node1".to_string(),
        };

        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_error_with_special_characters() {
        let error = KnowledgeGraphError::NodeNotFound {
            node_id: "node-with-🦀-emoji".to_string(),
        };
        let display = error.to_string();
        assert!(display.contains("🦀"));

        let error = KnowledgeGraphError::InvalidQuery {
            query: "SELECT * WHERE value = 'test\"quote'".to_string(),
            message: "Unescaped quote".to_string(),
        };
        let display = error.to_string();
        assert!(display.contains("test\"quote"));
    }

    #[test]
    fn test_error_memory_size() {
        // Ensure error types are reasonably sized
        assert!(std::mem::size_of::<KnowledgeGraphError>() <= 64);
    }

    #[test]
    fn test_result_unwrap_patterns() {
        let ok_result: KnowledgeGraphResult<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: KnowledgeGraphResult<i32> =
            Err(KnowledgeGraphError::Other("test".to_string()));
        assert!(err_result.is_err());

        // Test with different types
        let string_result: KnowledgeGraphResult<String> = Ok("hello".to_string());
        assert_eq!(string_result?, "hello");

        let vec_result: KnowledgeGraphResult<Vec<u8>> = Ok(vec![1, 2, 3]);
        assert_eq!(vec_result.unwrap().len(), 3);
    }

    #[test]
    fn test_error_propagation() {
        fn inner_function() -> KnowledgeGraphResult<()> {
            Err(KnowledgeGraphError::MemoryError {
                message: "Inner error".to_string(),
            })
        }

        fn outer_function() -> KnowledgeGraphResult<String> {
            inner_function()?;
            Ok("Never reached".to_string())
        }

        let result = outer_function();
        assert!(result.is_err());
        match result {
            Err(KnowledgeGraphError::MemoryError { message }) => {
                assert_eq!(message, "Inner error");
            }
            _ => panic!("Expected MemoryError"),
        }
    }

    #[test]
    fn test_error_formatting() {
        let error = KnowledgeGraphError::InvalidQuery {
            query: "A".repeat(100),
            message: "Query too long".to_string(),
        };

        let formatted = format!("{error}");
        assert!(formatted.contains("Query too long"));
        assert!(formatted.len() > 100); // Should contain the long query

        let debug_formatted = format!("{:?}", error);
        assert!(debug_formatted.contains("InvalidQuery"));
    }

    #[test]
    fn test_error_map_operations() {
        let result: KnowledgeGraphResult<i32> = Ok(10);
        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.unwrap(), 20);

        let error_result: KnowledgeGraphResult<i32> =
            Err(KnowledgeGraphError::Other("test".to_string()));
        let mapped_error = error_result.map(|x| x * 2);
        assert!(mapped_error.is_err());
    }

    #[test]
    fn test_complex_error_scenarios() {
        // Test chaining multiple operations that might fail
        fn parse_and_validate(input: &str) -> KnowledgeGraphResult<u32> {
            if input.is_empty() {
                return Err(KnowledgeGraphError::InvalidQuery {
                    query: input.to_string(),
                    message: "Empty input".to_string(),
                });
            }

            input
                .parse::<u32>()
                .map_err(|_| KnowledgeGraphError::Other("Parse error".to_string()))
        }

        assert!(parse_and_validate("42").is_ok());
        assert!(parse_and_validate("").is_err());
        assert!(parse_and_validate("not-a-number").is_err());
    }

    #[test]
    fn test_error_collection() {
        let errors: Vec<KnowledgeGraphError> = vec![
            KnowledgeGraphError::NodeNotFound {
                node_id: "1".to_string(),
            },
            KnowledgeGraphError::EdgeNotFound {
                edge_id: "2".to_string(),
            },
            KnowledgeGraphError::Other("3".to_string()),
        ];

        let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();

        assert_eq!(error_messages.len(), 3);
        assert!(error_messages[0].contains("Node not found"));
        assert!(error_messages[1].contains("Edge not found"));
        assert!(error_messages[2].contains("Knowledge graph error"));
    }
}
