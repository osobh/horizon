//! Storage error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Key not found: {key}")]
    KeyNotFound { key: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Storage full: {available} bytes available")]
    StorageFull { available: u64 },

    #[error("Invalid node: {id} - {reason}")]
    InvalidNode { id: u64, reason: String },

    #[error("Invalid edge: from {from} to {to} - {reason}")]
    InvalidEdge { from: u64, to: u64, reason: String },

    #[error("Storage not initialized")]
    StorageNotInitialized,

    #[error("Invalid data format: {reason}")]
    InvalidDataFormat { reason: String },

    #[error("WAL error: {reason}")]
    WALError { reason: String },

    #[error("Lock poisoned: {resource}")]
    LockPoisoned { resource: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error as IoError, ErrorKind};

    #[test]
    fn test_key_not_found_error() {
        let error = StorageError::KeyNotFound {
            key: "test_key".to_string(),
        };
        assert!(error.to_string().contains("Key not found: test_key"));
    }

    #[test]
    fn test_storage_full_error() {
        let error = StorageError::StorageFull { available: 1024 };
        assert!(error.to_string().contains("Storage full: 1024 bytes"));
    }

    #[test]
    fn test_invalid_node_error() {
        let error = StorageError::InvalidNode {
            id: 42,
            reason: "missing properties".to_string(),
        };
        assert!(error.to_string().contains("Invalid node: 42"));
        assert!(error.to_string().contains("missing properties"));
    }

    #[test]
    fn test_invalid_edge_error() {
        let error = StorageError::InvalidEdge {
            from: 1,
            to: 2,
            reason: "weight out of range".to_string(),
        };
        assert!(error.to_string().contains("Invalid edge: from 1 to 2"));
        assert!(error.to_string().contains("weight out of range"));
    }

    #[test]
    fn test_storage_not_initialized_error() {
        let error = StorageError::StorageNotInitialized;
        assert!(error.to_string().contains("Storage not initialized"));
    }

    #[test]
    fn test_invalid_data_format_error() {
        let error = StorageError::InvalidDataFormat {
            reason: "malformed JSON".to_string(),
        };
        assert!(error.to_string().contains("Invalid data format"));
        assert!(error.to_string().contains("malformed JSON"));
    }

    #[test]
    fn test_wal_error() {
        let error = StorageError::WALError {
            reason: "checkpoint failed".to_string(),
        };
        assert!(error.to_string().contains("WAL error"));
        assert!(error.to_string().contains("checkpoint failed"));
    }

    #[test]
    fn test_lock_poisoned_error() {
        let error = StorageError::LockPoisoned {
            resource: "graph_mutex".to_string(),
        };
        assert!(error.to_string().contains("Lock poisoned"));
        assert!(error.to_string().contains("graph_mutex"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = IoError::new(ErrorKind::NotFound, "file not found");
        let storage_error = StorageError::from(io_error);

        match storage_error {
            StorageError::Io(_) => {
                assert!(storage_error.to_string().contains("I/O error"));
            }
            _ => panic!("Expected Io variant"),
        }
    }

    #[test]
    fn test_serialization_error_conversion() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let storage_error = StorageError::from(json_error);

        match storage_error {
            StorageError::Serialization(_) => {
                assert!(storage_error.to_string().contains("Serialization error"));
            }
            _ => panic!("Expected Serialization variant"),
        }
    }

    #[test]
    fn test_error_debug_format() {
        let errors = vec![
            StorageError::KeyNotFound {
                key: "test".to_string(),
            },
            StorageError::StorageFull { available: 100 },
            StorageError::InvalidNode {
                id: 1,
                reason: "test".to_string(),
            },
            StorageError::InvalidEdge {
                from: 1,
                to: 2,
                reason: "test".to_string(),
            },
            StorageError::StorageNotInitialized,
            StorageError::InvalidDataFormat {
                reason: "test".to_string(),
            },
            StorageError::WALError {
                reason: "test".to_string(),
            },
            StorageError::LockPoisoned {
                resource: "test".to_string(),
            },
        ];

        for error in errors {
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
            // Debug format may not include "StorageError" prefix, but should include variant names
            assert!(
                debug_str.contains("KeyNotFound")
                    || debug_str.contains("StorageFull")
                    || debug_str.contains("InvalidNode")
                    || debug_str.contains("InvalidEdge")
                    || debug_str.contains("StorageNotInitialized")
                    || debug_str.contains("InvalidDataFormat")
                    || debug_str.contains("WALError")
                    || debug_str.contains("LockPoisoned")
            );
        }
    }

    #[test]
    fn test_error_source() {
        use std::error::Error;

        let io_error = IoError::new(ErrorKind::PermissionDenied, "access denied");
        let storage_error = StorageError::Io(io_error);

        assert!(storage_error.source().is_some());
    }

    #[test]
    fn test_error_chain() {
        use std::error::Error;

        let io_error = IoError::new(ErrorKind::BrokenPipe, "pipe broken");
        let storage_error = StorageError::Io(io_error);

        let mut error_chain = Vec::new();
        let mut current_error: &dyn Error = &storage_error;

        loop {
            error_chain.push(current_error.to_string());
            if let Some(source) = current_error.source() {
                current_error = source;
            } else {
                break;
            }
        }

        assert!(error_chain.len() >= 1);
        assert!(error_chain[0].contains("I/O error"));
    }

    #[test]
    fn test_edge_cases() {
        // Test with empty strings
        let error1 = StorageError::KeyNotFound { key: String::new() };
        assert!(error1.to_string().contains("Key not found:"));

        // Test with very large numbers
        let error2 = StorageError::StorageFull {
            available: u64::MAX,
        };
        assert!(error2.to_string().contains(&u64::MAX.to_string()));

        // Test with Unicode strings
        let error3 = StorageError::InvalidDataFormat {
            reason: "数据格式错误".to_string(),
        };
        assert!(error3.to_string().contains("数据格式错误"));
    }

    #[test]
    fn test_error_equality() {
        // Test that errors with same content are formatted consistently
        let error1 = StorageError::KeyNotFound {
            key: "test".to_string(),
        };
        let error2 = StorageError::KeyNotFound {
            key: "test".to_string(),
        };

        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<StorageError>();
        assert_sync::<StorageError>();
    }
}
