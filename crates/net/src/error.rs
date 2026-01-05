//! Network error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection failed: {endpoint}")]
    ConnectionFailed { endpoint: String },

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    #[error("Message serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // DoS protection errors

    #[error("Rate limited on endpoint {endpoint}, retry after {retry_after_ms}ms")]
    RateLimited { endpoint: String, retry_after_ms: u64 },

    #[error("Message too large: {size} bytes exceeds max {max_size} bytes")]
    MessageTooLarge { size: usize, max_size: usize },

    #[error("Queue full: {queue_size}/{max_size} messages")]
    QueueFull { queue_size: usize, max_size: usize },

    #[error("Buffer too large: {size} bytes exceeds max {max_size} bytes")]
    BufferTooLarge { size: usize, max_size: usize },

    #[error("Buffer pool exhausted: requested {requested} bytes, {available} available")]
    BufferPoolExhausted { requested: usize, available: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_connection_failed_error() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "192.168.1.100:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Connection failed: 192.168.1.100:8080");
    }

    #[test]
    fn test_connection_failed_empty_endpoint() {
        let error = NetworkError::ConnectionFailed {
            endpoint: String::new(),
        };
        assert_eq!(error.to_string(), "Connection failed: ");
    }

    #[test]
    fn test_connection_failed_unicode_endpoint() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "测试服务器:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Connection failed: 测试服务器:8080");
    }

    #[test]
    fn test_serialization_error_from_serde() {
        let serde_err = serde_json::from_str::<String>("invalid json").unwrap_err();
        let error = NetworkError::Serialization(serde_err);
        assert!(error.to_string().contains("Message serialization error"));
    }

    #[test]
    fn test_io_error_from_std() {
        let io_err = io::Error::new(io::ErrorKind::ConnectionRefused, "connection refused");
        let error = NetworkError::Io(io_err);
        assert!(error.to_string().contains("I/O error"));
        assert!(error.to_string().contains("connection refused"));
    }

    #[test]
    fn test_io_error_auto_conversion() {
        let io_err = io::Error::new(io::ErrorKind::TimedOut, "operation timed out");
        let error: NetworkError = io_err.into();
        assert!(matches!(error, NetworkError::Io(_)));
    }

    #[test]
    fn test_serde_error_auto_conversion() {
        let serde_err = serde_json::from_str::<Vec<u8>>("}invalid{").unwrap_err();
        let error: NetworkError = serde_err.into();
        assert!(matches!(error, NetworkError::Serialization(_)));
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "localhost:9999".to_string(),
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("ConnectionFailed"));
        assert!(debug_str.contains("localhost:9999"));
    }

    #[test]
    fn test_error_source_chain() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let error = NetworkError::Io(io_err);

        // Test that error source chain works
        let source = std::error::Error::source(&error);
        assert!(source.is_some());
    }

    #[test]
    fn test_connection_failed_special_characters() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "host!@#$%^&*():1234/path?query=value".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Connection failed: host!@#$%^&*():1234/path?query=value"
        );
    }

    #[test]
    fn test_connection_failed_ipv6() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "[2001:db8::1]:8080".to_string(),
        };
        assert_eq!(error.to_string(), "Connection failed: [2001:db8::1]:8080");
    }

    #[test]
    fn test_io_error_variants() {
        let errors = vec![
            io::Error::new(io::ErrorKind::NotFound, "file not found"),
            io::Error::new(io::ErrorKind::PermissionDenied, "permission denied"),
            io::Error::new(io::ErrorKind::ConnectionAborted, "connection aborted"),
            io::Error::new(io::ErrorKind::ConnectionReset, "connection reset"),
            io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"),
            io::Error::new(io::ErrorKind::AddrInUse, "address in use"),
            io::Error::new(io::ErrorKind::AddrNotAvailable, "address not available"),
            io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"),
        ];

        for io_err in errors {
            let error = NetworkError::Io(io_err);
            assert!(error.to_string().contains("I/O error"));
        }
    }

    #[test]
    fn test_serialization_error_different_types() {
        // Test with different JSON parsing errors
        let test_cases = vec![
            "null",
            "undefined",
            "{\"incomplete\":",
            "[1, 2, 3",
            "{'single_quotes': 'invalid'}",
            "NaN",
            "Infinity",
        ];

        for invalid_json in test_cases {
            let result = serde_json::from_str::<serde_json::Value>(invalid_json);
            if let Err(serde_err) = result {
                let error = NetworkError::Serialization(serde_err);
                assert!(error.to_string().contains("Message serialization error"));
            }
        }
    }

    #[test]
    fn test_error_downcasting() {
        let error = NetworkError::ConnectionFailed {
            endpoint: "test:8080".to_string(),
        };

        // Convert to Box<dyn Error>
        let boxed_error: Box<dyn std::error::Error> = Box::new(error);

        // Verify it can be downcast back
        assert!(boxed_error.downcast_ref::<NetworkError>().is_some());
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NetworkError>();
    }
}
