//! CUDA error types and result handling

use thiserror::Error;

/// CUDA operation errors
#[derive(Debug, Error)]
pub enum CudaError {
    /// CUDA toolkit not found
    #[error("CUDA toolkit not found")]
    ToolkitNotFound,

    /// Unsupported CUDA version
    #[error("Unsupported CUDA version: {version}")]
    UnsupportedVersion {
        /// CUDA version string
        version: String,
    },

    /// CUDA initialization failed
    #[error("CUDA initialization failed: {message}")]
    InitializationFailed {
        /// Error message
        message: String,
    },

    /// Invalid device
    #[error("Invalid CUDA device: {device}")]
    InvalidDevice {
        /// Device ID
        device: i32,
    },

    /// Out of memory
    #[error("CUDA out of memory: requested {requested} bytes")]
    OutOfMemory {
        /// Requested bytes
        requested: usize,
    },

    /// Kernel compilation error
    #[error("Kernel compilation failed: {message}")]
    CompilationError {
        /// Error message
        message: String,
    },

    /// Kernel not found
    #[error("Kernel not found: {name}")]
    KernelNotFound {
        /// Kernel name
        name: String,
    },

    /// Invalid kernel parameter
    #[error("Invalid kernel parameter: {message}")]
    InvalidParameter {
        /// Error message
        message: String,
    },

    /// Kernel execution error
    #[error("Kernel execution failed: {message}")]
    ExecutionError {
        /// Error message
        message: String,
    },

    /// Stream error
    #[error("Stream error: {message}")]
    StreamError {
        /// Error message
        message: String,
    },

    /// Context error
    #[error("Context error: {message}")]
    ContextError {
        /// Error message
        message: String,
    },

    /// PTX parsing error
    #[error("PTX parsing error: {message}")]
    PtxError {
        /// Error message
        message: String,
    },

    /// Mock mode error
    #[error("Operation not supported in mock mode")]
    MockModeError,

    /// File not found error
    #[error("File not found: {path}")]
    FileNotFound {
        /// File path
        path: String,
    },

    /// Invalid value error
    #[error("Invalid value for {parameter}")]
    InvalidValue {
        /// Parameter name
        parameter: String,
    },

    /// Generic CUDA error
    #[error("CUDA error: {code} - {message}")]
    CudaApiError {
        /// Error code
        code: i32,
        /// Error message
        message: String,
    },
}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, CudaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CudaError::ToolkitNotFound;
        assert_eq!(err.to_string(), "CUDA toolkit not found");

        let err = CudaError::UnsupportedVersion {
            version: "10.0".to_string(),
        };
        assert_eq!(err.to_string(), "Unsupported CUDA version: 10.0");

        let err = CudaError::OutOfMemory { requested: 1024 };
        assert_eq!(err.to_string(), "CUDA out of memory: requested 1024 bytes");
    }

    #[test]
    fn test_all_error_variants() {
        // ToolkitNotFound
        let err = CudaError::ToolkitNotFound;
        assert_eq!(err.to_string(), "CUDA toolkit not found");

        // UnsupportedVersion
        let err = CudaError::UnsupportedVersion {
            version: "11.2".to_string(),
        };
        assert_eq!(err.to_string(), "Unsupported CUDA version: 11.2");

        // InitializationFailed
        let err = CudaError::InitializationFailed {
            message: "Failed to initialize driver".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "CUDA initialization failed: Failed to initialize driver"
        );

        // InvalidDevice
        let err = CudaError::InvalidDevice { device: 5 };
        assert_eq!(err.to_string(), "Invalid CUDA device: 5");

        // CompilationError
        let err = CudaError::CompilationError {
            message: "Syntax error at line 10".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Kernel compilation failed: Syntax error at line 10"
        );

        // KernelNotFound
        let err = CudaError::KernelNotFound {
            name: "matmul_kernel".to_string(),
        };
        assert_eq!(err.to_string(), "Kernel not found: matmul_kernel");

        // InvalidParameter
        let err = CudaError::InvalidParameter {
            message: "Block size exceeds limit".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid kernel parameter: Block size exceeds limit"
        );

        // ExecutionError
        let err = CudaError::ExecutionError {
            message: "Launch failed".to_string(),
        };
        assert_eq!(err.to_string(), "Kernel execution failed: Launch failed");

        // StreamError
        let err = CudaError::StreamError {
            message: "Stream synchronization failed".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Stream error: Stream synchronization failed"
        );

        // ContextError
        let err = CudaError::ContextError {
            message: "Context creation failed".to_string(),
        };
        assert_eq!(err.to_string(), "Context error: Context creation failed");

        // PtxError
        let err = CudaError::PtxError {
            message: "Invalid PTX format".to_string(),
        };
        assert_eq!(err.to_string(), "PTX parsing error: Invalid PTX format");

        // MockModeError
        let err = CudaError::MockModeError;
        assert_eq!(err.to_string(), "Operation not supported in mock mode");

        // CudaApiError
        let err = CudaError::CudaApiError {
            code: 700,
            message: "Illegal memory access".to_string(),
        };
        assert_eq!(err.to_string(), "CUDA error: 700 - Illegal memory access");
    }

    #[test]
    fn test_error_debug_format() {
        let err = CudaError::OutOfMemory { requested: 1048576 };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("OutOfMemory"));
        assert!(debug_str.contains("1048576"));
    }

    #[test]
    fn test_error_source() {
        use std::error::Error;

        let err = CudaError::ToolkitNotFound;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_cuda_result_type() {
        fn success_function() -> CudaResult<i32> {
            Ok(42)
        }

        fn error_function() -> CudaResult<i32> {
            Err(CudaError::MockModeError)
        }

        assert!(success_function().is_ok());
        assert_eq!(success_function()?, 42);

        assert!(error_function().is_err());
        assert!(matches!(
            error_function().unwrap_err(),
            CudaError::MockModeError
        ));
    }

    #[test]
    fn test_error_pattern_matching() {
        let errors = vec![
            CudaError::ToolkitNotFound,
            CudaError::UnsupportedVersion {
                version: "9.0".to_string(),
            },
            CudaError::InvalidDevice { device: -1 },
            CudaError::OutOfMemory { requested: 0 },
            CudaError::MockModeError,
        ];

        for (i, err) in errors.into_iter().enumerate() {
            match err {
                CudaError::ToolkitNotFound => assert_eq!(i, 0),
                CudaError::UnsupportedVersion { version } => {
                    assert_eq!(i, 1);
                    assert_eq!(version, "9.0");
                }
                CudaError::InvalidDevice { device } => {
                    assert_eq!(i, 2);
                    assert_eq!(device, -1);
                }
                CudaError::OutOfMemory { requested } => {
                    assert_eq!(i, 3);
                    assert_eq!(requested, 0);
                }
                CudaError::MockModeError => assert_eq!(i, 4),
                _ => panic!("Unexpected error variant"),
            }
        }
    }

    #[test]
    fn test_error_chaining() {
        fn inner_function() -> CudaResult<()> {
            Err(CudaError::InitializationFailed {
                message: "Driver error".to_string(),
            })
        }

        fn outer_function() -> CudaResult<()> {
            inner_function().map_err(|_| CudaError::ContextError {
                message: "Failed due to initialization error".to_string(),
            })
        }

        let result = outer_function();
        assert!(result.is_err());

        match result.unwrap_err() {
            CudaError::ContextError { message } => {
                assert_eq!(message, "Failed due to initialization error");
            }
            _ => panic!("Expected ContextError"),
        }
    }

    #[test]
    fn test_error_edge_cases() {
        // Test with empty strings
        let err = CudaError::UnsupportedVersion {
            version: "".to_string(),
        };
        assert_eq!(err.to_string(), "Unsupported CUDA version: ");

        // Test with very large values
        let err = CudaError::OutOfMemory {
            requested: usize::MAX,
        };
        assert!(err.to_string().contains(&usize::MAX.to_string()));

        // Test with negative device ID
        let err = CudaError::InvalidDevice { device: i32::MIN };
        assert!(err.to_string().contains(&i32::MIN.to_string()));

        // Test with very long messages
        let long_message = "a".repeat(1000);
        let err = CudaError::CompilationError {
            message: long_message.clone(),
        };
        assert!(err.to_string().contains(&long_message));
    }

    #[test]
    fn test_error_conversion() {
        // Test converting to Box<dyn Error>
        let err: Box<dyn std::error::Error> = Box::new(CudaError::ToolkitNotFound);
        assert_eq!(err.to_string(), "CUDA toolkit not found");

        // Test with Send + Sync
        let err: Box<dyn std::error::Error + Send + Sync> = Box::new(CudaError::MockModeError);
        assert_eq!(err.to_string(), "Operation not supported in mock mode");
    }

    #[test]
    fn test_cuda_api_error_codes() {
        let common_error_codes = vec![
            (1, "Invalid value"),
            (2, "Memory allocation failed"),
            (3, "Initialization error"),
            (4, "Launch failure"),
            (700, "Illegal memory access"),
            (999, "Unknown error"),
        ];

        for (code, msg) in common_error_codes {
            let err = CudaError::CudaApiError {
                code,
                message: msg.to_string(),
            };
            let error_str = err.to_string();
            assert!(error_str.contains(&code.to_string()));
            assert!(error_str.contains(msg));
        }
    }

    #[test]
    fn test_error_equality() {
        // Since errors don't implement PartialEq, test through string representation
        let err1 = CudaError::KernelNotFound {
            name: "test_kernel".to_string(),
        };
        let err2 = CudaError::KernelNotFound {
            name: "test_kernel".to_string(),
        };
        assert_eq!(err1.to_string(), err2.to_string());

        let err3 = CudaError::KernelNotFound {
            name: "different_kernel".to_string(),
        };
        assert_ne!(err1.to_string(), err3.to_string());
    }

    #[test]
    fn test_result_propagation() {
        fn step1() -> CudaResult<i32> {
            Ok(10)
        }

        fn step2(value: i32) -> CudaResult<i32> {
            if value < 20 {
                Err(CudaError::InvalidParameter {
                    message: "Value too small".to_string(),
                })
            } else {
                Ok(value * 2)
            }
        }

        let result = step1().and_then(step2);
        assert!(result.is_err());

        match result.unwrap_err() {
            CudaError::InvalidParameter { message } => {
                assert_eq!(message, "Value too small");
            }
            _ => panic!("Expected InvalidParameter"),
        }
    }

    #[test]
    fn test_error_special_characters() {
        let special_messages = vec![
            "Error: newline\ncharacter",
            "Error: tab\tcharacter",
            "Error: quote\"character",
            "Error: unicode 中文 العربية",
            "Error: emoji 🚀 ⚡",
        ];

        for msg in special_messages {
            let err = CudaError::ExecutionError {
                message: msg.to_string(),
            };
            assert!(err.to_string().contains(msg));
        }
    }
}
