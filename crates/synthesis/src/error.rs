//! Synthesis error types

use thiserror::Error;

/// Synthesis pipeline errors
#[derive(Debug, Error)]
pub enum SynthesisError {
    /// Goal interpretation failed
    #[error("Failed to interpret goal: {message}")]
    InterpretationError { message: String },

    /// Kernel synthesis failed
    #[error("Failed to synthesize kernel: {message}")]
    SynthesisFailure { message: String },

    /// Compilation error
    #[error("Compilation failed: {message}")]
    CompilationError { message: String },

    /// Execution error
    #[error("Execution failed: {message}")]
    ExecutionError { message: String },

    /// Template not found
    #[error("Template not found: {name}")]
    TemplateNotFound { name: String },

    /// Invalid specification
    #[error("Invalid specification: {message}")]
    InvalidSpecification { message: String },

    /// LLM API error
    #[error("LLM API error: {message}")]
    LlmApiError { message: String },

    /// Performance target not met
    #[error("Performance target not met: achieved {achieved:.2}, target {target:.2}")]
    PerformanceTargetMissed { achieved: f64, target: f64 },

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {resource}")]
    ResourceLimitExceeded { resource: String },

    /// Optimization failed
    #[error("Optimization failed after {iterations} iterations")]
    OptimizationFailed { iterations: u32 },

    /// Cache error
    #[error("Cache error: {message}")]
    CacheError { message: String },

    /// CUDA error
    #[error("CUDA error: {message}")]
    CudaError {
        /// Error message describing the CUDA error
        message: String,
    },

    /// Agent core error
    #[error("Agent error: {message}")]
    AgentError {
        /// Error message describing the agent error
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Other error
    #[error("Synthesis error: {0}")]
    Other(String),
}

/// Result type for synthesis operations
pub type SynthesisResult<T> = Result<T, SynthesisError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SynthesisError::InterpretationError {
            message: "Invalid goal format".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to interpret goal: Invalid goal format"
        );

        let err = SynthesisError::PerformanceTargetMissed {
            achieved: 850.5,
            target: 1000.0,
        };
        assert_eq!(
            err.to_string(),
            "Performance target not met: achieved 850.50, target 1000.00"
        );
    }

    #[test]
    fn test_all_error_variants() {
        // InterpretationError
        let err = SynthesisError::InterpretationError {
            message: "Cannot parse goal".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to interpret goal: Cannot parse goal"
        );

        // SynthesisFailure
        let err = SynthesisError::SynthesisFailure {
            message: "Template generation failed".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to synthesize kernel: Template generation failed"
        );

        // CompilationError
        let err = SynthesisError::CompilationError {
            message: "NVCC failed".to_string(),
        };
        assert_eq!(err.to_string(), "Compilation failed: NVCC failed");

        // ExecutionError
        let err = SynthesisError::ExecutionError {
            message: "Kernel launch failed".to_string(),
        };
        assert_eq!(err.to_string(), "Execution failed: Kernel launch failed");

        // TemplateNotFound
        let err = SynthesisError::TemplateNotFound {
            name: "matrix_multiply".to_string(),
        };
        assert_eq!(err.to_string(), "Template not found: matrix_multiply");

        // InvalidSpecification
        let err = SynthesisError::InvalidSpecification {
            message: "Missing required fields".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid specification: Missing required fields"
        );

        // LlmApiError
        let err = SynthesisError::LlmApiError {
            message: "API timeout".to_string(),
        };
        assert_eq!(err.to_string(), "LLM API error: API timeout");

        // ResourceLimitExceeded
        let err = SynthesisError::ResourceLimitExceeded {
            resource: "GPU memory".to_string(),
        };
        assert_eq!(err.to_string(), "Resource limit exceeded: GPU memory");

        // OptimizationFailed
        let err = SynthesisError::OptimizationFailed { iterations: 100 };
        assert_eq!(err.to_string(), "Optimization failed after 100 iterations");

        // CacheError
        let err = SynthesisError::CacheError {
            message: "Cache corruption detected".to_string(),
        };
        assert_eq!(err.to_string(), "Cache error: Cache corruption detected");

        // Other
        let err = SynthesisError::Other("Custom error message".to_string());
        assert_eq!(err.to_string(), "Synthesis error: Custom error message");
    }

    #[test]
    fn test_performance_target_missed_formatting() {
        // Test various precision scenarios
        let test_cases = vec![
            (100.0, 200.0, "achieved 100.00, target 200.00"),
            (99.99, 100.0, "achieved 99.99, target 100.00"),
            (1000.123, 2000.456, "achieved 1000.12, target 2000.46"),
            (0.0, 1.0, "achieved 0.00, target 1.00"),
            (
                999999.99,
                1000000.0,
                "achieved 999999.99, target 1000000.00",
            ),
        ];

        for (achieved, target, expected) in test_cases {
            let err = SynthesisError::PerformanceTargetMissed { achieved, target };
            let error_str = err.to_string();
            assert!(
                error_str.contains(expected),
                "Expected '{}' to contain '{}'",
                error_str,
                expected
            );
        }
    }

    #[test]
    fn test_error_debug_format() {
        let err = SynthesisError::InterpretationError {
            message: "debug test".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InterpretationError"));
        assert!(debug_str.contains("debug test"));
    }

    #[test]
    fn test_error_chaining() {
        // Test error propagation patterns
        fn inner_function() -> SynthesisResult<()> {
            Err(SynthesisError::CompilationError {
                message: "Inner compilation failed".to_string(),
            })
        }

        fn outer_function() -> SynthesisResult<()> {
            inner_function().map_err(|_| SynthesisError::SynthesisFailure {
                message: "Outer synthesis failed due to compilation".to_string(),
            })
        }

        let result = outer_function();
        assert!(result.is_err());

        match result.unwrap_err() {
            SynthesisError::SynthesisFailure { message } => {
                assert_eq!(message, "Outer synthesis failed due to compilation");
            }
            _ => panic!("Expected SynthesisFailure"),
        }
    }

    #[test]
    fn test_io_error_conversion() {
        use std::io::{self, ErrorKind};

        let io_err = io::Error::new(ErrorKind::NotFound, "file not found");
        let synthesis_err: SynthesisError = io_err.into();

        assert!(matches!(synthesis_err, SynthesisError::IoError(_)));
        assert!(synthesis_err.to_string().contains("IO error"));
    }

    #[test]
    fn test_json_error_conversion() {
        // Create a JSON error by trying to parse invalid JSON
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let synthesis_err: SynthesisError = json_err.into();

        assert!(matches!(synthesis_err, SynthesisError::JsonError(_)));
        assert!(synthesis_err.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_pattern_matching() {
        let errors = vec![
            SynthesisError::InterpretationError {
                message: "test1".to_string(),
            },
            SynthesisError::SynthesisFailure {
                message: "test2".to_string(),
            },
            SynthesisError::TemplateNotFound {
                name: "test3".to_string(),
            },
            SynthesisError::OptimizationFailed { iterations: 42 },
            SynthesisError::Other("test5".to_string()),
        ];

        for (i, err) in errors.into_iter().enumerate() {
            match err {
                SynthesisError::InterpretationError { message } => {
                    assert_eq!(i, 0);
                    assert_eq!(message, "test1");
                }
                SynthesisError::SynthesisFailure { message } => {
                    assert_eq!(i, 1);
                    assert_eq!(message, "test2");
                }
                SynthesisError::TemplateNotFound { name } => {
                    assert_eq!(i, 2);
                    assert_eq!(name, "test3");
                }
                SynthesisError::OptimizationFailed { iterations } => {
                    assert_eq!(i, 3);
                    assert_eq!(iterations, 42);
                }
                SynthesisError::Other(msg) => {
                    assert_eq!(i, 4);
                    assert_eq!(msg, "test5");
                }
                _ => panic!("Unexpected error variant at index {}", i),
            }
        }
    }

    #[test]
    fn test_error_size() {
        use std::mem::size_of;

        // Ensure error enum is reasonably sized
        assert!(size_of::<SynthesisError>() < 256); // Should be compact
        assert!(size_of::<SynthesisResult<()>>() < 300);
    }

    #[test]
    fn test_error_message_variations() {
        let test_messages = vec![
            "",                                                   // Empty message
            "Short",                                              // Short message
            "A".repeat(1000),                                     // Very long message
            "Message with special chars: àáâãäå çčđ ñòóôõö ùúûü", // Unicode
            "Message\nwith\nnewlines",                            // Multiline
            "Message\twith\ttabs",                                // Tabs
        ];

        for message in test_messages {
            let err = SynthesisError::InterpretationError {
                message: message.clone(),
            };
            let error_str = err.to_string();
            assert!(error_str.contains(&message));
            assert!(!error_str.is_empty());
        }
    }

    #[test]
    fn test_performance_target_edge_cases() {
        let edge_cases = vec![
            (0.0, 0.0),
            (f64::MIN, f64::MAX),
            (1.0, 0.0), // Achieved > target (shouldn't happen but handle gracefully)
            (f64::EPSILON, 1.0),
            (1000000.0, 1000000.1), // Very close values
        ];

        for (achieved, target) in edge_cases {
            let err = SynthesisError::PerformanceTargetMissed { achieved, target };
            let error_str = err.to_string();
            assert!(error_str.contains("Performance target not met"));
            assert!(error_str.contains(&format!("{:.2}", achieved)));
            assert!(error_str.contains(&format!("{:.2}", target)));
        }
    }

    #[test]
    fn test_error_equality_through_string() {
        // Since errors don't implement PartialEq, test through string representation
        let err1 = SynthesisError::InterpretationError {
            message: "same message".to_string(),
        };
        let err2 = SynthesisError::InterpretationError {
            message: "same message".to_string(),
        };

        assert_eq!(err1.to_string(), err2.to_string());

        let err3 = SynthesisError::InterpretationError {
            message: "different message".to_string(),
        };

        assert_ne!(err1.to_string(), err3.to_string());
    }

    #[test]
    fn test_result_type_usage() {
        fn success_function() -> SynthesisResult<i32> {
            Ok(42)
        }

        fn error_function() -> SynthesisResult<i32> {
            Err(SynthesisError::Other("test error".to_string()))
        }

        // Test success case
        let result = success_function();
        assert!(result.is_ok());
        assert_eq!(result?, 42);

        // Test error case
        let result = error_function();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SynthesisError::Other(_)));
    }

    #[test]
    fn test_error_propagation_in_result_chains() {
        fn step1() -> SynthesisResult<i32> {
            Ok(10)
        }

        fn step2(value: i32) -> SynthesisResult<i32> {
            if value < 20 {
                Err(SynthesisError::InvalidSpecification {
                    message: "Value too small".to_string(),
                })
            } else {
                Ok(value * 2)
            }
        }

        let result = step1().and_then(step2);
        assert!(result.is_err());

        match result.unwrap_err() {
            SynthesisError::InvalidSpecification { message } => {
                assert_eq!(message, "Value too small");
            }
            _ => panic!("Expected InvalidSpecification"),
        }
    }

    #[test]
    fn test_complex_error_scenarios() {
        // Test nested error scenarios
        let scenarios = vec![
            (
                SynthesisError::CompilationError {
                    message: "NVCC version 11.2 required, found 10.1".to_string(),
                },
                "NVCC version 11.2 required",
            ),
            (
                SynthesisError::ResourceLimitExceeded {
                    resource: "Shared memory: 48KB requested, 32KB available".to_string(),
                },
                "48KB requested",
            ),
            (
                SynthesisError::LlmApiError {
                    message: "Rate limit exceeded: 1000 requests/hour".to_string(),
                },
                "Rate limit exceeded",
            ),
        ];

        for (error, expected_content) in scenarios {
            let error_str = error.to_string();
            assert!(
                error_str.contains(expected_content),
                "Error '{}' should contain '{}'",
                error_str,
                expected_content
            );
        }
    }

    #[test]
    fn test_error_source_chain() {
        use std::error::Error;

        // Test that IO errors maintain their source chain
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let synthesis_err: SynthesisError = io_err.into();

        // The source should be available
        assert!(synthesis_err.source().is_some());
    }

    #[test]
    fn test_optimization_iterations_variations() {
        let iteration_counts = vec![0, 1, 10, 100, 1000, 10000, u32::MAX];

        for count in iteration_counts {
            let err = SynthesisError::OptimizationFailed { iterations: count };
            let error_str = err.to_string();
            assert!(error_str.contains(&format!("after {} iterations", count)));
        }
    }

    #[test]
    fn test_template_name_variations() {
        let template_names = vec![
            "simple_kernel",
            "MatrixMultiply_FP16_TensorCore",
            "conv2d_winograd_3x3",
            "fft_1024_point_batched",
            "reduce_sum_fp32_warp_optimized",
            "", // Empty template name
            "template-with-dashes",
            "template.with.dots",
            "TEMPLATE_ALL_CAPS",
            "templateWithCamelCase",
        ];

        for name in template_names {
            let err = SynthesisError::TemplateNotFound {
                name: name.to_string(),
            };
            let error_str = err.to_string();
            assert!(error_str.contains("Template not found:"));
            if !name.is_empty() {
                assert!(error_str.contains(name));
            }
        }
    }

    #[test]
    fn test_cache_error_scenarios() {
        let cache_scenarios = vec![
            "Cache miss for kernel hash abc123def456",
            "Cache eviction failed: disk full",
            "Cache corruption detected: invalid checksum",
            "Cache lock timeout after 30 seconds",
            "Cache size limit exceeded: 10GB max",
        ];

        for scenario in cache_scenarios {
            let err = SynthesisError::CacheError {
                message: scenario.to_string(),
            };
            let error_str = err.to_string();
            assert!(error_str.contains("Cache error:"));
            assert!(error_str.contains(scenario));
        }
    }

    #[test]
    fn test_error_memory_usage() {
        // Test that creating many errors doesn't cause memory issues
        let mut errors = Vec::new();

        for i in 0..1000 {
            let err = SynthesisError::InterpretationError {
                message: format!("Error number {i}"),
            };
            errors.push(err);
        }

        // Verify all errors are distinct
        for (i, err) in errors.iter().enumerate() {
            let expected_msg = format!("Error number {i}");
            assert!(err.to_string().contains(&expected_msg));
        }
    }
}
