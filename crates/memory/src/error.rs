//! Memory allocation error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Invalid allocation size: {size} bytes")]
    InvalidSize { size: usize },

    #[error("Memory handle not found: {id}")]
    HandleNotFound { id: uuid::Uuid },

    #[cfg(feature = "cuda")]
    #[error("GPU context error: {0}")]
    GpuContext(#[from] cust::error::CudaError),

    #[error("Memory allocation failed: {reason}")]
    AllocationFailed { reason: String },

    #[error("Memory pool initialization failed: {reason}")]
    PoolInitFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_out_of_memory_error_display() {
        let error = MemoryError::OutOfMemory {
            requested: 2048,
            available: 1024,
        };

        let message = error.to_string();
        assert!(message.contains("Out of GPU memory"));
        assert!(message.contains("2048"));
        assert!(message.contains("1024"));
    }

    #[test]
    fn test_invalid_size_error_display() {
        let error = MemoryError::InvalidSize { size: 0 };

        let message = error.to_string();
        assert!(message.contains("Invalid allocation size"));
        assert!(message.contains("0 bytes"));
    }

    #[test]
    fn test_handle_not_found_error_display() {
        let id = uuid::Uuid::new_v4();
        let error = MemoryError::HandleNotFound { id };

        let message = error.to_string();
        assert!(message.contains("Memory handle not found"));
        assert!(message.contains(&id.to_string()));
    }

    #[test]
    fn test_allocation_failed_error_display() {
        let error = MemoryError::AllocationFailed {
            reason: "CUDA driver not available".to_string(),
        };

        let message = error.to_string();
        assert!(message.contains("Memory allocation failed"));
        assert!(message.contains("CUDA driver not available"));
    }

    #[test]
    fn test_pool_init_failed_error_display() {
        let error = MemoryError::PoolInitFailed {
            reason: "Mutex poisoned".to_string(),
        };

        let message = error.to_string();
        assert!(message.contains("Memory pool initialization failed"));
        assert!(message.contains("Mutex poisoned"));
    }

    #[test]
    fn test_error_debug_formatting() {
        let errors = vec![
            MemoryError::OutOfMemory {
                requested: 1024,
                available: 512,
            },
            MemoryError::InvalidSize { size: 0 },
            MemoryError::HandleNotFound {
                id: uuid::Uuid::new_v4(),
            },
            MemoryError::AllocationFailed {
                reason: "Test error".to_string(),
            },
            MemoryError::PoolInitFailed {
                reason: "Test pool error".to_string(),
            },
        ];

        for error in errors {
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
            // Debug format shows variant name, not enum name
            assert!(
                debug_str.contains("OutOfMemory")
                    || debug_str.contains("InvalidSize")
                    || debug_str.contains("HandleNotFound")
                    || debug_str.contains("AllocationFailed")
                    || debug_str.contains("PoolInitFailed")
            );
        }
    }

    #[test]
    fn test_error_conversion_to_trait_object() {
        let error: Box<dyn std::error::Error> = Box::new(MemoryError::InvalidSize { size: 42 });
        assert!(error.to_string().contains("Invalid allocation size"));
    }

    #[test]
    fn test_error_with_large_values() {
        let error = MemoryError::OutOfMemory {
            requested: usize::MAX,
            available: usize::MAX / 2,
        };

        let message = error.to_string();
        assert!(message.contains(&usize::MAX.to_string()));
    }

    #[test]
    fn test_error_with_empty_reason() {
        let error = MemoryError::AllocationFailed {
            reason: String::new(),
        };

        let message = error.to_string();
        assert!(message.contains("Memory allocation failed"));
        // Should handle empty reason gracefully
    }

    #[test]
    fn test_error_with_unicode_reason() {
        let error = MemoryError::PoolInitFailed {
            reason: "Ошибка инициализации пула памяти".to_string(),
        };

        let message = error.to_string();
        assert!(message.contains("Memory pool initialization failed"));
        assert!(message.contains("Ошибка"));
    }

    #[test]
    fn test_error_equality() {
        let id = uuid::Uuid::new_v4();

        let error1 = MemoryError::HandleNotFound { id };
        let error2 = MemoryError::HandleNotFound { id };

        // Errors with same data should format the same way
        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_error_send_sync() {
        // Test that MemoryError is Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<MemoryError>();
        assert_sync::<MemoryError>();
    }

    #[test]
    fn test_error_clone_partial_eq() {
        // Since MemoryError doesn't derive Clone or PartialEq,
        // we test that it can be converted to string for comparison
        let error1 = MemoryError::InvalidSize { size: 100 };
        let error2 = MemoryError::InvalidSize { size: 100 };

        assert_eq!(error1.to_string(), error2.to_string());
    }

    #[test]
    fn test_allocation_failed_with_long_reason() {
        let long_reason = "x".repeat(1000);
        let error = MemoryError::AllocationFailed {
            reason: long_reason.clone(),
        };

        let message = error.to_string();
        assert!(message.contains(&long_reason));
    }

    #[test]
    fn test_zero_values() {
        let error = MemoryError::OutOfMemory {
            requested: 0,
            available: 0,
        };

        let message = error.to_string();
        assert!(message.contains("requested 0 bytes"));
        assert!(message.contains("available 0 bytes"));
    }

    #[test]
    fn test_error_source() {
        let error = MemoryError::AllocationFailed {
            reason: "Root cause".to_string(),
        };

        // Test that the error can be used as error source
        assert!(error.source().is_none()); // This error doesn't wrap another error
    }
}
