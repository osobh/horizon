//! Tests for CUDA error handling

use crate::error::*;

#[test]
fn test_cuda_error_display() {
    let errors = vec![
        (CudaError::NotInitialized, "CUDA not initialized"),
        (CudaError::ToolkitNotFound, "CUDA toolkit not found"),
        (CudaError::InvalidDevice { id: 5 }, "Invalid CUDA device: 5"),
        (
            CudaError::OutOfMemory { requested: 1024 },
            "Out of GPU memory (requested 1024 bytes)",
        ),
        (
            CudaError::LaunchFailed {
                kernel: "test_kernel".to_string(),
            },
            "Kernel launch failed: test_kernel",
        ),
        (
            CudaError::InvalidValue {
                param: "block_size".to_string(),
            },
            "Invalid parameter value: block_size",
        ),
    ];

    for (error, expected_msg) in errors {
        assert_eq!(error.to_string(), expected_msg);
    }
}

#[test]
fn test_all_cuda_error_variants() {
    // NotInitialized
    let err = CudaError::NotInitialized;
    assert_eq!(err.to_string(), "CUDA not initialized");

    // ToolkitNotFound
    let err = CudaError::ToolkitNotFound;
    assert_eq!(err.to_string(), "CUDA toolkit not found");

    // UnsupportedVersion
    let err = CudaError::UnsupportedVersion {
        version: "10.0".to_string(),
    };
    assert_eq!(err.to_string(), "Unsupported CUDA version: 10.0");

    // InvalidDevice
    let err = CudaError::InvalidDevice { id: 999 };
    assert_eq!(err.to_string(), "Invalid CUDA device: 999");

    // OutOfMemory
    let err = CudaError::OutOfMemory {
        requested: 1073741824,
    };
    assert_eq!(
        err.to_string(),
        "Out of GPU memory (requested 1073741824 bytes)"
    );

    // LaunchFailed
    let err = CudaError::LaunchFailed {
        kernel: "matrix_multiply".to_string(),
    };
    assert_eq!(err.to_string(), "Kernel launch failed: matrix_multiply");

    // InvalidValue
    let err = CudaError::InvalidValue {
        param: "grid_size".to_string(),
    };
    assert_eq!(err.to_string(), "Invalid parameter value: grid_size");

    // InvalidAddress
    let err = CudaError::InvalidAddress {
        address: 0xDEADBEEF,
    };
    assert_eq!(err.to_string(), "Invalid memory address: 0xdeadbeef");

    // InitializationFailed
    let err = CudaError::InitializationFailed {
        message: "Failed to create context".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "CUDA initialization failed: Failed to create context"
    );

    // SynchronizationError
    let err = CudaError::SynchronizationError;
    assert_eq!(err.to_string(), "CUDA synchronization error");

    // CompilationFailed
    let err = CudaError::CompilationFailed {
        source: "kernel.cu".to_string(),
        error: "syntax error".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "CUDA compilation failed for kernel.cu: syntax error"
    );

    // RuntimeError
    let err = CudaError::RuntimeError {
        code: 700,
        message: "Unknown error".to_string(),
    };
    assert_eq!(err.to_string(), "CUDA runtime error (700): Unknown error");

    // Other
    let err = CudaError::Other("Custom error message".to_string());
    assert_eq!(err.to_string(), "CUDA error: Custom error message");
}

#[test]
fn test_cuda_result_type() {
    fn success_function() -> CudaResult<i32> {
        Ok(42)
    }

    fn error_function() -> CudaResult<i32> {
        Err(CudaError::NotInitialized)
    }

    assert!(success_function().is_ok());
    assert_eq!(success_function()?, 42);

    assert!(error_function().is_err());
    assert!(matches!(
        error_function().unwrap_err(),
        CudaError::NotInitialized
    ));
}

#[test]
fn test_error_debug_format() {
    let err = CudaError::InvalidDevice { id: 123 };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("InvalidDevice"));
    assert!(debug_str.contains("123"));
}

#[test]
fn test_error_pattern_matching() {
    let errors = vec![
        CudaError::NotInitialized,
        CudaError::InvalidDevice { id: 1 },
        CudaError::OutOfMemory { requested: 1024 },
        CudaError::Other("test".to_string()),
    ];

    for err in errors {
        match err {
            CudaError::NotInitialized => assert_eq!(err.to_string(), "CUDA not initialized"),
            CudaError::InvalidDevice { id } => assert_eq!(id, 1),
            CudaError::OutOfMemory { requested } => assert_eq!(requested, 1024),
            CudaError::Other(msg) => assert_eq!(msg, "test"),
            _ => panic!("Unexpected error variant"),
        }
    }
}

#[test]
fn test_error_chaining() {
    fn inner_function() -> CudaResult<()> {
        Err(CudaError::OutOfMemory { requested: 1024 })
    }

    fn outer_function() -> CudaResult<()> {
        inner_function().map_err(|_| CudaError::LaunchFailed {
            kernel: "failed due to OOM".to_string(),
        })
    }

    let result = outer_function();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        CudaError::LaunchFailed { .. }
    ));
}

#[test]
fn test_result_unwrap_operations() {
    let ok_result: CudaResult<i32> = Ok(100);
    assert_eq!(ok_result.unwrap_or(0), 100);
    assert_eq!(ok_result.unwrap_or_else(|_| 0), 100);

    let err_result: CudaResult<i32> = Err(CudaError::NotInitialized);
    assert_eq!(err_result.unwrap_or(0), 0);
    assert_eq!(err_result.unwrap_or_else(|_| -1), -1);
}

#[test]
fn test_result_map_operations() {
    let ok_result: CudaResult<i32> = Ok(10);
    let mapped = ok_result.map(|x| x * 2);
    assert_eq!(mapped.unwrap(), 20);

    let err_result: CudaResult<i32> = Err(CudaError::NotInitialized);
    let mapped = err_result.map(|x| x * 2);
    assert!(mapped.is_err());
}

#[test]
fn test_error_size() {
    use std::mem::size_of;

    // Ensure error enum is reasonably sized
    assert!(size_of::<CudaError>() <= 64); // Should be compact
}

#[test]
fn test_memory_error_formatting() {
    let sizes = vec![
        1024,               // 1 KB
        1024 * 1024,        // 1 MB
        1024 * 1024 * 1024, // 1 GB
        u64::MAX as usize,  // Max size
    ];

    for size in sizes {
        let err = CudaError::OutOfMemory { requested: size };
        let msg = err.to_string();
        assert!(msg.contains(&size.to_string()));
    }
}

#[test]
fn test_address_formatting() {
    let addresses = vec![0x0, 0xDEADBEEF, 0xCAFEBABE, usize::MAX];

    for addr in addresses {
        let err = CudaError::InvalidAddress { address: addr };
        let msg = err.to_string();
        assert!(msg.contains(&format!("0x{:x}", addr)));
    }
}

#[test]
fn test_runtime_error_codes() {
    let error_codes = vec![
        (1, "InvalidValue"),
        (2, "MemoryAllocation"),
        (700, "IllegalAddress"),
        (999, "Unknown"),
    ];

    for (code, desc) in error_codes {
        let err = CudaError::RuntimeError {
            code,
            message: desc.to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains(&code.to_string()));
        assert!(msg.contains(desc));
    }
}

#[test]
fn test_compilation_error_details() {
    let test_cases = vec![
        ("kernel.cu", "syntax error at line 42"),
        ("device_function.cuh", "undefined symbol: atomicAdd"),
        ("complex_kernel.ptx", "invalid PTX version"),
    ];

    for (source, error) in test_cases {
        let err = CudaError::CompilationFailed {
            source: source.to_string(),
            error: error.to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains(source));
        assert!(msg.contains(error));
    }
}

#[test]
fn test_error_conversion() {
    // Test conversion from other error types if implemented
    fn returns_io_error() -> Result<(), std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ))
    }

    fn convert_to_cuda_error() -> CudaResult<()> {
        returns_io_error().map_err(|e| CudaError::Other(e.to_string()))
    }

    let result = convert_to_cuda_error();
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), CudaError::Other(_)));
}

#[test]
fn test_question_mark_operator() {
    fn may_fail(should_fail: bool) -> CudaResult<i32> {
        if should_fail {
            Err(CudaError::NotInitialized)
        } else {
            Ok(42)
        }
    }

    fn using_question_mark(should_fail: bool) -> CudaResult<i32> {
        let value = may_fail(should_fail)?;
        Ok(value * 2)
    }

    assert_eq!(using_question_mark(false)?, 84);
    assert!(using_question_mark(true).is_err());
}

#[test]
fn test_error_equality() {
    // Note: CudaError might not implement PartialEq, but we can test string representations
    let err1 = CudaError::InvalidDevice { id: 5 };
    let err2 = CudaError::InvalidDevice { id: 5 };

    assert_eq!(err1.to_string(), err2.to_string());
}

#[test]
fn test_nested_results() {
    type NestedResult = CudaResult<CudaResult<i32>>;

    let nested_ok: NestedResult = Ok(Ok(42));
    assert!(nested_ok.is_ok());
    assert!(nested_ok?.is_ok());
    assert_eq!(nested_ok??, 42);

    let nested_inner_err: NestedResult = Ok(Err(CudaError::NotInitialized));
    assert!(nested_inner_err.is_ok());
    assert!(nested_inner_err?.is_err());

    let nested_outer_err: NestedResult = Err(CudaError::ToolkitNotFound);
    assert!(nested_outer_err.is_err());
}

#[test]
fn test_collect_results() {
    let results: Vec<CudaResult<i32>> = vec![Ok(1), Ok(2), Ok(3)];

    let collected: CudaResult<Vec<i32>> = results.into_iter().collect();
    assert!(collected.is_ok());
    assert_eq!(collected?, vec![1, 2, 3]);

    let results_with_error: Vec<CudaResult<i32>> =
        vec![Ok(1), Err(CudaError::NotInitialized), Ok(3)];

    let collected: CudaResult<Vec<i32>> = results_with_error.into_iter().collect();
    assert!(collected.is_err());
}

#[test]
fn test_error_context_information() {
    // Test that errors contain enough context for debugging
    let err = CudaError::LaunchFailed {
        kernel: "my_complex_kernel<<<256,128>>>".to_string(),
    };

    let msg = err.to_string();
    assert!(msg.contains("my_complex_kernel"));
    assert!(msg.contains("<<<256,128>>>"));
}

#[test]
fn test_error_thread_safety() {
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                let err = CudaError::InvalidDevice { id: i };
                err.to_string()
            })
        })
        .collect();

    for handle in handles {
        let _result = handle.join().unwrap();
    }
}
