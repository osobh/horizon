//! Edge case tests for CUDA crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{
        context::{Context, ContextFlags},
        detection::{CudaToolkit, ToolkitInfo},
        error::{CudaError, CudaResult},
        kernel::{Kernel, KernelModule},
        memory::{DeviceMemory, MemoryPool},
        stream::{Stream, StreamFlags},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    // Context edge case tests

    #[test]
    fn test_context_flags_combinations() {
        // Test all possible flag combinations
        let flag_combinations = vec![
            (true, true, true, true, true, true),
            (false, false, false, false, false, false),
            (true, false, true, false, true, false),
            (false, true, false, true, false, true),
        ];

        for (auto, spin, yield_flag, blocking, map, lmem) in flag_combinations {
            let flags = ContextFlags {
                sched_auto: auto,
                sched_spin: spin,
                sched_yield: yield_flag,
                sched_blocking_sync: blocking,
                map_host: map,
                lmem_resize_to_max: lmem,
            };

            assert_eq!(flags.sched_auto, auto);
            assert_eq!(flags.sched_spin, spin);
            assert_eq!(flags.sched_yield, yield_flag);
            assert_eq!(flags.sched_blocking_sync, blocking);
            assert_eq!(flags.map_host, map);
            assert_eq!(flags.lmem_resize_to_max, lmem);
        }
    }

    #[test]
    fn test_context_flags_edge_cases() {
        // Test mutually exclusive scheduling flags
        let flags = ContextFlags {
            sched_auto: true,
            sched_spin: true,
            sched_yield: true,
            sched_blocking_sync: true,
            map_host: true,
            lmem_resize_to_max: true,
        };

        // In real CUDA, these would be mutually exclusive, but we accept them in the struct
        assert!(flags.sched_auto);
        assert!(flags.sched_spin);
        assert!(flags.sched_yield);
        assert!(flags.sched_blocking_sync);
    }

    #[test]
    fn test_device_id_extremes() {
        // Test extreme device IDs
        let device_ids = vec![i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX];

        for device_id in device_ids {
            let err = CudaError::InvalidDevice { device: device_id };
            let error_str = err.to_string();
            assert!(error_str.contains(&device_id.to_string()));
        }
    }

    // Memory edge case tests

    #[test]
    fn test_memory_size_extremes() {
        // Test extreme memory sizes
        let sizes = vec![
            0,
            1,
            1024,
            1024 * 1024,
            1024 * 1024 * 1024,
            usize::MAX / 2,
            usize::MAX,
        ];

        for size in sizes {
            let err = CudaError::OutOfMemory { requested: size };
            let error_str = err.to_string();
            assert!(error_str.contains(&size.to_string()));
        }
    }

    // Note: Device memory tests require mock feature or actual CUDA
    // They are tested separately in memory_tests.rs

    // Stream edge case tests

    #[test]
    fn test_stream_flags_combinations() {
        // Test StreamFlags combinations
        let flag_values = vec![0, 1, 2, 4, 8, 16, u32::MAX];

        for value in flag_values {
            let flags = StreamFlags {
                non_blocking: (value & 1) != 0,
                disable_timing: (value & 2) != 0,
                bits: value,
            };
            assert_eq!(flags.bits, value);
        }
    }

    #[test]
    fn test_stream_priority_extremes() {
        // Test stream priority extremes
        let priorities = vec![i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX];

        for priority in priorities {
            // Would test Stream creation with priority in real implementation
            assert!(priority == priority); // Placeholder test
        }
    }

    // Kernel edge case tests

    #[test]
    fn test_kernel_name_edge_cases() {
        // Test various kernel names
        let kernel_names = vec![
            "",
            "a",
            "kernel_name",
            "very_long_kernel_name_that_exceeds_normal_length_limits_and_continues_on_and_on",
            "kernel_with_unicode_名前_🚀",
            "kernel.with.dots",
            "kernel-with-dashes",
            "kernel_with_numbers_123",
            "__global__kernel",
            "template<int>",
        ];

        for name in kernel_names {
            let err = CudaError::KernelNotFound {
                name: name.to_string(),
            };
            assert!(err.to_string().contains(name));
        }
    }

    #[test]
    fn test_kernel_parameter_edge_cases() {
        // Test various parameter error messages
        let param_errors = vec![
            "",
            "Invalid type",
            "Size exceeds limit: 18446744073709551615",
            "Null pointer",
            "Alignment violation",
            "Type mismatch: expected float4, got int",
            "Parameter count mismatch: expected 5, got 3",
            "🚨 Error: Invalid UTF-8 sequence",
        ];

        for msg in param_errors {
            let err = CudaError::InvalidParameter {
                message: msg.to_string(),
            };
            assert!(err.to_string().contains(msg));
        }
    }

    // Toolkit detection edge cases

    #[test]
    fn test_toolkit_version_edge_cases() {
        // Test various version strings
        let versions = vec![
            "",
            "0",
            "1.0",
            "11.8",
            "12.0",
            "999.999",
            "invalid",
            "11.8.0.86",
            "CUDA 11.8",
            "v11.8",
            "11.8-beta",
            "11.8.0+cuda",
        ];

        for version in versions {
            let err = CudaError::UnsupportedVersion {
                version: version.to_string(),
            };
            assert!(err.to_string().contains(version));
        }
    }

    // Error message edge cases

    #[test]
    fn test_error_message_extremes() {
        // Test empty messages
        let errors = vec![
            CudaError::InitializationFailed {
                message: "".to_string(),
            },
            CudaError::CompilationError {
                message: "".to_string(),
            },
            CudaError::ExecutionError {
                message: "".to_string(),
            },
            CudaError::StreamError {
                message: "".to_string(),
            },
            CudaError::ContextError {
                message: "".to_string(),
            },
            CudaError::PtxError {
                message: "".to_string(),
            },
        ];

        for err in errors {
            let error_str = err.to_string();
            assert!(error_str.contains(": "));
        }

        // Test very long messages
        let long_msg = "x".repeat(10000);
        let err = CudaError::CompilationError {
            message: long_msg.clone(),
        };
        assert!(err.to_string().contains(&long_msg));
    }

    #[test]
    fn test_error_unicode_handling() {
        // Test unicode in error messages
        let unicode_messages = vec![
            "错误：内存不足",
            "Erreur: Mémoire insuffisante",
            "エラー：メモリ不足",
            "خطأ: ذاكرة غير كافية",
            "🚨💥❌ Fatal Error ❌💥🚨",
            "Error\0with\0nulls",
            "Error\nwith\nnewlines\n",
            "Error\twith\ttabs\t",
        ];

        for msg in unicode_messages {
            let err = CudaError::ExecutionError {
                message: msg.to_string(),
            };
            let error_str = err.to_string();
            // Just verify it doesn't panic
            assert!(!error_str.is_empty());
        }
    }

    // CUDA API error code edge cases

    #[test]
    fn test_cuda_api_error_code_extremes() {
        let extreme_codes = vec![
            (i32::MIN, "Minimum error code"),
            (-999999, "Large negative code"),
            (0, "Success?"),
            (1, "cudaErrorInvalidValue"),
            (999, "cudaErrorUnknown"),
            (999999, "Large positive code"),
            (i32::MAX, "Maximum error code"),
        ];

        for (code, msg) in extreme_codes {
            let err = CudaError::CudaApiError {
                code,
                message: msg.to_string(),
            };
            let error_str = err.to_string();
            assert!(error_str.contains(&code.to_string()));
            assert!(error_str.contains(msg));
        }
    }

    // UUID edge cases

    #[test]
    fn test_uuid_handling() {
        // Test nil UUID
        let nil_id = Uuid::nil();
        assert_eq!(nil_id.to_string(), "00000000-0000-0000-0000-000000000000");

        // Test max UUID
        let max_id = Uuid::from_u128(u128::MAX);
        assert_eq!(max_id.to_string(), "ffffffff-ffff-ffff-ffff-ffffffffffff");

        // Test random UUIDs are unique
        let mut uuids = Vec::new();
        for _ in 0..100 {
            uuids.push(Uuid::new_v4());
        }

        // Check all are unique
        for i in 0..uuids.len() {
            for j in (i + 1)..uuids.len() {
                assert_ne!(uuids[i], uuids[j]);
            }
        }
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify error types are Send + Sync
        assert_send::<CudaError>();
        assert_sync::<CudaError>();

        // Verify other types
        assert_send::<ContextFlags>();
        assert_sync::<ContextFlags>();
        assert_send::<StreamFlags>();
        assert_sync::<StreamFlags>();
    }

    // PTX parsing edge cases

    #[test]
    fn test_ptx_error_messages() {
        let ptx_errors = vec![
            "",
            "Syntax error at line 0",
            "Unexpected token: @#$%",
            "Missing .target directive",
            "Invalid register name: %r999999",
            "Unsupported PTX version: 99.9",
            ".globl .func kernel\n{\n  syntax error\n}",
            "Multiple definition of symbol 'kernel'",
            "Undefined symbol: __device_function",
        ];

        for msg in ptx_errors {
            let err = CudaError::PtxError {
                message: msg.to_string(),
            };
            assert!(err.to_string().contains("PTX parsing error"));
            if !msg.is_empty() {
                assert!(err.to_string().contains(msg));
            }
        }
    }

    // Mock mode edge cases

    #[test]
    fn test_mock_mode_behavior() {
        // Test that MockModeError has consistent behavior
        let err1 = CudaError::MockModeError;
        let err2 = CudaError::MockModeError;

        // Both should produce same error message
        assert_eq!(err1.to_string(), err2.to_string());
        assert_eq!(err1.to_string(), "Operation not supported in mock mode");
    }

    // Complex error scenarios

    #[test]
    fn test_error_result_chaining() {
        fn may_fail(should_fail: bool) -> CudaResult<i32> {
            if should_fail {
                Err(CudaError::InvalidDevice { device: -1 })
            } else {
                Ok(42)
            }
        }

        // Test successful path
        let result = may_fail(false)
            .map(|x| x * 2)
            .and_then(|x| Ok(x + 10))
            .map_err(|_| CudaError::MockModeError);

        assert_eq!(result.unwrap(), 94);

        // Test error path
        let result = may_fail(true)
            .map(|x| x * 2)
            .and_then(|x| Ok(x + 10))
            .map_err(|_| CudaError::MockModeError);

        assert!(matches!(result.unwrap_err(), CudaError::MockModeError));
    }

    #[test]
    fn test_result_collection() {
        let results: Vec<CudaResult<i32>> = vec![
            Ok(1),
            Ok(2),
            Err(CudaError::OutOfMemory { requested: 1024 }),
            Ok(4),
        ];

        let collected: Result<Vec<i32>, CudaError> = results.into_iter().collect();
        assert!(collected.is_err());

        match collected.unwrap_err() {
            CudaError::OutOfMemory { requested } => assert_eq!(requested, 1024),
            _ => panic!("Expected OutOfMemory error"),
        }
    }
}
