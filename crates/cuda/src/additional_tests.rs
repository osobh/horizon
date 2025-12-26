//! Additional tests for CUDA crate to enhance coverage to 90%+

#[cfg(test)]
mod tests {
    use crate::detection::CudaVersion;
    use crate::*;

    // Test all ContextFlags combinations
    #[test]
    fn test_context_flags_comprehensive() {
        // Test default
        let default_flags = ContextFlags::default();
        assert!(default_flags.sched_auto);
        assert!(!default_flags.sched_spin);

        // Test all combinations
        for i in 0..64 {
            let flags = ContextFlags {
                sched_auto: (i & 1) != 0,
                sched_spin: (i & 2) != 0,
                sched_yield: (i & 4) != 0,
                sched_blocking_sync: (i & 8) != 0,
                map_host: (i & 16) != 0,
                lmem_resize_to_max: (i & 32) != 0,
            };

            // Clone and equality
            let cloned = flags.clone();
            assert_eq!(flags.sched_auto, cloned.sched_auto);
            assert_eq!(flags.sched_spin, cloned.sched_spin);
        }
    }

    // Test StreamFlags edge cases
    #[test]
    fn test_stream_flags_comprehensive() {
        let default_flags = StreamFlags::default();
        assert!(default_flags.non_blocking);
        assert!(!default_flags.disable_timing);
        assert_eq!(default_flags.bits, 0);

        // Test different bit patterns
        for bits in [0, 1, 2, 4, 8, 16, 255, u32::MAX] {
            let flags = StreamFlags {
                non_blocking: (bits & 1) != 0,
                disable_timing: (bits & 2) != 0,
                bits,
            };
            assert_eq!(flags.bits, bits);
        }
    }

    // Test SourceType enum
    #[test]
    fn test_source_type_comprehensive() {
        use kernel::SourceType;

        let types = vec![SourceType::CudaC, SourceType::Ptx, SourceType::Cubin];

        for (i, src_type) in types.iter().enumerate() {
            match src_type {
                SourceType::CudaC => assert_eq!(i, 0),
                SourceType::Ptx => assert_eq!(i, 1),
                SourceType::Cubin => assert_eq!(i, 2),
            }
        }
    }

    // Test CompileOptions
    #[test]
    fn test_compile_options_comprehensive() {
        let default_opts = CompileOptions::default();
        assert_eq!(default_opts.arch, "sm_80");
        assert_eq!(default_opts.opt_level, 3);
        assert!(default_opts.fast_math);

        // Test various configurations
        let configs = vec![
            ("sm_60", 0, false, Some(16)),
            ("sm_70", 1, true, Some(32)),
            ("sm_80", 2, false, None),
            ("sm_90", 3, true, Some(64)),
        ];

        for (arch, opt, fast, regs) in configs {
            let opts = CompileOptions {
                arch: arch.to_string(),
                opt_level: opt,
                fast_math: fast,
                max_registers: regs,
                extra_flags: vec![],
            };
            assert_eq!(opts.arch, arch);
            assert_eq!(opts.opt_level, opt);
        }
    }

    // Test KernelMetadata
    #[test]
    fn test_kernel_metadata_comprehensive() {
        let metadata = KernelMetadata {
            name: "test_kernel".to_string(),
            source_type: SourceType::Ptx,
            compile_options: CompileOptions::default(),
            registers_used: 32,
            shared_memory: 4096,
            constant_memory: 1024,
            local_memory: 512,
            max_threads: 1024,
        };

        assert_eq!(metadata.name, "test_kernel");
        assert_eq!(metadata.registers_used, 32);
        assert_eq!(metadata.shared_memory, 4096);

        // Test with extreme values
        let extreme_metadata = KernelMetadata {
            name: String::new(),
            source_type: SourceType::Cubin,
            compile_options: CompileOptions {
                arch: "compute_90".to_string(),
                opt_level: 0,
                fast_math: false,
                max_registers: Some(255),
                extra_flags: vec!["-g".to_string(); 10],
            },
            registers_used: 255,
            shared_memory: 49152,
            constant_memory: 65536,
            local_memory: 0,
            max_threads: 2048,
        };

        assert!(extreme_metadata.name.is_empty());
        assert_eq!(extreme_metadata.compile_options.extra_flags.len(), 10);
    }

    // Test CudaVersion
    #[test]
    fn test_cuda_version_comprehensive() {
        let v1 = CudaVersion::new(11, 8, 0);
        assert_eq!(v1.to_string(), "11.8.0");

        let v2 = CudaVersion::new(12, 0, 1);
        assert!(v2.meets_minimum(11, 0));
        assert!(v2.meets_minimum(12, 0));
        assert!(!v2.meets_minimum(12, 1));

        // Edge cases
        let v3 = CudaVersion::new(0, 0, 0);
        assert_eq!(v3.to_string(), "0.0.0");
        assert!(!v3.meets_minimum(1, 0));

        let v4 = CudaVersion::new(999, 999, 999);
        assert!(v4.meets_minimum(999, 999));
    }

    // Test ToolkitInfo
    #[test]
    fn test_toolkit_info_comprehensive() {
        let info = ToolkitInfo {
            version: CudaVersion::new(11, 8, 0),
            path: std::path::PathBuf::from("/usr/local/cuda"),
            device_count: 4,
            is_mock: true,
            driver_version: "525.105.17".to_string(),
        };

        assert_eq!(info.version.major, 11);
        assert_eq!(info.device_count, 4);
        assert!(info.is_mock);

        // Test with edge values
        let edge_info = ToolkitInfo {
            version: CudaVersion::new(0, 0, 0),
            path: std::path::PathBuf::new(),
            device_count: 0,
            is_mock: false,
            driver_version: String::new(),
        };

        assert_eq!(edge_info.device_count, 0);
        assert!(edge_info.path.as_os_str().is_empty());
    }

    // Test error variants comprehensively
    #[test]
    fn test_error_variants_comprehensive() {
        use std::error::Error;

        // Test all variants exist and display correctly
        let errors: Vec<(CudaError, &str)> = vec![
            (CudaError::ToolkitNotFound, "toolkit not found"),
            (
                CudaError::UnsupportedVersion {
                    version: "9.0".to_string(),
                },
                "9.0",
            ),
            (
                CudaError::InitializationFailed {
                    message: "test".to_string(),
                },
                "test",
            ),
            (CudaError::InvalidDevice { device: -1 }, "-1"),
            (CudaError::OutOfMemory { requested: 1024 }, "1024"),
            (
                CudaError::CompilationError {
                    message: "syntax".to_string(),
                },
                "syntax",
            ),
            (
                CudaError::KernelNotFound {
                    name: "kernel".to_string(),
                },
                "kernel",
            ),
            (
                CudaError::InvalidParameter {
                    message: "param".to_string(),
                },
                "param",
            ),
            (
                CudaError::ExecutionError {
                    message: "exec".to_string(),
                },
                "exec",
            ),
            (
                CudaError::StreamError {
                    message: "stream".to_string(),
                },
                "stream",
            ),
            (
                CudaError::ContextError {
                    message: "ctx".to_string(),
                },
                "ctx",
            ),
            (
                CudaError::PtxError {
                    message: "ptx".to_string(),
                },
                "ptx",
            ),
            (CudaError::MockModeError, "mock mode"),
            (
                CudaError::CudaApiError {
                    code: 700,
                    message: "api".to_string(),
                },
                "700",
            ),
        ];

        for (error, expected_substring) in errors {
            let display = error.to_string();
            assert!(display.contains(expected_substring));
            assert!(error.source().is_none());
        }
    }

    // Test error edge cases
    #[test]
    fn test_error_edge_cases() {
        // Empty strings
        let e1 = CudaError::InitializationFailed {
            message: String::new(),
        };
        assert!(e1.to_string().contains("initialization failed"));

        // Very long strings
        let long_msg = "x".repeat(10000);
        let e2 = CudaError::CompilationError {
            message: long_msg.clone(),
        };
        assert!(e2.to_string().contains(&long_msg));

        // Unicode
        let e3 = CudaError::ExecutionError {
            message: "错误 🚨 エラー".to_string(),
        };
        let display = e3.to_string();
        assert!(!display.is_empty());

        // Extreme numbers
        let e4 = CudaError::OutOfMemory {
            requested: usize::MAX,
        };
        assert!(e4.to_string().contains(&usize::MAX.to_string()));

        let e5 = CudaError::InvalidDevice { device: i32::MIN };
        assert!(e5.to_string().contains(&i32::MIN.to_string()));

        let e6 = CudaError::CudaApiError {
            code: i32::MAX,
            message: "max".to_string(),
        };
        assert!(e6.to_string().contains(&i32::MAX.to_string()));
    }

    // Test Kernel struct
    #[test]
    fn test_kernel_comprehensive() {
        let metadata = KernelMetadata {
            name: "test".to_string(),
            source_type: SourceType::Ptx,
            compile_options: CompileOptions::default(),
            registers_used: 32,
            shared_memory: 1024,
            constant_memory: 0,
            local_memory: 0,
            max_threads: 512,
        };

        let kernel = Kernel::new("test".to_string(), vec![1, 2, 3, 4], metadata);

        assert_eq!(kernel.name(), "test");
        assert_eq!(kernel.code(), &[1, 2, 3, 4]);
        assert!(!kernel.id.is_nil());
        assert_eq!(kernel.is_mock(), cfg!(cuda_mock));
    }

    // Test UUID handling
    #[test]
    fn test_uuid_edge_cases() {
        use uuid::Uuid;

        // Nil UUID
        let nil = Uuid::nil();
        assert_eq!(nil.to_string(), "00000000-0000-0000-0000-000000000000");

        // Max UUID
        let max = Uuid::from_u128(u128::MAX);
        assert_eq!(max.to_string(), "ffffffff-ffff-ffff-ffff-ffffffffffff");

        // Random UUIDs are unique
        let uuids: Vec<_> = (0..100).map(|_| Uuid::new_v4()).collect();
        for i in 0..uuids.len() {
            for j in (i + 1)..uuids.len() {
                assert_ne!(uuids[i], uuids[j]);
            }
        }
    }

    // Test Send + Sync traits
    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify types are Send + Sync
        assert_send::<CudaError>();
        assert_sync::<CudaError>();
        assert_send::<ContextFlags>();
        assert_sync::<ContextFlags>();
        assert_send::<StreamFlags>();
        assert_sync::<StreamFlags>();
        assert_send::<CompileOptions>();
        assert_sync::<CompileOptions>();
        assert_send::<KernelMetadata>();
        assert_sync::<KernelMetadata>();
        assert_send::<SourceType>();
        assert_sync::<SourceType>();
    }
}
