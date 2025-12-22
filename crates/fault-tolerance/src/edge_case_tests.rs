//! Edge case tests for fault-tolerance crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{
        checkpoint::{CheckpointId, ExecutionContext, GpuCheckpoint, KernelState},
        error::{FaultToleranceError, FtResult, HealthStatus},
        FaultToleranceManager,
    };
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use uuid::Uuid;

    impl Default for ExecutionContext {
        fn default() -> Self {
            ExecutionContext {
                device_id: 0,
                block_size: (1, 1, 1),
                grid_size: (1, 1, 1),
                shared_memory_bytes: 0,
            }
        }
    }

    // FaultToleranceError edge cases

    #[test]
    fn test_error_variants_all() {
        let errors = vec![
            FaultToleranceError::CheckpointFailed("".to_string()),
            FaultToleranceError::RecoveryFailed("Very long error message".repeat(100)),
            FaultToleranceError::CoordinationError("Unicode error: 错误 🚨".to_string()),
            FaultToleranceError::SerializationError("Error\nwith\nnewlines".to_string()),
            FaultToleranceError::StorageError("Error\twith\ttabs\tand spaces   ".to_string()),
            FaultToleranceError::GpuMemoryError(String::new()),
            FaultToleranceError::HealthCheckFailed("⏰ Failed after 1000000ms".to_string()),
            FaultToleranceError::CheckpointNotFound("🔍 Not found".to_string()),
            FaultToleranceError::InvalidCheckpoint("Invalid\0with\0nulls".to_string()),
        ];

        for err in errors {
            let error_str = err.to_string();
            assert!(!error_str.is_empty());

            // Test Debug format
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_error_extreme_messages() {
        // Test with extreme lengths
        let long_msg = "x".repeat(10000);
        let err = FaultToleranceError::CheckpointFailed(long_msg.clone());
        assert!(err.to_string().contains(&long_msg));

        // Test with special characters
        let special_chars = "Error: \r\n\t\u{1F4A5}💥❌";
        let err = FaultToleranceError::RecoveryFailed(special_chars.to_string());
        let _ = err.to_string(); // Just ensure it doesn't panic
    }

    #[test]
    fn test_io_error_conversion() {
        use std::io::{Error as IoError, ErrorKind};

        let io_errors = vec![
            IoError::new(ErrorKind::NotFound, "file not found"),
            IoError::new(ErrorKind::PermissionDenied, "access denied"),
            IoError::new(ErrorKind::AlreadyExists, "already exists"),
            IoError::new(ErrorKind::WouldBlock, "would block"),
            IoError::new(ErrorKind::InvalidInput, "invalid input"),
            IoError::new(ErrorKind::InvalidData, "invalid data"),
            IoError::new(ErrorKind::TimedOut, "timed out"),
            IoError::new(ErrorKind::WriteZero, "write zero"),
            IoError::new(ErrorKind::Interrupted, "interrupted"),
            IoError::new(ErrorKind::UnexpectedEof, "unexpected EOF"),
            IoError::new(ErrorKind::Other, "other error"),
        ];

        for io_err in io_errors {
            let ft_err = FaultToleranceError::from(io_err);
            assert!(matches!(ft_err, FaultToleranceError::IoError(_)));
            assert!(!ft_err.to_string().is_empty());
        }
    }

    // HealthStatus edge cases

    #[test]
    fn test_health_status_serialization() {
        let statuses = vec![
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Failed,
        ];

        for status in statuses {
            // Test serialization
            let json = serde_json::to_string(&status)?;
            let deserialized: HealthStatus = serde_json::from_str(&json)?;
            assert_eq!(status, deserialized);

            // Test clone
            let cloned = status.clone();
            assert_eq!(status, cloned);
        }
    }

    // CheckpointId edge cases

    #[test]
    fn test_checkpoint_id_edge_cases() {
        // Test nil UUID
        let nil_id = CheckpointId::from_uuid(Uuid::nil());
        assert_eq!(nil_id.as_uuid(), Uuid::nil());

        // Test max UUID
        let max_id = CheckpointId::from_uuid(Uuid::from_u128(u128::MAX));
        assert_eq!(max_id.as_uuid(), Uuid::from_u128(u128::MAX));

        // Test default
        let default_id = CheckpointId::default();
        let new_id = CheckpointId::new();
        assert_ne!(default_id, new_id); // Should be different random UUIDs

        // Test serialization
        let id = CheckpointId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: CheckpointId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_checkpoint_id_uniqueness() {
        let mut ids = Vec::new();
        for _ in 0..100 {
            ids.push(CheckpointId::new());
        }

        // Check all are unique
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    // GpuCheckpoint edge cases

    #[test]
    fn test_gpu_checkpoint_extremes() {
        let checkpoint = GpuCheckpoint {
            memory_snapshot: vec![],
            kernel_states: HashMap::new(),
            timestamp: 0,
            size_bytes: 0,
        };

        // Test serialization with empty data
        let serialized = bincode::serialize(&checkpoint)?;
        let deserialized: GpuCheckpoint = bincode::deserialize(&serialized)?;
        assert_eq!(deserialized.memory_snapshot.len(), 0);
        assert_eq!(deserialized.kernel_states.len(), 0);

        // Test with large data
        let large_checkpoint = GpuCheckpoint {
            memory_snapshot: vec![0u8; 1024 * 1024], // 1MB
            kernel_states: (0..100)
                .map(|i| {
                    (
                        format!("kernel_{}", i),
                        KernelState {
                            kernel_id: format!("id_{}", i),
                            parameters: vec![i as u8; 100],
                            execution_context: ExecutionContext::default(),
                        },
                    )
                })
                .collect(),
            timestamp: u64::MAX,
            size_bytes: usize::MAX,
        };

        let serialized = bincode::serialize(&large_checkpoint).unwrap();
        assert!(!serialized.is_empty());
    }

    #[test]
    fn test_kernel_state_variations() {
        let kernel_states = vec![
            KernelState {
                kernel_id: "".to_string(),
                parameters: vec![],
                execution_context: ExecutionContext::default(),
            },
            KernelState {
                kernel_id: "kernel_with_unicode_名前_🚀".to_string(),
                parameters: vec![0xFF; 1000],
                execution_context: ExecutionContext::default(),
            },
            KernelState {
                kernel_id: "very_long_kernel_id_that_exceeds_normal_length".repeat(10),
                parameters: (0..=255).collect(),
                execution_context: ExecutionContext::default(),
            },
        ];

        for state in kernel_states {
            let serialized = bincode::serialize(&state).unwrap();
            let deserialized: KernelState = bincode::deserialize(&serialized).unwrap();
            assert_eq!(state.kernel_id, deserialized.kernel_id);
            assert_eq!(state.parameters, deserialized.parameters);
        }
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify types are Send + Sync
        assert_send::<FaultToleranceError>();
        assert_sync::<FaultToleranceError>();
        assert_send::<HealthStatus>();
        assert_sync::<HealthStatus>();
        assert_send::<CheckpointId>();
        assert_sync::<CheckpointId>();
    }

    // Error result handling

    #[test]
    fn test_result_operations() {
        fn may_fail(should_fail: bool) -> FtResult<i32> {
            if should_fail {
                Err(FaultToleranceError::CheckpointFailed("test".to_string()))
            } else {
                Ok(42)
            }
        }

        // Test successful path
        let result = may_fail(false).map(|x| x * 2).and_then(|x| Ok(x + 10));
        assert_eq!(result?, 94);

        // Test error path
        let result = may_fail(true).map(|x| x * 2).and_then(|x| Ok(x + 10));
        assert!(result.is_err());

        // Test error mapping
        let result = may_fail(true)
            .map_err(|_| FaultToleranceError::RecoveryFailed("mapped error".to_string()));
        match result {
            Err(FaultToleranceError::RecoveryFailed(msg)) => {
                assert_eq!(msg, "mapped error");
            }
            _ => panic!("Expected RecoveryFailed error"),
        }
    }

    #[test]
    fn test_result_collection() {
        let results: Vec<FtResult<i32>> = vec![
            Ok(1),
            Ok(2),
            Err(FaultToleranceError::StorageError("disk full".to_string())),
            Ok(4),
        ];

        let collected: Result<Vec<i32>, FaultToleranceError> = results.into_iter().collect();
        assert!(collected.is_err());

        match collected.unwrap_err() {
            FaultToleranceError::StorageError(msg) => assert_eq!(msg, "disk full"),
            _ => panic!("Expected StorageError"),
        }
    }

    // Timestamp edge cases

    #[test]
    fn test_timestamp_extremes() {
        let timestamps = vec![
            0u64,
            1,
            1000,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                ?
                .as_secs(),
            u64::MAX - 1,
            u64::MAX,
        ];

        for ts in timestamps {
            let checkpoint = GpuCheckpoint {
                memory_snapshot: vec![],
                kernel_states: HashMap::new(),
                timestamp: ts,
                size_bytes: 0,
            };

            assert_eq!(checkpoint.timestamp, ts);
        }
    }

    // Memory snapshot patterns

    #[test]
    fn test_memory_snapshot_patterns() {
        let patterns = vec![
            vec![],
            vec![0u8],
            vec![0xFF],
            vec![0u8; 1024],
            vec![0xFF; 1024],
            (0..=255).collect::<Vec<u8>>(),
            vec![0xDE, 0xAD, 0xBE, 0xEF],
            vec![0xCA, 0xFE, 0xBA, 0xBE],
        ];

        for pattern in patterns {
            let checkpoint = GpuCheckpoint {
                memory_snapshot: pattern.clone(),
                kernel_states: HashMap::new(),
                timestamp: 0,
                size_bytes: pattern.len(),
            };

            assert_eq!(checkpoint.memory_snapshot, pattern);
            assert_eq!(checkpoint.size_bytes, pattern.len());
        }
    }

    // HashMap edge cases

    #[test]
    fn test_kernel_states_hashmap() {
        let mut states = HashMap::new();

        // Empty key
        states.insert(
            "".to_string(),
            KernelState {
                kernel_id: "empty_key".to_string(),
                parameters: vec![],
                execution_context: ExecutionContext::default(),
            },
        );

        // Unicode key
        states.insert(
            "核心_🚀".to_string(),
            KernelState {
                kernel_id: "unicode_key".to_string(),
                parameters: vec![1, 2, 3],
                execution_context: ExecutionContext::default(),
            },
        );

        // Very long key
        let long_key = "k".repeat(1000);
        states.insert(
            long_key.clone(),
            KernelState {
                kernel_id: "long_key".to_string(),
                parameters: vec![42],
                execution_context: ExecutionContext::default(),
            },
        );

        assert_eq!(states.len(), 3);
        assert!(states.contains_key(""));
        assert!(states.contains_key("核心_🚀"));
        assert!(states.contains_key(&long_key));
    }

    // Error conversion edge cases

    #[test]
    fn test_error_into_box() {
        let errors: Vec<Box<dyn std::error::Error + Send + Sync>> = vec![
            Box::new(FaultToleranceError::CheckpointFailed("test".to_string())),
            Box::new(FaultToleranceError::GpuMemoryError(
                "out of memory".to_string(),
            )),
            Box::new(FaultToleranceError::HealthCheckFailed(
                "unhealthy".to_string(),
            )),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    // Size edge cases

    #[test]
    fn test_size_bytes_extremes() {
        let sizes = vec![0, 1, 1024, 1024 * 1024, usize::MAX];

        for size in sizes {
            let checkpoint = GpuCheckpoint {
                memory_snapshot: vec![],
                kernel_states: HashMap::new(),
                timestamp: 0,
                size_bytes: size,
            };

            assert_eq!(checkpoint.size_bytes, size);
        }
    }

    // Concurrent operations simulation

    #[test]
    fn test_checkpoint_id_concurrent_generation() {
        use std::sync::Arc;
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|_| thread::spawn(|| CheckpointId::new()))
            .collect();

        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.join()?);
        }

        // All IDs should be unique
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    // Complex error scenarios

    #[test]
    fn test_nested_error_handling() {
        fn inner_operation() -> FtResult<String> {
            Err(FaultToleranceError::SerializationError(
                "inner error".to_string(),
            ))
        }

        fn middle_operation() -> FtResult<String> {
            inner_operation()
                .map_err(|e| FaultToleranceError::CheckpointFailed(format!("Middle: {}", e)))
        }

        fn outer_operation() -> FtResult<String> {
            middle_operation()
                .map_err(|e| FaultToleranceError::RecoveryFailed(format!("Outer: {}", e)))
        }

        let result = outer_operation();
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Recovery failed"));
        assert!(error_msg.contains("Outer:"));
    }

    // Edge case combinations

    #[test]
    fn test_checkpoint_with_all_edge_cases() {
        let checkpoint = GpuCheckpoint {
            memory_snapshot: vec![0u8; 0], // Empty snapshot
            kernel_states: {
                let mut states = HashMap::new();
                states.insert(
                    "".to_string(),
                    KernelState {
                        kernel_id: "".to_string(),
                        parameters: vec![],
                        execution_context: ExecutionContext::default(),
                    },
                );
                states
            },
            timestamp: u64::MAX,
            size_bytes: usize::MAX,
        };

        // Should handle all edge cases gracefully
        let serialized = bincode::serialize(&checkpoint).unwrap();
        let deserialized: GpuCheckpoint = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.memory_snapshot.len(), 0);
        assert_eq!(deserialized.kernel_states.len(), 1);
        assert_eq!(deserialized.timestamp, u64::MAX);
        assert_eq!(deserialized.size_bytes, usize::MAX);
    }
}
