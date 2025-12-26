//! Edge case tests for debug crate to enhance coverage to 90%+

#[cfg(test)]
mod edge_case_tests {
    use crate::{DebugError, DebugSession};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    // DebugError edge cases with unicode and special characters

    #[test]
    fn test_error_unicode_reasons() {
        let unicode_reasons = vec![
            "错误：快照失败 🚨",
            "Erreur: Échec de la capture 💾",
            "エラー：スナップショット失敗 ⚡",
            "خطأ: فشل اللقطة 🔥",
            "🚨💥❌ Critical Failure ❌💥🚨",
            "Error\0with\0embedded\0nulls",
            "Error\nwith\nmultiple\nnewlines\n",
            "Error\twith\ttabs\teverywhere\t",
            "Error\rwith\rcarriage\rreturns\r",
            "Mixed\n\r\t\0special chars",
        ];

        for reason in unicode_reasons {
            let errors = vec![
                DebugError::SnapshotFailed {
                    reason: reason.to_string(),
                },
                DebugError::ReplayFailed {
                    reason: reason.to_string(),
                },
                DebugError::AnalysisFailed {
                    reason: reason.to_string(),
                },
            ];

            for err in errors {
                let error_str = err.to_string();
                // Just verify it doesn't panic
                assert!(!error_str.is_empty());

                // Test Debug format too
                let debug_str = format!("{:?}", err);
                assert!(!debug_str.is_empty());
            }
        }
    }

    #[test]
    fn test_error_extremely_long_reasons() {
        // Test with very long reason strings
        let long_reasons = vec![
            "x".repeat(10000),
            "🚨".repeat(1000),
            "Error ".repeat(1000),
            format!("{}\n{}", "Line1".repeat(100), "Line2".repeat(100)),
        ];

        for reason in long_reasons {
            let err = DebugError::SnapshotFailed {
                reason: reason.clone(),
            };
            let error_str = err.to_string();
            assert!(error_str.len() > reason.len()); // Should include prefix
        }
    }

    // DebugSession edge cases

    #[test]
    fn test_session_id_edge_cases() {
        let edge_ids = vec![
            "",
            " ",
            "    ",
            "\n",
            "\t",
            "\r\n",
            "a",
            "A-Z0-9_-.",
            "session/with/slashes",
            "session\\with\\backslashes",
            "session:with:colons",
            "session;with;semicolons",
            "session<with>brackets",
            "session|with|pipes",
            "session?with?questions",
            "session*with*asterisks",
            "session\"with\"quotes",
            "session'with'apostrophes",
            "🚀🔥💾⚡🎯",
            "测试会话标识符",
            "セッション識別子",
            "معرف الجلسة",
            "\u{0000}\u{0001}\u{0002}", // Control characters
            "\u{FEFF}invisible",        // Zero-width space
        ];

        for id in edge_ids {
            let session = DebugSession::new(id.to_string());
            assert_eq!(session.session_id, id);

            // Test serialization
            let json = serde_json::to_string(&session).unwrap();
            let deserialized: DebugSession = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.session_id, id);
        }
    }

    #[test]
    fn test_session_timestamp_boundaries() {
        // Test with specific timestamps
        let mut session = DebugSession::new("timestamp-test".to_string());

        // Test various timestamp values
        let timestamps = vec![
            0u64,         // Unix epoch
            1,            // One second after epoch
            946684800,    // Y2K
            2147483647,   // 32-bit max (Y2038)
            4294967295,   // 32-bit unsigned max
            9999999999,   // Far future
            u64::MAX - 1, // Near max
            u64::MAX,     // Max value
        ];

        for ts in timestamps {
            session.start_time = ts;

            let json = serde_json::to_string(&session).unwrap();
            let deserialized: DebugSession = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.start_time, ts);
        }
    }

    #[test]
    fn test_session_container_id_variations() {
        let mut session = DebugSession::new("container-test".to_string());

        let container_ids = vec![
            None,
            Some("".to_string()),
            Some("a".to_string()),
            Some("container-123".to_string()),
            Some("CONTAINER_UPPER_CASE_123".to_string()),
            Some("container.with.dots".to_string()),
            Some("container-with-dashes".to_string()),
            Some("container_with_underscores".to_string()),
            Some("container/with/slashes".to_string()),
            Some("🚀-unicode-container-🔥".to_string()),
            Some("x".repeat(1000)), // Long ID
        ];

        for container_id in container_ids {
            session.container_id = container_id.clone();

            let json = serde_json::to_string(&session).unwrap();
            let deserialized: DebugSession = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.container_id, container_id);
        }
    }

    #[test]
    fn test_session_snapshot_count_extremes() {
        let mut session = DebugSession::new("count-test".to_string());

        let counts = vec![
            0,
            1,
            100,
            1000,
            10000,
            100000,
            1000000,
            usize::MAX / 2,
            usize::MAX - 1,
            usize::MAX,
        ];

        for count in counts {
            session.snapshot_count = count;
            assert_eq!(session.snapshot_count, count);

            // Test serialization
            let json = serde_json::to_string(&session).unwrap();
            let deserialized: DebugSession = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.snapshot_count, count);
        }
    }

    // Thread safety tests

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Verify types are Send + Sync
        assert_send::<DebugError>();
        assert_sync::<DebugError>();
        assert_send::<DebugSession>();
        assert_sync::<DebugSession>();
    }

    // Error pattern matching edge cases

    #[test]
    fn test_error_exhaustive_patterns() {
        fn classify_error(err: &DebugError) -> &'static str {
            match err {
                DebugError::SnapshotFailed { .. } => "snapshot",
                DebugError::ReplayFailed { .. } => "replay",
                DebugError::AnalysisFailed { .. } => "analysis",
            }
        }

        let errors = vec![
            (
                DebugError::SnapshotFailed {
                    reason: "test".to_string(),
                },
                "snapshot",
            ),
            (
                DebugError::ReplayFailed {
                    reason: "test".to_string(),
                },
                "replay",
            ),
            (
                DebugError::AnalysisFailed {
                    reason: "test".to_string(),
                },
                "analysis",
            ),
        ];

        for (err, expected) in errors {
            assert_eq!(classify_error(&err), expected);
        }
    }

    // JSON edge cases

    #[test]
    fn test_session_json_missing_fields() {
        // Test partial JSON deserialization
        let json_variants = vec![
            // Minimal valid JSON
            r#"{"session_id":"test","start_time":0,"snapshot_count":0}"#,
            // With null container_id
            r#"{"session_id":"test","start_time":0,"container_id":null,"snapshot_count":0}"#,
            // With extra unknown fields (should ignore)
            r#"{"session_id":"test","start_time":0,"container_id":null,"snapshot_count":0,"unknown_field":"ignored"}"#,
        ];

        for json in json_variants {
            let session: Result<DebugSession, _> = serde_json::from_str(json);
            assert!(session.is_ok());
        }
    }

    #[test]
    fn test_session_json_field_order() {
        let session = DebugSession {
            session_id: "order-test".to_string(),
            start_time: 12345,
            container_id: Some("container".to_string()),
            snapshot_count: 42,
        };

        // Create JSON with different field orders
        let json1 = serde_json::json!({
            "session_id": "order-test",
            "start_time": 12345,
            "container_id": "container",
            "snapshot_count": 42
        });

        let json2 = serde_json::json!({
            "snapshot_count": 42,
            "container_id": "container",
            "session_id": "order-test",
            "start_time": 12345
        });

        let parsed1: DebugSession = serde_json::from_value(json1).unwrap();
        let parsed2: DebugSession = serde_json::from_value(json2).unwrap();

        assert_eq!(parsed1.session_id, parsed2.session_id);
        assert_eq!(parsed1.start_time, parsed2.start_time);
        assert_eq!(parsed1.container_id, parsed2.container_id);
        assert_eq!(parsed1.snapshot_count, parsed2.snapshot_count);
    }

    // Error conversion edge cases

    #[test]
    fn test_error_trait_object_conversion() {
        let errors: Vec<Box<dyn std::error::Error + Send + Sync>> = vec![
            Box::new(DebugError::SnapshotFailed {
                reason: "test1".to_string(),
            }),
            Box::new(DebugError::ReplayFailed {
                reason: "test2".to_string(),
            }),
            Box::new(DebugError::AnalysisFailed {
                reason: "test3".to_string(),
            }),
        ];

        for err in errors {
            assert!(!err.to_string().is_empty());
            // Verify source is None (no nested errors)
            assert!(err.source().is_none());
        }
    }

    // Concurrent session operations

    #[test]
    fn test_concurrent_session_modifications() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let session = Arc::new(Mutex::new(DebugSession::new("concurrent".to_string())));
        let mut handles = vec![];

        // Spawn threads that increment snapshot count
        for i in 0..10 {
            let session_clone = session.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut sess = session_clone.lock().unwrap();
                    sess.snapshot_count = sess.snapshot_count.saturating_add(1);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_session = session.lock().unwrap();
        // Should have incremented 1000 times (10 threads * 100 increments)
        assert_eq!(final_session.snapshot_count, 1000);
    }

    // Time edge cases

    #[test]
    fn test_session_time_boundaries() {
        // Test creating session at specific system times
        let session = DebugSession::new("time-test".to_string());

        // Verify timestamp is reasonable (between 1970 and far future)
        assert!(session.start_time > 0);
        assert!(session.start_time < u64::MAX);

        // Test that timestamp increases for sequential sessions
        let session1 = DebugSession::new("seq1".to_string());
        std::thread::sleep(Duration::from_millis(10));
        let session2 = DebugSession::new("seq2".to_string());

        assert!(session2.start_time >= session1.start_time);
    }

    // Memory efficiency test

    #[test]
    fn test_session_memory_size() {
        use std::mem::size_of;

        // Ensure DebugSession is reasonably sized
        assert!(size_of::<DebugSession>() <= 128); // Should be compact

        // Test with different container_id sizes
        let mut session = DebugSession::new("mem-test".to_string());
        let original_size = size_of::<DebugSession>();

        session.container_id = Some("x".repeat(1000));
        // Size of struct should not change (String is heap-allocated)
        assert_eq!(size_of::<DebugSession>(), original_size);
    }

    // All combinations test

    #[test]
    fn test_session_all_edge_combinations() {
        let session = DebugSession {
            session_id: "".to_string(),         // Empty ID
            start_time: u64::MAX,               // Max timestamp
            container_id: Some("".to_string()), // Empty container
            snapshot_count: usize::MAX,         // Max snapshots
        };

        // Should handle all edge cases gracefully
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: DebugSession = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.session_id, "");
        assert_eq!(deserialized.start_time, u64::MAX);
        assert_eq!(deserialized.container_id, Some("".to_string()));
        assert_eq!(deserialized.snapshot_count, usize::MAX);
    }
}
