//! Debugging and introspection tools for GPU container systems
//!
//! This crate provides advanced debugging capabilities including time-travel debugging,
//! memory snapshots, and execution replay for GPU-accelerated applications.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod analyzer;
pub mod replay;
pub mod snapshot;

/// Debug errors
#[derive(Error, Debug)]
pub enum DebugError {
    #[error("Snapshot creation failed: {reason}")]
    SnapshotFailed { reason: String },

    #[error("Replay failed: {reason}")]
    ReplayFailed { reason: String },

    #[error("Analysis failed: {reason}")]
    AnalysisFailed { reason: String },

    #[error("Snapshot not found: {snapshot_id}")]
    SnapshotNotFound { snapshot_id: uuid::Uuid },

    #[error("Session not found: {container_id}")]
    SessionNotFound { container_id: uuid::Uuid },

    #[error("Serialization failed: {reason}")]
    SerializationFailed { reason: String },

    #[error("Storage error: {reason}")]
    StorageError { reason: String },
}

/// Debug session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSession {
    pub session_id: String,
    pub start_time: u64,
    pub container_id: Option<String>,
    pub snapshot_count: usize,
}

impl DebugSession {
    /// Create a new debug session
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            container_id: None,
            snapshot_count: 0,
        }
    }
}

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_session_creation() {
        let session = DebugSession::new("test-session".to_string());
        assert_eq!(session.session_id, "test-session");
        assert!(session.start_time > 0);
        assert_eq!(session.snapshot_count, 0);
        assert!(session.container_id.is_none());
    }

    #[test]
    fn test_debug_session_with_container() {
        let mut session = DebugSession::new("container-session".to_string());
        session.container_id = Some("container-123".to_string());
        session.snapshot_count = 5;

        assert_eq!(session.session_id, "container-session");
        assert_eq!(session.container_id, Some("container-123".to_string()));
        assert_eq!(session.snapshot_count, 5);
    }

    #[test]
    fn test_debug_session_serialization() {
        let session = DebugSession {
            session_id: "serialize-test".to_string(),
            start_time: 1000,
            container_id: Some("container-456".to_string()),
            snapshot_count: 10,
        };

        let serialized = serde_json::to_string(&session).unwrap();
        let deserialized: DebugSession = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.session_id, session.session_id);
        assert_eq!(deserialized.start_time, session.start_time);
        assert_eq!(deserialized.container_id, session.container_id);
        assert_eq!(deserialized.snapshot_count, session.snapshot_count);
    }

    #[test]
    fn test_debug_error_snapshot_failed() {
        let error = DebugError::SnapshotFailed {
            reason: "Out of memory".to_string(),
        };
        assert_eq!(error.to_string(), "Snapshot creation failed: Out of memory");
    }

    #[test]
    fn test_debug_error_replay_failed() {
        let error = DebugError::ReplayFailed {
            reason: "Invalid session".to_string(),
        };
        assert_eq!(error.to_string(), "Replay failed: Invalid session");
    }

    #[test]
    fn test_debug_error_analysis_failed() {
        let error = DebugError::AnalysisFailed {
            reason: "No data available".to_string(),
        };
        assert_eq!(error.to_string(), "Analysis failed: No data available");
    }

    #[test]
    fn test_debug_session_empty_id() {
        let session = DebugSession::new(String::new());
        assert_eq!(session.session_id, "");
        assert!(session.start_time > 0);
    }

    #[test]
    fn test_debug_session_long_id() {
        let long_id = "a".repeat(1000);
        let session = DebugSession::new(long_id.clone());
        assert_eq!(session.session_id, long_id);
    }

    #[test]
    fn test_debug_session_clone() {
        let session = DebugSession {
            session_id: "clone-test".to_string(),
            start_time: 2000,
            container_id: Some("container-789".to_string()),
            snapshot_count: 20,
        };

        let cloned = session.clone();
        assert_eq!(cloned.session_id, session.session_id);
        assert_eq!(cloned.start_time, session.start_time);
        assert_eq!(cloned.container_id, session.container_id);
        assert_eq!(cloned.snapshot_count, session.snapshot_count);
    }

    #[test]
    fn test_debug_session_timestamp() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let session = DebugSession::new("timestamp-test".to_string());

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        assert!(session.start_time >= before);
        assert!(session.start_time <= after);
    }

    #[test]
    fn test_debug_session_partial_json() {
        // Test deserializing with missing optional field
        let json = r#"{
            "session_id": "partial-test",
            "start_time": 3000,
            "snapshot_count": 15
        }"#;

        let session: DebugSession = serde_json::from_str(json).unwrap();
        assert_eq!(session.session_id, "partial-test");
        assert_eq!(session.start_time, 3000);
        assert!(session.container_id.is_none());
        assert_eq!(session.snapshot_count, 15);
    }

    #[test]
    fn test_debug_error_formatting() {
        let errors = vec![
            DebugError::SnapshotFailed {
                reason: "Disk full".to_string(),
            },
            DebugError::ReplayFailed {
                reason: "Corrupted data".to_string(),
            },
            DebugError::AnalysisFailed {
                reason: "Invalid metrics".to_string(),
            },
        ];

        for error in errors {
            // Test that Display trait is properly implemented
            let formatted = format!("{error}");
            assert!(!formatted.is_empty());

            // Test that Debug trait works
            let debug_formatted = format!("{:?}", error);
            assert!(!debug_formatted.is_empty());
        }
    }

    #[test]
    fn test_module_exports() {
        // Verify modules are exported
        use crate::{analyzer, replay, snapshot};

        // This test just ensures the modules are accessible
        let _ = analyzer::AnalysisType::PerformanceAnalysis;
        let _ = replay::ReplayStatus::Pending;
        let _ = snapshot::SnapshotConfig::default();
    }

    #[test]
    fn test_debug_session_max_values() {
        let mut session = DebugSession::new("max-test".to_string());
        session.snapshot_count = usize::MAX;

        assert_eq!(session.snapshot_count, usize::MAX);

        // Test serialization with max values
        let serialized = serde_json::to_string(&session).unwrap();
        let deserialized: DebugSession = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.snapshot_count, usize::MAX);
    }

    #[test]
    fn test_debug_error_empty_reasons() {
        let errors = vec![
            DebugError::SnapshotFailed {
                reason: String::new(),
            },
            DebugError::ReplayFailed {
                reason: String::new(),
            },
            DebugError::AnalysisFailed {
                reason: String::new(),
            },
        ];

        for error in errors {
            let formatted = error.to_string();
            assert!(formatted.contains(": "));
            // Empty reason should still produce valid error message
        }
    }

    #[test]
    fn test_debug_session_concurrent_creation() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let sessions = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        for i in 0..10 {
            let sessions_clone = sessions.clone();
            let handle = thread::spawn(move || {
                let session = DebugSession::new(format!("concurrent-{i}"));
                sessions_clone.lock().unwrap().push(session);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let sessions = sessions.lock().unwrap();
        assert_eq!(sessions.len(), 10);

        // All sessions should have unique IDs
        let mut ids: Vec<_> = sessions.iter().map(|s| &s.session_id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_debug_session_json_roundtrip() {
        let original = DebugSession {
            session_id: "roundtrip-test".to_string(),
            start_time: 5000,
            container_id: Some("container-999".to_string()),
            snapshot_count: 42,
        };

        // Test JSON roundtrip
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DebugSession = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.session_id, original.session_id);
        assert_eq!(parsed.start_time, original.start_time);
        assert_eq!(parsed.container_id, original.container_id);
        assert_eq!(parsed.snapshot_count, original.snapshot_count);
    }

    #[test]
    fn test_debug_session_update_counts() {
        let mut session = DebugSession::new("update-test".to_string());

        // Simulate taking snapshots
        for i in 1..=100 {
            session.snapshot_count = i;
            assert_eq!(session.snapshot_count, i);
        }
    }

    #[test]
    fn test_debug_error_long_reasons() {
        let long_reason = "Error: ".to_string() + &"x".repeat(1000);

        let error = DebugError::SnapshotFailed {
            reason: long_reason.clone(),
        };

        let formatted = error.to_string();
        assert!(formatted.contains(&long_reason));
    }

    #[test]
    fn test_debug_session_special_characters() {
        let special_chars = "test-session-!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let session = DebugSession::new(special_chars.to_string());
        assert_eq!(session.session_id, special_chars);

        // Test serialization with special characters
        let json = serde_json::to_string(&session).unwrap();
        let parsed: DebugSession = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.session_id, special_chars);
    }

    #[test]
    fn test_debug_session_unicode() {
        let unicode_id = "测试会话-🚀-тест";
        let session = DebugSession::new(unicode_id.to_string());
        assert_eq!(session.session_id, unicode_id);

        // Test serialization with unicode
        let json = serde_json::to_string(&session).unwrap();
        let parsed: DebugSession = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.session_id, unicode_id);
    }

    #[test]
    fn test_debug_error_conversion() {
        // Test that errors can be converted to Box<dyn Error>
        let error: Box<dyn std::error::Error> = Box::new(DebugError::SnapshotFailed {
            reason: "Test error".to_string(),
        });

        assert!(error.to_string().contains("Snapshot creation failed"));
    }

    #[test]
    fn test_debug_session_equality() {
        let session1 = DebugSession {
            session_id: "equal-test".to_string(),
            start_time: 6000,
            container_id: Some("container-eq".to_string()),
            snapshot_count: 25,
        };

        let session2 = DebugSession {
            session_id: "equal-test".to_string(),
            start_time: 6000,
            container_id: Some("container-eq".to_string()),
            snapshot_count: 25,
        };

        // Manual equality check (since PartialEq not derived)
        assert_eq!(session1.session_id, session2.session_id);
        assert_eq!(session1.start_time, session2.start_time);
        assert_eq!(session1.container_id, session2.container_id);
        assert_eq!(session1.snapshot_count, session2.snapshot_count);
    }

    #[test]
    fn test_crate_reexports() {
        // Test that main types are accessible from crate root
        let _session = DebugSession::new("reexport-test".to_string());
        let _error = DebugError::SnapshotFailed {
            reason: "test".to_string(),
        };
    }
}
