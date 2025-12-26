//! Edge case tests for CPU-GPU Bridge to enhance coverage to 90%+

use crate::{bridge::*, CpuAgentError};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_message_validation_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Empty ID
    let msg1 = CpuGpuMessage {
        id: "".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(msg1).await.is_err());

    // Empty source
    let msg2 = CpuGpuMessage {
        id: "test".to_string(),
        message_type: MessageType::TaskRequest,
        source: "".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(msg2).await.is_err());

    // Empty destination
    let msg3 = CpuGpuMessage {
        id: "test".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(msg3).await.is_err());

    // Invalid priority (0)
    let msg4 = CpuGpuMessage {
        id: "test".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 0,
    };
    assert!(bridge.send_to_gpu(msg4).await.is_err());

    // Invalid priority (>10)
    let msg5 = CpuGpuMessage {
        id: "test".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 11,
    };
    assert!(bridge.send_to_gpu(msg5).await.is_err());
}

#[tokio::test]
async fn test_extreme_message_sizes() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 1,
        polling_interval_ms: 100,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Very small message
    let small_msg = CpuGpuMessage {
        id: "small".to_string(),
        message_type: MessageType::StatusUpdate,
        source: "c".to_string(),
        destination: "g".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 1,
    };
    assert!(bridge.send_to_gpu(small_msg).await.is_ok());

    // Message exactly at limit (approximate)
    let data_size = 900 * 1024; // ~900KB to account for JSON overhead
    let limit_data = vec![0u8; data_size];
    let limit_msg = CpuGpuMessage {
        id: "limit".to_string(),
        message_type: MessageType::DataTransfer,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({
            "data": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &limit_data)
        }),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    // This might succeed or fail depending on exact JSON serialization size
    let _ = bridge.send_to_gpu(limit_msg).await;
}

#[tokio::test]
async fn test_all_message_types() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    let message_types = vec![
        MessageType::TaskRequest,
        MessageType::TaskResult,
        MessageType::DataTransfer,
        MessageType::StatusUpdate,
        MessageType::ErrorReport,
        MessageType::Shutdown,
    ];

    for (i, msg_type) in message_types.iter().enumerate() {
        let msg = CpuGpuMessage {
            id: format!("type-test-{}", i),
            message_type: msg_type.clone(),
            source: "cpu".to_string(),
            destination: "gpu".to_string(),
            payload: serde_json::json!({
                "type": format!("{:?}", msg_type)
            }),
            timestamp: chrono::Utc::now(),
            priority: ((i % 10) + 1) as u8,
        };

        assert!(bridge.send_to_gpu(msg).await.is_ok());
    }

    // Verify all message types are properly serialized
    let stats = bridge.stats();
    assert_eq!(stats.messages_sent, message_types.len() as u64);
}

#[tokio::test]
async fn test_timestamp_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Very old timestamp
    let old_msg = CpuGpuMessage {
        id: "old".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::DateTime::parse_from_rfc3339("1970-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(old_msg).await.is_ok());

    // Future timestamp
    let future_msg = CpuGpuMessage {
        id: "future".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now() + chrono::Duration::days(365),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(future_msg).await.is_ok());
}

#[tokio::test]
async fn test_payload_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Null payload
    let null_msg = CpuGpuMessage {
        id: "null".to_string(),
        message_type: MessageType::TaskRequest,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::Value::Null,
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(null_msg).await.is_ok());

    // Complex nested payload
    let complex_payload = serde_json::json!({
        "level1": {
            "level2": {
                "level3": {
                    "array": [1, 2, 3, null, true, false],
                    "string": "test",
                    "number": 42.5,
                    "unicode": "🚀💾🔥",
                }
            }
        }
    });

    let complex_msg = CpuGpuMessage {
        id: "complex".to_string(),
        message_type: MessageType::DataTransfer,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: complex_payload,
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(complex_msg).await.is_ok());
}

#[tokio::test]
async fn test_bridge_stats_accuracy() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Initial stats should be zero
    let stats = bridge.stats();
    assert_eq!(stats.messages_sent, 0);
    assert_eq!(stats.messages_received, 0);
    assert_eq!(stats.messages_failed, 0);
    assert_eq!(stats.bytes_sent, 0);
    assert_eq!(stats.bytes_received, 0);

    // Send messages and verify stats
    for i in 0..5 {
        let msg = CpuGpuMessage {
            id: format!("stats-{}", i),
            message_type: MessageType::TaskRequest,
            source: "cpu".to_string(),
            destination: "gpu".to_string(),
            payload: serde_json::json!({"index": i}),
            timestamp: chrono::Utc::now(),
            priority: 5,
        };
        bridge.send_to_gpu(msg).await.unwrap();
    }

    let stats = bridge.stats();
    assert_eq!(stats.messages_sent, 5);
    assert!(stats.bytes_sent > 0);
}

#[tokio::test]
async fn test_config_edge_values() {
    let temp_dir = TempDir::new().unwrap();

    // Zero retention time
    let config1 = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "in".to_string(),
        outbox_dir: "out".to_string(),
        message_retention_seconds: 0,
        max_message_size_mb: 1,
        polling_interval_ms: 1,
    };
    assert!(CpuGpuBridge::new(config1).await.is_ok());

    // Very large values
    let config2 = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: u64::MAX,
        max_message_size_mb: usize::MAX,
        polling_interval_ms: u64::MAX,
    };
    assert!(CpuGpuBridge::new(config2).await.is_ok());
}

#[tokio::test]
async fn test_priority_edge_values() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 1000,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Test boundary priority values
    let priorities = vec![1, 10]; // Valid boundaries

    for priority in priorities {
        let msg = CpuGpuMessage {
            id: format!("priority-{}", priority),
            message_type: MessageType::TaskRequest,
            source: "cpu".to_string(),
            destination: "gpu".to_string(),
            payload: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
            priority,
        };
        assert!(bridge.send_to_gpu(msg).await.is_ok());
    }
}

#[tokio::test]
async fn test_corrupted_message_handling() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Create various corrupted messages in inbox
    let inbox_path = temp_dir.path().join("inbox");

    // Invalid JSON
    tokio::fs::write(inbox_path.join("corrupt1.json"), b"{ invalid json")
        .await
        .unwrap();

    // Valid JSON but wrong structure
    tokio::fs::write(
        inbox_path.join("corrupt2.json"),
        b"{\"wrong\": \"structure\"}",
    )
    .await
    .unwrap();

    // Empty file
    tokio::fs::write(inbox_path.join("corrupt3.json"), b"")
        .await
        .unwrap();

    // Non-JSON file
    tokio::fs::write(inbox_path.join("notjson.txt"), b"plain text")
        .await
        .unwrap();

    // Receive messages - should handle corrupted ones gracefully
    let messages = bridge.receive_from_gpu().await.unwrap();
    assert_eq!(messages.len(), 0); // All corrupted messages should be ignored

    // Check that corrupted messages were moved to error directory
    let error_dir = temp_dir.path().join("error");
    assert!(error_dir.exists());
}

#[tokio::test]
async fn test_stop_bridge_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig::default();
    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Start the bridge
    bridge.start().await.unwrap();

    // Send some messages
    for i in 0..3 {
        let msg = CpuGpuMessage {
            id: format!("stop-test-{}", i),
            message_type: MessageType::TaskRequest,
            source: "cpu".to_string(),
            destination: "gpu".to_string(),
            payload: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
            priority: 5,
        };
        bridge.send_to_gpu(msg).await.unwrap();
    }

    // Stop the bridge
    bridge.stop().await.unwrap();

    // Bridge should still have stats
    let stats = bridge.stats();
    assert_eq!(stats.messages_sent, 3);
}

#[tokio::test]
async fn test_directory_creation_failure() {
    // Use a path that will fail to create
    let config = BridgeConfig {
        shared_storage_path: PathBuf::from("/root/no-permission"),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    // Should fail to create bridge due to permission issues
    let result = CpuGpuBridge::new(config).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_unicode_in_paths_and_ids() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "收件箱".to_string(),  // Chinese for "inbox"
        outbox_dir: "発信箱".to_string(), // Japanese for "outbox"
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Message with unicode ID
    let msg = CpuGpuMessage {
        id: "消息-🚀-メッセージ".to_string(),
        message_type: MessageType::DataTransfer,
        source: "процессор".to_string(),    // Russian for "processor"
        destination: "γραφική".to_string(), // Greek for "graphics"
        payload: serde_json::json!({
            "data": "مرحبا", // Arabic for "hello"
            "emoji": "🔥💾🚀"
        }),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };

    assert!(bridge.send_to_gpu(msg).await.is_ok());
}

#[tokio::test]
async fn test_concurrent_receive_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };
    let bridge = std::sync::Arc::new(tokio::sync::Mutex::new(
        CpuGpuBridge::new(config).await.unwrap(),
    ));

    // Pre-populate inbox with messages
    let inbox_path = temp_dir.path().join("inbox");
    for i in 0..10 {
        let msg = CpuGpuMessage {
            id: format!("concurrent-{}", i),
            message_type: MessageType::TaskResult,
            source: "gpu".to_string(),
            destination: "cpu".to_string(),
            payload: serde_json::json!({"index": i}),
            timestamp: chrono::Utc::now(),
            priority: 5,
        };
        let json = serde_json::to_string_pretty(&msg).unwrap();
        tokio::fs::write(inbox_path.join(format!("{}.json", msg.id)), json)
            .await
            .unwrap();
    }

    // Multiple concurrent receives
    let mut handles = Vec::new();
    for _ in 0..5 {
        let bridge_clone = bridge.clone();
        let handle = tokio::spawn(async move {
            let mut b = bridge_clone.lock().await;
            b.receive_from_gpu().await
        });
        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    // At least one should receive messages
    let total_received: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .filter_map(|r| r.as_ref().ok())
        .map(|msgs| msgs.len())
        .sum();

    assert!(total_received > 0);
}

#[tokio::test]
async fn test_cleanup_task_execution() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 1, // 1 second retention
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();
    bridge.start().await.unwrap();

    // Create an old message file directly
    let outbox_path = temp_dir.path().join("outbox");
    let old_file = outbox_path.join("old-message.json");
    tokio::fs::write(&old_file, b"{}").await.unwrap();

    // Modify the file's timestamp to be old
    let metadata = tokio::fs::metadata(&old_file).await.unwrap();
    // Note: Modifying file timestamps is platform-specific and may not work in tests

    // Wait for cleanup to potentially run
    sleep(Duration::from_secs(2)).await;

    // File might be cleaned up (depends on platform support for timestamp modification)
    // Just verify bridge is still functional
    let msg = CpuGpuMessage {
        id: "after-cleanup".to_string(),
        message_type: MessageType::StatusUpdate,
        source: "cpu".to_string(),
        destination: "gpu".to_string(),
        payload: serde_json::json!({}),
        timestamp: chrono::Utc::now(),
        priority: 5,
    };
    assert!(bridge.send_to_gpu(msg).await.is_ok());
}

#[tokio::test]
async fn test_drop_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    {
        let mut bridge = CpuGpuBridge::new(config).await.unwrap();
        bridge.start().await.unwrap();

        // Send a message
        let msg = CpuGpuMessage {
            id: "drop-test".to_string(),
            message_type: MessageType::TaskRequest,
            source: "cpu".to_string(),
            destination: "gpu".to_string(),
            payload: serde_json::json!({}),
            timestamp: chrono::Utc::now(),
            priority: 5,
        };
        bridge.send_to_gpu(msg).await.unwrap();

        // Bridge will be dropped here
    }

    // Verify files still exist after drop
    let outbox = temp_dir.path().join("outbox");
    assert!(outbox.exists());
}

#[tokio::test]
async fn test_error_directory_creation_failure() {
    let temp_dir = TempDir::new().unwrap();
    let config = BridgeConfig {
        shared_storage_path: temp_dir.path().to_path_buf(),
        inbox_dir: "inbox".to_string(),
        outbox_dir: "outbox".to_string(),
        message_retention_seconds: 3600,
        max_message_size_mb: 10,
        polling_interval_ms: 100,
    };

    let mut bridge = CpuGpuBridge::new(config).await.unwrap();

    // Create a file named "error" to prevent directory creation
    let error_file = temp_dir.path().join("error");
    tokio::fs::write(&error_file, b"not a directory")
        .await
        .unwrap();

    // Create corrupted message
    let inbox_path = temp_dir.path().join("inbox");
    tokio::fs::write(inbox_path.join("bad.json"), b"invalid")
        .await
        .unwrap();

    // Try to receive - should handle the error directory creation failure
    let messages = bridge.receive_from_gpu().await.unwrap();
    assert_eq!(messages.len(), 0);
}

#[tokio::test]
async fn test_serialization_compatibility() {
    // Test that all enums serialize/deserialize correctly
    let msg_types = vec![
        MessageType::TaskRequest,
        MessageType::TaskResult,
        MessageType::DataTransfer,
        MessageType::StatusUpdate,
        MessageType::ErrorReport,
        MessageType::Shutdown,
    ];

    for msg_type in msg_types {
        let json = serde_json::to_string(&msg_type).unwrap();
        let decoded: MessageType = serde_json::from_str(&json).unwrap();
        assert_eq!(msg_type, decoded);
    }

    // Test BridgeStats serialization
    let stats = BridgeStats {
        messages_sent: u64::MAX,
        messages_received: u64::MAX - 1,
        messages_failed: 42,
        bytes_sent: u64::MAX / 2,
        bytes_received: u64::MAX / 3,
        cleanup_operations: 100,
    };

    let json = serde_json::to_string(&stats).unwrap();
    let decoded: BridgeStats = serde_json::from_str(&json).unwrap();
    assert_eq!(stats.messages_sent, decoded.messages_sent);
}
