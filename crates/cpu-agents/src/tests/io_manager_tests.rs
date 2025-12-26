//! Tests for I/O Manager

use crate::{io_manager::*, Result};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_io_manager_creation() {
    let config = IoConfig {
        max_file_size_mb: 100,
        buffer_size_kb: 64,
        enable_compression: true,
        temp_dir: PathBuf::from("/tmp/exorust-io"),
    };

    let manager = IoManager::new(config.clone()).await.unwrap();
    assert_eq!(manager.config().max_file_size_mb, 100);
    assert!(manager.config().enable_compression);
}

#[tokio::test]
async fn test_file_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig {
        max_file_size_mb: 10,
        buffer_size_kb: 64,
        enable_compression: false,
        temp_dir: temp_dir.path().to_path_buf(),
    };

    let mut manager = IoManager::new(config).await.unwrap();

    // Test write operation
    let write_op = IoOperation::WriteFile {
        path: temp_dir.path().join("test.txt"),
        content: b"Hello, ExoRust!".to_vec(),
        create_dirs: true,
    };

    let result = manager.execute(write_op).await.unwrap();
    assert!(matches!(result, IoResult::Success { .. }));

    // Test read operation
    let read_op = IoOperation::ReadFile {
        path: temp_dir.path().join("test.txt"),
    };

    let result = manager.execute(read_op).await.unwrap();
    match result {
        IoResult::FileContent(content) => {
            assert_eq!(content, b"Hello, ExoRust!");
        }
        _ => panic!("Expected file content"),
    }

    // Test delete operation
    let delete_op = IoOperation::DeleteFile {
        path: temp_dir.path().join("test.txt"),
    };

    let result = manager.execute(delete_op).await.unwrap();
    assert!(matches!(result, IoResult::Success { .. }));
}

#[tokio::test]
async fn test_directory_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig::default();
    let mut manager = IoManager::new(config).await.unwrap();

    // Create directory
    let create_op = IoOperation::CreateDirectory {
        path: temp_dir.path().join("test_dir"),
        recursive: true,
    };

    let result = manager.execute(create_op).await.unwrap();
    assert!(matches!(result, IoResult::Success { .. }));

    // List directory
    let list_op = IoOperation::ListDirectory {
        path: temp_dir.path().to_path_buf(),
        recursive: false,
    };

    let result = manager.execute(list_op).await.unwrap();
    match result {
        IoResult::DirectoryListing(entries) => {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].file_name().unwrap(), "test_dir");
        }
        _ => panic!("Expected directory listing"),
    }
}

#[tokio::test]
async fn test_compression() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig {
        max_file_size_mb: 10,
        buffer_size_kb: 64,
        enable_compression: true,
        temp_dir: temp_dir.path().to_path_buf(),
    };

    let mut manager = IoManager::new(config).await.unwrap();

    // Write compressed
    let content = vec![42u8; 1024]; // Repetitive data compresses well
    let write_op = IoOperation::WriteCompressed {
        path: temp_dir.path().join("compressed.gz"),
        content: content.clone(),
    };

    let result = manager.execute(write_op).await.unwrap();
    assert!(matches!(result, IoResult::Success { .. }));

    // Read compressed
    let read_op = IoOperation::ReadCompressed {
        path: temp_dir.path().join("compressed.gz"),
    };

    let result = manager.execute(read_op).await.unwrap();
    match result {
        IoResult::FileContent(decompressed) => {
            assert_eq!(decompressed, content);
        }
        _ => panic!("Expected decompressed content"),
    }
}

#[tokio::test]
async fn test_batch_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig::default();
    let mut manager = IoManager::new(config).await.unwrap();

    // Create multiple files
    let mut operations = Vec::new();
    for i in 0..5 {
        operations.push(IoOperation::WriteFile {
            path: temp_dir.path().join(format!("file{}.txt", i)),
            content: format!("Content {}", i).into_bytes(),
            create_dirs: false,
        });
    }

    let results = manager.execute_batch(operations).await.unwrap();
    assert_eq!(results.len(), 5);
    assert!(results
        .iter()
        .all(|r| matches!(r, IoResult::Success { .. })));

    // Verify files exist
    for i in 0..5 {
        assert!(temp_dir.path().join(format!("file{}.txt", i)).exists());
    }
}

#[tokio::test]
async fn test_error_handling() {
    let config = IoConfig::default();
    let mut manager = IoManager::new(config).await.unwrap();

    // Try to read non-existent file
    let read_op = IoOperation::ReadFile {
        path: PathBuf::from("/nonexistent/file.txt"),
    };

    let result = manager.execute(read_op).await;
    assert!(result.is_err());

    // Try to write to invalid path
    let write_op = IoOperation::WriteFile {
        path: PathBuf::from("/root/no_permission.txt"),
        content: b"test".to_vec(),
        create_dirs: false,
    };

    let result = manager.execute(write_op).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_size_limits() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig {
        max_file_size_mb: 1, // 1MB limit
        buffer_size_kb: 64,
        enable_compression: false,
        temp_dir: temp_dir.path().to_path_buf(),
    };

    let mut manager = IoManager::new(config).await.unwrap();

    // Try to write file larger than limit
    let large_content = vec![0u8; 2 * 1024 * 1024]; // 2MB
    let write_op = IoOperation::WriteFile {
        path: temp_dir.path().join("large.txt"),
        content: large_content,
        create_dirs: false,
    };

    let result = manager.execute(write_op).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let temp_dir = TempDir::new().unwrap();
    let config = IoConfig::default();
    let manager = IoManager::new(config).await.unwrap();
    let manager = std::sync::Arc::new(tokio::sync::Mutex::new(manager));

    // Spawn multiple concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let manager_clone = manager.clone();
        let path = temp_dir.path().join(format!("concurrent{}.txt", i));
        let handle = tokio::spawn(async move {
            let op = IoOperation::WriteFile {
                path,
                content: format!("Concurrent {}", i).into_bytes(),
                create_dirs: false,
            };
            manager_clone.lock().await.execute(op).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    let results: Vec<_> = futures::future::join_all(handles).await;

    // Verify all succeeded
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.unwrap().is_ok());
    }
}
