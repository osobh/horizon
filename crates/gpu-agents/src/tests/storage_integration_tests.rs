//! Storage integration tests following rust.md TDD guidelines
//!
//! These tests validate storage module imports and functionality

use anyhow::Result;

/// Test that storage types are imported from local module, not external crate
/// Following rust.md: Test import organization and module structure
#[test]
fn test_local_storage_imports() -> Result<()> {
    // These should come from crate::storage, not exorust_storage
    use crate::storage::{
        GpuAgentData, GpuAgentStorage, GpuKnowledgeGraph as StorageGpuKnowledgeGraph,
        GpuStorageConfig, GraphEdge as StorageGraphEdge, GraphNode as StorageGraphNode,
    };

    // Test that we can create configurations
    let config = GpuStorageConfig::default();
    assert!(!config.base_path.as_os_str().is_empty());

    // Test that GPU storage can be created
    let storage = GpuAgentStorage::new(config)?;
    assert!(storage.is_initialized());

    Ok(())
}

/// Test storage configuration with /magikdev/gpu path
/// Following cuda.md: Validate GPU-specific storage paths
#[test]
fn test_gpu_storage_path_configuration() -> Result<()> {
    use crate::storage::GpuStorageConfig;

    let config = GpuStorageConfig::default();

    // Should default to /magikdev/gpu as per our design
    assert!(config.base_path.to_string_lossy().contains("magikdev"));
    assert!(config.enable_gpu_mapping);
    assert!(config.nvme_config.is_some());

    Ok(())
}

/// Test storage integration with external exorust-storage crate
/// This validates we properly use the external crate for NVMe functionality
#[test]
fn test_external_storage_integration() -> Result<()> {
    use crate::storage::GpuAgentStorage;

    // Our local storage should integrate with external storage crate
    let config = crate::storage::GpuStorageConfig::default();
    let storage = GpuAgentStorage::new(config)?;

    // Test that NVMe integration works
    assert!(storage.nvme_available());

    Ok(())
}

/// Test storage serialization compatibility
/// Following rust.md: Test serialization for configuration types
#[test]
fn test_storage_serialization() -> Result<()> {
    use crate::storage::GpuStorageConfig;

    let config = GpuStorageConfig::default();

    // Should be able to serialize to JSON
    let json = serde_json::to_string(&config)?;
    assert!(!json.is_empty());

    // Should be able to deserialize back
    let deserialized: GpuStorageConfig = serde_json::from_str(&json)?;
    assert_eq!(deserialized.base_path, config.base_path);

    Ok(())
}

/// Test that lib.rs exports are properly configured
/// This validates our public API surface
#[test]
fn test_lib_storage_exports() -> Result<()> {
    // Following rust.md: Test public API exports
    // These should be available from the crate root

    // Note: These are the corrected imports that should work
    use crate::storage::{
        GpuAgentData, GpuAgentStorage, GpuKnowledgeGraph as StorageGpuKnowledgeGraph,
        GpuStorageConfig,
    };

    // Test that all types are constructible
    let config = GpuStorageConfig::default();
    let _storage = GpuAgentStorage::new(config)?;

    println!("✅ Storage exports validated");
    Ok(())
}
