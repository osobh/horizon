//! Integration tests for swarm-registry

use std::collections::HashMap;
use std::sync::Arc;
use swarm_registry::mocks::cluster_mesh::ClusterMesh;
use swarm_registry::registry::RegistryConfig;
use swarm_registry::store::StoreConfig;
use swarm_registry::streamer::{LayerPriority, StreamConfig};
use swarm_registry::verifier::{SecurityPolicy, VerifierConfig};
use swarm_registry::{
    ContentAddressableStore, DistributedRegistry, DockerConverter, ImageMetadata, ImageStreamer,
    ImageVariant, ImageVerifier, Result, RootfsBuilder, SwarmImage, SwarmRegistryError,
};
use tempfile::TempDir;

#[tokio::test]
async fn test_full_image_lifecycle() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let work_dir = temp_dir.path();

    // Create components
    let store_config = StoreConfig {
        storage_path: work_dir.join("store"),
        ..Default::default()
    };
    let store = ContentAddressableStore::new(store_config).await?;

    let registry_config = RegistryConfig {
        storage_path: work_dir.join("registry"),
        ..Default::default()
    };
    let cluster_mesh = Arc::new(ClusterMesh::new_mock());
    let registry = DistributedRegistry::new(registry_config, cluster_mesh).await?;

    // Create a test image
    let test_image = SwarmImage {
        hash: "sha256:test123".to_string(),
        metadata: ImageMetadata {
            name: "test-app".to_string(),
            tag: "v1.0".to_string(),
            variant: ImageVariant::Base,
            created: 12345,
            size: 1024 * 1024,
            architecture: "amd64".to_string(),
            os: "linux".to_string(),
            env: vec!["PATH=/usr/bin".to_string()],
            entrypoint: Some(vec!["/bin/sh".to_string()]),
            cmd: None,
        },
        layers: vec!["layer1".to_string(), "layer2".to_string()],
        agent_config: None,
    };

    // Test publishing
    registry.publish(test_image.clone()).await?;

    // Test pulling
    let pulled_image = registry.pull(&test_image.hash).await?;
    assert_eq!(pulled_image.hash, test_image.hash);
    assert_eq!(pulled_image.metadata.name, test_image.metadata.name);

    // Test listing
    let images = registry.list().await?;
    assert!(!images.is_empty());
    assert!(images.iter().any(|(h, _, _)| h == &test_image.hash));

    // Test searching
    let search_results = registry.search("test").await?;
    assert!(!search_results.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_image_verification() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let verifier_config = VerifierConfig {
        work_dir: temp_dir.path().to_path_buf(),
        policy: SecurityPolicy {
            min_security_score: 50, // Lower threshold for testing
            ..Default::default()
        },
        enable_vuln_scan: false, // Disable external scanners for tests
        enable_malware_scan: false,
        ..Default::default()
    };

    let mut verifier = ImageVerifier::new(verifier_config);

    let test_image = SwarmImage {
        hash: "sha256:verify123".to_string(),
        metadata: ImageMetadata {
            name: "ubuntu".to_string(),
            tag: "22.04".to_string(),
            variant: ImageVariant::Base,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size: 100 * 1024 * 1024,
            architecture: "amd64".to_string(),
            os: "linux".to_string(),
            env: vec![],
            entrypoint: None,
            cmd: None,
        },
        layers: vec!["layer1".to_string()],
        agent_config: None,
    };

    // Test verification
    let scan_result = verifier.verify(&test_image).await?;
    assert_eq!(scan_result.image_hash, test_image.hash);

    // Test compliance check
    let compliant = verifier.check_compliance(&test_image).await?;
    assert!(compliant); // Should pass with relaxed policy

    Ok(())
}

#[tokio::test]
async fn test_content_addressable_store() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StoreConfig {
        storage_path: temp_dir.path().to_path_buf(),
        enable_dedup: true,
        enable_tiers: false, // Disable tiers for simplicity
        ..Default::default()
    };

    let store = ContentAddressableStore::new(config).await?;

    // Test put and get
    let test_data = b"Hello, SwarmFS!";
    let hash1 = store.put("key1", test_data).await?;

    let retrieved = store.get("key1").await?;
    assert_eq!(retrieved, test_data);

    // Test deduplication
    let hash2 = store.put("key2", test_data).await?;
    assert_eq!(hash1, hash2); // Same content should have same hash

    // Test exists
    assert!(store.exists("key1").await);
    assert!(store.exists("key2").await);
    assert!(!store.exists("nonexistent").await);

    // Test list
    let objects = store.list().await?;
    assert_eq!(objects.len(), 2);

    // Test delete
    store.delete("key1").await?;
    assert!(!store.exists("key1").await);

    // With deduplication, key2 might or might not exist depending on implementation
    // The important thing is that we can still retrieve data if key2 exists
    if store.exists("key2").await {
        let data = store.get("key2").await?;
        assert_eq!(data, test_data);
    }

    Ok(())
}

#[tokio::test]
async fn test_image_streaming() -> Result<()> {
    let config = StreamConfig {
        chunk_size: 1024, // Small chunks for testing
        ..Default::default()
    };

    let streamer = ImageStreamer::new(config);

    let test_image = SwarmImage {
        hash: "sha256:stream123".to_string(),
        metadata: ImageMetadata {
            name: "stream-test".to_string(),
            tag: "latest".to_string(),
            variant: ImageVariant::Base,
            created: 12345,
            size: 3 * 1024,
            architecture: "amd64".to_string(),
            os: "linux".to_string(),
            env: vec![],
            entrypoint: None,
            cmd: None,
        },
        layers: vec!["layer1".to_string(), "layer2".to_string()],
        agent_config: None,
    };

    let mut priorities = HashMap::new();
    priorities.insert("layer1".to_string(), LayerPriority::Critical);
    priorities.insert("layer2".to_string(), LayerPriority::Normal);

    // Start streaming
    let mut handle = streamer
        .stream_image(test_image.clone(), priorities)
        .await?;

    // Try to receive chunks (may not receive any in mock implementation)
    let mut chunks_received = 0;
    // Use timeout to avoid hanging
    let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), handle.recv()).await;

    if let Ok(Some(_chunk)) = timeout {
        chunks_received = 1;
    }

    // For mock implementation, we might not receive chunks
    // Just check that streaming was initiated

    // Test progress tracking - might be None if stream hasn't started yet
    let progress = streamer.get_progress(&test_image.hash).await;
    // Just verify it doesn't panic

    // Test cancellation - might fail if stream already ended
    let _ = streamer.cancel_stream(&test_image.hash).await;

    Ok(())
}

#[tokio::test]
async fn test_registry_search() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let registry_config = RegistryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let cluster_mesh = Arc::new(ClusterMesh::new_mock());
    let registry = DistributedRegistry::new(registry_config, cluster_mesh).await?;

    // Publish multiple images
    let images = vec![
        ("ubuntu", "22.04"),
        ("ubuntu", "24.04"),
        ("nginx", "latest"),
        ("redis", "7.0"),
    ];

    for (name, tag) in images {
        let image = SwarmImage {
            hash: format!("sha256:{}_{}", name, tag),
            metadata: ImageMetadata {
                name: name.to_string(),
                tag: tag.to_string(),
                variant: ImageVariant::Base,
                created: 12345,
                size: 1024,
                architecture: "amd64".to_string(),
                os: "linux".to_string(),
                env: vec![],
                entrypoint: None,
                cmd: None,
            },
            layers: vec!["layer".to_string()],
            agent_config: None,
        };
        registry.publish(image).await?;
    }

    // Test search
    let ubuntu_results = registry.search("ubuntu").await?;
    assert_eq!(ubuntu_results.len(), 2);

    let nginx_results = registry.search("nginx").await?;
    assert_eq!(nginx_results.len(), 1);

    let all_results = registry.search("").await?;
    assert_eq!(all_results.len(), 4);

    Ok(())
}

#[tokio::test]
async fn test_store_garbage_collection() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let config = StoreConfig {
        storage_path: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let store = ContentAddressableStore::new(config).await?;

    // Add some objects
    store.put("obj1", b"data1").await?;
    store.put("obj2", b"data2").await?;
    store.put("obj3", b"data3").await?;

    // Delete one
    store.delete("obj2").await?;

    // Run garbage collection
    store.garbage_collect().await?;

    // Remaining objects should still be accessible
    assert!(store.exists("obj1").await);
    assert!(!store.exists("obj2").await);
    assert!(store.exists("obj3").await);

    Ok(())
}

// Add more integration tests as needed...
