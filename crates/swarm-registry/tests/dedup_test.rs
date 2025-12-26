//! Test specifically for deduplication behavior

use swarm_registry::store::{ContentAddressableStore, StoreConfig};
use tempfile::TempDir;

#[tokio::test]
async fn test_deduplication_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let config = StoreConfig {
        storage_path: temp_dir.path().to_path_buf(),
        enable_dedup: true,
        enable_tiers: false,
        ..Default::default()
    };

    let store = ContentAddressableStore::new(config).await.unwrap();

    // Store same content with two different keys
    let data = b"duplicate content";
    let hash1 = store.put("key1", data).await.unwrap();
    let hash2 = store.put("key2", data).await.unwrap();

    // Hashes should be the same (deduplication)
    assert_eq!(hash1, hash2);

    // Both keys should exist
    assert!(store.exists("key1").await);
    assert!(store.exists("key2").await);

    // Both keys should return the same data
    let data1 = store.get("key1").await.unwrap();
    let data2 = store.get("key2").await.unwrap();
    assert_eq!(data1, data);
    assert_eq!(data2, data);

    // List should show both objects
    let objects = store.list().await.unwrap();
    assert_eq!(objects.len(), 2);

    // Delete key1
    store.delete("key1").await.unwrap();

    // key1 should not exist
    assert!(!store.exists("key1").await);

    // key2 should still exist because content is shared
    assert!(store.exists("key2").await);

    // key2 should still return the data
    let data2_after = store.get("key2").await.unwrap();
    assert_eq!(data2_after, data);
}
