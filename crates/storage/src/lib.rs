//! ExoRust GPU Storage Layer
//!
//! This crate provides storage layer with NVMe optimization and
//! knowledge graph storage primitives for the ExoRust system.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod benchmarks;
pub mod error;
pub mod gpu_cache;
pub mod graph;
pub mod graph_csr;
pub mod graph_format;
pub mod graph_storage;
pub mod graph_wal;
pub mod memory;
pub mod nvme;

#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod async_storage_tests;

#[cfg(test)]
mod async_concurrency_tests;

pub use benchmarks::{BenchmarkConfig, BenchmarkResults, FullBenchmarkResults, StorageBenchmark};
pub use error::StorageError;
pub use gpu_cache::{CacheEntry, CacheStats, GpuCache};
pub use graph::{GraphEdge, GraphNode, KnowledgeGraphStorage};
pub use graph_csr::{Edge, EdgeIterator, GraphCSR};
pub use graph_format::{node_flags, NodeRecord};
pub use graph_storage::GraphStorage;
pub use graph_wal::{GraphWAL, NodeUpdates, WALEntry, WALReader};
pub use memory::MemoryStorage;
pub use nvme::{NvmeConfig, NvmeStats, NvmeStorage};

/// Storage interface for ExoRust components
#[async_trait::async_trait]
pub trait Storage: Send + Sync {
    /// Store data with key
    async fn store(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;

    /// Retrieve data by key
    async fn retrieve(&self, key: &str) -> Result<Vec<u8>, StorageError>;

    /// Delete data by key
    async fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// List all keys with prefix
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>, StorageError>;

    /// Get storage statistics
    async fn stats(&self) -> Result<StorageStats, StorageError>;
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub total_files: u64,
    pub read_throughput_mbps: f64,
    pub write_throughput_mbps: f64,
}

impl StorageStats {
    pub fn utilization_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }

    pub fn free_space_percent(&self) -> f64 {
        100.0 - self.utilization_percent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_stats_creation() {
        let stats = StorageStats {
            total_bytes: 1024 * 1024,
            used_bytes: 512 * 1024,
            available_bytes: 512 * 1024,
            total_files: 100,
            read_throughput_mbps: 150.5,
            write_throughput_mbps: 120.3,
        };

        assert_eq!(stats.total_bytes, 1024 * 1024);
        assert_eq!(stats.used_bytes, 512 * 1024);
        assert_eq!(stats.available_bytes, 512 * 1024);
        assert_eq!(stats.total_files, 100);
        assert_eq!(stats.read_throughput_mbps, 150.5);
        assert_eq!(stats.write_throughput_mbps, 120.3);
    }

    #[test]
    fn test_storage_stats_utilization() {
        let stats = StorageStats {
            total_bytes: 2048,
            used_bytes: 1024,
            available_bytes: 1024,
            total_files: 50,
            read_throughput_mbps: 200.0,
            write_throughput_mbps: 180.0,
        };

        assert_eq!(stats.utilization_percent(), 50.0);
        assert_eq!(stats.free_space_percent(), 50.0);
    }

    #[test]
    fn test_storage_stats_zero_total() {
        let stats = StorageStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            total_files: 0,
            read_throughput_mbps: 0.0,
            write_throughput_mbps: 0.0,
        };

        assert_eq!(stats.utilization_percent(), 0.0);
        assert_eq!(stats.free_space_percent(), 100.0);
    }

    #[test]
    fn test_storage_stats_full_utilization() {
        let stats = StorageStats {
            total_bytes: 4096,
            used_bytes: 4096,
            available_bytes: 0,
            total_files: 200,
            read_throughput_mbps: 300.75,
            write_throughput_mbps: 250.25,
        };

        assert_eq!(stats.utilization_percent(), 100.0);
        assert_eq!(stats.free_space_percent(), 0.0);
    }

    #[test]
    fn test_storage_stats_clone() {
        let original = StorageStats {
            total_bytes: 8192,
            used_bytes: 4096,
            available_bytes: 4096,
            total_files: 75,
            read_throughput_mbps: 175.5,
            write_throughput_mbps: 155.5,
        };

        let cloned = original.clone();
        assert_eq!(original.total_bytes, cloned.total_bytes);
        assert_eq!(original.used_bytes, cloned.used_bytes);
        assert_eq!(original.available_bytes, cloned.available_bytes);
        assert_eq!(original.total_files, cloned.total_files);
        assert_eq!(original.read_throughput_mbps, cloned.read_throughput_mbps);
        assert_eq!(original.write_throughput_mbps, cloned.write_throughput_mbps);
    }

    #[test]
    fn test_storage_stats_debug() {
        let stats = StorageStats {
            total_bytes: 1024,
            used_bytes: 512,
            available_bytes: 512,
            total_files: 10,
            read_throughput_mbps: 100.0,
            write_throughput_mbps: 90.0,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("StorageStats"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("512"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("90"));
    }

    #[test]
    fn test_storage_stats_serialization() -> serde_json::Result<()> {
        let stats = StorageStats {
            total_bytes: 16384,
            used_bytes: 8192,
            available_bytes: 8192,
            total_files: 150,
            read_throughput_mbps: 250.75,
            write_throughput_mbps: 200.25,
        };

        let serialized = serde_json::to_string(&stats)?;
        let deserialized: StorageStats = serde_json::from_str(&serialized)?;

        assert_eq!(stats.total_bytes, deserialized.total_bytes);
        assert_eq!(stats.used_bytes, deserialized.used_bytes);
        assert_eq!(stats.available_bytes, deserialized.available_bytes);
        assert_eq!(stats.total_files, deserialized.total_files);
        assert_eq!(
            stats.read_throughput_mbps,
            deserialized.read_throughput_mbps
        );
        assert_eq!(
            stats.write_throughput_mbps,
            deserialized.write_throughput_mbps
        );
        Ok(())
    }

    #[test]
    fn test_storage_stats_edge_cases() {
        // Maximum values
        let max_stats = StorageStats {
            total_bytes: u64::MAX,
            used_bytes: u64::MAX,
            available_bytes: 0,
            total_files: u64::MAX,
            read_throughput_mbps: f64::MAX,
            write_throughput_mbps: f64::MAX,
        };

        assert_eq!(max_stats.utilization_percent(), 100.0);
        assert_eq!(max_stats.free_space_percent(), 0.0);

        // Precision test
        let precise_stats = StorageStats {
            total_bytes: 3,
            used_bytes: 1,
            available_bytes: 2,
            total_files: 1,
            read_throughput_mbps: 33.333333,
            write_throughput_mbps: 66.666666,
        };

        assert!((precise_stats.utilization_percent() - 33.333333333333336).abs() < 0.001);
        assert!((precise_stats.free_space_percent() - 66.666666666666664).abs() < 0.001);
    }

    #[test]
    fn test_storage_stats_negative_throughput() {
        // Some systems might report negative throughput in error conditions
        let stats = StorageStats {
            total_bytes: 1024,
            used_bytes: 512,
            available_bytes: 512,
            total_files: 10,
            read_throughput_mbps: -1.0,
            write_throughput_mbps: -1.0,
        };

        assert_eq!(stats.read_throughput_mbps, -1.0);
        assert_eq!(stats.write_throughput_mbps, -1.0);
        assert_eq!(stats.utilization_percent(), 50.0);
    }

    #[test]
    fn test_module_exports() {
        // Test that all public types are accessible
        let _error = StorageError::StorageNotInitialized;

        let _stats = StorageStats {
            total_bytes: 1024,
            used_bytes: 512,
            available_bytes: 512,
            total_files: 5,
            read_throughput_mbps: 100.0,
            write_throughput_mbps: 80.0,
        };

        // Test re-exports are available
        let _node = GraphNode {
            id: uuid::Uuid::new_v4(),
            data: vec![1, 2, 3],
            node_type: "test".to_string(),
        };

        let _edge = GraphEdge {
            id: uuid::Uuid::new_v4(),
            from_node: uuid::Uuid::new_v4(),
            to_node: uuid::Uuid::new_v4(),
            edge_type: "test".to_string(),
            weight: 1.0,
        };
    }

    #[test]
    fn test_storage_trait_object_bounds() {
        // Test that Storage trait can be used as a trait object
        fn accepts_storage(_storage: &dyn Storage) {}

        // This test ensures the trait bounds are correct for trait objects
        // We can't test actual implementation here since we don't have concrete types
        // but we can verify the trait is object-safe
    }

    #[test]
    fn test_storage_stats_consistency() {
        let stats = StorageStats {
            total_bytes: 1000,
            used_bytes: 300,
            available_bytes: 700,
            total_files: 20,
            read_throughput_mbps: 150.0,
            write_throughput_mbps: 120.0,
        };

        // Verify consistency: used + available should equal total
        assert_eq!(stats.used_bytes + stats.available_bytes, stats.total_bytes);

        // Verify utilization calculation
        let expected_utilization = (300.0 / 1000.0) * 100.0;
        assert_eq!(stats.utilization_percent(), expected_utilization);

        // Verify free space calculation
        assert_eq!(stats.free_space_percent(), 100.0 - expected_utilization);
    }

    #[test]
    fn test_storage_stats_nan_infinity_handling() {
        // Test with NaN values
        let stats_nan = StorageStats {
            total_bytes: 1000,
            used_bytes: 500,
            available_bytes: 500,
            total_files: 10,
            read_throughput_mbps: f64::NAN,
            write_throughput_mbps: f64::INFINITY,
        };

        // Operations should still work with NaN/Infinity
        assert_eq!(stats_nan.utilization_percent(), 50.0);
        assert_eq!(stats_nan.free_space_percent(), 50.0);
        assert!(stats_nan.read_throughput_mbps.is_nan());
        assert!(stats_nan.write_throughput_mbps.is_infinite());
    }

    #[test]
    fn test_storage_stats_extreme_values() {
        // Test with maximum values
        let max_stats = StorageStats {
            total_bytes: u64::MAX,
            used_bytes: u64::MAX - 1,
            available_bytes: 1,
            total_files: u64::MAX,
            read_throughput_mbps: f64::MAX,
            write_throughput_mbps: f64::MIN,
        };

        let utilization = max_stats.utilization_percent();
        assert!(utilization > 99.0); // Very close to 100%
        assert!(utilization <= 100.0);
    }

    #[test]
    fn test_storage_stats_overflow_protection() {
        // Test potential overflow scenarios
        let overflow_stats = StorageStats {
            total_bytes: 1,
            used_bytes: u64::MAX, // More used than total (invalid but testing robustness)
            available_bytes: 0,
            total_files: 1,
            read_throughput_mbps: 1.0,
            write_throughput_mbps: 1.0,
        };

        // Should handle overflow gracefully
        let utilization = overflow_stats.utilization_percent();
        assert!(utilization.is_finite()); // Should not be infinite
    }

    #[test]
    fn test_storage_stats_precision() {
        // Test floating point precision edge cases
        let precision_stats = StorageStats {
            total_bytes: 3,
            used_bytes: 1,
            available_bytes: 2,
            total_files: 1,
            read_throughput_mbps: f64::MIN_POSITIVE,
            write_throughput_mbps: f64::EPSILON,
        };

        let utilization = precision_stats.utilization_percent();
        let expected = (1.0 / 3.0) * 100.0;
        assert!((utilization - expected).abs() < f64::EPSILON * 100.0);
    }

    #[test]
    fn test_storage_trait_send_sync() {
        // Verify Storage trait is Send + Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // Can't directly test trait object, but verify bounds compile
        fn _verify_storage_bounds<T: Storage>() {
            assert_send::<T>();
            assert_sync::<T>();
        }
    }

    #[test]
    fn test_storage_stats_default_values() {
        // Test stats with default/zero values
        let default_stats = StorageStats {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            total_files: 0,
            read_throughput_mbps: 0.0,
            write_throughput_mbps: 0.0,
        };

        assert_eq!(default_stats.utilization_percent(), 0.0);
        assert_eq!(default_stats.free_space_percent(), 100.0);
    }

    #[test]
    fn test_storage_stats_partial_eq_semantics() {
        let stats1 = StorageStats {
            total_bytes: 1000,
            used_bytes: 500,
            available_bytes: 500,
            total_files: 10,
            read_throughput_mbps: 100.0,
            write_throughput_mbps: 80.0,
        };

        let stats2 = StorageStats {
            total_bytes: 1000,
            used_bytes: 500,
            available_bytes: 500,
            total_files: 10,
            read_throughput_mbps: 100.0,
            write_throughput_mbps: 80.0,
        };

        // Test that clone produces identical stats
        let cloned = stats1.clone();
        assert_eq!(stats1.total_bytes, cloned.total_bytes);
        assert_eq!(stats1.utilization_percent(), cloned.utilization_percent());

        // Test that identical stats produce same calculations
        assert_eq!(stats1.utilization_percent(), stats2.utilization_percent());
        assert_eq!(stats1.free_space_percent(), stats2.free_space_percent());
    }

    #[test]
    fn test_storage_error_integration() {
        // Test that all public error types are accessible
        let errors = vec![
            StorageError::StorageNotInitialized,
            StorageError::KeyNotFound {
                key: "test".to_string(),
            },
            StorageError::StorageFull { available: 100 },
            StorageError::InvalidNode {
                id: 1,
                reason: "test".to_string(),
            },
            StorageError::InvalidEdge {
                from: 1,
                to: 2,
                reason: "test".to_string(),
            },
            StorageError::InvalidDataFormat {
                reason: "test".to_string(),
            },
            StorageError::WALError {
                reason: "test".to_string(),
            },
            StorageError::LockPoisoned {
                resource: "test".to_string(),
            },
        ];

        // Verify all error types can be created and displayed
        for error in errors {
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
        }
    }
}
