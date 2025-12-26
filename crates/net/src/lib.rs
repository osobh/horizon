//! ExoRust Zero-Copy Networking
//!
//! This crate provides zero-copy networking for GPU containers
//! and inter-node communication in the ExoRust system.

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod benchmarks;
pub mod error;
pub mod memory;
pub mod protocol;
pub mod zero_copy;

#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod additional_tests_simple;

pub use benchmarks::{
    FullNetworkBenchmarkResults, NetworkBenchmark, NetworkBenchmarkConfig, NetworkBenchmarkResults,
};
pub use error::NetworkError;
pub use memory::MemoryNetwork;
pub use protocol::{Message, MessageType};
pub use zero_copy::ZeroCopyTransport;

/// Network interface for ExoRust components
#[async_trait::async_trait]
pub trait Network: Send + Sync {
    /// Send message to endpoint
    async fn send(&self, endpoint: &str, message: Message) -> Result<(), NetworkError>;

    /// Receive message from any endpoint
    async fn receive(&self) -> Result<(String, Message), NetworkError>;

    /// Get network statistics
    async fn stats(&self) -> Result<NetworkStats, NetworkError>;
}

/// Network statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub average_latency_us: f64,
    pub throughput_mbps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_stats_default() {
        let stats = NetworkStats::default();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.average_latency_us, 0.0);
        assert_eq!(stats.throughput_mbps, 0.0);
    }

    #[test]
    fn test_network_stats_creation() {
        let stats = NetworkStats {
            bytes_sent: 1024,
            bytes_received: 2048,
            messages_sent: 10,
            messages_received: 20,
            average_latency_us: 150.5,
            throughput_mbps: 1000.0,
        };

        assert_eq!(stats.bytes_sent, 1024);
        assert_eq!(stats.bytes_received, 2048);
        assert_eq!(stats.messages_sent, 10);
        assert_eq!(stats.messages_received, 20);
        assert_eq!(stats.average_latency_us, 150.5);
        assert_eq!(stats.throughput_mbps, 1000.0);
    }

    #[test]
    fn test_network_stats_serialization() {
        let stats = NetworkStats {
            bytes_sent: 1_000_000,
            bytes_received: 2_000_000,
            messages_sent: 100,
            messages_received: 200,
            average_latency_us: 75.25,
            throughput_mbps: 10_000.0,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: NetworkStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_network_stats_clone() {
        let stats = NetworkStats {
            bytes_sent: 5000,
            bytes_received: 6000,
            messages_sent: 50,
            messages_received: 60,
            average_latency_us: 200.0,
            throughput_mbps: 5000.0,
        };

        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_network_stats_debug() {
        let stats = NetworkStats {
            bytes_sent: 100,
            bytes_received: 200,
            messages_sent: 1,
            messages_received: 2,
            average_latency_us: 50.0,
            throughput_mbps: 100.0,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("bytes_sent: 100"));
        assert!(debug_str.contains("bytes_received: 200"));
        assert!(debug_str.contains("messages_sent: 1"));
        assert!(debug_str.contains("messages_received: 2"));
        assert!(debug_str.contains("average_latency_us: 50.0"));
        assert!(debug_str.contains("throughput_mbps: 100.0"));
    }

    #[test]
    fn test_network_stats_max_values() {
        let stats = NetworkStats {
            bytes_sent: u64::MAX,
            bytes_received: u64::MAX,
            messages_sent: u64::MAX,
            messages_received: u64::MAX,
            average_latency_us: f64::MAX,
            throughput_mbps: f64::MAX,
        };

        assert_eq!(stats.bytes_sent, u64::MAX);
        assert_eq!(stats.average_latency_us, f64::MAX);
    }

    #[test]
    fn test_network_stats_special_float_values() {
        let stats = NetworkStats {
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            average_latency_us: f64::INFINITY,
            throughput_mbps: f64::NEG_INFINITY,
        };

        assert!(stats.average_latency_us.is_infinite());
        assert!(stats.throughput_mbps.is_infinite());
    }

    #[test]
    fn test_network_stats_nan_values() {
        let stats = NetworkStats {
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            average_latency_us: f64::NAN,
            throughput_mbps: 0.0,
        };

        assert!(stats.average_latency_us.is_nan());
    }

    #[test]
    fn test_module_reexports() {
        // Verify that all public modules and types are accessible
        let _msg_type = MessageType::AgentSpawn;
        let _error = NetworkError::ConnectionFailed {
            endpoint: "test".to_string(),
        };
    }

    #[test]
    fn test_network_trait_object_safety() {
        // This test verifies that Network trait is object-safe
        fn _accepts_network_trait(network: Box<dyn Network>) {
            // If this compiles, the trait is object-safe
            let _ = network;
        }
    }

    #[test]
    fn test_network_stats_partial_update() {
        let mut stats = NetworkStats::default();

        // Simulate partial updates
        stats.bytes_sent = 1000;
        assert_eq!(stats.bytes_sent, 1000);
        assert_eq!(stats.bytes_received, 0);

        stats.messages_sent = 10;
        stats.average_latency_us = 100.0;
        assert_eq!(stats.messages_sent, 10);
        assert_eq!(stats.average_latency_us, 100.0);
    }

    #[test]
    fn test_network_stats_json_field_names() {
        let stats = NetworkStats {
            bytes_sent: 123,
            bytes_received: 456,
            messages_sent: 7,
            messages_received: 8,
            average_latency_us: 99.9,
            throughput_mbps: 1234.5,
        };

        let json = serde_json::to_string(&stats).unwrap();

        // Verify JSON field names
        assert!(json.contains("\"bytes_sent\":123"));
        assert!(json.contains("\"bytes_received\":456"));
        assert!(json.contains("\"messages_sent\":7"));
        assert!(json.contains("\"messages_received\":8"));
        assert!(json.contains("\"average_latency_us\":99.9"));
        assert!(json.contains("\"throughput_mbps\":1234.5"));
    }

    #[test]
    fn test_network_stats_accumulation() {
        let mut stats = NetworkStats::default();

        // Simulate accumulating stats over time
        for i in 1..=10 {
            stats.bytes_sent += i * 100;
            stats.bytes_received += i * 200;
            stats.messages_sent += 1;
            stats.messages_received += 2;
        }

        assert_eq!(stats.bytes_sent, 5500); // Sum of 100 to 1000
        assert_eq!(stats.bytes_received, 11000); // Sum of 200 to 2000
        assert_eq!(stats.messages_sent, 10);
        assert_eq!(stats.messages_received, 20);
    }

    #[test]
    fn test_network_stats_performance_calculations() {
        let stats = NetworkStats {
            bytes_sent: 1_000_000,     // 1 MB
            bytes_received: 2_000_000, // 2 MB
            messages_sent: 1000,
            messages_received: 2000,
            average_latency_us: 50.0, // 50 microseconds
            throughput_mbps: 800.0,   // 800 Mbps
        };

        // Verify performance metrics make sense
        assert!(stats.average_latency_us > 0.0);
        assert!(stats.throughput_mbps > 0.0);

        // Calculate theoretical max messages per second based on latency
        let max_messages_per_second = 1_000_000.0 / stats.average_latency_us;
        assert!(max_messages_per_second > 0.0);
    }
}
