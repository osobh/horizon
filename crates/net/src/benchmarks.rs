//! Network performance benchmarking infrastructure

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{MemoryNetwork, Message, MessageType, Network, ZeroCopyTransport};

/// Network benchmark configuration
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NetworkBenchmarkConfig {
    pub num_messages: u64,
    pub message_sizes: Vec<usize>,
    pub message_size: usize,
    pub concurrent_connections: usize,
    pub parallel_connections: usize,
    pub warmup_iterations: usize,
    pub warmup_messages: u64,
    pub batch_size: usize,
}

impl Default for NetworkBenchmarkConfig {
    fn default() -> Self {
        Self {
            num_messages: 1000,
            message_sizes: vec![1024],
            message_size: 1024, // 1KB
            concurrent_connections: 10,
            parallel_connections: 10,
            warmup_iterations: 100,
            warmup_messages: 100,
            batch_size: 10,
        }
    }
}

/// Network benchmark results (detailed)
#[derive(Debug, Clone)]
pub struct DetailedNetworkBenchmarkResults {
    pub total_duration: Duration,
    pub messages_per_second: f64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_mbps: f64,
}

/// Network benchmark results (simplified for tests)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkBenchmarkResults {
    pub throughput_mbps: f64,
    pub latency_us: f64,
    pub messages_per_second: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
}

impl DetailedNetworkBenchmarkResults {
    pub fn new(durations: &[Duration], message_size: usize, total_duration: Duration) -> Self {
        let mut sorted_durations = durations.to_vec();
        sorted_durations.sort();

        let total_messages = durations.len();
        let messages_per_second = total_messages as f64 / total_duration.as_secs_f64();

        let avg_latency_ms = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / total_messages as f64;

        let min_latency_ms = sorted_durations[0].as_secs_f64() * 1000.0;
        let max_latency_ms = sorted_durations[total_messages - 1].as_secs_f64() * 1000.0;

        let p95_idx = (total_messages as f64 * 0.95) as usize;
        let p99_idx = (total_messages as f64 * 0.99) as usize;
        let p95_latency_ms =
            sorted_durations[p95_idx.min(total_messages - 1)].as_secs_f64() * 1000.0;
        let p99_latency_ms =
            sorted_durations[p99_idx.min(total_messages - 1)].as_secs_f64() * 1000.0;

        let total_bytes = total_messages * message_size;
        let throughput_mbps = (total_bytes as f64 / 1024.0 / 1024.0) / total_duration.as_secs_f64();

        Self {
            total_duration,
            messages_per_second,
            avg_latency_ms,
            min_latency_ms,
            max_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            throughput_mbps,
        }
    }
}

/// Network benchmark suite
pub struct NetworkBenchmark {
    config: NetworkBenchmarkConfig,
}

impl NetworkBenchmark {
    pub fn new(config: NetworkBenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark message sending throughput
    pub async fn benchmark_send_throughput(
        &self,
        network: &MemoryNetwork,
    ) -> DetailedNetworkBenchmarkResults {
        let test_data = vec![0u8; self.config.message_size];
        let mut durations = Vec::new();

        // Warmup
        for i in 0..self.config.warmup_iterations {
            let message = Message::new(MessageType::ResourceRequest, test_data.clone());
            let endpoint = format!("warmup_{i}");
            network.send(&endpoint, message).await.unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for i in 0..self.config.num_messages {
            let message = Message::new(MessageType::ResourceRequest, test_data.clone());
            let endpoint = format!("bench_{i}");

            let op_start = Instant::now();
            network.send(&endpoint, message).await.unwrap();
            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        DetailedNetworkBenchmarkResults::new(&durations, self.config.message_size, total_duration)
    }

    /// Benchmark message receiving throughput
    pub async fn benchmark_receive_throughput(
        &self,
        network: &MemoryNetwork,
    ) -> DetailedNetworkBenchmarkResults {
        let test_data = vec![1u8; self.config.message_size];
        let mut durations = Vec::new();

        // Pre-populate messages for receiving
        for i in 0..self.config.num_messages {
            let message = Message::new(MessageType::AgentSpawn, test_data.clone());
            let endpoint = format!("receive_bench_{i}");
            network.simulate_receive(endpoint, message).await.unwrap();
        }

        // Warmup
        for _i in 0..self
            .config
            .warmup_iterations
            .min(self.config.num_messages as usize)
        {
            let _message = network.receive().await.unwrap();
        }

        // Re-populate after warmup
        for i in 0..self.config.num_messages {
            let message = Message::new(MessageType::AgentSpawn, test_data.clone());
            let endpoint = format!("receive_bench_real_{i}");
            network.simulate_receive(endpoint, message).await.unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for _i in 0..self.config.num_messages {
            let op_start = Instant::now();
            let _message = network.receive().await.unwrap();
            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        DetailedNetworkBenchmarkResults::new(&durations, self.config.message_size, total_duration)
    }

    /// Benchmark zero-copy transport operations
    pub async fn benchmark_zero_copy(
        &self,
        transport: &ZeroCopyTransport,
    ) -> DetailedNetworkBenchmarkResults {
        let test_data = vec![2u8; self.config.message_size];
        let mut durations = Vec::new();

        // Warmup
        for i in 0..self.config.warmup_iterations {
            let message = Message::new(MessageType::KnowledgeSync, test_data.clone());
            let endpoint = format!("warmup_{i}");
            transport.send(&endpoint, message).await.unwrap();
        }

        // Actual benchmark
        let start_time = Instant::now();

        for i in 0..self.config.num_messages {
            let message = Message::new(MessageType::KnowledgeSync, test_data.clone());
            let endpoint = format!("zero_copy_{i}");

            let op_start = Instant::now();
            transport.send(&endpoint, message).await.unwrap();
            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        DetailedNetworkBenchmarkResults::new(&durations, self.config.message_size, total_duration)
    }

    /// Benchmark concurrent network operations
    pub async fn benchmark_concurrent_ops(
        &self,
        network: Arc<MemoryNetwork>,
    ) -> DetailedNetworkBenchmarkResults {
        let test_data = vec![3u8; self.config.message_size];
        let messages_per_connection =
            self.config.num_messages as usize / self.config.concurrent_connections;

        let start_time = Instant::now();
        let mut handles = Vec::new();

        for conn_id in 0..self.config.concurrent_connections {
            let network_clone = network.clone();
            let data_clone = test_data.clone();

            let handle = tokio::spawn(async move {
                let mut connection_durations = Vec::new();

                for i in 0..messages_per_connection {
                    let message = Message::new(MessageType::KnowledgeSync, data_clone.clone());
                    let endpoint = format!("concurrent_{conn_id}_{i}");

                    let op_start = Instant::now();
                    network_clone.send(&endpoint, message).await.unwrap();
                    connection_durations.push(op_start.elapsed());
                }

                connection_durations
            });

            handles.push(handle);
        }

        let mut all_durations = Vec::new();

        for handle in handles {
            let connection_durations = handle.await.unwrap();
            all_durations.extend(connection_durations);
        }

        let total_duration = start_time.elapsed();
        DetailedNetworkBenchmarkResults::new(
            &all_durations,
            self.config.message_size,
            total_duration,
        )
    }

    /// Benchmark batch operations
    pub async fn benchmark_batch_ops(
        &self,
        network: &MemoryNetwork,
    ) -> DetailedNetworkBenchmarkResults {
        let test_data = vec![4u8; self.config.message_size];
        let mut durations = Vec::new();

        let batches = self.config.num_messages as usize / self.config.batch_size;

        let start_time = Instant::now();

        for batch_id in 0..batches {
            let batch_start = Instant::now();

            // Send batch of messages
            for i in 0..self.config.batch_size {
                let message = Message::new(MessageType::ResourceResponse, test_data.clone());
                let endpoint = format!("batch_{batch_id}_msg_{i}");
                network.send(&endpoint, message).await.unwrap();
            }

            durations.push(batch_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        // Calculate per-message latency by dividing batch latency by batch size
        let per_message_durations: Vec<Duration> = durations
            .iter()
            .map(|d| Duration::from_nanos(d.as_nanos() as u64 / self.config.batch_size as u64))
            .collect();

        DetailedNetworkBenchmarkResults::new(
            &per_message_durations,
            self.config.message_size,
            total_duration,
        )
    }

    /// Run comprehensive network benchmark suite
    pub async fn run_full_suite(
        &self,
        network: Arc<MemoryNetwork>,
        transport: &ZeroCopyTransport,
    ) -> DetailedFullNetworkBenchmarkResults {
        println!("Running network benchmark suite...");

        println!("  Running send throughput benchmark...");
        let send_results = self.benchmark_send_throughput(network.as_ref()).await;

        println!("  Running receive throughput benchmark...");
        let receive_results = self.benchmark_receive_throughput(network.as_ref()).await;

        println!("  Running zero-copy benchmark...");
        let zero_copy_results = self.benchmark_zero_copy(transport).await;

        println!("  Running concurrent operations benchmark...");
        let concurrent_results = self.benchmark_concurrent_ops(network.clone()).await;

        println!("  Running batch operations benchmark...");
        let batch_results = self.benchmark_batch_ops(network.as_ref()).await;

        DetailedFullNetworkBenchmarkResults {
            send_results,
            receive_results,
            zero_copy_results,
            concurrent_results,
            batch_results,
            config: self.config.clone(),
        }
    }
}

/// Complete network benchmark results (detailed)
#[derive(Debug, Clone)]
pub struct DetailedFullNetworkBenchmarkResults {
    pub send_results: DetailedNetworkBenchmarkResults,
    pub receive_results: DetailedNetworkBenchmarkResults,
    pub zero_copy_results: DetailedNetworkBenchmarkResults,
    pub concurrent_results: DetailedNetworkBenchmarkResults,
    pub batch_results: DetailedNetworkBenchmarkResults,
    pub config: NetworkBenchmarkConfig,
}

/// Full network benchmark results (for tests)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FullNetworkBenchmarkResults {
    pub results_by_size: std::collections::HashMap<usize, NetworkBenchmarkResults>,
    pub total_duration: std::time::Duration,
    pub total_messages: u64,
    pub total_bytes: u64,
}

impl DetailedFullNetworkBenchmarkResults {
    /// Print formatted benchmark results
    pub fn print_results(&self) {
        println!("\n========== Network Benchmark Results ==========");
        println!("Configuration:");
        println!("  Messages: {}", self.config.num_messages);
        println!("  Message size: {} bytes", self.config.message_size);
        println!(
            "  Concurrent connections: {}",
            self.config.concurrent_connections
        );
        println!("  Batch size: {}", self.config.batch_size);
        println!();

        self.print_single_result("Send Throughput", &self.send_results);
        self.print_single_result("Receive Throughput", &self.receive_results);
        self.print_single_result("Zero-Copy Operations", &self.zero_copy_results);
        self.print_single_result("Concurrent Operations", &self.concurrent_results);
        self.print_single_result("Batch Operations", &self.batch_results);

        println!("===============================================\n");
    }

    fn print_single_result(&self, name: &str, results: &DetailedNetworkBenchmarkResults) {
        println!("{name}:");
        println!("  Msgs/sec: {:.2}", results.messages_per_second);
        println!("  Throughput: {:.2} MB/s", results.throughput_mbps);
        println!(
            "  Latency - Avg: {:.2}ms, P95: {:.2}ms, P99: {:.2}ms",
            results.avg_latency_ms, results.p95_latency_ms, results.p99_latency_ms
        );
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_benchmark_config_default() {
        let config = NetworkBenchmarkConfig::default();
        assert_eq!(config.num_messages, 1000);
        assert_eq!(config.message_size, 1024);
        assert_eq!(config.concurrent_connections, 10);
        assert_eq!(config.warmup_iterations, 100);
        assert_eq!(config.batch_size, 10);
    }

    #[tokio::test]
    async fn test_network_benchmark_results_calculation() {
        let durations = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
            Duration::from_millis(5),
        ];
        let total_duration = Duration::from_millis(20);
        let message_size = 1024;

        let results =
            DetailedNetworkBenchmarkResults::new(&durations, message_size, total_duration);

        assert!(results.messages_per_second > 0.0);
        assert!(results.avg_latency_ms > 0.0);
        assert!(results.min_latency_ms <= results.max_latency_ms);
        assert!(results.p95_latency_ms >= results.avg_latency_ms);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_send_throughput_benchmark() {
        let network = MemoryNetwork::new("test_endpoint".to_string());
        let config = NetworkBenchmarkConfig {
            num_messages: 100,
            message_size: 256,
            warmup_iterations: 10,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.benchmark_send_throughput(&network).await;

        assert!(results.messages_per_second > 0.0);
        assert!(results.avg_latency_ms >= 0.0);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_receive_throughput_benchmark() {
        let network = MemoryNetwork::new("test_endpoint".to_string());
        let config = NetworkBenchmarkConfig {
            num_messages: 100,
            message_size: 256,
            warmup_iterations: 10,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.benchmark_receive_throughput(&network).await;

        assert!(results.messages_per_second > 0.0);
        assert!(results.avg_latency_ms >= 0.0);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_zero_copy_benchmark() {
        let transport = ZeroCopyTransport::new();
        let config = NetworkBenchmarkConfig {
            num_messages: 100,
            message_size: 256,
            warmup_iterations: 10,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.benchmark_zero_copy(&transport).await;

        assert!(results.messages_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_concurrent_benchmark() {
        let network = Arc::new(MemoryNetwork::new("test_concurrent".to_string()));
        let config = NetworkBenchmarkConfig {
            num_messages: 100,
            concurrent_connections: 4,
            message_size: 256,
            warmup_iterations: 0,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.benchmark_concurrent_ops(network).await;

        assert!(results.messages_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_batch_benchmark() {
        let network = MemoryNetwork::new("test_batch".to_string());
        let config = NetworkBenchmarkConfig {
            num_messages: 100,
            batch_size: 10,
            message_size: 256,
            warmup_iterations: 0,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.benchmark_batch_ops(&network).await;

        assert!(results.messages_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_full_benchmark_suite() {
        let network = Arc::new(MemoryNetwork::new("test_full".to_string()));
        let transport = ZeroCopyTransport::new();
        let config = NetworkBenchmarkConfig {
            num_messages: 50,
            concurrent_connections: 2,
            message_size: 256,
            warmup_iterations: 5,
            batch_size: 5,
            ..Default::default()
        };

        let benchmark = NetworkBenchmark::new(config);
        let results = benchmark.run_full_suite(network, &transport).await;

        assert!(results.send_results.messages_per_second > 0.0);
        assert!(results.receive_results.messages_per_second > 0.0);
        assert!(results.zero_copy_results.messages_per_second > 0.0);
        assert!(results.concurrent_results.messages_per_second > 0.0);
        assert!(results.batch_results.messages_per_second > 0.0);

        // Test print functionality (just ensure it doesn't panic)
        results.print_results();
    }

    #[tokio::test]
    async fn test_performance_comparison() {
        let network = Arc::new(MemoryNetwork::new("test_perf".to_string()));

        let small_config = NetworkBenchmarkConfig {
            num_messages: 100,
            message_size: 64,
            concurrent_connections: 1,
            warmup_iterations: 10,
            batch_size: 1,
            ..Default::default()
        };

        let large_config = NetworkBenchmarkConfig {
            num_messages: 100,
            message_size: 4096,
            concurrent_connections: 1,
            warmup_iterations: 10,
            batch_size: 1,
            ..Default::default()
        };

        let small_benchmark = NetworkBenchmark::new(small_config);
        let large_benchmark = NetworkBenchmark::new(large_config);

        let small_results = small_benchmark
            .benchmark_send_throughput(network.as_ref())
            .await;
        let large_results = large_benchmark
            .benchmark_send_throughput(network.as_ref())
            .await;

        // Both should have valid results
        assert!(small_results.messages_per_second > 0.0);
        assert!(large_results.messages_per_second > 0.0);

        // Larger messages should have higher throughput in MB/s but possibly lower message rate
        println!(
            "Small messages: {:.2} msgs/sec, {:.2} MB/s",
            small_results.messages_per_second, small_results.throughput_mbps
        );
        println!(
            "Large messages: {:.2} msgs/sec, {:.2} MB/s",
            large_results.messages_per_second, large_results.throughput_mbps
        );
    }
}
