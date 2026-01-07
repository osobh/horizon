//! Storage performance benchmarking infrastructure

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{Storage, StorageError};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub num_operations: usize,
    pub data_size: usize,
    pub concurrent_ops: usize,
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_operations: 1000,
            data_size: 1024, // 1KB
            concurrent_ops: 10,
            warmup_iterations: 100,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub total_duration: Duration,
    pub operations_per_second: f64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_mbps: f64,
}

impl BenchmarkResults {
    pub fn new(durations: &[Duration], data_size: usize, total_duration: Duration) -> Self {
        let mut sorted_durations = durations.to_vec();
        sorted_durations.sort();

        let total_ops = durations.len();
        let operations_per_second = total_ops as f64 / total_duration.as_secs_f64();

        let avg_latency_ms = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / total_ops as f64;

        let min_latency_ms = sorted_durations[0].as_secs_f64() * 1000.0;
        let max_latency_ms = sorted_durations[total_ops - 1].as_secs_f64() * 1000.0;

        let p95_idx = (total_ops as f64 * 0.95) as usize;
        let p99_idx = (total_ops as f64 * 0.99) as usize;
        let p95_latency_ms = sorted_durations[p95_idx.min(total_ops - 1)].as_secs_f64() * 1000.0;
        let p99_latency_ms = sorted_durations[p99_idx.min(total_ops - 1)].as_secs_f64() * 1000.0;

        let total_bytes = total_ops * data_size;
        let throughput_mbps = (total_bytes as f64 / 1024.0 / 1024.0) / total_duration.as_secs_f64();

        Self {
            total_duration,
            operations_per_second,
            avg_latency_ms,
            min_latency_ms,
            max_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            throughput_mbps,
        }
    }
}

/// Storage benchmark suite
pub struct StorageBenchmark {
    config: BenchmarkConfig,
}

impl StorageBenchmark {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Benchmark write operations
    pub async fn benchmark_writes<S: Storage>(
        &self,
        storage: &S,
    ) -> Result<BenchmarkResults, StorageError> {
        let test_data = vec![0u8; self.config.data_size];
        let mut durations = Vec::new();

        // Warmup
        for i in 0..self.config.warmup_iterations {
            let key = format!("warmup_{i}");
            storage.store(&key, &test_data).await?;
        }

        // Actual benchmark
        let start_time = Instant::now();

        for i in 0..self.config.num_operations {
            let key = format!("write_bench_{i}");
            let op_start = Instant::now();
            storage.store(&key, &test_data).await?;
            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        Ok(BenchmarkResults::new(
            &durations,
            self.config.data_size,
            total_duration,
        ))
    }

    /// Benchmark read operations
    pub async fn benchmark_reads<S: Storage>(
        &self,
        storage: &S,
    ) -> Result<BenchmarkResults, StorageError> {
        let test_data = vec![1u8; self.config.data_size];
        let mut durations = Vec::new();

        // Setup data for reading
        for i in 0..self.config.num_operations {
            let key = format!("read_bench_{i}");
            storage.store(&key, &test_data).await?;
        }

        // Warmup
        for i in 0..self
            .config
            .warmup_iterations
            .min(self.config.num_operations)
        {
            let key = format!("read_bench_{i}");
            let _ = storage.retrieve(&key).await?;
        }

        // Actual benchmark
        let start_time = Instant::now();

        for i in 0..self.config.num_operations {
            let key = format!("read_bench_{i}");
            let op_start = Instant::now();
            let _data = storage.retrieve(&key).await?;
            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        Ok(BenchmarkResults::new(
            &durations,
            self.config.data_size,
            total_duration,
        ))
    }

    /// Benchmark concurrent operations
    pub async fn benchmark_concurrent_ops<S: Storage + Send + Sync + 'static>(
        &self,
        storage: Arc<S>,
    ) -> Result<BenchmarkResults, StorageError> {
        let test_data = vec![2u8; self.config.data_size];
        let ops_per_task = self.config.num_operations / self.config.concurrent_ops;

        let start_time = Instant::now();
        let mut handles = Vec::new();

        for task_id in 0..self.config.concurrent_ops {
            let storage_clone = storage.clone();
            let data_clone = test_data.clone();
            let _config = self.config.clone();

            let handle = tokio::spawn(async move {
                let mut task_durations = Vec::new();

                for i in 0..ops_per_task {
                    let key = format!("concurrent_{task_id}_{i}");
                    let op_start = Instant::now();
                    storage_clone.store(&key, &data_clone).await?;
                    task_durations.push(op_start.elapsed());
                }

                Ok::<Vec<Duration>, StorageError>(task_durations)
            });

            handles.push(handle);
        }

        let mut all_durations = Vec::new();

        for handle in handles {
            let task_durations = handle.await.map_err(|_| StorageError::InvalidDataFormat {
                reason: "Task join error in concurrent benchmark".to_string(),
            })??;
            all_durations.extend(task_durations);
        }

        let total_duration = start_time.elapsed();
        Ok(BenchmarkResults::new(
            &all_durations,
            self.config.data_size,
            total_duration,
        ))
    }

    /// Benchmark mixed workload (read/write)
    pub async fn benchmark_mixed_workload<S: Storage>(
        &self,
        storage: &S,
    ) -> Result<BenchmarkResults, StorageError> {
        let test_data = vec![3u8; self.config.data_size];
        let mut durations = Vec::new();

        // Pre-populate some data for reads
        for i in 0..self.config.num_operations / 2 {
            let key = format!("mixed_read_{i}");
            storage.store(&key, &test_data).await?;
        }

        // Mixed operations: 70% reads, 30% writes
        let start_time = Instant::now();

        for i in 0..self.config.num_operations {
            let op_start = Instant::now();

            if i % 10 < 7 {
                // Read operation
                let read_key = format!("mixed_read_{}", i % (self.config.num_operations / 2));
                let _data = storage.retrieve(&read_key).await?;
            } else {
                // Write operation
                let write_key = format!("mixed_write_{i}");
                storage.store(&write_key, &test_data).await?;
            }

            durations.push(op_start.elapsed());
        }

        let total_duration = start_time.elapsed();
        Ok(BenchmarkResults::new(
            &durations,
            self.config.data_size,
            total_duration,
        ))
    }

    /// Run comprehensive benchmark suite
    pub async fn run_full_suite<S: Storage + Send + Sync + 'static>(
        &self,
        storage: Arc<S>,
        storage_name: &str,
    ) -> Result<FullBenchmarkResults, StorageError> {
        println!("Running {storage_name} benchmark suite...");

        println!("  Running write benchmark...");
        let write_results = self.benchmark_writes(storage.as_ref()).await?;

        println!("  Running read benchmark...");
        let read_results = self.benchmark_reads(storage.as_ref()).await?;

        println!("  Running concurrent benchmark...");
        let concurrent_results = self.benchmark_concurrent_ops(storage.clone()).await?;

        println!("  Running mixed workload benchmark...");
        let mixed_results = self.benchmark_mixed_workload(storage.as_ref()).await?;

        Ok(FullBenchmarkResults {
            storage_name: storage_name.to_string(),
            write_results,
            read_results,
            concurrent_results,
            mixed_results,
            config: self.config.clone(),
        })
    }
}

/// Complete benchmark results
#[derive(Debug, Clone)]
pub struct FullBenchmarkResults {
    pub storage_name: String,
    pub write_results: BenchmarkResults,
    pub read_results: BenchmarkResults,
    pub concurrent_results: BenchmarkResults,
    pub mixed_results: BenchmarkResults,
    pub config: BenchmarkConfig,
}

impl FullBenchmarkResults {
    /// Print formatted benchmark results
    pub fn print_results(&self) {
        println!(
            "\n========== {} Benchmark Results ==========",
            self.storage_name
        );
        println!("Configuration:");
        println!("  Operations: {}", self.config.num_operations);
        println!("  Data size: {} bytes", self.config.data_size);
        println!("  Concurrent ops: {}", self.config.concurrent_ops);
        println!();

        self.print_single_result("Write Operations", &self.write_results);
        self.print_single_result("Read Operations", &self.read_results);
        self.print_single_result("Concurrent Operations", &self.concurrent_results);
        self.print_single_result("Mixed Workload", &self.mixed_results);

        println!("==============================================\n");
    }

    fn print_single_result(&self, name: &str, results: &BenchmarkResults) {
        println!("{name}:");
        println!("  Ops/sec: {:.2}", results.operations_per_second);
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
    use crate::{MemoryStorage, NvmeConfig, NvmeStorage};
    use std::time::Duration;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.num_operations, 1000);
        assert_eq!(config.data_size, 1024);
        assert_eq!(config.concurrent_ops, 10);
        assert_eq!(config.warmup_iterations, 100);
    }

    #[tokio::test]
    async fn test_benchmark_results_calculation() {
        let durations = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
            Duration::from_millis(5),
        ];
        let total_duration = Duration::from_millis(20);
        let data_size = 1024;

        let results = BenchmarkResults::new(&durations, data_size, total_duration);

        assert!(results.operations_per_second > 0.0);
        assert!(results.avg_latency_ms > 0.0);
        assert!(results.min_latency_ms <= results.max_latency_ms);
        assert!(results.p95_latency_ms >= results.avg_latency_ms);
        assert!(results.throughput_mbps > 0.0);
    }

    #[tokio::test]
    async fn test_memory_storage_write_benchmark() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(10 * 1024 * 1024); // 10MB
        let config = BenchmarkConfig {
            num_operations: 100,
            data_size: 256,
            warmup_iterations: 10,
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_writes(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.avg_latency_ms >= 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_storage_read_benchmark() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(10 * 1024 * 1024); // 10MB
        let config = BenchmarkConfig {
            num_operations: 100,
            data_size: 256,
            warmup_iterations: 10,
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_reads(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.avg_latency_ms >= 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_nvme_storage_benchmark() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let nvme_config = NvmeConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let storage = NvmeStorage::with_config(nvme_config).await?;

        let bench_config = BenchmarkConfig {
            num_operations: 50,
            data_size: 512,
            warmup_iterations: 5,
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(bench_config);
        let results = benchmark.benchmark_writes(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_benchmark() -> anyhow::Result<()> {
        let storage = Arc::new(MemoryStorage::new(10 * 1024 * 1024));
        let config = BenchmarkConfig {
            num_operations: 100,
            concurrent_ops: 4,
            data_size: 256,
            warmup_iterations: 0,
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_concurrent_ops(storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_mixed_workload_benchmark() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(10 * 1024 * 1024);
        let config = BenchmarkConfig {
            num_operations: 100,
            data_size: 256,
            warmup_iterations: 0,
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_mixed_workload(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_full_benchmark_suite() {
        let storage = Arc::new(MemoryStorage::new(10 * 1024 * 1024));
        let config = BenchmarkConfig {
            num_operations: 50,
            concurrent_ops: 2,
            data_size: 256,
            warmup_iterations: 5,
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark
            .run_full_suite(storage, "Test Memory Storage")
            .await
            .unwrap();

        assert_eq!(results.storage_name, "Test Memory Storage");
        assert!(results.write_results.operations_per_second > 0.0);
        assert!(results.read_results.operations_per_second > 0.0);
        assert!(results.concurrent_results.operations_per_second > 0.0);
        assert!(results.mixed_results.operations_per_second > 0.0);

        // Test print functionality (just ensure it doesn't panic)
        results.print_results();
    }

    #[tokio::test]
    async fn test_benchmark_comparison() -> anyhow::Result<()> {
        let memory_storage = Arc::new(MemoryStorage::new(10 * 1024 * 1024));
        let temp_dir = tempdir()?;
        let nvme_config = NvmeConfig {
            base_path: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let nvme_storage = Arc::new(NvmeStorage::with_config(nvme_config).await?);

        let config = BenchmarkConfig {
            num_operations: 100,
            concurrent_ops: 2,
            data_size: 1024,
            warmup_iterations: 10,
        };

        let benchmark = StorageBenchmark::new(config);

        let memory_results = benchmark
            .run_full_suite(memory_storage, "Memory Storage")
            .await
            .unwrap();
        let nvme_results = benchmark
            .run_full_suite(nvme_storage, "NVMe Storage")
            .await
            .unwrap();

        // Both should have valid results
        assert!(memory_results.write_results.operations_per_second > 0.0);
        assert!(nvme_results.write_results.operations_per_second > 0.0);

        // Memory storage should generally be faster for small operations
        // (though this is not guaranteed in tests due to various factors)
        println!(
            "Memory OPS: {:.2}",
            memory_results.write_results.operations_per_second
        );
        println!(
            "NVMe OPS: {:.2}",
            nvme_results.write_results.operations_per_second
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_benchmark_config_edge_cases() -> anyhow::Result<()> {
        // Test with minimal operations
        let min_config = BenchmarkConfig {
            num_operations: 1,
            data_size: 1,
            concurrent_ops: 1,
            warmup_iterations: 0,
        };

        let storage = MemoryStorage::new(1024);
        let benchmark = StorageBenchmark::new(min_config);
        let results = benchmark.benchmark_writes(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.avg_latency_ms >= 0.0);
        assert_eq!(results.min_latency_ms, results.max_latency_ms); // Only one operation
        Ok(())
    }

    #[tokio::test]
    async fn test_benchmark_results_edge_cases() {
        // Test with single duration
        let single_duration = vec![Duration::from_nanos(1)];
        let total_duration = Duration::from_nanos(1);
        let results = BenchmarkResults::new(&single_duration, 1, total_duration);

        assert_eq!(results.min_latency_ms, results.max_latency_ms);
        assert_eq!(results.p95_latency_ms, results.min_latency_ms);
        assert_eq!(results.p99_latency_ms, results.min_latency_ms);

        // Test with zero duration (edge case)
        let zero_duration = vec![Duration::from_nanos(0)];
        let results_zero = BenchmarkResults::new(&zero_duration, 1, Duration::from_nanos(1));
        assert_eq!(results_zero.min_latency_ms, 0.0);
    }

    #[tokio::test]
    async fn test_benchmark_large_data_size() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(10 * 1024 * 1024); // 10MB
        let config = BenchmarkConfig {
            num_operations: 5,
            data_size: 1024 * 1024, // 1MB per operation
            concurrent_ops: 1,
            warmup_iterations: 0,
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_writes(&storage).await?;

        assert!(results.throughput_mbps > 0.0);
        assert!(results.operations_per_second > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_benchmark_concurrent_error_handling() {
        // Test with storage that will fail due to capacity
        let small_storage = Arc::new(MemoryStorage::new(100)); // Very small
        let config = BenchmarkConfig {
            num_operations: 100,
            concurrent_ops: 4,
            data_size: 1024, // Data too large for storage
            warmup_iterations: 0,
        };

        let benchmark = StorageBenchmark::new(config);
        let result = benchmark.benchmark_concurrent_ops(small_storage).await;

        // Should handle storage errors gracefully
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_benchmark_percentile_calculation() {
        // Test percentile calculations with known data
        let durations: Vec<Duration> = (1..=100).map(|i| Duration::from_millis(i)).collect();

        let total_duration = Duration::from_secs(1);
        let results = BenchmarkResults::new(&durations, 1024, total_duration);

        // With 100 items, P95 should be around 95th item, P99 around 99th
        assert!(results.p95_latency_ms >= 90.0); // 95th percentile
        assert!(results.p99_latency_ms >= 98.0); // 99th percentile
        assert!(results.p99_latency_ms >= results.p95_latency_ms);
    }

    #[tokio::test]
    async fn test_benchmark_mixed_workload_edge_cases() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(1024 * 1024);

        // Test with operations that are exactly divisible by 10
        let config = BenchmarkConfig {
            num_operations: 20, // 14 reads (70%), 6 writes (30%)
            data_size: 256,
            warmup_iterations: 0,
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config);
        let results = benchmark.benchmark_mixed_workload(&storage).await?;

        assert!(results.operations_per_second > 0.0);
        assert!(results.throughput_mbps > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_full_benchmark_results_print() {
        let storage = Arc::new(MemoryStorage::new(1024 * 1024));
        let config = BenchmarkConfig {
            num_operations: 10,
            concurrent_ops: 2,
            data_size: 64,
            warmup_iterations: 1,
        };

        let benchmark = StorageBenchmark::new(config.clone());
        let results = benchmark
            .run_full_suite(storage, "Test Storage")
            .await
            .unwrap();

        // Test all fields are properly set
        assert_eq!(results.storage_name, "Test Storage");
        assert_eq!(results.config.num_operations, config.num_operations);
        assert_eq!(results.config.data_size, config.data_size);

        // Test the print functionality doesn't panic
        results.print_results(); // This should complete without panic
    }

    #[tokio::test]
    async fn test_benchmark_zero_warmup_iterations() -> anyhow::Result<()> {
        let storage = MemoryStorage::new(1024 * 1024);
        let config = BenchmarkConfig {
            num_operations: 10,
            data_size: 128,
            warmup_iterations: 0, // No warmup
            ..Default::default()
        };

        let benchmark = StorageBenchmark::new(config);

        // Should work without warmup
        let write_results = benchmark.benchmark_writes(&storage).await?;
        let read_results = benchmark.benchmark_reads(&storage).await?;

        assert!(write_results.operations_per_second > 0.0);
        assert!(read_results.operations_per_second > 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_benchmark_results_clone() {
        let durations = vec![Duration::from_millis(1), Duration::from_millis(2)];
        let total_duration = Duration::from_millis(10);
        let original = BenchmarkResults::new(&durations, 1024, total_duration);

        let cloned = original.clone();

        assert_eq!(original.total_duration, cloned.total_duration);
        assert_eq!(original.operations_per_second, cloned.operations_per_second);
        assert_eq!(original.avg_latency_ms, cloned.avg_latency_ms);
        assert_eq!(original.throughput_mbps, cloned.throughput_mbps);
    }

    #[tokio::test]
    async fn test_benchmark_config_clone() {
        let original = BenchmarkConfig {
            num_operations: 500,
            data_size: 2048,
            concurrent_ops: 8,
            warmup_iterations: 50,
        };

        let cloned = original.clone();

        assert_eq!(original.num_operations, cloned.num_operations);
        assert_eq!(original.data_size, cloned.data_size);
        assert_eq!(original.concurrent_ops, cloned.concurrent_ops);
        assert_eq!(original.warmup_iterations, cloned.warmup_iterations);
    }
}
