//! Kernel Optimizer for GPU Utilization
//!
//! Optimizes kernel launch configurations to maximize GPU utilization.

use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Kernel configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Number of threads per block
    pub block_size: u32,
    /// Number of blocks in grid
    pub grid_size: u32,
    /// Shared memory per block (bytes)
    pub shared_mem_size: u32,
    /// Registers per thread
    pub registers_per_thread: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 256,
            shared_mem_size: 0,
            registers_per_thread: 32,
        }
    }
}

/// Kernel performance metrics
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f32,
    /// Achieved occupancy (0.0 - 1.0)
    pub occupancy: f32,
    /// Memory throughput (GB/s)
    pub memory_throughput: f32,
    /// Compute throughput (GFLOPS)
    pub compute_throughput: f32,
    /// Warp efficiency (0.0 - 1.0)
    pub warp_efficiency: f32,
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            occupancy: 0.0,
            memory_throughput: 0.0,
            compute_throughput: 0.0,
            warp_efficiency: 0.0,
        }
    }
}

/// Kernel optimizer
pub struct KernelOptimizer {
    device: Arc<CudaContext>,
    kernel_configs: Arc<DashMap<String, KernelConfig>>,
    kernel_metrics: Arc<DashMap<String, Vec<KernelMetrics>>>,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub kernel_name: String,
    pub original_config: KernelConfig,
    pub optimized_config: KernelConfig,
    pub improvement: f32, // Percentage improvement
    pub metric: String,
}

impl KernelOptimizer {
    /// Create new kernel optimizer
    pub fn new(device: Arc<CudaContext>) -> Self {
        Self {
            device,
            kernel_configs: Arc::new(DashMap::new()),
            kernel_metrics: Arc::new(DashMap::new()),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register kernel with default configuration
    pub async fn register_kernel(&self, name: &str, config: KernelConfig) -> Result<()> {
        self.kernel_configs.insert(name.to_string(), config);
        Ok(())
    }

    /// Get optimal kernel configuration
    pub async fn get_optimal_config(&self, kernel_name: &str) -> Result<KernelConfig> {
        self.kernel_configs
            .get(kernel_name)
            .map(|entry| *entry.value())
            .ok_or_else(|| anyhow::anyhow!("Kernel {} not registered", kernel_name))
    }

    /// Record kernel execution metrics
    pub async fn record_metrics(
        &self,
        kernel_name: &str,
        _config: KernelConfig,
        metrics: KernelMetrics,
    ) -> Result<()> {
        // Get or create metrics entry
        let mut entry = self.kernel_metrics.entry(kernel_name.to_string()).or_insert_with(Vec::new);
        let kernel_metrics = entry.value_mut();

        // Keep last 100 measurements
        if kernel_metrics.len() >= 100 {
            kernel_metrics.remove(0);
        }

        kernel_metrics.push(metrics);

        // Update configuration if this performed better
        if let Some(best_config) = self.find_best_config(kernel_metrics).await {
            self.kernel_configs.insert(kernel_name.to_string(), best_config);
        }

        Ok(())
    }

    /// Find best configuration from metrics
    async fn find_best_config(&self, _metrics: &[KernelMetrics]) -> Option<KernelConfig> {
        // In real implementation, analyze metrics to find best config
        // For now, return None to keep current config
        None
    }

    /// Optimize kernel configuration through auto-tuning
    pub async fn auto_tune_kernel(
        &self,
        kernel_name: &str,
        test_workload: impl Fn(KernelConfig) -> Result<KernelMetrics>,
    ) -> Result<KernelConfig> {
        let original_config = self.get_optimal_config(kernel_name).await?;
        let mut best_config = original_config;
        let mut best_score = 0.0f32;

        // Test different block sizes
        let block_sizes = vec![64, 128, 256, 512, 1024];

        for &block_size in &block_sizes {
            if block_size > 1024 {
                continue; // Max threads per block
            }

            let mut test_config = original_config;
            test_config.block_size = block_size;

            // Adjust grid size to maintain total thread count
            let total_threads = original_config.block_size * original_config.grid_size;
            test_config.grid_size = (total_threads + block_size - 1) / block_size;

            // Test configuration
            match test_workload(test_config) {
                Ok(metrics) => {
                    let score = self.calculate_performance_score(&metrics);
                    if score > best_score {
                        best_score = score;
                        best_config = test_config;
                    }

                    // Record metrics
                    self.record_metrics(kernel_name, test_config, metrics)
                        .await?;
                }
                Err(_) => continue, // Skip failed configurations
            }
        }

        // Record optimization result
        if best_config.block_size != original_config.block_size {
            let improvement =
                ((best_score / self.calculate_performance_score(&KernelMetrics::default())) - 1.0)
                    * 100.0;

            let result = OptimizationResult {
                kernel_name: kernel_name.to_string(),
                original_config,
                optimized_config: best_config,
                improvement,
                metric: "performance_score".to_string(),
            };

            let mut history = self.optimization_history.write().await;
            history.push(result);
        }

        // Update stored configuration
        self.kernel_configs.insert(kernel_name.to_string(), best_config);

        Ok(best_config)
    }

    /// Calculate performance score from metrics
    fn calculate_performance_score(&self, metrics: &KernelMetrics) -> f32 {
        // Weighted score based on multiple factors
        let time_score = 1000.0 / (metrics.execution_time_ms + 1.0); // Lower time is better
        let occupancy_score = metrics.occupancy * 100.0;
        let efficiency_score = metrics.warp_efficiency * 100.0;
        let throughput_score = (metrics.memory_throughput + metrics.compute_throughput) / 10.0;

        // Weighted average
        time_score * 0.4 + occupancy_score * 0.3 + efficiency_score * 0.2 + throughput_score * 0.1
    }

    /// Get occupancy calculator for kernel
    pub fn calculate_occupancy(&self, config: KernelConfig) -> f32 {
        // RTX 5090 specifications
        let max_threads_per_sm = 2048;
        let max_blocks_per_sm = 32;
        let max_shared_mem_per_sm = 164 * 1024; // 164KB
        let max_registers_per_sm = 65536;

        // Calculate limitations
        let threads_per_block = config.block_size;
        let blocks_by_threads = max_threads_per_sm / threads_per_block;
        let blocks_by_limit = max_blocks_per_sm;

        let blocks_by_shared_mem = if config.shared_mem_size > 0 {
            max_shared_mem_per_sm / config.shared_mem_size
        } else {
            u32::MAX
        };

        let blocks_by_registers = if config.registers_per_thread > 0 {
            max_registers_per_sm / (config.registers_per_thread * threads_per_block)
        } else {
            u32::MAX
        };

        // Actual blocks per SM
        let blocks_per_sm = blocks_by_threads
            .min(blocks_by_limit)
            .min(blocks_by_shared_mem)
            .min(blocks_by_registers);

        // Calculate occupancy
        let active_warps = (blocks_per_sm * threads_per_block) / 32;
        let max_warps = max_threads_per_sm / 32;

        active_warps as f32 / max_warps as f32
    }

    /// Suggest optimal launch configuration
    pub fn suggest_launch_config(
        &self,
        total_work_items: u32,
        items_per_thread: u32,
    ) -> LaunchConfig {
        let total_threads = (total_work_items + items_per_thread - 1) / items_per_thread;

        // Choose block size based on workload
        let block_size = if total_threads < 1024 {
            128 // Small workload
        } else if total_threads < 65536 {
            256 // Medium workload
        } else {
            512 // Large workload
        };

        let grid_size = (total_threads + block_size - 1) / block_size;

        LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Generate optimization report
    pub async fn generate_report(&self) -> String {
        let history = self.optimization_history.read().await;

        let mut report = String::from("Kernel Optimization Report:\n\n");

        // Current configurations
        report.push_str("Current Kernel Configurations:\n");
        for entry in self.kernel_configs.iter() {
            let name = entry.key();
            let config = entry.value();
            let occupancy = self.calculate_occupancy(*config);
            report.push_str(&format!(
                "  {}: block_size={}, grid_size={}, occupancy={:.1}%\n",
                name,
                config.block_size,
                config.grid_size,
                occupancy * 100.0
            ));
        }

        // Optimization history
        if !history.is_empty() {
            report.push_str("\nOptimization History:\n");
            for result in history.iter().take(10) {
                report.push_str(&format!(
                    "  {}: {:.1}% improvement (block {} -> {})\n",
                    result.kernel_name,
                    result.improvement,
                    result.original_config.block_size,
                    result.optimized_config.block_size
                ));
            }
        }

        report
    }
}

/// Grid size calculator for different workload patterns
pub struct GridSizeCalculator;

impl GridSizeCalculator {
    /// Calculate grid size for 1D workload
    pub fn calculate_1d(total_items: u32, block_size: u32) -> u32 {
        (total_items + block_size - 1) / block_size
    }

    /// Calculate grid size for 2D workload
    pub fn calculate_2d(
        width: u32,
        height: u32,
        block_width: u32,
        block_height: u32,
    ) -> (u32, u32) {
        let grid_x = (width + block_width - 1) / block_width;
        let grid_y = (height + block_height - 1) / block_height;
        (grid_x, grid_y)
    }

    /// Calculate grid size for 3D workload
    pub fn calculate_3d(
        width: u32,
        height: u32,
        depth: u32,
        block_width: u32,
        block_height: u32,
        block_depth: u32,
    ) -> (u32, u32, u32) {
        let grid_x = (width + block_width - 1) / block_width;
        let grid_y = (height + block_height - 1) / block_height;
        let grid_z = (depth + block_depth - 1) / block_depth;
        (grid_x, grid_y, grid_z)
    }
}
