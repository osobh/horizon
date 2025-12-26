//! Performance Profiler Tests for Production Self-Evolution
//!
//! Tests real-time performance monitoring and feedback loops for autonomous optimization.
//! These tests verify GPU performance metrics collection and automatic tuning capabilities.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{sleep, interval, timeout};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceMetrics {
    pub timestamp: u64,
    pub kernel_id: String,
    pub execution_time_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub memory_bandwidth_gb_per_sec: f64,
    pub gpu_utilization_percent: f64,
    pub memory_utilization_percent: f64,
    pub power_consumption_watts: f64,
    pub temperature_celsius: f64,
    pub cache_hit_rate: f64,
    pub warp_efficiency: f64,
    pub occupancy_percent: f64,
}

/// Performance profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub sampling_interval_ms: u64,
    pub metrics_buffer_size: usize,
    pub performance_threshold: f64,
    pub optimization_trigger_threshold: f64,
    pub enable_real_time_optimization: bool,
    pub enable_predictive_scaling: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sampling_interval_ms: 10,
            metrics_buffer_size: 10000,
            performance_threshold: 80.0,
            optimization_trigger_threshold: 60.0,
            enable_real_time_optimization: true,
            enable_predictive_scaling: true,
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub kernel_id: String,
    pub trend_direction: TrendDirection,
    pub confidence: f64,
    pub predicted_performance_in_5min: f64,
    pub recommended_action: OptimizationAction,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

#[derive(Debug, Clone)]
pub enum OptimizationAction {
    IncreaseThreads,
    DecreaseThreads,
    OptimizeMemoryAccess,
    ReduceRegisterUsage,
    EnableL1Cache,
    TuneSharedMemory,
    ScaleUp,
    ScaleDown,
    NoAction,
}

/// Real-time performance profiler
pub struct GpuPerformanceProfiler {
    config: ProfilerConfig,
    metrics_buffer: Arc<RwLock<VecDeque<GpuPerformanceMetrics>>>,
    active_kernels: Arc<RwLock<HashMap<String, String>>>,
    performance_trends: Arc<RwLock<HashMap<String, PerformanceTrend>>>,
    optimization_sender: mpsc::UnboundedSender<OptimizationCommand>,
    profiling_active: Arc<Mutex<bool>>,
}

#[derive(Debug, Clone)]
pub enum OptimizationCommand {
    TuneKernel { kernel_id: String, action: OptimizationAction },
    RecompileWithOptimizations { kernel_id: String, optimizations: Vec<String> },
    ScaleResources { kernel_id: String, scale_factor: f64 },
    Emergency { kernel_id: String, issue: String },
}

impl GpuPerformanceProfiler {
    pub fn new(
        config: ProfilerConfig,
        optimization_sender: mpsc::UnboundedSender<OptimizationCommand>,
    ) -> Self {
        Self {
            config,
            metrics_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            active_kernels: Arc::new(RwLock::new(HashMap::new())),
            performance_trends: Arc::new(RwLock::new(HashMap::new())),
            optimization_sender,
            profiling_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start real-time performance profiling
    pub async fn start_profiling(&self) -> Result<()> {
        {
            let mut active = self.profiling_active.lock()?;
            *active = true;
        }

        // This will fail in RED phase - no actual GPU profiling implemented
        self.initialize_gpu_profiler().await?;
        
        // Start metrics collection loop
        let profiler = self.clone();
        tokio::spawn(async move {
            profiler.metrics_collection_loop().await;
        });

        // Start trend analysis loop
        let profiler = self.clone();
        tokio::spawn(async move {
            profiler.trend_analysis_loop().await;
        });

        Ok(())
    }

    /// Stop performance profiling
    pub async fn stop_profiling(&self) -> Result<()> {
        {
            let mut active = self.profiling_active.lock()?;
            *active = false;
        }
        
        self.cleanup_gpu_profiler().await?;
        Ok(())
    }

    /// Collect performance metrics for a specific kernel
    pub async fn profile_kernel_execution(
        &self,
        kernel_id: String,
        execution_time: Duration,
    ) -> Result<GpuPerformanceMetrics> {
        // This will fail in RED phase - no actual GPU metrics collection
        let gpu_metrics = self.collect_gpu_metrics(&kernel_id).await?;
        
        let metrics = GpuPerformanceMetrics {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            kernel_id: kernel_id.clone(),
            execution_time_ns: execution_time.as_nanos() as u64,
            throughput_ops_per_sec: gpu_metrics.throughput,
            memory_bandwidth_gb_per_sec: gpu_metrics.memory_bandwidth,
            gpu_utilization_percent: gpu_metrics.gpu_utilization,
            memory_utilization_percent: gpu_metrics.memory_utilization,
            power_consumption_watts: gpu_metrics.power_consumption,
            temperature_celsius: gpu_metrics.temperature,
            cache_hit_rate: gpu_metrics.cache_hit_rate,
            warp_efficiency: gpu_metrics.warp_efficiency,
            occupancy_percent: gpu_metrics.occupancy,
        };

        // Add to buffer
        {
            let mut buffer = self.metrics_buffer.write().await;
            buffer.push_back(metrics.clone());
            
            // Maintain buffer size
            while buffer.len() > self.config.metrics_buffer_size {
                buffer.pop_front();
            }
        }

        Ok(metrics)
    }

    /// Analyze performance trends and trigger optimizations
    pub async fn analyze_performance_trends(&self, kernel_id: &str) -> Result<PerformanceTrend> {
        let recent_metrics = self.get_recent_metrics(kernel_id, 100).await?;
        
        if recent_metrics.is_empty() {
            return Err(anyhow!("No metrics available for kernel {}", kernel_id));
        }

        // This will fail in RED phase - no trend analysis implemented
        let trend = self.compute_performance_trend(&recent_metrics).await?;

        // Store trend
        {
            let mut trends = self.performance_trends.write().await;
            trends.insert(kernel_id.to_string(), trend.clone());
        }

        // Trigger optimization if needed
        if self.config.enable_real_time_optimization {
            self.trigger_optimization_if_needed(&trend).await?;
        }

        Ok(trend)
    }

    /// Get recent performance metrics for a kernel
    pub async fn get_recent_metrics(
        &self,
        kernel_id: &str,
        count: usize,
    ) -> Result<Vec<GpuPerformanceMetrics>> {
        let buffer = self.metrics_buffer.read().await;
        
        let recent: Vec<GpuPerformanceMetrics> = buffer
            .iter()
            .filter(|m| m.kernel_id == kernel_id)
            .rev()
            .take(count)
            .cloned()
            .collect();
        
        Ok(recent)
    }

    /// Get performance statistics for a kernel
    pub async fn get_performance_stats(&self, kernel_id: &str) -> Result<PerformanceStats> {
        let metrics = self.get_recent_metrics(kernel_id, 1000).await?;
        
        if metrics.is_empty() {
            return Err(anyhow!("No performance data for kernel {}", kernel_id));
        }

        let execution_times: Vec<f64> = metrics
            .iter()
            .map(|m| m.execution_time_ns as f64 / 1_000_000.0) // Convert to ms
            .collect();

        let throughputs: Vec<f64> = metrics
            .iter()
            .map(|m| m.throughput_ops_per_sec)
            .collect();

        Ok(PerformanceStats {
            kernel_id: kernel_id.to_string(),
            sample_count: metrics.len(),
            avg_execution_time_ms: execution_times.iter().sum::<f64>() / execution_times.len() as f64,
            min_execution_time_ms: execution_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_execution_time_ms: execution_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            avg_throughput: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
            peak_throughput: throughputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            avg_gpu_utilization: metrics.iter().map(|m| m.gpu_utilization_percent).sum::<f64>() / metrics.len() as f64,
            avg_power_consumption: metrics.iter().map(|m| m.power_consumption_watts).sum::<f64>() / metrics.len() as f64,
        })
    }

    /// Enable predictive performance scaling
    pub async fn enable_predictive_scaling(&self, kernel_id: &str) -> Result<()> {
        // This will fail in RED phase - no predictive scaling implemented
        self.setup_predictive_model(kernel_id).await?;
        Ok(())
    }

    // Implementation methods that will fail in RED phase
    async fn initialize_gpu_profiler(&self) -> Result<()> {
        Err(anyhow!("GPU profiler initialization not implemented - RED phase failure"))
    }

    async fn cleanup_gpu_profiler(&self) -> Result<()> {
        Err(anyhow!("GPU profiler cleanup not implemented - RED phase failure"))
    }

    async fn collect_gpu_metrics(&self, _kernel_id: &str) -> Result<RawGpuMetrics> {
        Err(anyhow!("GPU metrics collection not implemented - RED phase failure"))
    }

    async fn compute_performance_trend(&self, _metrics: &[GpuPerformanceMetrics]) -> Result<PerformanceTrend> {
        Err(anyhow!("Performance trend analysis not implemented - RED phase failure"))
    }

    async fn trigger_optimization_if_needed(&self, trend: &PerformanceTrend) -> Result<()> {
        if matches!(trend.trend_direction, TrendDirection::Degrading) 
            && trend.confidence > 0.8 {
            self.optimization_sender.send(OptimizationCommand::TuneKernel {
                kernel_id: trend.kernel_id.clone(),
                action: trend.recommended_action.clone(),
            })?;
        }
        Ok(())
    }

    async fn setup_predictive_model(&self, _kernel_id: &str) -> Result<()> {
        Err(anyhow!("Predictive scaling setup not implemented - RED phase failure"))
    }

    async fn metrics_collection_loop(&self) {
        let mut interval = interval(Duration::from_millis(self.config.sampling_interval_ms));
        
        while {
            let active = self.profiling_active.lock()?;
            *active
        } {
            interval.tick().await;
            
            // This loop will effectively do nothing in RED phase
            // Real implementation would collect GPU metrics here
        }
    }

    async fn trend_analysis_loop(&self) {
        let mut interval = interval(Duration::from_secs(5)); // Analyze trends every 5 seconds
        
        while {
            let active = self.profiling_active.lock()?;
            *active
        } {
            interval.tick().await;
            
            // This loop will effectively do nothing in RED phase
            // Real implementation would analyze trends here
        }
    }
}

impl Clone for GpuPerformanceProfiler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_buffer: self.metrics_buffer.clone(),
            active_kernels: self.active_kernels.clone(),
            performance_trends: self.performance_trends.clone(),
            optimization_sender: self.optimization_sender.clone(),
            profiling_active: self.profiling_active.clone(),
        }
    }
}

/// Raw GPU metrics structure
struct RawGpuMetrics {
    throughput: f64,
    memory_bandwidth: f64,
    gpu_utilization: f64,
    memory_utilization: f64,
    power_consumption: f64,
    temperature: f64,
    cache_hit_rate: f64,
    warp_efficiency: f64,
    occupancy: f64,
}

/// Performance statistics summary
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub kernel_id: String,
    pub sample_count: usize,
    pub avg_execution_time_ms: f64,
    pub min_execution_time_ms: f64,
    pub max_execution_time_ms: f64,
    pub avg_throughput: f64,
    pub peak_throughput: f64,
    pub avg_gpu_utilization: f64,
    pub avg_power_consumption: f64,
}

/// Autonomous optimization engine
pub struct AutonomousOptimizer {
    profiler: Arc<GpuPerformanceProfiler>,
    optimization_history: Arc<RwLock<HashMap<String, Vec<OptimizationAttempt>>>>,
    learning_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct OptimizationAttempt {
    pub timestamp: u64,
    pub kernel_id: String,
    pub action: OptimizationAction,
    pub pre_optimization_perf: f64,
    pub post_optimization_perf: f64,
    pub success: bool,
}

impl AutonomousOptimizer {
    pub fn new(profiler: Arc<GpuPerformanceProfiler>) -> Self {
        Self {
            profiler,
            optimization_history: Arc::new(RwLock::new(HashMap::new())),
            learning_enabled: true,
        }
    }

    /// Apply optimization based on performance feedback
    pub async fn optimize_kernel(&self, kernel_id: &str, action: OptimizationAction) -> Result<bool> {
        // Record pre-optimization performance
        let pre_stats = self.profiler.get_performance_stats(kernel_id).await?;
        
        // This will fail in RED phase - no actual optimization implementation
        let success = self.apply_optimization(kernel_id, &action).await?;
        
        // Record post-optimization performance
        sleep(Duration::from_secs(1)).await; // Wait for metrics
        let post_stats = self.profiler.get_performance_stats(kernel_id).await?;
        
        // Record optimization attempt
        let attempt = OptimizationAttempt {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            kernel_id: kernel_id.to_string(),
            action,
            pre_optimization_perf: pre_stats.avg_throughput,
            post_optimization_perf: post_stats.avg_throughput,
            success,
        };
        
        {
            let mut history = self.optimization_history.write().await;
            history.entry(kernel_id.to_string()).or_insert_with(Vec::new).push(attempt);
        }
        
        Ok(success)
    }

    /// Learn from optimization history
    pub async fn learn_from_history(&self, kernel_id: &str) -> Result<Vec<OptimizationAction>> {
        let history = self.optimization_history.read().await;
        let attempts = history.get(kernel_id).cloned().unwrap_or_default();
        
        // This will fail in RED phase - no learning algorithm implemented
        let recommended_actions = self.analyze_optimization_history(&attempts).await?;
        
        Ok(recommended_actions)
    }

    async fn apply_optimization(&self, _kernel_id: &str, _action: &OptimizationAction) -> Result<bool> {
        Err(anyhow!("Optimization application not implemented - RED phase failure"))
    }

    async fn analyze_optimization_history(&self, _attempts: &[OptimizationAttempt]) -> Result<Vec<OptimizationAction>> {
        Err(anyhow!("Optimization history analysis not implemented - RED phase failure"))
    }
}

/// Performance feedback loop coordinator
pub struct PerformanceFeedbackLoop {
    profiler: Arc<GpuPerformanceProfiler>,
    optimizer: Arc<AutonomousOptimizer>,
    feedback_active: Arc<Mutex<bool>>,
}

impl PerformanceFeedbackLoop {
    pub fn new(profiler: Arc<GpuPerformanceProfiler>, optimizer: Arc<AutonomousOptimizer>) -> Self {
        Self {
            profiler,
            optimizer,
            feedback_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Start autonomous feedback loop
    pub async fn start_feedback_loop(&self) -> Result<()> {
        {
            let mut active = self.feedback_active.lock()?;
            *active = true;
        }

        // This will fail in RED phase - no feedback loop implementation
        self.initialize_feedback_system().await?;
        
        Ok(())
    }

    /// Stop feedback loop
    pub async fn stop_feedback_loop(&self) -> Result<()> {
        {
            let mut active = self.feedback_active.lock()?;
            *active = false;
        }
        Ok(())
    }

    async fn initialize_feedback_system(&self) -> Result<()> {
        Err(anyhow!("Feedback loop system not implemented - RED phase failure"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    /// Test real-time GPU performance profiling
    #[tokio::test]
    async fn test_real_time_performance_profiling() {
        let (opt_sender, mut opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig::default();
        let profiler = GpuPerformanceProfiler::new(config, opt_sender);
        
        // Start profiling - should fail in RED phase
        let result = profiler.start_profiling().await;
        assert!(result.is_err(), "Profiling should fail in RED phase");
        
        // Try to profile kernel execution
        let kernel_execution_result = profiler.profile_kernel_execution(
            "test_kernel".to_string(),
            Duration::from_micros(500),
        ).await;
        
        // Should fail because GPU metrics collection isn't implemented
        assert!(kernel_execution_result.is_err(), 
            "Kernel profiling should fail in RED phase");
    }

    /// Test performance trend analysis and optimization triggers
    #[tokio::test]
    async fn test_performance_trend_analysis() {
        let (opt_sender, mut opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig {
            enable_real_time_optimization: true,
            optimization_trigger_threshold: 70.0,
            ..ProfilerConfig::default()
        };
        let profiler = Arc::new(GpuPerformanceProfiler::new(config, opt_sender));
        
        // Simulate degrading performance metrics
        let degrading_metrics = vec![
            GpuPerformanceMetrics {
                timestamp: 1000,
                kernel_id: "degrading_kernel".to_string(),
                execution_time_ns: 1_000_000,
                throughput_ops_per_sec: 100.0,
                memory_bandwidth_gb_per_sec: 500.0,
                gpu_utilization_percent: 90.0,
                memory_utilization_percent: 80.0,
                power_consumption_watts: 250.0,
                temperature_celsius: 75.0,
                cache_hit_rate: 0.95,
                warp_efficiency: 0.85,
                occupancy_percent: 90.0,
            },
            GpuPerformanceMetrics {
                timestamp: 2000,
                kernel_id: "degrading_kernel".to_string(),
                execution_time_ns: 1_500_000,
                throughput_ops_per_sec: 80.0,
                memory_bandwidth_gb_per_sec: 450.0,
                gpu_utilization_percent: 85.0,
                memory_utilization_percent: 85.0,
                power_consumption_watts: 260.0,
                temperature_celsius: 78.0,
                cache_hit_rate: 0.90,
                warp_efficiency: 0.80,
                occupancy_percent: 85.0,
            },
        ];
        
        // Add metrics to buffer manually for testing
        {
            let mut buffer = profiler.metrics_buffer.write().await;
            for metric in degrading_metrics {
                buffer.push_back(metric);
            }
        }
        
        // Try to analyze trends - should fail in RED phase
        let trend_result = profiler.analyze_performance_trends("degrading_kernel").await;
        assert!(trend_result.is_err(), "Trend analysis should fail in RED phase");
    }

    /// Test autonomous kernel optimization
    #[tokio::test]
    async fn test_autonomous_kernel_optimization() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig::default();
        let profiler = Arc::new(GpuPerformanceProfiler::new(config, opt_sender));
        let optimizer = AutonomousOptimizer::new(profiler.clone());
        
        // Try to optimize a kernel - should fail in RED phase
        let optimization_result = optimizer.optimize_kernel(
            "target_kernel",
            OptimizationAction::IncreaseThreads,
        ).await;
        
        assert!(optimization_result.is_err(), 
            "Kernel optimization should fail in RED phase");
    }

    /// Test performance statistics collection
    #[tokio::test]
    async fn test_performance_statistics_collection() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig::default();
        let profiler = GpuPerformanceProfiler::new(config, opt_sender);
        
        // Add some test metrics
        let test_metrics = vec![
            GpuPerformanceMetrics {
                timestamp: 1000,
                kernel_id: "stats_kernel".to_string(),
                execution_time_ns: 1_000_000,
                throughput_ops_per_sec: 1000.0,
                memory_bandwidth_gb_per_sec: 800.0,
                gpu_utilization_percent: 95.0,
                memory_utilization_percent: 75.0,
                power_consumption_watts: 300.0,
                temperature_celsius: 70.0,
                cache_hit_rate: 0.98,
                warp_efficiency: 0.90,
                occupancy_percent: 95.0,
            },
            GpuPerformanceMetrics {
                timestamp: 2000,
                kernel_id: "stats_kernel".to_string(),
                execution_time_ns: 900_000,
                throughput_ops_per_sec: 1100.0,
                memory_bandwidth_gb_per_sec: 850.0,
                gpu_utilization_percent: 97.0,
                memory_utilization_percent: 73.0,
                power_consumption_watts: 295.0,
                temperature_celsius: 69.0,
                cache_hit_rate: 0.99,
                warp_efficiency: 0.92,
                occupancy_percent: 97.0,
            },
        ];
        
        {
            let mut buffer = profiler.metrics_buffer.write().await;
            for metric in test_metrics {
                buffer.push_back(metric);
            }
        }
        
        // Try to get performance stats
        let stats_result = profiler.get_performance_stats("stats_kernel").await;
        
        // This should succeed because it only processes existing data
        assert!(stats_result.is_ok(), "Statistics calculation should work with existing data");
        
        let stats = stats_result.unwrap();
        assert_eq!(stats.kernel_id, "stats_kernel");
        assert_eq!(stats.sample_count, 2);
        assert!(stats.avg_throughput > 0.0);
    }

    /// Test predictive performance scaling
    #[tokio::test]
    async fn test_predictive_performance_scaling() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig {
            enable_predictive_scaling: true,
            ..ProfilerConfig::default()
        };
        let profiler = GpuPerformanceProfiler::new(config, opt_sender);
        
        // Try to enable predictive scaling - should fail in RED phase
        let scaling_result = profiler.enable_predictive_scaling("predictive_kernel").await;
        assert!(scaling_result.is_err(), 
            "Predictive scaling should fail in RED phase");
    }

    /// Test performance feedback loop
    #[tokio::test]
    async fn test_performance_feedback_loop() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig::default();
        let profiler = Arc::new(GpuPerformanceProfiler::new(config, opt_sender));
        let optimizer = Arc::new(AutonomousOptimizer::new(profiler.clone()));
        let feedback_loop = PerformanceFeedbackLoop::new(profiler, optimizer);
        
        // Try to start feedback loop - should fail in RED phase
        let feedback_result = feedback_loop.start_feedback_loop().await;
        assert!(feedback_result.is_err(), 
            "Feedback loop should fail in RED phase");
    }

    /// Test optimization learning from history
    #[tokio::test]
    async fn test_optimization_learning() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig::default();
        let profiler = Arc::new(GpuPerformanceProfiler::new(config, opt_sender));
        let optimizer = AutonomousOptimizer::new(profiler.clone());
        
        // Try to learn from optimization history - should fail in RED phase
        let learning_result = optimizer.learn_from_history("learning_kernel").await;
        assert!(learning_result.is_err(), 
            "Learning from history should fail in RED phase");
    }

    /// Test concurrent performance profiling of multiple kernels
    #[tokio::test]
    async fn test_concurrent_multi_kernel_profiling() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig {
            sampling_interval_ms: 5,
            metrics_buffer_size: 5000,
            ..ProfilerConfig::default()
        };
        let profiler = Arc::new(GpuPerformanceProfiler::new(config, opt_sender));
        
        // Launch concurrent profiling tasks for multiple kernels
        let mut handles = Vec::new();
        for i in 0..5 {
            let profiler_clone = profiler.clone();
            let kernel_id = format!("concurrent_kernel_{}", i);
            
            let handle = tokio::spawn(async move {
                profiler_clone.profile_kernel_execution(
                    kernel_id,
                    Duration::from_micros(100 + i * 50),
                ).await
            });
            handles.push(handle);
        }
        
        // All tasks should fail in RED phase
        for handle in handles {
            let result = handle.await.expect("Task should complete");
            assert!(result.is_err(), "Concurrent profiling should fail in RED phase");
        }
    }

    /// Test memory-efficient metrics buffering
    #[tokio::test]
    async fn test_metrics_buffer_management() {
        let (opt_sender, _opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig {
            metrics_buffer_size: 3, // Small buffer for testing
            ..ProfilerConfig::default()
        };
        let profiler = GpuPerformanceProfiler::new(config, opt_sender);
        
        // Add more metrics than buffer can hold
        let test_metrics = (0..5).map(|i| GpuPerformanceMetrics {
            timestamp: 1000 + i,
            kernel_id: "buffer_test".to_string(),
            execution_time_ns: 1_000_000 + i * 100_000,
            throughput_ops_per_sec: 100.0 + i as f64 * 10.0,
            memory_bandwidth_gb_per_sec: 500.0,
            gpu_utilization_percent: 90.0,
            memory_utilization_percent: 80.0,
            power_consumption_watts: 250.0,
            temperature_celsius: 70.0,
            cache_hit_rate: 0.95,
            warp_efficiency: 0.85,
            occupancy_percent: 90.0,
        }).collect::<Vec<_>>();
        
        // Add all metrics
        {
            let mut buffer = profiler.metrics_buffer.write().await;
            for metric in test_metrics {
                buffer.push_back(metric);
                
                // Simulate buffer size management
                while buffer.len() > 3 {
                    buffer.pop_front();
                }
            }
        }
        
        // Verify buffer size is maintained
        {
            let buffer = profiler.metrics_buffer.read().await;
            assert_eq!(buffer.len(), 3, "Buffer size should be maintained at 3");
            
            // Verify we have the most recent metrics (timestamps 3, 4, 5)
            let timestamps: Vec<u64> = buffer.iter().map(|m| m.timestamp).collect();
            assert_eq!(timestamps, vec![1003, 1004, 1005]);
        }
    }

    /// Test real-time optimization triggering
    #[tokio::test]
    async fn test_real_time_optimization_triggering() {
        let (opt_sender, mut opt_receiver) = mpsc::unbounded_channel();
        let config = ProfilerConfig {
            enable_real_time_optimization: true,
            optimization_trigger_threshold: 75.0,
            ..ProfilerConfig::default()
        };
        let profiler = GpuPerformanceProfiler::new(config, opt_sender);
        
        // Create a performance trend that should trigger optimization
        let degrading_trend = PerformanceTrend {
            kernel_id: "trigger_test".to_string(),
            trend_direction: TrendDirection::Degrading,
            confidence: 0.9,
            predicted_performance_in_5min: 60.0, // Below threshold
            recommended_action: OptimizationAction::IncreaseThreads,
        };
        
        // This should succeed since it's just sending a message
        let trigger_result = profiler.trigger_optimization_if_needed(&degrading_trend).await;
        assert!(trigger_result.is_ok(), "Optimization triggering should succeed");
        
        // Verify optimization command was sent
        let command = timeout(Duration::from_millis(100), opt_receiver.recv()).await;
        assert!(command.is_ok(), "Should receive optimization command");
        
        if let Ok(Some(OptimizationCommand::TuneKernel { kernel_id, action })) = command {
            assert_eq!(kernel_id, "trigger_test");
            assert!(matches!(action, OptimizationAction::IncreaseThreads));
        }
    }
}