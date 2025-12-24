//! GPU utilization analysis and optimization
//!
//! Monitors GPU usage patterns, identifies bottlenecks, and provides
//! optimization recommendations to achieve 90% utilization target.

use super::*;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

/// GPU utilization analyzer and optimizer
pub struct GpuUtilizationAnalyzer {
    config: UtilizationConfig,
    device: Option<Arc<CudaDevice>>,
    is_monitoring: Arc<AtomicBool>,
    monitor_handle: Option<JoinHandle<()>>,
    utilization_history: Arc<Mutex<VecDeque<UtilizationSample>>>,
    kernel_stats: Arc<DashMap<String, KernelStats>>,
    optimization_stats: Arc<UtilizationStats>,
    current_strategies: Arc<RwLock<Vec<OptimizationStrategy>>>,
}

impl GpuUtilizationAnalyzer {
    /// Create new GPU utilization analyzer
    pub fn new(config: UtilizationConfig) -> Self {
        Self {
            config,
            device: None,
            is_monitoring: Arc::new(AtomicBool::new(false)),
            monitor_handle: None,
            utilization_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            kernel_stats: Arc::new(DashMap::new()),
            optimization_stats: Arc::new(UtilizationStats::default()),
            current_strategies: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize with CUDA device
    pub fn with_device(mut self, device: Arc<CudaDevice>) -> Self {
        self.device = Some(device);
        self
    }

    /// Start GPU utilization monitoring
    pub async fn start(&self) -> Result<()> {
        self.is_monitoring.store(true, Ordering::Relaxed);

        // Start monitoring thread
        let is_monitoring = self.is_monitoring.clone();
        let config = self.config.clone();
        let device = self.device.clone();
        let utilization_history = self.utilization_history.clone();
        let kernel_stats = self.kernel_stats.clone();
        let optimization_stats = self.optimization_stats.clone();

        let monitor_handle = tokio::spawn(async move {
            Self::monitoring_loop(
                is_monitoring,
                config,
                device,
                utilization_history,
                kernel_stats,
                optimization_stats,
            )
            .await;
        });

        Ok(())
    }

    /// Stop GPU utilization monitoring
    pub async fn stop(&self) -> Result<()> {
        self.is_monitoring.store(false, Ordering::Relaxed);

        if let Some(handle) = &self.monitor_handle {
            handle.abort();
        }

        Ok(())
    }

    /// Get current GPU utilization
    pub fn get_current_utilization(&self) -> f32 {
        // In a real implementation, would query NVIDIA ML or similar
        // For now, simulate based on kernel activity
        let kernel_activity = self.estimate_kernel_activity();
        let memory_activity = self.estimate_memory_activity();
        let compute_activity = self.estimate_compute_activity();

        // Weighted average: 40% compute, 30% memory, 30% kernel scheduling
        compute_activity * 0.4 + memory_activity * 0.3 + kernel_activity * 0.3
    }

    /// Record kernel execution for analysis
    pub async fn record_kernel_execution(
        &self,
        kernel_name: &str,
        execution_time: Duration,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        shared_memory: usize,
        occupancy: f32,
    ) -> Result<()> {
        self.kernel_stats
            .entry(kernel_name.to_string())
            .and_modify(|kernel_stats| {
                kernel_stats.execution_count += 1;
                kernel_stats.total_execution_time += execution_time;
                kernel_stats.total_threads += (grid_size.0 * grid_size.1 * grid_size.2) as u64
                    * (block_size.0 * block_size.1 * block_size.2) as u64;
                kernel_stats.average_occupancy = (kernel_stats.average_occupancy
                    * (kernel_stats.execution_count - 1) as f32
                    + occupancy)
                    / kernel_stats.execution_count as f32;
                kernel_stats.shared_memory_usage = shared_memory;
            })
            .or_insert_with(|| {
                let mut ks = KernelStats::default();
                ks.execution_count = 1;
                ks.total_execution_time = execution_time;
                ks.total_threads = (grid_size.0 * grid_size.1 * grid_size.2) as u64
                    * (block_size.0 * block_size.1 * block_size.2) as u64;
                ks.average_occupancy = occupancy;
                ks.shared_memory_usage = shared_memory;
                ks
            });

        // Update optimization statistics
        self.optimization_stats
            .kernels_analyzed
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Analyze kernel performance bottlenecks
    pub async fn analyze_kernel_bottlenecks(&self) -> Result<Vec<KernelBottleneck>> {
        let mut bottlenecks = vec![];

        for entry in self.kernel_stats.iter() {
            let kernel_name = entry.key();
            let kernel_stats = entry.value();
            let avg_execution_time = kernel_stats.total_execution_time.as_micros() as f64
                / kernel_stats.execution_count as f64;

            // Check for low occupancy
            if kernel_stats.average_occupancy < self.config.min_occupancy_threshold {
                bottlenecks.push(KernelBottleneck {
                    kernel_name: kernel_name.clone(),
                    bottleneck_type: BottleneckType::LowOccupancy,
                    severity: if kernel_stats.average_occupancy < 0.25 {
                        BottleneckSeverity::Critical
                    } else {
                        BottleneckSeverity::Major
                    },
                    current_value: kernel_stats.average_occupancy,
                    target_value: self.config.min_occupancy_threshold,
                    recommendation: format!(
                        "Increase occupancy from {:.1}% to {:.1}%",
                        kernel_stats.average_occupancy * 100.0,
                        self.config.min_occupancy_threshold * 100.0
                    ),
                });
            }

            // Check for excessive execution time
            if avg_execution_time > self.config.max_kernel_time_us as f64 {
                bottlenecks.push(KernelBottleneck {
                    kernel_name: kernel_name.clone(),
                    bottleneck_type: BottleneckType::SlowExecution,
                    severity: BottleneckSeverity::Major,
                    current_value: avg_execution_time as f32,
                    target_value: self.config.max_kernel_time_us as f32,
                    recommendation: format!(
                        "Optimize kernel to reduce execution time from {:.0}μs to <{:.0}μs",
                        avg_execution_time, self.config.max_kernel_time_us
                    ),
                });
            }

            // Check for memory bandwidth limitations
            let memory_throughput = self.estimate_memory_throughput(kernel_stats);
            if memory_throughput < self.config.min_memory_throughput_gbps {
                bottlenecks.push(KernelBottleneck {
                    kernel_name: kernel_name.clone(),
                    bottleneck_type: BottleneckType::MemoryBandwidth,
                    severity: BottleneckSeverity::Medium,
                    current_value: memory_throughput,
                    target_value: self.config.min_memory_throughput_gbps,
                    recommendation: format!("Improve memory access patterns to increase throughput from {:.1} GB/s to >{:.1} GB/s",
                                          memory_throughput, self.config.min_memory_throughput_gbps),
                });
            }
        }

        // Sort by severity
        bottlenecks.sort_by(|a, b| {
            b.severity.cmp(&a.severity).then(
                b.current_value
                    .partial_cmp(&a.current_value)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        Ok(bottlenecks)
    }

    /// Generate optimization recommendations
    pub async fn get_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = vec![];
        let current_util = self.get_current_utilization();
        let bottlenecks = self.analyze_kernel_bottlenecks().await?;

        // General utilization recommendations
        if current_util < self.config.target_utilization {
            let util_gap = self.config.target_utilization - current_util;

            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::GpuUtilization,
                description: format!(
                    "Increase GPU utilization from {:.1}% to {:.1}%",
                    current_util * 100.0,
                    self.config.target_utilization * 100.0
                ),
                estimated_impact: util_gap,
                implementation_cost: 0.3,
                priority: if util_gap > 0.2 {
                    RecommendationPriority::Critical
                } else {
                    RecommendationPriority::High
                },
                parameters: [
                    (
                        "current_utilization".to_string(),
                        (current_util * 100.0).to_string(),
                    ),
                    (
                        "target_utilization".to_string(),
                        (self.config.target_utilization * 100.0).to_string(),
                    ),
                ]
                .into(),
            });
        }

        // Kernel-specific recommendations
        for bottleneck in bottlenecks {
            let (description, parameters) = match bottleneck.bottleneck_type {
                BottleneckType::LowOccupancy => (
                    format!("Optimize {} for higher occupancy", bottleneck.kernel_name),
                    [
                        ("kernel".to_string(), bottleneck.kernel_name.clone()),
                        ("optimization".to_string(), "occupancy".to_string()),
                        (
                            "current_occupancy".to_string(),
                            (bottleneck.current_value * 100.0).to_string(),
                        ),
                        (
                            "target_occupancy".to_string(),
                            (bottleneck.target_value * 100.0).to_string(),
                        ),
                    ]
                    .into(),
                ),
                BottleneckType::SlowExecution => (
                    format!("Optimize {} execution speed", bottleneck.kernel_name),
                    [
                        ("kernel".to_string(), bottleneck.kernel_name.clone()),
                        ("optimization".to_string(), "execution_speed".to_string()),
                        (
                            "current_time_us".to_string(),
                            bottleneck.current_value.to_string(),
                        ),
                        (
                            "target_time_us".to_string(),
                            bottleneck.target_value.to_string(),
                        ),
                    ]
                    .into(),
                ),
                BottleneckType::MemoryBandwidth => (
                    format!("Optimize {} memory access patterns", bottleneck.kernel_name),
                    [
                        ("kernel".to_string(), bottleneck.kernel_name.clone()),
                        ("optimization".to_string(), "memory_bandwidth".to_string()),
                        (
                            "current_throughput_gbps".to_string(),
                            bottleneck.current_value.to_string(),
                        ),
                        (
                            "target_throughput_gbps".to_string(),
                            bottleneck.target_value.to_string(),
                        ),
                    ]
                    .into(),
                ),
            };

            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::GpuUtilization,
                description,
                estimated_impact: (bottleneck.target_value - bottleneck.current_value).abs()
                    / bottleneck.target_value,
                implementation_cost: match bottleneck.severity {
                    BottleneckSeverity::Critical => 0.5,
                    BottleneckSeverity::Major => 0.3,
                    BottleneckSeverity::Medium => 0.2,
                    BottleneckSeverity::Minor => 0.1,
                },
                priority: match bottleneck.severity {
                    BottleneckSeverity::Critical => RecommendationPriority::Critical,
                    BottleneckSeverity::Major => RecommendationPriority::High,
                    BottleneckSeverity::Medium => RecommendationPriority::Medium,
                    BottleneckSeverity::Minor => RecommendationPriority::Low,
                },
                parameters,
            });
        }

        // Stream optimization recommendations
        recommendations.extend(self.generate_stream_recommendations().await?);

        // Memory optimization recommendations
        recommendations.extend(self.generate_memory_recommendations().await?);

        Ok(recommendations)
    }

    /// Trigger immediate optimization
    pub async fn trigger_optimization(&self) -> Result<()> {
        let recommendations = self.get_recommendations().await?;

        for recommendation in recommendations {
            if recommendation.priority == RecommendationPriority::Critical {
                self.apply_optimization(recommendation).await?;
            }
        }

        self.optimization_stats
            .optimizations_triggered
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Apply a specific optimization
    pub async fn apply_optimization(
        &self,
        recommendation: OptimizationRecommendation,
    ) -> Result<()> {
        let mut strategies = self.current_strategies.write().await;

        let strategy = match recommendation
            .parameters
            .get("optimization")
            .map(|s| s.as_str())
        {
            Some("occupancy") => OptimizationStrategy::IncreaseOccupancy {
                kernel_name: recommendation.parameters.get("kernel")?.clone(),
                target_occupancy: recommendation
                    .parameters
                    .get("target_occupancy")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(75.0)
                    / 100.0,
            },
            Some("execution_speed") => OptimizationStrategy::OptimizeKernelSpeed {
                kernel_name: recommendation.parameters.get("kernel")?.clone(),
                target_time_us: recommendation
                    .parameters
                    .get("target_time_us")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1000.0),
            },
            Some("memory_bandwidth") => OptimizationStrategy::OptimizeMemoryAccess {
                kernel_name: recommendation.parameters.get("kernel")?.clone(),
                target_throughput: recommendation
                    .parameters
                    .get("target_throughput_gbps")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(500.0),
            },
            _ => OptimizationStrategy::GeneralUtilization {
                target_utilization: recommendation
                    .parameters
                    .get("target_utilization")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(90.0)
                    / 100.0,
            },
        };

        strategies.push(strategy);
        self.optimization_stats
            .optimizations_applied
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Generate utilization report
    pub async fn generate_report(&self) -> Result<UtilizationReport> {
        let current_util = self.get_current_utilization();
        let history = self.utilization_history.lock().await;

        let average_utilization = if !history.is_empty() {
            history.iter().map(|s| s.utilization).sum::<f32>() / history.len() as f32
        } else {
            current_util
        };

        let peak_utilization = history
            .iter()
            .map(|s| s.utilization)
            .fold(0.0f32, |a, b| a.max(b));

        let utilization_variance = if history.len() > 1 {
            let mean = average_utilization;
            let variance = history
                .iter()
                .map(|s| (s.utilization - mean).powi(2))
                .sum::<f32>()
                / (history.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        Ok(UtilizationReport {
            current_utilization: current_util,
            average_utilization,
            peak_utilization,
            target_utilization: self.config.target_utilization,
            utilization_variance,
            optimizations_attempted: self
                .optimization_stats
                .optimizations_triggered
                .load(Ordering::Relaxed),
            optimizations_applied: self
                .optimization_stats
                .optimizations_applied
                .load(Ordering::Relaxed),
            improvement_achieved: (current_util - 0.5).max(0.0), // Assume baseline 50%
            kernel_count: self.kernel_stats.len(),
            bottlenecks_identified: self.analyze_kernel_bottlenecks().await?.len(),
        })
    }

    // Helper methods

    /// Main monitoring loop
    async fn monitoring_loop(
        is_monitoring: Arc<AtomicBool>,
        config: UtilizationConfig,
        device: Option<Arc<CudaDevice>>,
        utilization_history: Arc<Mutex<VecDeque<UtilizationSample>>>,
        kernel_stats: Arc<DashMap<String, KernelStats>>,
        optimization_stats: Arc<UtilizationStats>,
    ) {
        while is_monitoring.load(Ordering::Relaxed) {
            // Sample current utilization
            let utilization = if device.is_some() {
                Self::query_gpu_utilization(&device)
            } else {
                Self::simulate_gpu_utilization()
            };

            let sample = UtilizationSample {
                timestamp: Instant::now(),
                utilization,
                active_kernels: Self::count_active_kernels(&kernel_stats).await,
                memory_utilization: Self::query_memory_utilization(&device),
            };

            // Store sample
            {
                let mut history = utilization_history.lock().await;
                history.push_back(sample);
                if history.len() > 1000 {
                    history.pop_front();
                }
            }

            optimization_stats
                .samples_collected
                .fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(config.monitoring_interval).await;
        }
    }

    fn query_gpu_utilization(device: &Option<Arc<CudaDevice>>) -> f32 {
        // In a real implementation, would use NVIDIA ML API
        // For now, simulate based on device activity
        if device.is_some() {
            0.75 // Simulate 75% utilization
        } else {
            0.50 // Default simulation
        }
    }

    fn simulate_gpu_utilization() -> f32 {
        // Simulate utilization with some variance
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let random = (hasher.finish() % 100) as f32 / 100.0;

        0.6 + 0.3 * random // 60-90% range
    }

    async fn count_active_kernels(kernel_stats: &Arc<DashMap<String, KernelStats>>) -> u32 {
        kernel_stats.len() as u32
    }

    fn query_memory_utilization(device: &Option<Arc<CudaDevice>>) -> f32 {
        // In a real implementation, would query actual memory usage
        if device.is_some() {
            0.65 // Simulate 65% memory utilization
        } else {
            0.40 // Default simulation
        }
    }

    fn estimate_kernel_activity(&self) -> f32 {
        // Estimate based on recent kernel executions
        0.70 // Simulate 70% kernel activity
    }

    fn estimate_memory_activity(&self) -> f32 {
        // Estimate based on memory transfer patterns
        0.80 // Simulate 80% memory activity
    }

    fn estimate_compute_activity(&self) -> f32 {
        // Estimate based on compute unit utilization
        0.75 // Simulate 75% compute activity
    }

    fn estimate_memory_throughput(&self, _kernel_stats: &KernelStats) -> f32 {
        // Estimate memory throughput in GB/s
        400.0 // Simulate 400 GB/s
    }

    async fn generate_stream_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        // Generate CUDA stream optimization recommendations
        Ok(vec![OptimizationRecommendation {
            optimization_type: OptimizationType::GpuUtilization,
            description: "Use multiple CUDA streams for concurrent execution".to_string(),
            estimated_impact: 0.15,
            implementation_cost: 0.25,
            priority: RecommendationPriority::Medium,
            parameters: [
                ("optimization".to_string(), "streams".to_string()),
                ("recommended_streams".to_string(), "4".to_string()),
            ]
            .into(),
        }])
    }

    async fn generate_memory_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        // Generate memory optimization recommendations
        Ok(vec![OptimizationRecommendation {
            optimization_type: OptimizationType::GpuUtilization,
            description: "Optimize memory coalescing for better bandwidth utilization".to_string(),
            estimated_impact: 0.10,
            implementation_cost: 0.20,
            priority: RecommendationPriority::Medium,
            parameters: [
                ("optimization".to_string(), "memory_coalescing".to_string()),
                ("target_efficiency".to_string(), "90".to_string()),
            ]
            .into(),
        }])
    }
}

/// Utilization configuration
#[derive(Clone)]
pub struct UtilizationConfig {
    pub target_utilization: f32,
    pub monitoring_interval: Duration,
    pub min_occupancy_threshold: f32,
    pub max_kernel_time_us: u64,
    pub min_memory_throughput_gbps: f32,
}

impl Default for UtilizationConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.90, // 90%
            monitoring_interval: Duration::from_millis(100),
            min_occupancy_threshold: 0.50,     // 50%
            max_kernel_time_us: 10000,         // 10ms
            min_memory_throughput_gbps: 300.0, // 300 GB/s
        }
    }
}

/// Utilization sample
#[derive(Debug, Clone)]
pub struct UtilizationSample {
    pub timestamp: Instant,
    pub utilization: f32,
    pub active_kernels: u32,
    pub memory_utilization: f32,
}

/// Kernel performance statistics
#[derive(Debug, Default)]
pub struct KernelStats {
    pub execution_count: u64,
    pub total_execution_time: Duration,
    pub total_threads: u64,
    pub average_occupancy: f32,
    pub shared_memory_usage: usize,
}

/// Kernel bottleneck information
#[derive(Debug, Clone)]
pub struct KernelBottleneck {
    pub kernel_name: String,
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub current_value: f32,
    pub target_value: f32,
    pub recommendation: String,
}

/// Types of kernel bottlenecks
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckType {
    LowOccupancy,
    SlowExecution,
    MemoryBandwidth,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    Critical,
    Major,
    Medium,
    Minor,
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    IncreaseOccupancy {
        kernel_name: String,
        target_occupancy: f32,
    },
    OptimizeKernelSpeed {
        kernel_name: String,
        target_time_us: f32,
    },
    OptimizeMemoryAccess {
        kernel_name: String,
        target_throughput: f32,
    },
    GeneralUtilization {
        target_utilization: f32,
    },
}

/// Utilization report
#[derive(Debug)]
pub struct UtilizationReport {
    pub current_utilization: f32,
    pub average_utilization: f32,
    pub peak_utilization: f32,
    pub target_utilization: f32,
    pub utilization_variance: f32,
    pub optimizations_attempted: u64,
    pub optimizations_applied: u64,
    pub improvement_achieved: f32,
    pub kernel_count: usize,
    pub bottlenecks_identified: usize,
}

/// Utilization statistics
#[derive(Default)]
pub struct UtilizationStats {
    pub samples_collected: AtomicU64,
    pub kernels_analyzed: AtomicU64,
    pub optimizations_triggered: AtomicU64,
    pub optimizations_applied: AtomicU64,
}
