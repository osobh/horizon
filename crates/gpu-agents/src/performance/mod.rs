//! Performance optimization module for GPU agents
//!
//! This module provides comprehensive performance analysis, tuning, and optimization
//! for the ExoRust GPU agent system to achieve 90% GPU utilization targets.

use crate::GpuAgent;
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

mod compression_tuning;
mod double_buffering;
mod job_batching;
mod memory_optimization;
mod memory_optimization_types;
mod utilization;

pub use compression_tuning::*;
pub use double_buffering::*;
pub use job_batching::*;
pub use memory_optimization::*;
pub use utilization::*;

/// Performance optimization manager
pub struct PerformanceOptimizer {
    config: OptimizationConfig,
    utilization_analyzer: Arc<GpuUtilizationAnalyzer>,
    memory_optimizer: Arc<MemoryTierOptimizer>,
    job_batcher: Arc<JobBatchOptimizer>,
    double_buffer_manager: Arc<DoubleBufferManager>,
    compression_tuner: Arc<CompressionTuner>,
    is_running: Arc<AtomicBool>,
    optimization_handle: Option<JoinHandle<()>>,
    stats: Arc<PerformanceStats>,
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            utilization_analyzer: Arc::new(GpuUtilizationAnalyzer::new(
                config.utilization_config.clone(),
            )),
            memory_optimizer: Arc::new(MemoryTierOptimizer::new(config.memory_config.clone())),
            job_batcher: Arc::new(JobBatchOptimizer::new(config.batching_config.clone())),
            double_buffer_manager: Arc::new(DoubleBufferManager::new(
                config.buffering_config.clone(),
            )),
            compression_tuner: Arc::new(CompressionTuner::new(config.compression_config.clone())),
            config,
            is_running: Arc::new(AtomicBool::new(false)),
            optimization_handle: None,
            stats: Arc::new(PerformanceStats::default()),
        }
    }

    /// Start performance optimization
    pub async fn start(&mut self) -> Result<()> {
        self.is_running.store(true, Ordering::Relaxed);

        // Start all optimizers
        self.utilization_analyzer.start().await?;
        self.memory_optimizer.start().await?;
        self.job_batcher.start().await?;
        self.double_buffer_manager.start().await?;
        self.compression_tuner.start().await?;

        // Start main optimization loop
        let is_running = self.is_running.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let utilization_analyzer = self.utilization_analyzer.clone();
        let memory_optimizer = self.memory_optimizer.clone();
        let job_batcher = self.job_batcher.clone();

        self.optimization_handle = Some(tokio::spawn(async move {
            Self::optimization_loop(
                is_running,
                config,
                stats,
                utilization_analyzer,
                memory_optimizer,
                job_batcher,
            )
            .await;
        }));

        Ok(())
    }

    /// Stop performance optimization
    pub async fn stop(&mut self) -> Result<()> {
        self.is_running.store(false, Ordering::Relaxed);

        // Stop all optimizers
        self.utilization_analyzer.stop().await?;
        self.memory_optimizer.stop().await?;
        self.job_batcher.stop().await?;
        self.double_buffer_manager.stop().await?;
        self.compression_tuner.stop().await?;

        if let Some(handle) = self.optimization_handle.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            gpu_utilization: self.utilization_analyzer.get_current_utilization(),
            memory_efficiency: self.memory_optimizer.get_efficiency_metrics(),
            batch_efficiency: self.job_batcher.get_batch_metrics(),
            buffer_hit_rate: self.double_buffer_manager.get_hit_rate(),
            compression_ratio: self.compression_tuner.get_compression_ratio(),
            optimization_cycles: self.stats.optimization_cycles.load(Ordering::Relaxed),
            improvements_applied: self.stats.improvements_applied.load(Ordering::Relaxed),
        }
    }

    /// Generate optimization report
    pub async fn generate_optimization_report(&self) -> Result<OptimizationReport> {
        let utilization_report = self.utilization_analyzer.generate_report().await?;
        let memory_report = self.memory_optimizer.generate_report().await?;
        let batching_report = self.job_batcher.generate_report().await?;
        let buffering_report = self.double_buffer_manager.generate_report().await?;
        let compression_report = self.compression_tuner.generate_report().await?;

        Ok(OptimizationReport {
            timestamp: std::time::SystemTime::now(),
            overall_score: self.calculate_optimization_score(),
            utilization_report,
            memory_report,
            batching_report,
            buffering_report,
            compression_report,
            recommendations: self.generate_recommendations().await?,
        })
    }

    /// Apply optimization recommendations
    pub async fn apply_optimizations(
        &self,
        recommendations: Vec<OptimizationRecommendation>,
    ) -> Result<()> {
        for recommendation in recommendations {
            match recommendation.optimization_type {
                OptimizationType::GpuUtilization => {
                    self.utilization_analyzer
                        .apply_optimization(recommendation)
                        .await?;
                }
                OptimizationType::MemoryTier => {
                    self.memory_optimizer
                        .apply_optimization(recommendation)
                        .await?;
                }
                OptimizationType::JobBatching => {
                    self.job_batcher.apply_optimization(recommendation).await?;
                }
                OptimizationType::DoubleBuffering => {
                    self.double_buffer_manager
                        .apply_optimization(recommendation)
                        .await?;
                }
                OptimizationType::Compression => {
                    self.compression_tuner
                        .apply_optimization(recommendation)
                        .await?;
                }
            }

            self.stats
                .improvements_applied
                .fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Main optimization loop
    async fn optimization_loop(
        is_running: Arc<AtomicBool>,
        config: OptimizationConfig,
        stats: Arc<PerformanceStats>,
        utilization_analyzer: Arc<GpuUtilizationAnalyzer>,
        memory_optimizer: Arc<MemoryTierOptimizer>,
        job_batcher: Arc<JobBatchOptimizer>,
    ) {
        while is_running.load(Ordering::Relaxed) {
            // Collect current metrics
            let gpu_util = utilization_analyzer.get_current_utilization();
            let memory_eff = memory_optimizer.get_efficiency_metrics();
            let batch_eff = job_batcher.get_batch_metrics();

            // Check if optimizations are needed
            if gpu_util < config.target_gpu_utilization {
                // Trigger GPU utilization optimization
                if let Err(e) = utilization_analyzer.trigger_optimization().await {
                    eprintln!("GPU utilization optimization failed: {}", e);
                }
            }

            if memory_eff.cache_hit_rate < config.target_memory_efficiency {
                // Trigger memory optimization
                if let Err(e) = memory_optimizer.trigger_optimization().await {
                    eprintln!("Memory optimization failed: {}", e);
                }
            }

            if batch_eff.efficiency < config.target_batch_efficiency {
                // Trigger batch optimization
                if let Err(e) = job_batcher.trigger_optimization().await {
                    eprintln!("Job batching optimization failed: {}", e);
                }
            }

            stats.optimization_cycles.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(config.optimization_interval).await;
        }
    }

    /// Calculate overall optimization score
    fn calculate_optimization_score(&self) -> f32 {
        let metrics = self.get_performance_metrics();

        // Weighted scoring based on importance
        let gpu_score = metrics.gpu_utilization * 0.4; // 40% weight
        let memory_score = metrics.memory_efficiency.cache_hit_rate * 0.25; // 25% weight
        let batch_score = metrics.batch_efficiency.efficiency * 0.2; // 20% weight
        let buffer_score = metrics.buffer_hit_rate * 0.1; // 10% weight
        let compression_score = (metrics.compression_ratio - 1.0).min(3.0) / 3.0 * 0.05; // 5% weight

        gpu_score + memory_score + batch_score + buffer_score + compression_score
    }

    /// Generate optimization recommendations
    async fn generate_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = vec![];

        // GPU utilization recommendations
        let util_recs = self.utilization_analyzer.get_recommendations().await?;
        recommendations.extend(util_recs);

        // Memory optimization recommendations
        let mem_recs = self.memory_optimizer.get_recommendations().await?;
        recommendations.extend(mem_recs);

        // Batching recommendations
        let batch_recs = self.job_batcher.get_recommendations().await?;
        recommendations.extend(batch_recs);

        // Buffering recommendations
        let buffer_recs = self.double_buffer_manager.get_recommendations().await?;
        recommendations.extend(buffer_recs);

        // Compression recommendations
        let comp_recs = self.compression_tuner.get_recommendations().await?;
        recommendations.extend(comp_recs);

        // Sort by priority and impact
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.estimated_impact
                    .partial_cmp(&a.estimated_impact)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        Ok(recommendations)
    }
}

/// Optimization configuration
#[derive(Clone)]
pub struct OptimizationConfig {
    pub target_gpu_utilization: f32,
    pub target_memory_efficiency: f32,
    pub target_batch_efficiency: f32,
    pub optimization_interval: Duration,
    pub utilization_config: UtilizationConfig,
    pub memory_config: MemoryOptimizationConfig,
    pub batching_config: BatchingConfig,
    pub buffering_config: BufferingConfig,
    pub compression_config: CompressionConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target_gpu_utilization: 0.90,   // 90% target
            target_memory_efficiency: 0.85, // 85% cache hit rate
            target_batch_efficiency: 0.80,  // 80% batch efficiency
            optimization_interval: Duration::from_secs(5),
            utilization_config: UtilizationConfig::default(),
            memory_config: MemoryOptimizationConfig::default(),
            batching_config: BatchingConfig::default(),
            buffering_config: BufferingConfig::default(),
            compression_config: CompressionConfig::default(),
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub gpu_utilization: f32,
    pub memory_efficiency: MemoryEfficiencyMetrics,
    pub batch_efficiency: BatchEfficiencyMetrics,
    pub buffer_hit_rate: f32,
    pub compression_ratio: f32,
    pub optimization_cycles: u64,
    pub improvements_applied: u64,
}

/// Optimization report
#[derive(Debug)]
pub struct OptimizationReport {
    pub timestamp: std::time::SystemTime,
    pub overall_score: f32,
    pub utilization_report: UtilizationReport,
    pub memory_report: MemoryOptimizationReport,
    pub batching_report: BatchingReport,
    pub buffering_report: BufferingReport,
    pub compression_report: CompressionReport,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub optimization_type: OptimizationType,
    pub description: String,
    pub estimated_impact: f32,
    pub implementation_cost: f32,
    pub priority: RecommendationPriority,
    pub parameters: HashMap<String, String>,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    GpuUtilization,
    MemoryTier,
    JobBatching,
    DoubleBuffering,
    Compression,
}

/// Recommendation priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Performance statistics
#[derive(Default)]
pub struct PerformanceStats {
    pub optimization_cycles: AtomicU64,
    pub improvements_applied: AtomicU64,
    pub gpu_utilization_samples: AtomicU64,
    pub memory_optimizations: AtomicU64,
    pub batch_optimizations: AtomicU64,
    pub buffer_optimizations: AtomicU64,
    pub compression_optimizations: AtomicU64,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod benchmark_tests;
