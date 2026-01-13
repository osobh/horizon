//! Utilization Controller for GPU
//!
//! Advanced controller that coordinates all optimization strategies to achieve
//! and maintain 90%+ GPU utilization.

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

use super::{
    kernel_optimizer::{KernelConfig, KernelOptimizer},
    resource_monitor::{ResourceLimits, ResourceMonitor, ResourceType},
    workload_balancer::{WorkloadBalancer, WorkloadConfig},
    OptimizationStrategy, UtilizationManager, TARGET_UTILIZATION,
};

/// Controller configuration
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Target GPU utilization (0.0 - 1.0)
    pub target_utilization: f32,
    /// Control loop interval
    pub control_interval: Duration,
    /// Enable aggressive optimization
    pub aggressive_mode: bool,
    /// Enable predictive optimization
    pub predictive_mode: bool,
    /// History window for predictions
    pub history_window: Duration,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            target_utilization: TARGET_UTILIZATION,
            control_interval: Duration::from_millis(500),
            aggressive_mode: false,
            predictive_mode: true,
            history_window: Duration::from_secs(30),
        }
    }
}

/// Controller state
#[derive(Debug, Clone)]
pub struct ControllerState {
    /// Current utilization
    pub current_utilization: f32,
    /// Predicted utilization
    pub predicted_utilization: f32,
    /// Active optimization strategy
    pub active_strategy: OptimizationStrategy,
    /// Control actions taken
    pub actions_taken: u64,
    /// Time in optimal range
    pub time_optimal: Duration,
    /// Last update time
    pub last_update: Instant,
}

/// Advanced utilization controller
pub struct UtilizationController {
    config: ControllerConfig,
    device: Arc<CudaContext>,
    utilization_manager: Arc<UtilizationManager>,
    workload_balancer: Arc<WorkloadBalancer>,
    kernel_optimizer: Arc<KernelOptimizer>,
    resource_monitor: Arc<ResourceMonitor>,
    state: Arc<RwLock<ControllerState>>,
    is_running: Arc<AtomicBool>,
    optimization_history: Arc<Mutex<Vec<OptimizationEvent>>>,
}

/// Optimization event for history tracking
#[derive(Debug, Clone)]
struct OptimizationEvent {
    timestamp: Instant,
    utilization_before: f32,
    utilization_after: f32,
    strategy: OptimizationStrategy,
    success: bool,
}

impl UtilizationController {
    /// Create new utilization controller
    pub async fn new(device: Arc<CudaContext>, config: ControllerConfig) -> Result<Self> {
        let utilization_manager = Arc::new(UtilizationManager::new(device.clone())?);
        let workload_balancer = Arc::new(WorkloadBalancer::new(
            device.clone(),
            WorkloadConfig::default(),
        ));
        let kernel_optimizer = Arc::new(KernelOptimizer::new(device.clone()));
        let resource_monitor = Arc::new(ResourceMonitor::new(
            device.clone(),
            ResourceLimits::default(),
        ));

        let state = Arc::new(RwLock::new(ControllerState {
            current_utilization: 0.0,
            predicted_utilization: 0.0,
            active_strategy: OptimizationStrategy::Maintain,
            actions_taken: 0,
            time_optimal: Duration::ZERO,
            last_update: Instant::now(),
        }));

        Ok(Self {
            config,
            device,
            utilization_manager,
            workload_balancer,
            kernel_optimizer,
            resource_monitor,
            state,
            is_running: Arc::new(AtomicBool::new(false)),
            optimization_history: Arc::new(Mutex::new(Vec::with_capacity(1000))),
        })
    }

    /// Start the controller
    pub async fn start(&self) -> Result<()> {
        // Start sub-components
        self.utilization_manager.start_monitoring().await?;
        self.resource_monitor.start_monitoring().await?;

        // Start control loop
        if self.is_running.swap(true, Ordering::Relaxed) {
            return Ok(()); // Already running
        }

        let controller = self.clone_for_async();
        tokio::spawn(async move {
            controller.control_loop().await;
        });

        Ok(())
    }

    /// Stop the controller
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        self.utilization_manager.stop_monitoring();
        self.resource_monitor.stop_monitoring();
    }

    /// Main control loop
    async fn control_loop(&self) {
        let mut last_optimization = Instant::now();

        while self.is_running.load(Ordering::Relaxed) {
            let loop_start = Instant::now();

            // Get current metrics
            let metrics = self.utilization_manager.get_metrics().await;
            let current_util = metrics.current_utilization;

            // Update state
            {
                let mut state = self.state.write().await;
                state.current_utilization = current_util;
                state.last_update = Instant::now();

                // Track time in optimal range
                if (current_util - self.config.target_utilization).abs() < 0.05 {
                    state.time_optimal += loop_start.duration_since(last_optimization);
                }
            }

            // Predict future utilization
            if self.config.predictive_mode {
                if let Ok(predicted) = self.predict_utilization().await {
                    let mut state = self.state.write().await;
                    state.predicted_utilization = predicted;
                }
            }

            // Determine optimization strategy
            let strategy = self.determine_strategy().await;

            // Apply optimization if needed
            if strategy != OptimizationStrategy::Maintain {
                if let Err(e) = self.apply_strategy(strategy).await {
                    eprintln!("Failed to apply optimization: {}", e);
                }
                last_optimization = Instant::now();
            }

            // Wait for next iteration
            let elapsed = loop_start.elapsed();
            if elapsed < self.config.control_interval {
                tokio::time::sleep(self.config.control_interval - elapsed).await;
            }
        }
    }

    /// Determine optimization strategy
    async fn determine_strategy(&self) -> OptimizationStrategy {
        let state = self.state.read().await;
        let current = state.current_utilization;
        let predicted = state.predicted_utilization;
        let target = self.config.target_utilization;

        // Use predicted value if available
        let effective_util = if self.config.predictive_mode && predicted > 0.0 {
            predicted
        } else {
            current
        };

        // Check resource constraints first
        if let Some(memory_alert) = self.check_resource_constraints().await {
            return OptimizationStrategy::DecreaseWorkload;
        }

        // Determine strategy based on utilization
        let diff = target - effective_util;

        if self.config.aggressive_mode {
            // Aggressive optimization
            match diff {
                d if d > 0.15 => OptimizationStrategy::IncreaseWorkload,
                d if d > 0.05 => OptimizationStrategy::BalanceWorkload,
                d if d > -0.02 => OptimizationStrategy::Maintain,
                _ => OptimizationStrategy::DecreaseWorkload,
            }
        } else {
            // Conservative optimization
            self.utilization_manager.get_optimization_strategy().await
        }
    }

    /// Apply optimization strategy
    async fn apply_strategy(&self, strategy: OptimizationStrategy) -> Result<()> {
        let util_before = self.state.read().await.current_utilization;

        // Record start of optimization
        let start = Instant::now();

        // Apply strategy
        match strategy {
            OptimizationStrategy::IncreaseWorkload => {
                self.increase_workload().await?;
            }
            OptimizationStrategy::DecreaseWorkload => {
                self.decrease_workload().await?;
            }
            OptimizationStrategy::OptimizeKernels => {
                self.optimize_kernels().await?;
            }
            OptimizationStrategy::OptimizeMemory => {
                self.optimize_memory().await?;
            }
            OptimizationStrategy::BalanceWorkload => {
                self.balance_workload().await?;
            }
            OptimizationStrategy::Maintain => {
                // No action
            }
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.active_strategy = strategy;
            state.actions_taken += 1;
        }

        // Wait for effect
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Measure result
        let util_after = self
            .utilization_manager
            .get_metrics()
            .await
            .current_utilization;
        let success = (util_after - self.config.target_utilization).abs()
            < (util_before - self.config.target_utilization).abs();

        // Record event
        let event = OptimizationEvent {
            timestamp: start,
            utilization_before: util_before,
            utilization_after: util_after,
            strategy,
            success,
        };

        let mut history = self.optimization_history.lock().await;
        if history.len() >= 1000 {
            history.remove(0);
        }
        history.push(event);

        Ok(())
    }

    /// Increase workload
    async fn increase_workload(&self) -> Result<()> {
        // Adjust workload multiplier
        self.utilization_manager
            .apply_optimization(OptimizationStrategy::IncreaseWorkload)
            .await?;

        // Increase batch size
        let current_util = self.state.read().await.current_utilization;
        self.workload_balancer.adjust_batch_size(current_util);

        Ok(())
    }

    /// Decrease workload
    async fn decrease_workload(&self) -> Result<()> {
        // Adjust workload multiplier
        self.utilization_manager
            .apply_optimization(OptimizationStrategy::DecreaseWorkload)
            .await?;

        // Decrease batch size
        let current_util = self.state.read().await.current_utilization;
        self.workload_balancer.adjust_batch_size(current_util);

        Ok(())
    }

    /// Optimize kernels
    async fn optimize_kernels(&self) -> Result<()> {
        // Auto-tune key kernels
        // In real implementation, identify and optimize hot kernels
        Ok(())
    }

    /// Optimize memory access
    async fn optimize_memory(&self) -> Result<()> {
        // Optimize memory access patterns
        // In real implementation, adjust memory coalescing, prefetching, etc.
        Ok(())
    }

    /// Balance workload
    async fn balance_workload(&self) -> Result<()> {
        // Balance work across streams and SMs
        let stream_batches = self.workload_balancer.balance_across_streams(4).await?;
        // Process batches...
        Ok(())
    }

    /// Predict future utilization
    async fn predict_utilization(&self) -> Result<f32> {
        // Simple prediction based on recent history
        let history_duration = Duration::from_secs(10);
        let measurements = self
            .resource_monitor
            .get_measurement_history(ResourceType::Compute, history_duration)
            .await;

        if measurements.len() < 10 {
            return Ok(self.state.read().await.current_utilization);
        }

        // Calculate trend
        let values: Vec<f32> = measurements.iter().map(|m| m.value / 100.0).collect();
        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, v)| i as f32 * v).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict next value
        let next_x = values.len() as f32;
        let predicted = intercept + slope * next_x;

        Ok(predicted.clamp(0.0, 1.0))
    }

    /// Check resource constraints
    async fn check_resource_constraints(&self) -> Option<()> {
        let alerts = self.resource_monitor.get_recent_alerts(1).await;
        if !alerts.is_empty() {
            Some(())
        } else {
            None
        }
    }

    /// Generate controller report
    pub async fn generate_report(&self) -> String {
        let state = self.state.read().await;
        let history = self.optimization_history.lock().await;

        let successful_optimizations = history.iter().filter(|e| e.success).count();
        let total_optimizations = history.len();
        let success_rate = if total_optimizations > 0 {
            successful_optimizations as f32 / total_optimizations as f32 * 100.0
        } else {
            0.0
        };

        format!(
            "Utilization Controller Report:\n\
             \n\
             Current State:\n\
             - Utilization: {:.1}% (target: {:.1}%)\n\
             - Predicted: {:.1}%\n\
             - Active Strategy: {:?}\n\
             - Actions Taken: {}\n\
             - Time Optimal: {:?}\n\
             \n\
             Performance:\n\
             - Success Rate: {:.1}%\n\
             - Total Optimizations: {}\n\
             - Workload Multiplier: {:.2}x\n\
             \n\
             Configuration:\n\
             - Control Interval: {:?}\n\
             - Aggressive Mode: {}\n\
             - Predictive Mode: {}",
            state.current_utilization * 100.0,
            self.config.target_utilization * 100.0,
            state.predicted_utilization * 100.0,
            state.active_strategy,
            state.actions_taken,
            state.time_optimal,
            success_rate,
            total_optimizations,
            self.utilization_manager.get_workload_multiplier(),
            self.config.control_interval,
            self.config.aggressive_mode,
            self.config.predictive_mode,
        )
    }

    /// Clone controller for async tasks
    fn clone_for_async(&self) -> Self {
        Self {
            config: self.config.clone(),
            device: self.device.clone(),
            utilization_manager: self.utilization_manager.clone(),
            workload_balancer: self.workload_balancer.clone(),
            kernel_optimizer: self.kernel_optimizer.clone(),
            resource_monitor: self.resource_monitor.clone(),
            state: self.state.clone(),
            is_running: self.is_running.clone(),
            optimization_history: self.optimization_history.clone(),
        }
    }
}

/// High-level optimization recommendations
#[derive(Debug)]
pub struct OptimizationRecommendations {
    pub kernel_configs: Vec<(String, KernelConfig)>,
    pub workload_adjustment: f32,
    pub memory_optimizations: Vec<String>,
    pub scheduling_changes: Vec<String>,
}

impl UtilizationController {
    /// Get optimization recommendations
    pub async fn get_recommendations(&self) -> OptimizationRecommendations {
        let state = self.state.read().await;
        let mut recommendations = OptimizationRecommendations {
            kernel_configs: Vec::new(),
            workload_adjustment: 1.0,
            memory_optimizations: Vec::new(),
            scheduling_changes: Vec::new(),
        };

        // Workload adjustment
        let util_diff = self.config.target_utilization - state.current_utilization;
        recommendations.workload_adjustment = 1.0 + util_diff;

        // Memory optimizations
        if state.current_utilization < 0.7 {
            recommendations
                .memory_optimizations
                .push("Enable memory prefetching".to_string());
            recommendations
                .memory_optimizations
                .push("Increase L2 cache usage".to_string());
        }

        // Scheduling changes
        if state.current_utilization < 0.8 {
            recommendations
                .scheduling_changes
                .push("Increase concurrent kernel execution".to_string());
            recommendations
                .scheduling_changes
                .push("Use multiple CUDA streams".to_string());
        }

        recommendations
    }
}
