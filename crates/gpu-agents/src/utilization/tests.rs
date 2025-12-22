//! Tests for GPU Utilization Optimization
//!
//! Comprehensive test suite for achieving and maintaining 90%+ GPU utilization.

use super::*;
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
mod utilization_manager_tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    fn test_utilization_manager_creation() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = UtilizationManager::new(device)?;

        assert_eq!(manager.target_utilization, TARGET_UTILIZATION);
        assert!(!manager.is_monitoring.load(Ordering::Relaxed));

        Ok(())
    }

    #[tokio::test]
    async fn test_monitoring_start_stop() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = UtilizationManager::new(device)?;

        // Start monitoring
        manager.start_monitoring().await?;
        assert!(manager.is_monitoring.load(Ordering::Relaxed));

        // Wait for some measurements
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check metrics
        let metrics = manager.get_metrics().await;
        assert!(metrics.current_utilization >= 0.0);
        assert!(metrics.current_utilization <= 1.0);

        // Stop monitoring
        manager.stop_monitoring();
        assert!(!manager.is_monitoring.load(Ordering::Relaxed));

        Ok(())
    }

    #[tokio::test]
    async fn test_optimization_strategy() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = UtilizationManager::new(device)?;

        // Test different utilization levels
        let test_cases = vec![
            (0.5, OptimizationStrategy::IncreaseWorkload), // 50% -> increase
            (0.75, OptimizationStrategy::BalanceWorkload), // 75% -> balance
            (0.88, OptimizationStrategy::Maintain),        // 88% -> maintain
            (0.98, OptimizationStrategy::DecreaseWorkload), // 98% -> decrease
        ];

        for (util, expected_strategy) in test_cases {
            // Set current utilization
            {
                let mut metrics = manager.metrics.write().await;
                metrics.current_utilization = util;
            }

            let strategy = manager.get_optimization_strategy().await;
            assert_eq!(strategy, expected_strategy, "For utilization {}", util);
        }

        Ok(())
    }

    #[test]
    fn test_workload_multiplier() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = UtilizationManager::new(device)?;

        // Initial multiplier
        assert_eq!(manager.get_workload_multiplier(), 1.0);

        // Set custom multiplier
        manager.workload_multiplier.store(150, Ordering::Relaxed);
        assert_eq!(manager.get_workload_multiplier(), 1.5);

        Ok(())
    }
}

#[cfg(test)]
mod workload_balancer_tests {
    use super::*;
    use crate::utilization::workload_balancer::*;

    #[tokio::test]
    async fn test_workload_submission() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = WorkloadConfig::default();
        let balancer = WorkloadBalancer::new(device, config);

        // Submit work items
        let id1 = balancer.submit_work(1024, 100, 1).await?;
        let id2 = balancer.submit_work(2048, 200, 2).await?;
        let id3 = balancer.submit_work(512, 50, 3).await?;

        assert!(id1 < id2);
        assert!(id2 < id3);

        // Check queue depth
        let stats = balancer.get_stats().await;
        assert_eq!(stats.queue_depth, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_retrieval() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = WorkloadConfig::default();
        let balancer = WorkloadBalancer::new(device, config);

        // Submit items with different priorities
        balancer.submit_work(1024, 100, 1).await?;
        balancer.submit_work(1024, 100, 3).await?; // Higher priority
        balancer.submit_work(1024, 100, 2).await?;

        // Get batch - should be priority ordered
        let batch = balancer.get_next_batch().await?;
        assert!(!batch.is_empty());
        assert_eq!(batch[0].priority, 3); // Highest priority first

        Ok(())
    }

    #[test]
    fn test_batch_size_adjustment() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let config = WorkloadConfig::default();
        let balancer = WorkloadBalancer::new(device, config);

        // Low utilization -> increase batch size
        balancer.adjust_batch_size(0.6);
        let (batch_size, _) = balancer.get_current_config();
        assert!(batch_size > 1024); // Should be increased

        // High utilization -> decrease batch size
        balancer.adjust_batch_size(0.96);
        let (batch_size, _) = balancer.get_current_config();
        assert!(batch_size < 2048); // Should be decreased
    }

    #[test]
    fn test_optimal_batch_estimation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let config = WorkloadConfig::default();
        let balancer = WorkloadBalancer::new(device, config);

        let optimal = balancer.estimate_optimal_batch_size();

        // RTX 5090: 128 SMs * 64 warps/SM * 32 threads/warp * 0.8
        let expected = (128 * 64 * 32) as f32 * 0.8;
        assert!((optimal as f32 - expected).abs() < 1000.0);
    }
}

#[cfg(test)]
mod kernel_optimizer_tests {
    use super::*;
    use crate::utilization::kernel_optimizer::*;

    #[tokio::test]
    async fn test_kernel_registration() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let optimizer = KernelOptimizer::new(device);

        let config = KernelConfig {
            block_size: 256,
            grid_size: 512,
            shared_mem_size: 4096,
            registers_per_thread: 32,
        };

        optimizer.register_kernel("test_kernel", config).await?;

        let retrieved = optimizer.get_optimal_config("test_kernel").await?;
        assert_eq!(retrieved.block_size, config.block_size);

        Ok(())
    }

    #[test]
    fn test_occupancy_calculation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let optimizer = KernelOptimizer::new(device);

        let config = KernelConfig {
            block_size: 256,
            grid_size: 1024,
            shared_mem_size: 0,
            registers_per_thread: 32,
        };

        let occupancy = optimizer.calculate_occupancy(config);
        assert!(occupancy > 0.0);
        assert!(occupancy <= 1.0);
    }

    #[test]
    fn test_launch_config_suggestion() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let optimizer = KernelOptimizer::new(device);

        let launch_config = optimizer.suggest_launch_config(1_000_000, 10);

        assert!(launch_config.block_dim.0 > 0);
        assert!(launch_config.grid_dim.0 > 0);
        assert_eq!(launch_config.block_dim.1, 1);
        assert_eq!(launch_config.block_dim.2, 1);
    }

    #[test]
    fn test_grid_size_calculator() {
        // 1D grid
        let grid_1d = GridSizeCalculator::calculate_1d(10000, 256);
        assert_eq!(grid_1d, (10000 + 255) / 256);

        // 2D grid
        let (grid_x, grid_y) = GridSizeCalculator::calculate_2d(1920, 1080, 16, 16);
        assert_eq!(grid_x, (1920 + 15) / 16);
        assert_eq!(grid_y, (1080 + 15) / 16);
    }
}

#[cfg(test)]
mod resource_monitor_tests {
    use super::*;
    use crate::utilization::resource_monitor::*;

    #[tokio::test]
    async fn test_resource_monitoring() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let limits = ResourceLimits::default();
        let monitor = ResourceMonitor::new(device, limits);

        // Start monitoring
        monitor.start_monitoring().await?;

        // Wait for measurements
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Get latest measurements
        let compute = monitor.get_latest_measurement(ResourceType::Compute).await;
        assert!(compute.is_some());

        let memory = monitor.get_latest_measurement(ResourceType::Memory).await;
        assert!(memory.is_some());

        // Stop monitoring
        monitor.stop_monitoring();

        Ok(())
    }

    #[tokio::test]
    async fn test_resource_statistics() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let limits = ResourceLimits::default();
        let monitor = ResourceMonitor::new(device, limits);

        monitor.start_monitoring().await?;
        tokio::time::sleep(Duration::from_secs(1)).await;

        let stats = monitor
            .calculate_statistics(ResourceType::Compute, Duration::from_secs(1))
            .await;

        assert!(stats.sample_count > 0);
        assert!(stats.average >= 0.0);
        assert!(stats.min <= stats.max);

        monitor.stop_monitoring();

        Ok(())
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();

        assert_eq!(limits.max_memory, 32 * 1024 * 1024 * 1024); // 32GB
        assert_eq!(limits.memory_warning_threshold, 0.9);
        assert_eq!(limits.max_temperature, 83.0);
        assert_eq!(limits.max_power, 450.0);
    }
}

#[cfg(test)]
mod utilization_controller_tests {
    use super::*;
    use crate::utilization::utilization_controller::*;

    #[tokio::test]
    async fn test_controller_creation() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = ControllerConfig::default();
        let controller = UtilizationController::new(device, config).await?;

        assert!(!controller.is_running.load(Ordering::Relaxed));

        Ok(())
    }

    #[tokio::test]
    async fn test_controller_start_stop() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = ControllerConfig {
            control_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let controller = UtilizationController::new(device, config).await?;

        // Start controller
        controller.start().await?;
        assert!(controller.is_running.load(Ordering::Relaxed));

        // Let it run
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check state
        let state = controller.state.read().await;
        assert!(state.actions_taken > 0);

        // Stop controller
        controller.stop();
        assert!(!controller.is_running.load(Ordering::Relaxed));

        Ok(())
    }

    #[tokio::test]
    async fn test_optimization_recommendations() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = ControllerConfig::default();
        let controller = UtilizationController::new(device, config).await?;

        // Set low utilization
        {
            let mut state = controller.state.write().await;
            state.current_utilization = 0.6;
        }

        let recommendations = controller.get_recommendations().await;

        assert!(recommendations.workload_adjustment > 1.0);
        assert!(!recommendations.memory_optimizations.is_empty());
        assert!(!recommendations.scheduling_changes.is_empty());

        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_optimization() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);

        // Create controller with aggressive optimization
        let config = ControllerConfig {
            aggressive_mode: true,
            control_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let controller = UtilizationController::new(device.clone(), config).await?;

        // Start optimization
        controller.start().await?;

        // Submit workload
        let balancer = controller.workload_balancer.clone();
        for i in 0..100 {
            balancer.submit_work(1024, 100, (i % 5) as u8).await?;
        }

        // Run for a while
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Generate reports
        let controller_report = controller.generate_report().await;
        println!("Controller Report:\n{}", controller_report);

        let utilization_report = controller.utilization_manager.generate_report().await;
        println!("\nUtilization Report:\n{}", utilization_report);

        let workload_report = balancer.generate_report().await;
        println!("\nWorkload Report:\n{}", workload_report);

        // Stop
        controller.stop();

        Ok(())
    }

    #[tokio::test]
    async fn test_auto_tuning() -> Result<()> {
        let device = Arc::new(CudaDevice::new(0)?);
        let manager = Arc::new(UtilizationManager::new(device)?);
        let auto_tuner = AutoTuningController::new(manager.clone());

        // Start monitoring and auto-tuning
        manager.start_monitoring().await?;
        auto_tuner.start().await?;

        // Run for a while
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Check that optimization happened
        let workload_multiplier = manager.get_workload_multiplier();
        println!("Final workload multiplier: {:.2}x", workload_multiplier);

        // Stop
        auto_tuner.stop();
        manager.stop_monitoring();

        Ok(())
    }
}
