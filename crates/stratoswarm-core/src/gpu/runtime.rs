//! GPU runtime service for processing commands from channels.
//!
//! This module provides the main GPU runtime that processes commands
//! through async channels, broadcasts events, and tracks metrics.

use crate::channels::{GpuCommand, SystemEvent};
use crate::gpu::command::{CommandProcessor, GpuDevice};
use crate::gpu::memory::UnifiedMemoryPool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Configuration for GPU runtime.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// GPU device ID to use
    pub device_id: u32,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Interval for broadcasting utilization updates
    pub utilization_broadcast_interval: Duration,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            utilization_broadcast_interval: Duration::from_secs(1),
        }
    }
}

/// GPU runtime metrics.
#[derive(Debug, Clone, Copy)]
pub struct GpuMetrics {
    /// Device ID
    pub device_id: u32,
    /// Number of kernels launched
    pub kernels_launched: u64,
    /// Number of memory transfers
    pub memory_transfers: u64,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Number of synchronizations
    pub synchronizations: u64,
}

impl GpuMetrics {
    #[allow(dead_code)]
    fn new(device_id: u32) -> Self {
        Self {
            device_id,
            kernels_launched: 0,
            memory_transfers: 0,
            bytes_transferred: 0,
            synchronizations: 0,
        }
    }
}

/// Shared metrics for the GPU runtime.
#[derive(Debug, Clone)]
pub struct SharedMetrics {
    device_id: u32,
    kernels_launched: Arc<AtomicU64>,
    memory_transfers: Arc<AtomicU64>,
    bytes_transferred: Arc<AtomicU64>,
    synchronizations: Arc<AtomicU64>,
}

impl SharedMetrics {
    fn new(device_id: u32) -> Self {
        Self {
            device_id,
            kernels_launched: Arc::new(AtomicU64::new(0)),
            memory_transfers: Arc::new(AtomicU64::new(0)),
            bytes_transferred: Arc::new(AtomicU64::new(0)),
            synchronizations: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get current metrics snapshot.
    pub fn metrics(&self) -> GpuMetrics {
        GpuMetrics {
            device_id: self.device_id,
            kernels_launched: self.kernels_launched.load(Ordering::Relaxed),
            memory_transfers: self.memory_transfers.load(Ordering::Relaxed),
            bytes_transferred: self.bytes_transferred.load(Ordering::Relaxed),
            synchronizations: self.synchronizations.load(Ordering::Relaxed),
        }
    }

    fn increment_kernels(&self) {
        self.kernels_launched.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_transfers(&self, bytes: u64) {
        self.memory_transfers.fetch_add(1, Ordering::Relaxed);
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
    }

    fn increment_syncs(&self) {
        self.synchronizations.fetch_add(1, Ordering::Relaxed);
    }
}

/// GPU runtime service.
///
/// Processes GPU commands from a channel and broadcasts events.
#[derive(Debug)]
pub struct GpuRuntime<D: GpuDevice> {
    config: GpuConfig,
    processor: CommandProcessor<D>,
    memory_pool: UnifiedMemoryPool,
    metrics: SharedMetrics,
}

impl<D: GpuDevice> GpuRuntime<D> {
    /// Create a new GPU runtime.
    #[must_use]
    pub fn new(config: GpuConfig, device: D) -> Self {
        let device_id = device.device_id();
        Self {
            memory_pool: UnifiedMemoryPool::new(config.memory_pool_size),
            processor: CommandProcessor::new(device),
            metrics: SharedMetrics::new(device_id),
            config,
        }
    }

    /// Get current metrics.
    #[must_use]
    pub fn metrics(&self) -> GpuMetrics {
        self.metrics.metrics()
    }

    /// Clone metrics handle for external access.
    #[must_use]
    pub fn clone_metrics(&self) -> SharedMetrics {
        self.metrics.clone()
    }

    /// Get current timestamp in milliseconds.
    fn timestamp_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Process a single GPU command.
    #[allow(clippy::too_many_lines)]
    async fn process_command(
        &mut self,
        command: GpuCommand,
        events_tx: &broadcast::Sender<SystemEvent>,
    ) {
        match command {
            GpuCommand::LaunchKernel {
                kernel_id,
                grid_dim,
                block_dim,
                params,
            } => {
                debug!(
                    "Launching kernel '{}' with grid {:?}, block {:?}",
                    kernel_id, grid_dim, block_dim
                );

                let start = std::time::Instant::now();
                let result = self
                    .processor
                    .process_launch_kernel(kernel_id.clone(), grid_dim, block_dim, params)
                    .await;

                #[allow(clippy::cast_possible_truncation)]
                let duration_us = start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        self.metrics.increment_kernels();

                        // Broadcast kernel completed event
                        let event = SystemEvent::KernelCompleted {
                            kernel_id,
                            duration_us,
                            success: true,
                            timestamp: Self::timestamp_ms(),
                        };

                        if let Err(e) = events_tx.send(event) {
                            warn!("Failed to broadcast kernel completed event: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Kernel launch failed: {}", e);

                        let event = SystemEvent::KernelCompleted {
                            kernel_id,
                            duration_us,
                            success: false,
                            timestamp: Self::timestamp_ms(),
                        };

                        let _ = events_tx.send(event);
                    }
                }
            }

            GpuCommand::TransferToDevice {
                buffer_id,
                data,
                offset,
            } => {
                debug!(
                    "Transferring {} bytes to buffer '{}' at offset {}",
                    data.len(),
                    buffer_id,
                    offset
                );

                let bytes = data.len() as u64;

                // Ensure buffer is allocated in memory pool
                if !self.memory_pool.is_allocated(&buffer_id) {
                    // Auto-allocate buffer
                    let size = offset + data.len();
                    if let Err(e) = self.memory_pool.allocate(&buffer_id, size) {
                        error!("Failed to allocate buffer '{}': {}", buffer_id, e);
                        return;
                    }
                }

                // Write to memory pool
                if let Err(e) = self.memory_pool.write(&buffer_id, data.clone(), offset) {
                    error!("Failed to write to memory pool: {}", e);
                    return;
                }

                // Transfer to device
                let result = self
                    .processor
                    .process_transfer_to_device(buffer_id, data, offset)
                    .await;

                match result {
                    Ok(()) => {
                        self.metrics.increment_transfers(bytes);
                    }
                    Err(e) => {
                        error!("Transfer to device failed: {}", e);
                    }
                }
            }

            GpuCommand::TransferFromDevice {
                buffer_id,
                size,
                offset,
            } => {
                debug!(
                    "Transferring {} bytes from buffer '{}' at offset {}",
                    size, buffer_id, offset
                );

                let result = self
                    .processor
                    .process_transfer_from_device(buffer_id, size, offset)
                    .await;

                match result {
                    Ok(_data) => {
                        self.metrics.increment_transfers(size as u64);
                    }
                    Err(e) => {
                        error!("Transfer from device failed: {}", e);
                    }
                }
            }

            GpuCommand::Synchronize { stream_id } => {
                debug!("Synchronizing stream {:?}", stream_id);

                let result = self.processor.process_synchronize(stream_id).await;

                match result {
                    Ok(()) => {
                        self.metrics.increment_syncs();
                    }
                    Err(e) => {
                        error!("Synchronization failed: {}", e);
                    }
                }
            }
        }
    }

    /// Broadcast GPU utilization event.
    fn broadcast_utilization(&self, events_tx: &broadcast::Sender<SystemEvent>) {
        let utilization = self.processor.device().utilization();

        let event = SystemEvent::GpuUtilization {
            device_id: self.config.device_id,
            utilization,
            timestamp: Self::timestamp_ms(),
        };

        if let Err(e) = events_tx.send(event) {
            debug!("Failed to broadcast utilization (no subscribers): {}", e);
        }
    }

    /// Check and broadcast memory pressure if needed.
    fn check_memory_pressure(&self, events_tx: &broadcast::Sender<SystemEvent>) {
        let pressure = self.memory_pool.pressure();

        // Broadcast if pressure is above 80%
        if pressure > 0.8 {
            let stats = self.memory_pool.stats();

            let event = SystemEvent::MemoryPressure {
                usage_percent: pressure * 100.0,
                available_bytes: stats.available as u64,
                timestamp: Self::timestamp_ms(),
            };

            let _ = events_tx.send(event);
        }
    }

    /// Run the GPU runtime service.
    ///
    /// This is the main event loop that processes commands and broadcasts events.
    /// It will run until the command channel is closed.
    pub async fn run(
        mut self,
        mut command_rx: broadcast::Receiver<GpuCommand>,
        events_tx: broadcast::Sender<SystemEvent>,
    ) {
        info!(
            "GPU runtime started for device {}",
            self.config.device_id
        );

        let mut utilization_timer = interval(self.config.utilization_broadcast_interval);

        loop {
            tokio::select! {
                // Process incoming commands
                result = command_rx.recv() => {
                    match result {
                        Ok(command) => {
                            self.process_command(command, &events_tx).await;
                            self.check_memory_pressure(&events_tx);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!("Command channel closed, shutting down GPU runtime");
                            break;
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            warn!("GPU runtime lagged, skipped {} messages", skipped);
                        }
                    }
                }

                // Broadcast utilization periodically
                _ = utilization_timer.tick() => {
                    self.broadcast_utilization(&events_tx);
                }
            }
        }

        info!("GPU runtime shutdown complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::MockDevice;

    #[test]
    fn test_gpu_config_creation() {
        let config = GpuConfig {
            device_id: 1,
            memory_pool_size: 2048,
            utilization_broadcast_interval: Duration::from_secs(5),
        };

        assert_eq!(config.device_id, 1);
        assert_eq!(config.memory_pool_size, 2048);
        assert_eq!(config.utilization_broadcast_interval, Duration::from_secs(5));
    }

    #[test]
    fn test_gpu_metrics_creation() {
        let metrics = GpuMetrics::new(0);
        assert_eq!(metrics.device_id, 0);
        assert_eq!(metrics.kernels_launched, 0);
        assert_eq!(metrics.memory_transfers, 0);
        assert_eq!(metrics.bytes_transferred, 0);
    }

    #[test]
    fn test_shared_metrics() {
        let metrics = SharedMetrics::new(0);

        metrics.increment_kernels();
        metrics.increment_kernels();
        assert_eq!(metrics.metrics().kernels_launched, 2);

        metrics.increment_transfers(1024);
        assert_eq!(metrics.metrics().memory_transfers, 1);
        assert_eq!(metrics.metrics().bytes_transferred, 1024);

        metrics.increment_syncs();
        assert_eq!(metrics.metrics().synchronizations, 1);
    }

    #[test]
    fn test_runtime_creation() {
        let config = GpuConfig::default();
        let device = MockDevice::new(0, 1024 * 1024);
        let runtime = GpuRuntime::new(config, device);

        let metrics = runtime.metrics();
        assert_eq!(metrics.device_id, 0);
        assert_eq!(metrics.kernels_launched, 0);
    }
}
