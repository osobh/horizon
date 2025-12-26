//! GPU command processing and device abstraction.
//!
//! This module provides the `GpuDevice` trait for abstracting GPU operations
//! and a `MockDevice` implementation for testing without actual GPU hardware.

use bytes::Bytes;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, String>;

/// Trait for GPU device operations.
///
/// This trait abstracts GPU operations to enable testing without actual hardware.
/// Implementations must be `Send + Sync` for use in async contexts.
#[async_trait::async_trait]
pub trait GpuDevice: Send + Sync {
    /// Get device ID.
    fn device_id(&self) -> u32;

    /// Get total memory in bytes.
    fn total_memory(&self) -> u64;

    /// Get currently used memory in bytes.
    fn used_memory(&self) -> u64;

    /// Get GPU utilization percentage (0.0 to 100.0).
    fn utilization(&self) -> f64;

    /// Get memory pressure (0.0 to 1.0).
    fn memory_pressure(&self) -> f64 {
        if self.total_memory() == 0 {
            0.0
        } else {
            self.used_memory() as f64 / self.total_memory() as f64
        }
    }

    /// Allocate a buffer on the device.
    async fn allocate_buffer(&mut self, buffer_id: &str, size: usize) -> GpuResult<()>;

    /// Deallocate a buffer from the device.
    async fn deallocate_buffer(&mut self, buffer_id: &str) -> GpuResult<()>;

    /// Transfer data to device.
    async fn transfer_to_device(
        &mut self,
        buffer_id: &str,
        data: Bytes,
        offset: usize,
    ) -> GpuResult<()>;

    /// Transfer data from device.
    async fn transfer_from_device(
        &mut self,
        buffer_id: &str,
        size: usize,
        offset: usize,
    ) -> GpuResult<Bytes>;

    /// Launch a kernel on the device.
    ///
    /// Returns the execution duration in microseconds.
    async fn launch_kernel(
        &mut self,
        kernel_id: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        params: Bytes,
    ) -> GpuResult<u64>;

    /// Synchronize device execution.
    async fn synchronize(&mut self, stream_id: Option<u32>) -> GpuResult<()>;

    /// Reset device state (for testing).
    async fn reset(&mut self);

    /// Get number of kernels launched (for testing).
    fn kernel_count(&self) -> usize;
}

/// Mock GPU device for testing.
///
/// This implementation simulates GPU behavior without requiring actual hardware.
/// It tracks memory allocations, simulates kernel execution timing, and maintains
/// realistic utilization metrics.
#[derive(Debug)]
pub struct MockDevice {
    device_id: u32,
    total_memory: u64,
    used_memory: AtomicU64,
    utilization: AtomicU64, // Stored as fixed-point (value * 100)
    kernel_count: AtomicUsize,
    buffers: Arc<DashMap<String, BufferInfo>>,
}

#[derive(Debug, Clone)]
struct BufferInfo {
    size: usize,
    data: Vec<u8>,
}

impl MockDevice {
    /// Create a new mock device with specified memory capacity.
    #[must_use]
    pub fn new(device_id: u32, total_memory: u64) -> Self {
        Self {
            device_id,
            total_memory,
            used_memory: AtomicU64::new(0),
            utilization: AtomicU64::new(0),
            kernel_count: AtomicUsize::new(0),
            buffers: Arc::new(DashMap::new()),
        }
    }

    /// Get buffer data for testing.
    #[must_use]
    pub fn get_buffer_data(&self, buffer_id: &str) -> Option<Vec<u8>> {
        self.buffers.get(buffer_id).map(|entry| entry.data.clone())
    }

    /// Calculate simulated kernel execution time.
    fn simulate_kernel_time(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> u64 {
        // Simulate based on thread count
        let total_threads = u64::from(grid_dim.0)
            * u64::from(grid_dim.1)
            * u64::from(grid_dim.2)
            * u64::from(block_dim.0)
            * u64::from(block_dim.1)
            * u64::from(block_dim.2);

        // Base time + time per thread (microseconds)
        100 + (total_threads / 1000)
    }

    /// Update utilization based on kernel launch.
    fn update_utilization(&self, grid_dim: (u32, u32, u32), block_dim: (u32, u32, u32)) {
        let total_threads = u64::from(grid_dim.0)
            * u64::from(grid_dim.1)
            * u64::from(grid_dim.2)
            * u64::from(block_dim.0)
            * u64::from(block_dim.1)
            * u64::from(block_dim.2);

        // Simulate utilization increase (capped at 100%)
        let new_util = ((total_threads as f64).log2() * 10.0).min(100.0);
        let stored = (new_util * 100.0) as u64;
        self.utilization.store(stored, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl GpuDevice for MockDevice {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn total_memory(&self) -> u64 {
        self.total_memory
    }

    fn used_memory(&self) -> u64 {
        self.used_memory.load(Ordering::Relaxed)
    }

    fn utilization(&self) -> f64 {
        self.utilization.load(Ordering::Relaxed) as f64 / 100.0
    }

    async fn allocate_buffer(&mut self, buffer_id: &str, size: usize) -> GpuResult<()> {
        // Check if buffer already exists
        if self.buffers.contains_key(buffer_id) {
            return Err(format!("Buffer '{}' already exists", buffer_id));
        }

        // Check if we have enough memory
        let current_used = self.used_memory.load(Ordering::Relaxed);
        if current_used + size as u64 > self.total_memory {
            return Err(format!(
                "out of memory: requested {}, available {}",
                size,
                self.total_memory - current_used
            ));
        }

        // Allocate buffer
        let buffer_info = BufferInfo {
            size,
            data: vec![0; size],
        };

        self.buffers.insert(buffer_id.to_string(), buffer_info);
        self.used_memory.fetch_add(size as u64, Ordering::Relaxed);

        // Simulate allocation time
        sleep(Duration::from_micros(10)).await;

        Ok(())
    }

    async fn deallocate_buffer(&mut self, buffer_id: &str) -> GpuResult<()> {
        let (_, buffer_info) = self
            .buffers
            .remove(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        self.used_memory
            .fetch_sub(buffer_info.size as u64, Ordering::Relaxed);

        // Simulate deallocation time
        sleep(Duration::from_micros(5)).await;

        Ok(())
    }

    async fn transfer_to_device(
        &mut self,
        buffer_id: &str,
        data: Bytes,
        offset: usize,
    ) -> GpuResult<()> {
        let mut buffer_entry = self
            .buffers
            .get_mut(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        // Check bounds
        if offset + data.len() > buffer_entry.size {
            return Err(format!(
                "Transfer exceeds buffer size: {} + {} > {}",
                offset,
                data.len(),
                buffer_entry.size
            ));
        }

        // Copy data
        buffer_entry.data[offset..offset + data.len()].copy_from_slice(&data);

        // Simulate transfer time (1 GB/s)
        let transfer_time_us = (data.len() as u64 * 1000) / (1024 * 1024);
        sleep(Duration::from_micros(transfer_time_us.max(1))).await;

        Ok(())
    }

    async fn transfer_from_device(
        &mut self,
        buffer_id: &str,
        size: usize,
        offset: usize,
    ) -> GpuResult<Bytes> {
        let buffer_entry = self
            .buffers
            .get(buffer_id)
            .ok_or_else(|| format!("Buffer '{}' not found", buffer_id))?;

        // Check bounds
        if offset + size > buffer_entry.size {
            return Err(format!(
                "Transfer exceeds buffer size: {} + {} > {}",
                offset, size, buffer_entry.size
            ));
        }

        // Copy data
        let data = Bytes::copy_from_slice(&buffer_entry.data[offset..offset + size]);

        // Simulate transfer time
        let transfer_time_us = (size as u64 * 1000) / (1024 * 1024);
        sleep(Duration::from_micros(transfer_time_us.max(1))).await;

        Ok(data)
    }

    async fn launch_kernel(
        &mut self,
        _kernel_id: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        _params: Bytes,
    ) -> GpuResult<u64> {
        let duration_us = self.simulate_kernel_time(grid_dim, block_dim);

        // Simulate kernel execution
        sleep(Duration::from_micros(duration_us.min(1000))).await;

        // Update metrics
        self.kernel_count.fetch_add(1, Ordering::Relaxed);
        self.update_utilization(grid_dim, block_dim);

        Ok(duration_us)
    }

    async fn synchronize(&mut self, _stream_id: Option<u32>) -> GpuResult<()> {
        // Simulate synchronization delay
        sleep(Duration::from_micros(50)).await;
        Ok(())
    }

    async fn reset(&mut self) {
        self.buffers.clear();
        self.used_memory.store(0, Ordering::Relaxed);
        self.utilization.store(0, Ordering::Relaxed);
        self.kernel_count.store(0, Ordering::Relaxed);
    }

    fn kernel_count(&self) -> usize {
        self.kernel_count.load(Ordering::Relaxed)
    }
}

/// Command processor for handling GPU commands.
///
/// Processes incoming GPU commands and delegates to the underlying device.
#[derive(Debug)]
pub struct CommandProcessor<D: GpuDevice> {
    device: D,
}

impl<D: GpuDevice> CommandProcessor<D> {
    /// Create a new command processor with the given device.
    #[must_use]
    pub fn new(device: D) -> Self {
        Self { device }
    }

    /// Get reference to the underlying device.
    #[must_use]
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Get mutable reference to the underlying device.
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    /// Process a launch kernel command.
    pub async fn process_launch_kernel(
        &mut self,
        kernel_id: String,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        params: Bytes,
    ) -> GpuResult<u64> {
        self.device
            .launch_kernel(&kernel_id, grid_dim, block_dim, params)
            .await
    }

    /// Process a transfer to device command.
    pub async fn process_transfer_to_device(
        &mut self,
        buffer_id: String,
        data: Bytes,
        offset: usize,
    ) -> GpuResult<()> {
        self.device
            .transfer_to_device(&buffer_id, data, offset)
            .await
    }

    /// Process a transfer from device command.
    pub async fn process_transfer_from_device(
        &mut self,
        buffer_id: String,
        size: usize,
        offset: usize,
    ) -> GpuResult<Bytes> {
        self.device
            .transfer_from_device(&buffer_id, size, offset)
            .await
    }

    /// Process a synchronize command.
    pub async fn process_synchronize(&mut self, stream_id: Option<u32>) -> GpuResult<()> {
        self.device.synchronize(stream_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_command_processor_creation() {
        let device = MockDevice::new(0, 1024 * 1024);
        let processor = CommandProcessor::new(device);
        assert_eq!(processor.device().device_id(), 0);
    }

    #[tokio::test]
    async fn test_command_processor_launch_kernel() {
        let device = MockDevice::new(0, 1024 * 1024);
        let mut processor = CommandProcessor::new(device);

        let params = Bytes::from(vec![1, 2, 3]);
        let result = processor
            .process_launch_kernel("test".to_string(), (1, 1, 1), (32, 1, 1), params)
            .await;

        assert!(result.is_ok());
        assert_eq!(processor.device().kernel_count(), 1);
    }

    #[tokio::test]
    async fn test_command_processor_transfer_to_device() {
        let mut device = MockDevice::new(0, 1024 * 1024);
        device.allocate_buffer("buf", 1024).await.unwrap();
        let mut processor = CommandProcessor::new(device);

        let data = Bytes::from(vec![5, 6, 7, 8]);
        let result = processor
            .process_transfer_to_device("buf".to_string(), data, 0)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_command_processor_synchronize() {
        let device = MockDevice::new(0, 1024 * 1024);
        let mut processor = CommandProcessor::new(device);

        let result = processor.process_synchronize(None).await;
        assert!(result.is_ok());
    }
}
