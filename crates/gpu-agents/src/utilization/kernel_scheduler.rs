//! Advanced kernel scheduler for optimal GPU utilization
//!
//! Implements intelligent kernel scheduling to maximize GPU throughput

use crate::utilization::kernel_optimizer::KernelConfig;
use anyhow::Result;
use cudarc::driver::CudaDevice;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use stratoswarm_core::priority_queue::{PrioritySchedulerQueue, SchedulerPriority};
use tokio::sync::Mutex;

/// Advanced kernel scheduler for maximizing GPU utilization
pub struct AdvancedKernelScheduler {
    device: Arc<CudaDevice>,
    /// Stream IDs for concurrent execution (in production would use real streams)
    streams: Vec<usize>,
    /// Kernel queue organized by priority (branch-prediction-friendly)
    kernel_queue: Arc<Mutex<PrioritySchedulerQueue<ScheduledKernel>>>,
    /// Active kernels per stream
    active_kernels: Arc<DashMap<usize, ActiveKernel>>,
    /// Scheduling statistics
    stats: Arc<SchedulingStats>,
    /// Scheduler configuration
    pub config: SchedulerConfig,
}

/// Configuration for kernel scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of CUDA streams to use
    pub num_streams: usize,
    /// Enable kernel fusion optimization
    pub enable_fusion: bool,
    /// Enable dynamic load balancing
    pub enable_load_balancing: bool,
    /// Maximum kernels to batch together
    pub max_batch_size: usize,
    /// Target stream utilization (0.0 - 1.0)
    pub target_stream_utilization: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            enable_fusion: true,
            enable_load_balancing: true,
            max_batch_size: 8,
            target_stream_utilization: 0.95,
        }
    }
}

/// Scheduled kernel with priority and metadata
#[derive(Debug, Clone)]
pub struct ScheduledKernel {
    pub id: u64,
    pub name: String,
    pub priority: KernelPriority,
    pub config: KernelConfig,
    pub dependencies: Vec<u64>,
    pub estimated_time: Duration,
    pub submitted_at: Instant,
    pub data_size: usize,
}

impl PartialEq for ScheduledKernel {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledKernel {}

// Note: Ord/PartialOrd removed - PrioritySchedulerQueue handles ordering
// via the priority parameter at enqueue time (branch-prediction-friendly)

/// Kernel priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl From<KernelPriority> for SchedulerPriority {
    #[inline]
    fn from(priority: KernelPriority) -> Self {
        match priority {
            KernelPriority::Low => SchedulerPriority::Low,
            KernelPriority::Normal => SchedulerPriority::Normal,
            KernelPriority::High => SchedulerPriority::High,
            KernelPriority::Critical => SchedulerPriority::Critical,
        }
    }
}

/// Active kernel information
#[derive(Debug, Clone)]
struct ActiveKernel {
    kernel: ScheduledKernel,
    stream_id: usize,
    start_time: Instant,
    estimated_completion: Instant,
}

/// Scheduling statistics
#[derive(Default)]
struct SchedulingStats {
    kernels_scheduled: std::sync::atomic::AtomicU64,
    kernels_completed: std::sync::atomic::AtomicU64,
    total_wait_time: std::sync::atomic::AtomicU64,
    stream_switches: std::sync::atomic::AtomicU64,
}

impl AdvancedKernelScheduler {
    /// Create new advanced kernel scheduler
    pub fn new(device: Arc<CudaDevice>, config: SchedulerConfig) -> Result<Self> {
        // Create stream IDs (in production would create real CUDA streams)
        let mut streams = Vec::with_capacity(config.num_streams);
        for i in 0..config.num_streams {
            streams.push(i);
        }

        Ok(Self {
            device,
            streams,
            kernel_queue: Arc::new(Mutex::new(PrioritySchedulerQueue::new())),
            active_kernels: Arc::new(DashMap::new()),
            stats: Arc::new(SchedulingStats::default()),
            config,
        })
    }

    /// Submit kernel for scheduling
    pub async fn submit_kernel(&self, kernel: ScheduledKernel) -> Result<u64> {
        let kernel_id = kernel.id;
        let priority: SchedulerPriority = kernel.priority.into();

        // Check dependencies
        if !kernel.dependencies.is_empty() {
            self.wait_for_dependencies(&kernel.dependencies).await?;
        }

        // Add to queue (O(1) enqueue with branch-prediction-friendly structure)
        let mut queue = self.kernel_queue.lock().await;
        queue.enqueue(kernel, priority);

        // Trigger scheduling
        drop(queue); // Release lock before scheduling
        self.schedule_kernels().await?;

        Ok(kernel_id)
    }

    /// Schedule kernels from queue to streams
    async fn schedule_kernels(&self) -> Result<()> {
        let mut queue = self.kernel_queue.lock().await;

        // Find available streams
        let mut available_streams = Vec::new();
        for stream_id in 0..self.config.num_streams {
            if !self.active_kernels.contains_key(&stream_id) {
                available_streams.push(stream_id);
            }
        }

        // Schedule kernels to available streams (O(1) dequeue - branch-prediction-friendly)
        while !queue.is_empty() && !available_streams.is_empty() {
            if let Some(kernel) = queue.dequeue() {
                let stream_id = self
                    .select_optimal_stream(&available_streams, &kernel)
                    .await?;

                // Launch kernel on selected stream
                self.launch_kernel_on_stream(kernel, stream_id).await?;

                // Remove stream from available list
                available_streams.retain(|&id| id != stream_id);
            }
        }

        // Try kernel fusion if enabled
        if self.config.enable_fusion && queue.len() >= 2 {
            self.try_kernel_fusion(&mut queue).await?;
        }

        Ok(())
    }

    /// Select optimal stream for kernel
    async fn select_optimal_stream(
        &self,
        available_streams: &[usize],
        _kernel: &ScheduledKernel,
    ) -> Result<usize> {
        if !self.config.enable_load_balancing {
            // Simple round-robin
            return Ok(available_streams[0]);
        }

        // Load balancing based on estimated completion times
        let mut best_stream = available_streams[0];
        let mut earliest_available = Instant::now() + Duration::from_secs(3600);

        for &stream_id in available_streams {
            // Check when this stream will be available
            let availability = if let Some(active) = self.active_kernels.get(&stream_id) {
                active.estimated_completion
            } else {
                Instant::now()
            };

            if availability < earliest_available {
                earliest_available = availability;
                best_stream = stream_id;
            }
        }

        Ok(best_stream)
    }

    /// Launch kernel on specific stream
    async fn launch_kernel_on_stream(
        &self,
        kernel: ScheduledKernel,
        stream_id: usize,
    ) -> Result<()> {
        // In production would use actual CUDA stream
        let start_time = Instant::now();
        let estimated_completion = start_time + kernel.estimated_time;

        // Record active kernel
        self.active_kernels.insert(
            stream_id,
            ActiveKernel {
                kernel: kernel.clone(),
                stream_id,
                start_time,
                estimated_completion,
            },
        );

        // Update statistics
        let wait_time = start_time.duration_since(kernel.submitted_at);
        self.stats
            .kernels_scheduled
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.total_wait_time.fetch_add(
            wait_time.as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Launch kernel asynchronously
        let active_kernels = self.active_kernels.clone();
        let stats = self.stats.clone();
        let _kernel_id = kernel.id;

        tokio::spawn(async move {
            // Simulate kernel execution
            tokio::time::sleep(kernel.estimated_time).await;

            // Remove from active kernels
            active_kernels.remove(&stream_id);

            // Update completion stats
            stats
                .kernels_completed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });

        Ok(())
    }

    /// Try to fuse compatible kernels
    async fn try_kernel_fusion(
        &self,
        queue: &mut PrioritySchedulerQueue<ScheduledKernel>,
    ) -> Result<()> {
        if queue.len() < 2 {
            return Ok(());
        }

        // Look for fusable kernel pairs - drain all items
        let kernels: Vec<_> = queue.drain().collect();
        let mut fused = Vec::new();
        let mut remaining = Vec::new();

        let mut i = 0;
        while i < kernels.len() {
            if i + 1 < kernels.len() && self.can_fuse(&kernels[i], &kernels[i + 1]) {
                // Create fused kernel
                let fused_kernel = self.create_fused_kernel(&kernels[i], &kernels[i + 1])?;
                fused.push(fused_kernel);
                i += 2;
            } else {
                remaining.push(kernels[i].clone());
                i += 1;
            }
        }

        // Re-add kernels to queue
        for kernel in fused.into_iter().chain(remaining.into_iter()) {
            let priority: SchedulerPriority = kernel.priority.into();
            queue.enqueue(kernel, priority);
        }

        Ok(())
    }

    /// Check if two kernels can be fused
    fn can_fuse(&self, kernel1: &ScheduledKernel, kernel2: &ScheduledKernel) -> bool {
        // Check compatibility criteria
        kernel1.priority == kernel2.priority
            && kernel1.config.block_size == kernel2.config.block_size
            && kernel1.data_size + kernel2.data_size < 1024 * 1024 * 512 // 512MB limit
    }

    /// Create fused kernel from two kernels
    fn create_fused_kernel(
        &self,
        kernel1: &ScheduledKernel,
        kernel2: &ScheduledKernel,
    ) -> Result<ScheduledKernel> {
        Ok(ScheduledKernel {
            id: kernel1.id * 1000 + kernel2.id, // Combined ID
            name: format!("{}+{}", kernel1.name, kernel2.name),
            priority: kernel1.priority,
            config: kernel1.config,
            dependencies: kernel1
                .dependencies
                .iter()
                .chain(kernel2.dependencies.iter())
                .cloned()
                .collect(),
            estimated_time: kernel1.estimated_time.max(kernel2.estimated_time),
            submitted_at: kernel1.submitted_at.min(kernel2.submitted_at),
            data_size: kernel1.data_size + kernel2.data_size,
        })
    }

    /// Wait for kernel dependencies
    async fn wait_for_dependencies(&self, dependencies: &[u64]) -> Result<()> {
        // In a real implementation, would track kernel completion
        // For now, simulate with a short wait
        if !dependencies.is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok(())
    }

    /// Get scheduling statistics
    pub fn get_stats(&self) -> SchedulingStatistics {
        let scheduled = self
            .stats
            .kernels_scheduled
            .load(std::sync::atomic::Ordering::Relaxed);
        let completed = self
            .stats
            .kernels_completed
            .load(std::sync::atomic::Ordering::Relaxed);
        let total_wait_ms = self
            .stats
            .total_wait_time
            .load(std::sync::atomic::Ordering::Relaxed);
        let switches = self
            .stats
            .stream_switches
            .load(std::sync::atomic::Ordering::Relaxed);

        SchedulingStatistics {
            kernels_scheduled: scheduled,
            kernels_completed: completed,
            kernels_in_flight: scheduled - completed,
            average_wait_time: if scheduled > 0 {
                Duration::from_millis(total_wait_ms / scheduled)
            } else {
                Duration::ZERO
            },
            stream_switches: switches,
            stream_utilization: self.calculate_stream_utilization(),
        }
    }

    /// Calculate current stream utilization
    fn calculate_stream_utilization(&self) -> f32 {
        // Estimate based on active kernels (DashMap provides lock-free len())
        let active_count = self.active_kernels.len();
        active_count as f32 / self.config.num_streams as f32
    }

    /// Optimize scheduling parameters based on current performance
    pub async fn auto_tune(&mut self) -> Result<()> {
        let stats = self.get_stats();

        // Adjust number of streams based on utilization
        if stats.stream_utilization > 0.95 && self.config.num_streams < 8 {
            self.config.num_streams += 1;
            self.add_stream().await?;
        } else if stats.stream_utilization < 0.5 && self.config.num_streams > 2 {
            self.config.num_streams -= 1;
            self.remove_stream().await?;
        }

        // Adjust batch size based on queue depth
        let queue_depth = self.kernel_queue.lock().await.len();
        if queue_depth > 100 && self.config.max_batch_size < 16 {
            self.config.max_batch_size += 2;
        } else if queue_depth < 10 && self.config.max_batch_size > 4 {
            self.config.max_batch_size -= 1;
        }

        Ok(())
    }

    /// Add a new stream
    async fn add_stream(&mut self) -> Result<()> {
        let new_stream_id = self.streams.len();
        self.streams.push(new_stream_id);
        Ok(())
    }

    /// Remove a CUDA stream
    async fn remove_stream(&mut self) -> Result<()> {
        if self.streams.len() > 1 {
            // Wait for the last stream to be idle (DashMap contains_key is lock-free)
            let stream_id = self.streams.len() - 1;
            while self.active_kernels.contains_key(&stream_id) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.streams.pop();
        }
        Ok(())
    }
}

/// Scheduling statistics
#[derive(Debug, Clone)]
pub struct SchedulingStatistics {
    pub kernels_scheduled: u64,
    pub kernels_completed: u64,
    pub kernels_in_flight: u64,
    pub average_wait_time: Duration,
    pub stream_switches: u64,
    pub stream_utilization: f32,
}

/// Kernel batch for optimized execution
pub struct KernelBatch {
    pub kernels: Vec<ScheduledKernel>,
    pub total_data_size: usize,
    pub estimated_time: Duration,
}

impl KernelBatch {
    /// Create new kernel batch
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            total_data_size: 0,
            estimated_time: Duration::ZERO,
        }
    }

    /// Add kernel to batch
    pub fn add_kernel(&mut self, kernel: ScheduledKernel) {
        self.total_data_size += kernel.data_size;
        self.estimated_time = self.estimated_time.max(kernel.estimated_time);
        self.kernels.push(kernel);
    }

    /// Check if batch can accept more kernels
    pub fn can_add(&self, kernel: &ScheduledKernel, max_size: usize) -> bool {
        self.kernels.len() < max_size
            && self.total_data_size + kernel.data_size < 1024 * 1024 * 1024 // 1GB limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_creation() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let config = SchedulerConfig::default();
        let scheduler = AdvancedKernelScheduler::new(device, config)?;

        assert_eq!(scheduler.streams.len(), 4);
    }

    #[tokio::test]
    async fn test_kernel_submission() -> Result<(), Box<dyn std::error::Error>> {
        let device = Arc::new(CudaDevice::new(0)?);
        let scheduler = AdvancedKernelScheduler::new(device, SchedulerConfig::default())?;

        let kernel = ScheduledKernel {
            id: 1,
            name: "test_kernel".to_string(),
            priority: KernelPriority::Normal,
            config: KernelConfig::default(),
            dependencies: vec![],
            estimated_time: Duration::from_millis(10),
            submitted_at: Instant::now(),
            data_size: 1024 * 1024,
        };

        let kernel_id = scheduler.submit_kernel(kernel).await?;
        assert_eq!(kernel_id, 1);

        // Wait for kernel to complete
        tokio::time::sleep(Duration::from_millis(50)).await;

        let stats = scheduler.get_stats();
        assert_eq!(stats.kernels_scheduled, 1);
        assert_eq!(stats.kernels_completed, 1);
    }

    #[test]
    fn test_kernel_priority_ordering() {
        // Test that KernelPriority ordering is correct
        assert!(KernelPriority::High > KernelPriority::Low);
        assert!(KernelPriority::Critical > KernelPriority::High);
        assert!(KernelPriority::Normal > KernelPriority::Low);

        // Test conversion to SchedulerPriority
        let high_scheduler: SchedulerPriority = KernelPriority::High.into();
        let low_scheduler: SchedulerPriority = KernelPriority::Low.into();
        assert!(high_scheduler > low_scheduler);
    }
}
