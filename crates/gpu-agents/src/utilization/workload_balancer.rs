//! Workload Balancer for GPU Utilization
//!
//! Dynamically adjusts workload distribution to maintain optimal GPU utilization.

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaStream};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// Workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Base number of agents per kernel
    pub base_agents_per_kernel: u32,
    /// Base number of iterations per agent
    pub base_iterations: u32,
    /// Minimum batch size
    pub min_batch_size: u32,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Work queue capacity
    pub queue_capacity: usize,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            base_agents_per_kernel: 1024,
            base_iterations: 100,
            min_batch_size: 256,
            max_batch_size: 65536,
            queue_capacity: 1000,
        }
    }
}

/// Work item for GPU processing
#[derive(Clone)]
pub struct WorkItem {
    pub id: u64,
    pub agent_count: u32,
    pub iterations: u32,
    pub priority: u8,
    pub created_at: Instant,
}

/// Workload statistics
#[derive(Debug, Default, Clone)]
pub struct WorkloadStats {
    pub total_items_processed: u64,
    pub average_processing_time: Duration,
    pub queue_depth: usize,
    pub throughput: f32, // items per second
    pub efficiency: f32, // 0.0 - 1.0
}

/// Workload balancer
pub struct WorkloadBalancer {
    config: WorkloadConfig,
    device: Arc<CudaDevice>,
    work_queue: Arc<Mutex<VecDeque<WorkItem>>>,
    current_batch_size: Arc<AtomicU64>,
    current_iterations: Arc<AtomicU64>,
    stats: Arc<RwLock<WorkloadStats>>,
    next_item_id: Arc<AtomicU64>,
}

impl WorkloadBalancer {
    /// Create new workload balancer
    pub fn new(device: Arc<CudaDevice>, config: WorkloadConfig) -> Self {
        Self {
            device,
            work_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity))),
            current_batch_size: Arc::new(AtomicU64::new(config.base_agents_per_kernel as u64)),
            current_iterations: Arc::new(AtomicU64::new(config.base_iterations as u64)),
            stats: Arc::new(RwLock::new(WorkloadStats::default())),
            next_item_id: Arc::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Submit work item
    pub async fn submit_work(
        &self,
        agent_count: u32,
        iterations: u32,
        priority: u8,
    ) -> Result<u64> {
        let id = self.next_item_id.fetch_add(1, Ordering::Relaxed);

        let item = WorkItem {
            id,
            agent_count,
            iterations,
            priority,
            created_at: Instant::now(),
        };

        let mut queue = self.work_queue.lock().await;

        // Insert based on priority
        let insert_pos = queue
            .iter()
            .position(|item| item.priority < priority)
            .unwrap_or(queue.len());

        queue.insert(insert_pos, item);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.queue_depth = queue.len();

        Ok(id)
    }

    /// Get next batch of work
    pub async fn get_next_batch(&self) -> Result<Vec<WorkItem>> {
        let batch_size = self.current_batch_size.load(Ordering::Relaxed) as usize;
        let mut queue = self.work_queue.lock().await;
        let mut batch = Vec::with_capacity(batch_size);

        // Collect items up to batch size
        let mut total_agents = 0u32;
        while let Some(item) = queue.front() {
            if total_agents + item.agent_count > batch_size as u32 && !batch.is_empty() {
                break; // Batch is full
            }

            if let Some(item) = queue.pop_front() {
                total_agents += item.agent_count;
                batch.push(item);
            }
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.queue_depth = queue.len();

        Ok(batch)
    }

    /// Adjust batch size based on utilization
    pub fn adjust_batch_size(&self, utilization: f32) {
        let current = self.current_batch_size.load(Ordering::Relaxed);

        let new_size = if utilization < 0.8 {
            // Increase batch size
            (current as f32 * 1.2).min(self.config.max_batch_size as f32) as u64
        } else if utilization > 0.95 {
            // Decrease batch size slightly
            (current as f32 * 0.95).max(self.config.min_batch_size as f32) as u64
        } else {
            current
        };

        self.current_batch_size.store(new_size, Ordering::Relaxed);
    }

    /// Adjust iteration count based on kernel time
    pub fn adjust_iterations(&self, kernel_time_ms: f32, target_time_ms: f32) {
        let current = self.current_iterations.load(Ordering::Relaxed);

        let ratio = target_time_ms / kernel_time_ms.max(0.1);
        let new_iterations = (current as f32 * ratio).clamp(10.0, 10000.0) as u64;

        self.current_iterations
            .store(new_iterations, Ordering::Relaxed);
    }

    /// Get current configuration
    pub fn get_current_config(&self) -> (u32, u32) {
        (
            self.current_batch_size.load(Ordering::Relaxed) as u32,
            self.current_iterations.load(Ordering::Relaxed) as u32,
        )
    }

    /// Update processing statistics
    pub async fn update_stats(
        &self,
        items_processed: u64,
        processing_time: Duration,
    ) -> Result<()> {
        let mut stats = self.stats.write().await;

        stats.total_items_processed += items_processed;

        // Update average processing time (exponential moving average)
        let alpha = 0.1;
        let new_avg = if stats.average_processing_time == Duration::ZERO {
            processing_time
        } else {
            let current_ms = stats.average_processing_time.as_millis() as f32;
            let new_ms = processing_time.as_millis() as f32;
            let avg_ms = current_ms * (1.0 - alpha) + new_ms * alpha;
            Duration::from_millis(avg_ms as u64)
        };
        stats.average_processing_time = new_avg;

        // Calculate throughput
        if processing_time.as_secs_f32() > 0.0 {
            stats.throughput = items_processed as f32 / processing_time.as_secs_f32();
        }

        Ok(())
    }

    /// Get workload statistics
    pub async fn get_stats(&self) -> WorkloadStats {
        let stats_guard = self.stats.read().await;
        stats_guard.clone()
    }

    /// Balance workload across multiple streams
    pub async fn balance_across_streams(&self, num_streams: usize) -> Result<Vec<Vec<WorkItem>>> {
        let batch = self.get_next_batch().await?;
        let mut stream_batches = vec![Vec::new(); num_streams];

        // Distribute work items across streams
        for (i, item) in batch.into_iter().enumerate() {
            let stream_idx = i % num_streams;
            stream_batches[stream_idx].push(item);
        }

        Ok(stream_batches)
    }

    /// Estimate optimal batch size based on GPU properties
    pub fn estimate_optimal_batch_size(&self) -> u32 {
        // RTX 5090 specifications
        let max_threads_per_block = 1024;
        let num_sms = 128;
        let warps_per_sm = 64;

        // Calculate optimal occupancy
        let total_warps = num_sms * warps_per_sm;
        let warp_size = 32;
        let optimal_threads = total_warps * warp_size;

        // Adjust for memory constraints
        let memory_factor = 0.8; // Use 80% to leave room for other data
        let optimal_batch = (optimal_threads as f32 * memory_factor) as u32;

        optimal_batch.clamp(self.config.min_batch_size, self.config.max_batch_size)
    }

    /// Generate workload report
    pub async fn generate_report(&self) -> String {
        let stats = self.stats.read().await;
        let (batch_size, iterations) = self.get_current_config();

        format!(
            "Workload Balancer Report:\n\
             - Current batch size: {}\n\
             - Current iterations: {}\n\
             - Queue depth: {}\n\
             - Total processed: {}\n\
             - Average processing time: {:?}\n\
             - Throughput: {:.2} items/sec\n\
             - Efficiency: {:.1}%",
            batch_size,
            iterations,
            stats.queue_depth,
            stats.total_items_processed,
            stats.average_processing_time,
            stats.throughput,
            stats.efficiency * 100.0,
        )
    }
}

/// Dynamic workload generator for testing
pub struct WorkloadGenerator {
    balancer: Arc<WorkloadBalancer>,
    generation_rate: f32, // items per second
    is_running: Arc<AtomicBool>,
}

impl WorkloadGenerator {
    /// Create new workload generator
    pub fn new(balancer: Arc<WorkloadBalancer>, generation_rate: f32) -> Self {
        Self {
            balancer,
            generation_rate,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start generating workload
    pub async fn start(&self) -> Result<()> {
        use std::sync::atomic::AtomicBool;

        if self.is_running.swap(true, Ordering::Relaxed) {
            return Ok(()); // Already running
        }

        let balancer = self.balancer.clone();
        let is_running = self.is_running.clone();
        let interval = Duration::from_secs_f32(1.0 / self.generation_rate);

        tokio::spawn(async move {
            while is_running.load(Ordering::Relaxed) {
                // Generate random work item
                let agent_count = 256 + (rand::random::<u32>() % 768);
                let iterations = 50 + (rand::random::<u32>() % 150);
                let priority = (rand::random::<u8>() % 5) + 1;

                if let Err(e) = balancer
                    .submit_work(agent_count, iterations, priority)
                    .await
                {
                    eprintln!("Failed to submit work: {}", e);
                }

                tokio::time::sleep(interval).await;
            }
        });

        Ok(())
    }

    /// Stop generating workload
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
}
