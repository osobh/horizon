//! GPU Workload Generator using Real CUDA Kernels
//!
//! Generates realistic GPU workloads using actual pattern matching kernels

use crate::synthesis::{AstNode, NodeType, Pattern};
use crate::utilization::kernel_optimizer::KernelConfig;
use crate::utilization::real_kernel_scheduler::{
    KernelPriority, RealKernelScheduler, ScheduledKernel, SchedulerConfig,
};
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// GPU workload generator for testing optimization
pub struct GpuWorkloadGenerator {
    device: Arc<CudaDevice>,
    scheduler: Arc<RwLock<RealKernelScheduler>>,
    /// Workload intensity multiplier
    pub intensity: f32,
    /// Pattern buffer for reuse
    pattern_buffer: Option<CudaSlice<u8>>,
    /// AST buffer for reuse
    ast_buffer: Option<CudaSlice<u8>>,
    /// Match results buffer
    match_buffer: Option<CudaSlice<u32>>,
}

impl GpuWorkloadGenerator {
    /// Create new workload generator
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let scheduler_config = SchedulerConfig {
            num_streams: 4,
            enable_fusion: true,
            enable_load_balancing: true,
            ..Default::default()
        };

        let scheduler = Arc::new(RwLock::new(RealKernelScheduler::new(
            device.clone(),
            scheduler_config,
        )?));

        Ok(Self {
            device,
            scheduler,
            intensity: 1.0,
            pattern_buffer: None,
            ast_buffer: None,
            match_buffer: None,
        })
    }

    /// Set workload intensity (1.0 = normal, 2.0 = double, etc.)
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }

    /// Generate continuous GPU workload
    pub async fn generate_continuous_workload(
        &mut self,
        duration: Duration,
    ) -> Result<WorkloadStats> {
        let start_time = Instant::now();
        let mut stats = WorkloadStats::default();

        // Pre-allocate buffers for efficiency
        self.allocate_buffers().await?;

        while start_time.elapsed() < duration {
            // Generate variable-sized workloads
            let workload_size = self.calculate_workload_size();

            // Submit pattern matching kernels
            self.submit_pattern_matching_workload(workload_size, &mut stats)
                .await?;

            // Small delay to allow scheduler to work
            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        // Wait for all kernels to complete
        self.scheduler.read().await.synchronize_all().await?;

        stats.total_duration = start_time.elapsed();
        Ok(stats)
    }

    /// Generate burst workload for stress testing
    pub async fn generate_burst_workload(&mut self, burst_size: usize) -> Result<WorkloadStats> {
        let start_time = Instant::now();
        let mut stats = WorkloadStats::default();

        // Pre-allocate buffers
        self.allocate_buffers().await?;

        // Submit all kernels at once
        for i in 0..burst_size {
            let priority = match i % 4 {
                0 => KernelPriority::Critical,
                1 => KernelPriority::High,
                2 => KernelPriority::Normal,
                _ => KernelPriority::Low,
            };

            self.submit_single_kernel(i as u64, priority, &mut stats)
                .await?;
        }

        // Wait for completion
        self.scheduler.read().await.synchronize_all().await?;

        stats.total_duration = start_time.elapsed();
        Ok(stats)
    }

    /// Allocate GPU buffers for workload
    async fn allocate_buffers(&mut self) -> Result<()> {
        const MAX_PATTERNS: usize = 64;
        const MAX_NODES: usize = 10000;
        const NODE_SIZE: usize = 64;

        // Allocate pattern buffer if needed
        if self.pattern_buffer.is_none() {
            let pattern_buffer = unsafe { self.device.alloc::<u8>(MAX_PATTERNS * NODE_SIZE) }
                .context("Failed to allocate pattern buffer")?;
            self.pattern_buffer = Some(pattern_buffer);
        }

        // Allocate AST buffer if needed
        if self.ast_buffer.is_none() {
            let ast_buffer = unsafe { self.device.alloc::<u8>(MAX_NODES * NODE_SIZE) }
                .context("Failed to allocate AST buffer")?;
            self.ast_buffer = Some(ast_buffer);
        }

        // Allocate match buffer if needed
        if self.match_buffer.is_none() {
            let match_buffer = self
                .device
                .alloc_zeros::<u32>(MAX_NODES * 2)
                .context("Failed to allocate match buffer")?;
            self.match_buffer = Some(match_buffer);
        }

        Ok(())
    }

    /// Calculate workload size based on intensity
    fn calculate_workload_size(&self) -> WorkloadSize {
        let base_patterns = 32;
        let base_nodes = 5000;

        WorkloadSize {
            num_patterns: (base_patterns as f32 * self.intensity) as usize,
            num_nodes: (base_nodes as f32 * self.intensity) as usize,
        }
    }

    /// Submit pattern matching workload
    async fn submit_pattern_matching_workload(
        &mut self,
        size: WorkloadSize,
        stats: &mut WorkloadStats,
    ) -> Result<()> {
        // Generate test data
        let patterns = self.generate_test_patterns(size.num_patterns);
        let ast_nodes = self.generate_test_ast_nodes(size.num_nodes);

        // Encode data
        let pattern_data = self.encode_patterns(&patterns)?;
        let ast_data = self.encode_ast_nodes(&ast_nodes)?;

        // Copy to GPU buffers
        if let (Some(pattern_buf), Some(ast_buf)) = (&mut self.pattern_buffer, &mut self.ast_buffer)
        {
            self.device
                .htod_copy_into(pattern_data.clone(), pattern_buf)?;
            self.device.htod_copy_into(ast_data.clone(), ast_buf)?;
        }

        // Create kernel with launch function
        let kernel = ScheduledKernel {
            id: stats.kernels_submitted,
            name: format!("pattern_match_{}", stats.kernels_submitted),
            priority: KernelPriority::Normal,
            config: KernelConfig {
                block_size: 256,
                grid_size: ((size.num_nodes + 1023) / 1024) as u32,
                shared_mem_size: 2048, // 32 patterns * 64 bytes
                registers_per_thread: 32,
            },
            dependencies: vec![],
            estimated_time: Duration::from_micros(500),
            submitted_at: Instant::now(),
            data_size: pattern_data.len() + ast_data.len(),
            ptx_code: None, // Using pre-compiled kernel
            function_name: Some("launch_match_patterns_fast".to_string()),
        };

        // Submit to scheduler
        self.scheduler.write().await.submit_kernel(kernel).await?;

        stats.kernels_submitted += 1;
        stats.total_data_processed += pattern_data.len() + ast_data.len();

        Ok(())
    }

    /// Submit a single kernel for testing
    async fn submit_single_kernel(
        &mut self,
        id: u64,
        priority: KernelPriority,
        stats: &mut WorkloadStats,
    ) -> Result<()> {
        let size = self.calculate_workload_size();

        let kernel = ScheduledKernel {
            id,
            name: format!("burst_kernel_{}", id),
            priority,
            config: KernelConfig {
                block_size: 256,
                grid_size: ((size.num_nodes + 1023) / 1024) as u32,
                shared_mem_size: 2048,
                registers_per_thread: 32,
            },
            dependencies: if id > 0 && id % 5 == 0 {
                vec![id - 1] // Create some dependencies
            } else {
                vec![]
            },
            estimated_time: Duration::from_micros(300 + (id % 4) * 100),
            submitted_at: Instant::now(),
            data_size: 1024 * 1024, // 1MB
            ptx_code: None,
            function_name: Some("launch_match_patterns_fast".to_string()),
        };

        self.scheduler.write().await.submit_kernel(kernel).await?;
        stats.kernels_submitted += 1;

        Ok(())
    }

    /// Generate test patterns
    fn generate_test_patterns(&self, count: usize) -> Vec<Pattern> {
        (0..count)
            .map(|i| Pattern {
                node_type: match i % 4 {
                    0 => NodeType::Function,
                    1 => NodeType::Variable,
                    2 => NodeType::BinaryOp,
                    _ => NodeType::Literal,
                },
                children: vec![],
                value: Some(format!("pattern_{}", i)),
            })
            .collect()
    }

    /// Generate test AST nodes
    fn generate_test_ast_nodes(&self, count: usize) -> Vec<AstNode> {
        (0..count)
            .map(|i| AstNode {
                node_type: match i % 5 {
                    0 => NodeType::Function,
                    1 => NodeType::Variable,
                    2 => NodeType::BinaryOp,
                    3 => NodeType::Literal,
                    _ => NodeType::Call,
                },
                children: vec![],
                value: Some(format!("node_{}", i)),
            })
            .collect()
    }

    /// Encode patterns for GPU
    fn encode_patterns(&self, patterns: &[Pattern]) -> Result<Vec<u8>> {
        const NODE_SIZE: usize = 64;
        let mut buffer = Vec::with_capacity(patterns.len() * NODE_SIZE);

        for pattern in patterns {
            let start = buffer.len();
            buffer.resize(start + NODE_SIZE, 0);

            // Write pattern data
            self.write_u32(&mut buffer, start, pattern.node_type as u32);
            self.write_u32(&mut buffer, start + 4, self.hash_string(&pattern.value));
            self.write_u32(&mut buffer, start + 8, 0); // No children for simplicity
        }

        Ok(buffer)
    }

    /// Encode AST nodes for GPU
    fn encode_ast_nodes(&self, nodes: &[AstNode]) -> Result<Vec<u8>> {
        const NODE_SIZE: usize = 64;
        let mut buffer = Vec::with_capacity(nodes.len() * NODE_SIZE);

        for node in nodes {
            let start = buffer.len();
            buffer.resize(start + NODE_SIZE, 0);

            // Write node data
            self.write_u32(&mut buffer, start, node.node_type as u32);
            self.write_u32(&mut buffer, start + 4, self.hash_string(&node.value));
            self.write_u32(&mut buffer, start + 8, 0); // No children
        }

        Ok(buffer)
    }

    /// Write u32 to buffer
    fn write_u32(&self, buffer: &mut [u8], offset: usize, value: u32) {
        buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Hash string value
    fn hash_string(&self, value: &Option<String>) -> u32 {
        value
            .as_ref()
            .map(|s| {
                let mut hash = 0u32;
                for byte in s.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
                }
                hash
            })
            .unwrap_or(0)
    }

    /// Get the scheduler for external access
    pub fn scheduler(&self) -> Arc<RwLock<RealKernelScheduler>> {
        self.scheduler.clone()
    }
}

/// Workload size parameters
#[derive(Debug, Clone)]
struct WorkloadSize {
    num_patterns: usize,
    num_nodes: usize,
}

/// Workload generation statistics
#[derive(Debug, Default)]
pub struct WorkloadStats {
    pub kernels_submitted: u64,
    pub total_data_processed: usize,
    pub total_duration: Duration,
}

impl WorkloadStats {
    /// Calculate throughput in kernels/second
    pub fn kernels_per_second(&self) -> f64 {
        self.kernels_submitted as f64 / self.total_duration.as_secs_f64()
    }

    /// Calculate data throughput in MB/s
    pub fn megabytes_per_second(&self) -> f64 {
        (self.total_data_processed as f64 / (1024.0 * 1024.0)) / self.total_duration.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workload_generator() -> Result<(), Box<dyn std::error::Error>>  {
        let device = CudaDevice::new(0)?;
        let mut generator = GpuWorkloadGenerator::new(Arc::new(device)).unwrap();

        // Test burst workload
        let stats = generator.generate_burst_workload(10).await?;
        assert_eq!(stats.kernels_submitted, 10);
    }
}
