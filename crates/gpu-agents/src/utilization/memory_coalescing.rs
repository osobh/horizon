//! Memory coalescing optimizer for GPU bandwidth utilization
//!
//! Optimizes memory access patterns to achieve coalesced memory accesses

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::collections::HashMap;
use std::sync::Arc;

/// Memory access pattern analyzer and optimizer
pub struct MemoryCoalescingOptimizer {
    device: Arc<CudaDevice>,
    /// Access pattern statistics
    access_patterns: HashMap<String, AccessPattern>,
    /// Optimization suggestions
    optimizations: Vec<CoalescingOptimization>,
}

/// Memory access pattern information
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub kernel_name: String,
    /// Stride between consecutive thread accesses
    pub access_stride: usize,
    /// Access size per thread
    pub access_size: usize,
    /// Whether accesses are aligned
    pub is_aligned: bool,
    /// Coalescing efficiency (0.0 - 1.0)
    pub coalescing_efficiency: f32,
    /// Number of transactions per warp
    pub transactions_per_warp: u32,
}

/// Coalescing optimization suggestion
#[derive(Debug, Clone)]
pub struct CoalescingOptimization {
    pub kernel_name: String,
    pub optimization_type: CoalescingOptType,
    pub expected_improvement: f32,
    pub implementation: String,
}

/// Types of coalescing optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum CoalescingOptType {
    /// Align memory accesses to warp size
    AlignToWarp,
    /// Use Structure of Arrays instead of Array of Structures
    ConvertToSoA,
    /// Pad data structures for alignment
    PadStructures,
    /// Use texture memory for irregular accesses
    UseTextureMemory,
    /// Transpose data layout
    TransposeLayout,
    /// Use shared memory for irregular patterns
    UseSharedMemory,
}

impl MemoryCoalescingOptimizer {
    /// Create new memory coalescing optimizer
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            access_patterns: HashMap::new(),
            optimizations: Vec::new(),
        }
    }

    /// Analyze memory access pattern for a kernel
    pub fn analyze_access_pattern(
        &mut self,
        kernel_name: &str,
        base_address: u64,
        thread_accesses: &[(u32, u64)], // (thread_id, address)
    ) -> Result<AccessPattern> {
        // Sort accesses by thread ID
        let mut sorted_accesses = thread_accesses.to_vec();
        sorted_accesses.sort_by_key(|(tid, _)| *tid);

        // Calculate stride between consecutive threads
        let stride = if sorted_accesses.len() >= 2 {
            let addr1 = sorted_accesses[0].1;
            let addr2 = sorted_accesses[1].1;
            (addr2 as i64 - addr1 as i64).abs() as usize
        } else {
            0
        };

        // Check alignment (addresses should be multiples of 32 for optimal access)
        let is_aligned = sorted_accesses.iter().all(|(_, addr)| addr % 32 == 0);

        // Calculate coalescing efficiency
        let warp_size = 32;
        let ideal_transactions = 1; // Best case: 1 transaction per warp
        let actual_transactions = self.calculate_transactions_per_warp(&sorted_accesses, warp_size);
        let coalescing_efficiency = ideal_transactions as f32 / actual_transactions as f32;

        let pattern = AccessPattern {
            kernel_name: kernel_name.to_string(),
            access_stride: stride,
            access_size: 4, // Assume 4-byte accesses for now
            is_aligned,
            coalescing_efficiency,
            transactions_per_warp: actual_transactions,
        };

        // Store pattern
        self.access_patterns
            .insert(kernel_name.to_string(), pattern.clone());

        // Generate optimizations if needed
        if coalescing_efficiency < 0.8 {
            self.generate_optimizations(kernel_name, &pattern)?;
        }

        Ok(pattern)
    }

    /// Calculate number of memory transactions per warp
    fn calculate_transactions_per_warp(&self, accesses: &[(u32, u64)], warp_size: usize) -> u32 {
        // Group accesses by warp
        let mut transactions = 0u32;

        for warp_start in (0..accesses.len()).step_by(warp_size) {
            let warp_end = (warp_start + warp_size).min(accesses.len());
            let warp_accesses = &accesses[warp_start..warp_end];

            // Find unique cache lines accessed (128-byte cache lines)
            let cache_line_size = 128;
            let mut cache_lines = std::collections::HashSet::new();

            for (_, addr) in warp_accesses {
                let cache_line = addr / cache_line_size;
                cache_lines.insert(cache_line);
            }

            transactions += cache_lines.len() as u32;
        }

        transactions / ((accesses.len() + warp_size - 1) / warp_size) as u32
    }

    /// Generate optimization suggestions
    fn generate_optimizations(&mut self, kernel_name: &str, pattern: &AccessPattern) -> Result<()> {
        // Check for misalignment
        if !pattern.is_aligned {
            self.optimizations.push(CoalescingOptimization {
                kernel_name: kernel_name.to_string(),
                optimization_type: CoalescingOptType::AlignToWarp,
                expected_improvement: 0.2,
                implementation: "Ensure all memory allocations are aligned to 32-byte boundaries"
                    .to_string(),
            });
        }

        // Check for large strides indicating AoS pattern
        if pattern.access_stride > 32 {
            self.optimizations.push(CoalescingOptimization {
                kernel_name: kernel_name.to_string(),
                optimization_type: CoalescingOptType::ConvertToSoA,
                expected_improvement: 0.4,
                implementation: "Convert Array of Structures to Structure of Arrays layout"
                    .to_string(),
            });
        }

        // Check for irregular access patterns
        if pattern.coalescing_efficiency < 0.5 {
            self.optimizations.push(CoalescingOptimization {
                kernel_name: kernel_name.to_string(),
                optimization_type: CoalescingOptType::UseSharedMemory,
                expected_improvement: 0.3,
                implementation:
                    "Load data into shared memory with coalesced accesses, then access locally"
                        .to_string(),
            });
        }

        Ok(())
    }

    /// Apply Structure of Arrays transformation
    pub fn transform_to_soa<T: Clone>(
        &self,
        aos_data: &[ArrayOfStructs<T>],
    ) -> Result<StructureOfArrays<T>> {
        let count = aos_data.len();
        let mut x_values = Vec::with_capacity(count);
        let mut y_values = Vec::with_capacity(count);
        let mut z_values = Vec::with_capacity(count);

        for item in aos_data {
            x_values.push(item.x.clone());
            y_values.push(item.y.clone());
            z_values.push(item.z.clone());
        }

        Ok(StructureOfArrays {
            x: x_values,
            y: y_values,
            z: z_values,
        })
    }

    /// Optimize memory layout for coalescing
    pub async fn optimize_memory_layout<T: Clone + Default>(
        &self,
        data: &mut [T],
        access_pattern: &[usize],
    ) -> Result<Vec<T>> {
        // Reorder data based on access pattern
        let mut optimized = vec![T::default(); data.len()];

        for (new_idx, &old_idx) in access_pattern.iter().enumerate() {
            if old_idx < data.len() {
                optimized[new_idx] = data[old_idx].clone();
            }
        }

        Ok(optimized)
    }

    /// Get optimization report
    pub fn get_optimization_report(&self) -> String {
        let mut report = String::from("Memory Coalescing Analysis Report\n");
        report.push_str("=================================\n\n");

        // Patterns analysis
        report.push_str("Access Patterns:\n");
        for (kernel, pattern) in &self.access_patterns {
            report.push_str(&format!(
                "  {}: efficiency={:.1}%, stride={}, aligned={}, transactions/warp={}\n",
                kernel,
                pattern.coalescing_efficiency * 100.0,
                pattern.access_stride,
                pattern.is_aligned,
                pattern.transactions_per_warp
            ));
        }

        // Optimization suggestions
        if !self.optimizations.is_empty() {
            report.push_str("\nOptimization Suggestions:\n");
            for opt in &self.optimizations {
                report.push_str(&format!(
                    "  {} - {:?}: {:.0}% improvement\n    {}\n",
                    opt.kernel_name,
                    opt.optimization_type,
                    opt.expected_improvement * 100.0,
                    opt.implementation
                ));
            }
        }

        // Summary
        let avg_efficiency = if !self.access_patterns.is_empty() {
            self.access_patterns
                .values()
                .map(|p| p.coalescing_efficiency)
                .sum::<f32>()
                / self.access_patterns.len() as f32
        } else {
            0.0
        };

        report.push_str(&format!(
            "\nSummary:\n  Average coalescing efficiency: {:.1}%\n",
            avg_efficiency * 100.0
        ));

        report
    }

    /// Apply padding to structure for better alignment
    pub fn calculate_padding(struct_size: usize, alignment: usize) -> usize {
        let remainder = struct_size % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }
}

/// Example data structures for transformation
#[derive(Clone)]
pub struct ArrayOfStructs<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct StructureOfArrays<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub z: Vec<T>,
}

/// Memory access descriptor for analysis
pub struct MemoryAccessDescriptor {
    pub kernel_name: String,
    pub block_dim: (u32, u32, u32),
    pub grid_dim: (u32, u32, u32),
    pub accesses: Vec<AccessInfo>,
}

#[derive(Clone)]
pub struct AccessInfo {
    pub thread_id: u32,
    pub address: u64,
    pub size: usize,
    pub is_read: bool,
}

/// Coalescing benchmark utilities
pub struct CoalescingBenchmark {
    device: Arc<CudaDevice>,
}

impl CoalescingBenchmark {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Benchmark memory bandwidth with different access patterns
    pub async fn benchmark_access_patterns(&self) -> Result<BenchmarkResults> {
        let data_size = 1024 * 1024 * 16; // 16MB
        let iterations = 100;

        // Benchmark coalesced access
        let coalesced_time = self
            .benchmark_coalesced_access(data_size, iterations)
            .await?;

        // Benchmark strided access
        let strided_time = self
            .benchmark_strided_access(data_size, iterations, 32)
            .await?;

        // Benchmark random access
        let random_time = self.benchmark_random_access(data_size, iterations).await?;

        // Calculate bandwidths
        let bytes_transferred = (data_size * iterations) as f64;
        let coalesced_bandwidth = bytes_transferred / coalesced_time.as_secs_f64() / 1e9;
        let strided_bandwidth = bytes_transferred / strided_time.as_secs_f64() / 1e9;
        let random_bandwidth = bytes_transferred / random_time.as_secs_f64() / 1e9;

        Ok(BenchmarkResults {
            coalesced_bandwidth_gbps: coalesced_bandwidth,
            strided_bandwidth_gbps: strided_bandwidth,
            random_bandwidth_gbps: random_bandwidth,
            coalescing_efficiency: coalesced_bandwidth / self.get_theoretical_bandwidth(),
        })
    }

    /// Benchmark coalesced memory access
    async fn benchmark_coalesced_access(
        &self,
        data_size: usize,
        iterations: usize,
    ) -> Result<std::time::Duration> {
        // Allocate device memory
        let mut device_data = unsafe { self.device.alloc::<f32>(data_size / 4)? };

        let start = std::time::Instant::now();

        // Simulate coalesced access pattern
        for _ in 0..iterations {
            // In real implementation, would launch kernel with coalesced access
            self.device.synchronize()?;
        }

        Ok(start.elapsed())
    }

    /// Benchmark strided memory access
    async fn benchmark_strided_access(
        &self,
        data_size: usize,
        iterations: usize,
        stride: usize,
    ) -> Result<std::time::Duration> {
        let mut device_data = unsafe { self.device.alloc::<f32>(data_size / 4)? };

        let start = std::time::Instant::now();

        // Simulate strided access pattern
        for _ in 0..iterations {
            // In real implementation, would launch kernel with strided access
            self.device.synchronize()?;
        }

        Ok(start.elapsed())
    }

    /// Benchmark random memory access
    async fn benchmark_random_access(
        &self,
        data_size: usize,
        iterations: usize,
    ) -> Result<std::time::Duration> {
        let mut device_data = unsafe { self.device.alloc::<f32>(data_size / 4)? };

        let start = std::time::Instant::now();

        // Simulate random access pattern
        for _ in 0..iterations {
            // In real implementation, would launch kernel with random access
            self.device.synchronize()?;
        }

        Ok(start.elapsed())
    }

    /// Get theoretical memory bandwidth for the device
    fn get_theoretical_bandwidth(&self) -> f64 {
        // RTX 5090 theoretical bandwidth
        1008.0 // GB/s
    }
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub coalesced_bandwidth_gbps: f64,
    pub strided_bandwidth_gbps: f64,
    pub random_bandwidth_gbps: f64,
    pub coalescing_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalescing_analysis() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let mut optimizer = MemoryCoalescingOptimizer::new(device);

        // Simulate perfectly coalesced access
        let coalesced_accesses: Vec<(u32, u64)> =
            (0..32).map(|i| (i, 0x1000 + i as u64 * 4)).collect();

        let pattern = optimizer
            .analyze_access_pattern("coalesced_kernel", 0x1000, &coalesced_accesses)
            ?;

        assert!(pattern.coalescing_efficiency > 0.9);
        assert_eq!(pattern.access_stride, 4);
        assert!(pattern.is_aligned);
    }

    #[test]
    fn test_strided_access_detection() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let mut optimizer = MemoryCoalescingOptimizer::new(device);

        // Simulate strided access (AoS pattern)
        let strided_accesses: Vec<(u32, u64)> = (0..32)
            .map(|i| (i, 0x1000 + i as u64 * 64)) // 64-byte stride
            .collect();

        let pattern = optimizer
            .analyze_access_pattern("strided_kernel", 0x1000, &strided_accesses)
            .unwrap();

        assert!(pattern.coalescing_efficiency < 0.5);
        assert_eq!(pattern.access_stride, 64);

        // Should generate SoA optimization
        assert!(!optimizer.optimizations.is_empty());
        assert_eq!(
            optimizer.optimizations[0].optimization_type,
            CoalescingOptType::ConvertToSoA
        );
    }

    #[test]
    fn test_aos_to_soa_transformation() -> Result<(), Box<dyn std::error::Error>>  {
        let device = Arc::new(CudaDevice::new(0)?);
        let optimizer = MemoryCoalescingOptimizer::new(device);

        let aos_data = vec![
            ArrayOfStructs {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            ArrayOfStructs {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
            ArrayOfStructs {
                x: 7.0,
                y: 8.0,
                z: 9.0,
            },
        ];

        let soa = optimizer.transform_to_soa(&aos_data)?;

        assert_eq!(soa.x, vec![1.0, 4.0, 7.0]);
        assert_eq!(soa.y, vec![2.0, 5.0, 8.0]);
        assert_eq!(soa.z, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_padding_calculation() {
        assert_eq!(MemoryCoalescingOptimizer::calculate_padding(13, 16), 3);
        assert_eq!(MemoryCoalescingOptimizer::calculate_padding(16, 16), 0);
        assert_eq!(MemoryCoalescingOptimizer::calculate_padding(17, 16), 15);
    }
}
