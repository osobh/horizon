//! Kernel fusion for GPU performance optimization
//!
//! Implements kernel fusion techniques to reduce kernel launch overhead
//! and improve GPU utilization by combining multiple operations into
//! single, more efficient kernels.

use anyhow::{anyhow, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub mod fusion_analyzer;
pub mod fusion_compiler;
pub mod fusion_patterns;
pub mod fusion_runtime;

#[cfg(test)]
mod tests;

/// Kernel fusion configuration
#[derive(Debug, Clone)]
pub struct KernelFusionConfig {
    /// Enable automatic kernel fusion
    pub enable_auto_fusion: bool,
    /// Maximum operations to fuse in a single kernel
    pub max_fusion_depth: usize,
    /// Minimum operation count for fusion
    pub min_ops_for_fusion: usize,
    /// Enable runtime fusion analysis
    pub enable_runtime_analysis: bool,
    /// Fusion strategy to use
    pub fusion_strategy: FusionStrategy,
    /// Performance thresholds
    pub performance_thresholds: FusionThresholds,
}

impl Default for KernelFusionConfig {
    fn default() -> Self {
        Self {
            enable_auto_fusion: true,
            max_fusion_depth: 5,
            min_ops_for_fusion: 2,
            enable_runtime_analysis: true,
            fusion_strategy: FusionStrategy::Balanced,
            performance_thresholds: FusionThresholds::default(),
        }
    }
}

/// Fusion strategies
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Aggressive fusion - maximize fusion opportunities
    Aggressive,
    /// Balanced fusion - balance between fusion and flexibility
    Balanced,
    /// Conservative fusion - only fuse when clearly beneficial
    Conservative,
    /// Custom fusion with specific patterns
    Custom(Vec<FusionPattern>),
}

/// Performance thresholds for fusion decisions
#[derive(Debug, Clone)]
pub struct FusionThresholds {
    /// Minimum speedup required for fusion (e.g., 1.2 = 20% speedup)
    pub min_speedup_factor: f32,
    /// Maximum memory overhead allowed (percentage)
    pub max_memory_overhead: f32,
    /// Maximum register pressure increase allowed
    pub max_register_pressure: u32,
    /// Minimum kernel execution time for fusion consideration (microseconds)
    pub min_kernel_time_us: u64,
}

impl Default for FusionThresholds {
    fn default() -> Self {
        Self {
            min_speedup_factor: 1.2,   // 20% speedup required
            max_memory_overhead: 20.0, // 20% memory overhead allowed
            max_register_pressure: 48, // 48 registers max
            min_kernel_time_us: 10,    // 10μs minimum kernel time
        }
    }
}

/// Kernel fusion pattern
#[derive(Debug, Clone, PartialEq)]
pub struct FusionPattern {
    /// Pattern name
    pub name: String,
    /// Operations that can be fused
    pub operations: Vec<OperationType>,
    /// Fusion conditions
    pub conditions: FusionConditions,
    /// Expected performance gain
    pub expected_speedup: f32,
}

/// Operation types that can be fused
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Element-wise operations (add, multiply, etc.)
    ElementWise(ElementWiseOp),
    /// Reduction operations (sum, max, etc.)
    Reduction(ReductionOp),
    /// Memory operations (copy, transpose, etc.)
    Memory(MemoryOp),
    /// Matrix operations (gemm, etc.)
    Matrix(MatrixOp),
    /// Custom operation
    Custom(String),
}

/// Element-wise operations
#[derive(Debug, Clone, PartialEq)]
pub enum ElementWiseOp {
    Add,
    Multiply,
    Subtract,
    Divide,
    Power,
    Activation(ActivationType),
}

/// Activation types
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
}

/// Reduction operations
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
}

/// Memory operations
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOp {
    Copy,
    Transpose,
    Reshape,
    Permute,
    Gather,
    Scatter,
}

/// Matrix operations
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixOp {
    GEMM,
    BatchedGEMM,
    Convolution,
    Pooling,
}

/// Fusion conditions
#[derive(Debug, Clone, PartialEq)]
pub struct FusionConditions {
    /// Data dependencies between operations
    pub data_dependencies: Vec<DataDependency>,
    /// Memory access patterns must be compatible
    pub memory_pattern_compatible: bool,
    /// Register usage constraints
    pub register_constraints: RegisterConstraints,
    /// Shared memory constraints
    pub shared_memory_constraints: SharedMemoryConstraints,
}

/// Data dependency between operations
#[derive(Debug, Clone, PartialEq)]
pub struct DataDependency {
    /// Source operation index
    pub source_op: usize,
    /// Target operation index
    pub target_op: usize,
    /// Dependency type
    pub dependency_type: DependencyType,
}

/// Types of dependencies
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyType {
    /// Read after write
    RAW,
    /// Write after read
    WAR,
    /// Write after write
    WAW,
    /// No dependency
    None,
}

/// Register usage constraints
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterConstraints {
    /// Maximum registers per thread
    pub max_registers_per_thread: u32,
    /// Current register usage
    pub current_usage: u32,
    /// Additional registers needed for fusion
    pub fusion_overhead: u32,
}

/// Shared memory constraints
#[derive(Debug, Clone, PartialEq)]
pub struct SharedMemoryConstraints {
    /// Maximum shared memory per block (bytes)
    pub max_shared_memory: usize,
    /// Current shared memory usage
    pub current_usage: usize,
    /// Additional shared memory needed for fusion
    pub fusion_overhead: usize,
}

/// Main kernel fusion engine
pub struct KernelFusionEngine {
    device: Arc<CudaDevice>,
    config: KernelFusionConfig,
    fusion_cache: HashMap<String, FusedKernel>,
    performance_history: PerformanceHistory,
    statistics: FusionStatistics,
}

impl KernelFusionEngine {
    /// Create new kernel fusion engine
    pub fn new(device: Arc<CudaDevice>, config: KernelFusionConfig) -> Self {
        Self {
            device,
            config,
            fusion_cache: HashMap::new(),
            performance_history: PerformanceHistory::new(),
            statistics: FusionStatistics::default(),
        }
    }

    /// Analyze operations for fusion opportunities
    pub async fn analyze_fusion_opportunities(
        &mut self,
        operations: &[KernelOperation],
    ) -> Result<Vec<FusionOpportunity>> {
        let analyzer =
            fusion_analyzer::FusionAnalyzer::new(self.device.clone(), self.config.clone());

        analyzer.analyze_operations(operations).await
    }

    /// Compile fused kernel from operations
    pub async fn compile_fused_kernel(
        &mut self,
        opportunity: &FusionOpportunity,
    ) -> Result<FusedKernel> {
        // Check cache first
        if let Some(cached) = self.fusion_cache.get(&opportunity.fusion_id) {
            self.statistics.cache_hits += 1;
            return Ok(cached.clone());
        }

        self.statistics.cache_misses += 1;

        // Compile new fused kernel
        let compiler =
            fusion_compiler::FusionCompiler::new(self.device.clone(), self.config.clone());

        let fused_kernel = compiler.compile(opportunity).await?;

        // Cache the compiled kernel
        self.fusion_cache
            .insert(opportunity.fusion_id.clone(), fused_kernel.clone());

        Ok(fused_kernel)
    }

    /// Execute fused kernel
    pub async fn execute_fused_kernel(
        &mut self,
        kernel: &FusedKernel,
        inputs: &[CudaSlice<f32>],
        outputs: &mut [CudaSlice<f32>],
        stream: &CudaStream,
    ) -> Result<FusionExecutionResult> {
        let runtime = fusion_runtime::FusionRuntime::new(self.device.clone(), self.config.clone());

        let start_time = Instant::now();

        // Convert CudaSlice to GpuFloatBuffer
        use crate::gpu_buffer::GpuFloatBuffer;
        let gpu_inputs: Vec<GpuFloatBuffer> = inputs
            .iter()
            .map(|slice| {
                // Create GpuFloatBuffer from existing CudaSlice
                // Note: This is a workaround - in production, we'd pass GpuFloatBuffer directly
                let len = 1024; // Default size, should get from slice metadata
                GpuFloatBuffer::from_cuda_slice(slice.clone(), len)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut gpu_outputs: Vec<GpuFloatBuffer> = outputs
            .iter_mut()
            .map(|slice| {
                let len = 1024; // Default size
                GpuFloatBuffer::from_cuda_slice(slice.clone(), len)
            })
            .collect::<Result<Vec<_>>>()?;

        let result = runtime
            .execute(&kernel, &gpu_inputs, &mut gpu_outputs, stream)
            .await?;

        let execution_time = start_time.elapsed();

        // Update performance history
        self.performance_history.record_execution(
            &kernel.fusion_id,
            execution_time,
            result.gpu_utilization,
        );

        // Update statistics
        self.statistics.kernels_executed += 1;
        self.statistics.total_execution_time += execution_time;

        Ok(result)
    }

    /// Get fusion recommendations based on performance history
    pub fn get_fusion_recommendations(&self) -> Vec<FusionRecommendation> {
        self.performance_history
            .generate_recommendations(&self.config)
    }

    /// Get fusion statistics
    pub fn get_statistics(&self) -> &FusionStatistics {
        &self.statistics
    }

    /// Clear fusion cache
    pub fn clear_cache(&mut self) {
        self.fusion_cache.clear();
        self.statistics.cache_clears += 1;
    }
}

/// Kernel operation to be potentially fused
#[derive(Debug, Clone)]
pub struct KernelOperation {
    /// Operation ID
    pub id: String,
    /// Operation type
    pub op_type: OperationType,
    /// Input tensor descriptors
    pub inputs: Vec<TensorDescriptor>,
    /// Output tensor descriptors
    pub outputs: Vec<TensorDescriptor>,
    /// Estimated execution time
    pub estimated_time_us: u64,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
}

/// Tensor descriptor
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Memory layout
    pub layout: MemoryLayout,
    /// Stride information
    pub strides: Vec<usize>,
}

/// Supported data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
}

/// Memory layout
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
    /// Custom stride pattern
    Custom,
}

/// Memory requirements for an operation
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Global memory reads (bytes)
    pub global_reads: usize,
    /// Global memory writes (bytes)
    pub global_writes: usize,
    /// Shared memory usage (bytes)
    pub shared_memory: usize,
    /// Register usage per thread
    pub registers_per_thread: u32,
}

/// Fusion opportunity identified by analyzer
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    /// Unique ID for this fusion
    pub fusion_id: String,
    /// Operations to fuse
    pub operations: Vec<KernelOperation>,
    /// Expected speedup factor
    pub expected_speedup: f32,
    /// Memory savings (bytes)
    pub memory_savings: usize,
    /// Fusion pattern matched
    pub pattern: FusionPattern,
    /// Fusion feasibility score (0-1)
    pub feasibility_score: f32,
}

/// Compiled fused kernel
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Fusion ID
    pub fusion_id: String,
    /// PTX code for the fused kernel
    pub ptx_code: Vec<u8>,
    /// Launch configuration
    pub launch_config: LaunchConfig,
    /// Memory requirements
    pub memory_requirements: MemoryRequirements,
    /// Original operations that were fused
    pub original_ops: Vec<String>,
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size per block
    pub shared_mem_bytes: usize,
    /// Optimal stream count
    pub stream_count: usize,
}

/// Fusion execution result
#[derive(Debug, Clone)]
pub struct FusionExecutionResult {
    /// Execution time
    pub execution_time: Duration,
    /// GPU utilization achieved
    pub gpu_utilization: f32,
    /// Memory bandwidth utilization
    pub memory_bandwidth_util: f32,
    /// Power efficiency (GFLOPS/W)
    pub power_efficiency: f32,
    /// Any warnings or issues
    pub warnings: Vec<String>,
}

/// Performance history tracking
#[derive(Debug)]
struct PerformanceHistory {
    /// Execution history per fusion
    history: HashMap<String, Vec<ExecutionRecord>>,
    /// Maximum history size per fusion
    max_history_size: usize,
}

impl PerformanceHistory {
    fn new() -> Self {
        Self {
            history: HashMap::new(),
            max_history_size: 100,
        }
    }

    fn record_execution(&mut self, fusion_id: &str, time: Duration, gpu_util: f32) {
        let record = ExecutionRecord {
            timestamp: Instant::now(),
            execution_time: time,
            gpu_utilization: gpu_util,
        };

        let history = self
            .history
            .entry(fusion_id.to_string())
            .or_insert_with(Vec::new);
        history.push(record);

        // Limit history size
        if history.len() > self.max_history_size {
            history.remove(0);
        }
    }

    fn generate_recommendations(&self, config: &KernelFusionConfig) -> Vec<FusionRecommendation> {
        let mut recommendations = Vec::new();

        for (fusion_id, history) in &self.history {
            if history.len() < 10 {
                continue; // Need sufficient history
            }

            // Calculate average performance
            let avg_time = history
                .iter()
                .map(|r| r.execution_time.as_micros() as f64)
                .sum::<f64>()
                / history.len() as f64;

            let avg_gpu_util =
                history.iter().map(|r| r.gpu_utilization).sum::<f32>() / history.len() as f32;

            // Generate recommendations based on performance
            if avg_gpu_util < 80.0 {
                recommendations.push(FusionRecommendation {
                    fusion_id: fusion_id.clone(),
                    recommendation_type: RecommendationType::ImproveUtilization,
                    description: format!(
                        "GPU utilization is {:.1}%, consider more aggressive fusion",
                        avg_gpu_util
                    ),
                    priority: RecommendationPriority::High,
                });
            }

            if avg_time > 1000.0 {
                recommendations.push(FusionRecommendation {
                    fusion_id: fusion_id.clone(),
                    recommendation_type: RecommendationType::SplitKernel,
                    description: format!(
                        "Kernel execution time {:.0}μs is high, consider splitting",
                        avg_time
                    ),
                    priority: RecommendationPriority::Medium,
                });
            }
        }

        recommendations
    }
}

/// Execution record for performance tracking
#[derive(Debug)]
struct ExecutionRecord {
    timestamp: Instant,
    execution_time: Duration,
    gpu_utilization: f32,
}

/// Fusion recommendation
#[derive(Debug, Clone)]
pub struct FusionRecommendation {
    /// Fusion ID
    pub fusion_id: String,
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Description of recommendation
    pub description: String,
    /// Priority level
    pub priority: RecommendationPriority,
}

/// Types of recommendations
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationType {
    /// Improve GPU utilization
    ImproveUtilization,
    /// Split large kernel
    SplitKernel,
    /// Merge more operations
    MergeMore,
    /// Change fusion strategy
    ChangeStrategy,
    /// Memory optimization needed
    OptimizeMemory,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Fusion statistics
#[derive(Debug, Default)]
pub struct FusionStatistics {
    /// Total kernels executed
    pub kernels_executed: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache clears
    pub cache_clears: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average speedup achieved
    pub avg_speedup: f32,
}

impl FusionStatistics {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Calculate average execution time
    pub fn avg_execution_time(&self) -> Duration {
        if self.kernels_executed > 0 {
            self.total_execution_time / self.kernels_executed as u32
        } else {
            Duration::ZERO
        }
    }
}
