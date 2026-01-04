//! Core Metal backend traits.
//!
//! These traits abstract over Metal 3 and Metal 4 implementations,
//! allowing code to work with either version.

use crate::buffer::MetalBuffer;
use crate::command::MetalCommandQueue;
use crate::compute::MetalComputePipeline;
use crate::error::Result;
use crate::tensor::{MetalTensor, TensorDescriptor};

/// Information about a Metal device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name (e.g., "Apple M3 Max").
    pub name: String,
    /// Whether the device supports unified memory.
    pub unified_memory: bool,
    /// Maximum buffer length in bytes.
    pub max_buffer_length: u64,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: u64,
    /// Maximum threadgroup memory length.
    pub max_threadgroup_memory_length: u32,
    /// Metal GPU family.
    pub gpu_family: GpuFamily,
}

/// Metal GPU family classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    /// Apple Silicon (M1, M2, M3, M4).
    Apple(u8),
    /// Mac GPU family.
    Mac(u8),
    /// Common family.
    Common(u8),
    /// Unknown family.
    Unknown,
}

/// Trait for Metal device abstraction.
pub trait MetalDevice: Send + Sync {
    /// Get device information.
    fn info(&self) -> &DeviceInfo;

    /// Get the device name.
    fn name(&self) -> &str {
        &self.info().name
    }

    /// Check if device supports unified memory.
    fn supports_unified_memory(&self) -> bool {
        self.info().unified_memory
    }

    /// Get maximum buffer size.
    fn max_buffer_length(&self) -> u64 {
        self.info().max_buffer_length
    }
}

/// Main Metal backend trait.
///
/// This is the primary interface for GPU operations. Implementations
/// handle the differences between Metal 3 and Metal 4.
pub trait MetalBackend: Send + Sync + Sized {
    /// Buffer type for this backend.
    type Buffer: MetalBuffer;
    /// Command queue type for this backend.
    type CommandQueue: MetalCommandQueue;
    /// Compute pipeline type for this backend.
    type ComputePipeline: MetalComputePipeline;
    /// Tensor type for this backend.
    type Tensor: MetalTensor;

    /// Get the underlying device.
    fn device(&self) -> &dyn MetalDevice;

    /// Create a GPU buffer with the given element count.
    ///
    /// The buffer uses unified memory on Apple Silicon, meaning
    /// CPU and GPU can access it without explicit copies.
    fn create_buffer<T: bytemuck::Pod>(&self, count: usize) -> Result<Self::Buffer>;

    /// Create a GPU buffer initialized with the given data.
    fn create_buffer_with_data<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Self::Buffer>;

    /// Create a command queue for submitting GPU work.
    fn create_command_queue(&self) -> Result<Self::CommandQueue>;

    /// Create a compute pipeline from shader source.
    ///
    /// # Arguments
    /// * `source` - Metal Shading Language source code
    /// * `function_name` - Name of the kernel function to use
    fn create_compute_pipeline(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<Self::ComputePipeline>;

    /// Create a compute pipeline from precompiled library.
    fn create_compute_pipeline_from_library(
        &self,
        library_data: &[u8],
        function_name: &str,
    ) -> Result<Self::ComputePipeline>;

    /// Create a tensor for ML operations.
    fn create_tensor(&self, desc: TensorDescriptor) -> Result<Self::Tensor>;

    /// Synchronize all pending GPU work.
    fn synchronize(&self) -> Result<()>;

    /// Get the Metal version supported by this backend.
    fn metal_version(&self) -> MetalVersion;
}

/// Metal API version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalVersion {
    /// Metal 3 (macOS 14+, iOS 17+).
    Metal3,
    /// Metal 3.1.
    Metal3_1,
    /// Metal 3.2 (macOS 15+).
    Metal3_2,
    /// Metal 4 (macOS 26+).
    Metal4,
}

impl MetalVersion {
    /// Check if this version supports a feature.
    pub fn supports(&self, feature: MetalFeature) -> bool {
        match feature {
            MetalFeature::UnifiedMemory => true, // All versions on Apple Silicon
            MetalFeature::ArgumentBuffers => true,
            MetalFeature::IndirectCommandBuffers => true,
            MetalFeature::RayTracing => *self >= MetalVersion::Metal3,
            MetalFeature::MeshShaders => *self >= MetalVersion::Metal3,
            MetalFeature::NativeTensors => *self >= MetalVersion::Metal4,
            MetalFeature::ShaderML => *self >= MetalVersion::Metal4,
            MetalFeature::ArgumentTables => *self >= MetalVersion::Metal4,
            MetalFeature::ResidencySets => *self >= MetalVersion::Metal4,
        }
    }
}

/// Metal features that may vary by version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalFeature {
    /// Unified CPU/GPU memory.
    UnifiedMemory,
    /// Argument buffers for resource binding.
    ArgumentBuffers,
    /// Indirect command buffers.
    IndirectCommandBuffers,
    /// Ray tracing support.
    RayTracing,
    /// Mesh shaders.
    MeshShaders,
    /// Native MTLTensor type.
    NativeTensors,
    /// Shader ML for embedded neural networks.
    ShaderML,
    /// MTL4ArgumentTable for efficient binding.
    ArgumentTables,
    /// MTLResidencySet for memory management.
    ResidencySets,
}

/// Trait for backends that support ML operations.
pub trait MetalMLBackend: MetalBackend {
    /// Run a neural network encoded in the given CoreML package.
    fn run_ml_network(
        &self,
        network_path: &str,
        inputs: &[&Self::Tensor],
    ) -> Result<Vec<Self::Tensor>>;
}
