//! Compute pipeline abstraction.
//!
//! Compute pipelines encapsulate a compiled compute shader
//! and its execution configuration.

/// Trait for Metal compute pipelines.
///
/// A compute pipeline represents a compiled kernel function
/// that can be executed on the GPU.
pub trait MetalComputePipeline: Send + Sync {
    /// Get the function name.
    fn function_name(&self) -> &str;

    /// Get the maximum total threads per threadgroup.
    fn max_total_threads_per_threadgroup(&self) -> u64;

    /// Get the threadgroup memory length used by the shader.
    fn threadgroup_memory_length(&self) -> u32;

    /// Get the optimal threads per threadgroup for 1D dispatch.
    fn optimal_threads_per_threadgroup_1d(&self) -> u64 {
        self.max_total_threads_per_threadgroup().min(256)
    }

    /// Get the optimal threadgroup size for 2D dispatch.
    fn optimal_threads_per_threadgroup_2d(&self) -> (u64, u64) {
        let max = self.max_total_threads_per_threadgroup();
        if max >= 256 {
            (16, 16)
        } else if max >= 64 {
            (8, 8)
        } else {
            (4, 4)
        }
    }
}

/// Options for creating a compute pipeline.
#[derive(Debug, Clone, Default)]
pub struct ComputePipelineOptions {
    /// Label for debugging.
    pub label: Option<String>,
    /// Maximum call stack depth for recursive functions.
    pub max_call_stack_depth: Option<u32>,
    /// Thread execution width hint.
    pub thread_execution_width: Option<u32>,
}

impl ComputePipelineOptions {
    /// Create new options with a label.
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            ..Default::default()
        }
    }
}
