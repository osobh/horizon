//! Metal 3 compute pipeline implementation.

use crate::compute::MetalComputePipeline;
use crate::error::{MetalError, Result};
use crate::metal3::Metal3Device;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary};

/// Metal 3 compute pipeline.
pub struct Metal3ComputePipeline {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    function_name: String,
    max_threads: u64,
    threadgroup_memory: u32,
}

// SAFETY: MTLComputePipelineState is thread-safe
unsafe impl Send for Metal3ComputePipeline {}
unsafe impl Sync for Metal3ComputePipeline {}

impl Metal3ComputePipeline {
    /// Create a compute pipeline from source code.
    pub fn from_source(device: &Metal3Device, source: &str, function_name: &str) -> Result<Self> {
        // Compile the source to a library
        let source_ns = NSString::from_str(source);
        let options = MTLCompileOptions::new();

        let library = device
            .raw()
            .newLibraryWithSource_options_error(&source_ns, Some(&options))
            .map_err(|e| MetalError::shader_error(e.to_string()))?;

        Self::from_library(device, &library, function_name)
    }

    /// Create a compute pipeline from precompiled library data (metallib).
    ///
    /// # Note
    /// This function creates a library from pre-compiled Metal library data (.metallib files).
    /// For now, we use URL-based loading as an alternative to dispatch_data.
    pub fn from_library_data(
        _device: &Metal3Device,
        _data: &[u8],
        _function_name: &str,
    ) -> Result<Self> {
        // TODO: Implement when dispatch_data API stabilizes in dispatch2
        // For now, prefer from_source() which compiles MSL at runtime
        Err(MetalError::creation_failed(
            "library",
            "from_library_data not yet implemented - use from_source() instead",
        ))
    }

    /// Create a compute pipeline from a library.
    fn from_library(
        device: &Metal3Device,
        library: &Retained<ProtocolObject<dyn MTLLibrary>>,
        function_name: &str,
    ) -> Result<Self> {
        // Get the function
        let function_name_ns = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName(&function_name_ns)
            .ok_or_else(|| MetalError::FunctionNotFound(function_name.to_string()))?;

        // Create the pipeline state
        let pipeline = device
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::creation_failed("compute pipeline", e.to_string()))?;

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let threadgroup_memory = pipeline.staticThreadgroupMemoryLength() as u32;

        Ok(Self {
            pipeline,
            function_name: function_name.to_string(),
            max_threads,
            threadgroup_memory,
        })
    }

    /// Get the raw MTLComputePipelineState.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
    }
}

impl MetalComputePipeline for Metal3ComputePipeline {
    fn function_name(&self) -> &str {
        &self.function_name
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        self.max_threads
    }

    fn threadgroup_memory_length(&self) -> u32 {
        self.threadgroup_memory
    }
}

impl std::fmt::Debug for Metal3ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal3ComputePipeline")
            .field("function_name", &self.function_name)
            .field("max_threads", &self.max_threads)
            .field("threadgroup_memory", &self.threadgroup_memory)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_KERNEL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_arrays(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            result[id] = a[id] + b[id];
        }
    "#;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_pipeline_creation() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let pipeline = Metal3ComputePipeline::from_source(&device, SIMPLE_KERNEL, "add_arrays");

        assert!(
            pipeline.is_ok(),
            "Failed to create pipeline: {:?}",
            pipeline.err()
        );

        let pipeline = pipeline.unwrap();
        assert_eq!(pipeline.function_name(), "add_arrays");
        assert!(pipeline.max_total_threads_per_threadgroup() > 0);
    }
}
