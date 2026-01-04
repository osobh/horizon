//! Metal 4 backend implementation (feature-gated).
//!
//! This module provides the Metal 4 backend for macOS 26+ (when available).
//! Metal 4 introduces significant improvements:
//!
//! # Metal 4 Features
//!
//! - **Native MTLTensor**: First-class tensor support (replaces MPS matrices)
//! - **MTL4MachineLearningCommandEncoder**: Efficient ML inference
//! - **MTL4ArgumentTable**: Faster argument binding
//! - **MTLResidencySet**: Explicit memory residency control
//! - **MTL4CommandAllocator**: Per-frame memory management
//! - **Explicit Barriers**: Fine-grained synchronization control
//! - **Shader ML**: Inline tensor operations in compute shaders
//!
//! # Usage
//!
//! Enable the `metal4` feature in Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! stratoswarm-metal-core = { version = "0.1", features = ["metal4"] }
//! ```
//!
//! # Current Status
//!
//! This module implements Metal 4 patterns using Metal 3 fallbacks.
//! When macOS 26 ships with the Metal 4 SDK, native implementations
//! will be activated automatically via runtime detection.

mod allocator;
mod argument_table;
mod barrier;
mod buffer;
mod command;
mod detection;
mod device;
mod residency;
mod tensor;

// Core types
pub use buffer::Metal4Buffer;
pub use command::{Metal4CommandBuffer, Metal4CommandQueue, Metal4ComputeEncoder};
pub use device::Metal4Device;
pub use tensor::Metal4Tensor;

// Metal 4 specific types
pub use allocator::Metal4CommandAllocator;
pub use argument_table::{ArgumentTableDescriptor, Metal4ArgumentTable};
pub use barrier::{Barrier, BarrierBatch, BarrierBuilder, ResourceRef, ResourceType};
pub use detection::{
    is_argument_table_available, is_command_allocator_available, is_metal4_available,
    is_native_tensor_available, is_residency_set_available, Metal4Features, MetalGeneration,
};
pub use residency::{Metal4ResidencySet, ResidencySetBuilder, ResourceUsage, TrackedResource};

use crate::backend::{MetalBackend, MetalDevice, MetalVersion};
use crate::error::{MetalError, Result};
use crate::tensor::TensorDescriptor;

use std::sync::Arc;

/// Check if Metal 4 is available on this system.
///
/// Metal 4 requires macOS 26 or later.
pub fn is_available() -> bool {
    detection::is_metal4_available()
}

/// Metal 4 backend implementation.
///
/// This is the future-ready backend for macOS 26+.
/// It provides native tensor support and ML command encoding.
///
/// On systems without Metal 4, this backend still works using
/// Metal 3 fallback implementations for:
/// - Command allocators (frame tracking)
/// - Argument tables (GPU address buffers)
/// - Residency sets (resource tracking)
/// - Barriers (memory barriers)
#[derive(Clone)]
pub struct Metal4Backend {
    device: Arc<Metal4Device>,
}

impl Metal4Backend {
    /// Create a new Metal 4 backend using the default system device.
    ///
    /// Note: This will succeed even on systems without native Metal 4 support,
    /// using Metal 3 fallback implementations. Use `is_available()` to check
    /// for native Metal 4 support.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn new() -> Result<Self> {
        let device = Metal4Device::system_default()?;
        Ok(Self {
            device: Arc::new(device),
        })
    }

    /// Create a new Metal 4 backend, failing if native Metal 4 is not available.
    ///
    /// Use this when you specifically need Metal 4 features and cannot
    /// use the fallback implementations.
    pub fn new_native() -> Result<Self> {
        if !is_available() {
            return Err(MetalError::creation_failed(
                "Metal4Backend",
                "Native Metal 4 is not available on this system (requires macOS 26+)",
            ));
        }

        Self::new()
    }

    /// Create a backend from an existing device.
    pub fn from_device(device: Metal4Device) -> Self {
        Self {
            device: Arc::new(device),
        }
    }

    /// Get a reference to the device.
    pub fn device(&self) -> &Arc<Metal4Device> {
        &self.device
    }

    /// Get the raw Metal 4 device.
    pub fn raw_device(&self) -> &Metal4Device {
        &self.device
    }

    /// Check if this backend is using native Metal 4 APIs.
    ///
    /// Returns `true` if running on macOS 26+ with full Metal 4 support,
    /// `false` if using Metal 3 fallback implementations.
    pub fn is_native(&self) -> bool {
        is_available()
    }

    /// Get detected Metal 4 features.
    pub fn features(&self) -> &'static Metal4Features {
        Metal4Features::detect()
    }
}

impl MetalBackend for Metal4Backend {
    type Buffer = Metal4Buffer;
    type CommandQueue = Metal4CommandQueue;
    type ComputePipeline = crate::metal3::Metal3ComputePipeline; // Reuse Metal 3 pipelines
    type Tensor = Metal4Tensor;

    fn device(&self) -> &dyn MetalDevice {
        self.device.as_ref()
    }

    fn create_buffer<T: bytemuck::Pod>(&self, count: usize) -> Result<Self::Buffer> {
        Metal4Buffer::new(&self.device, count * std::mem::size_of::<T>())
    }

    fn create_buffer_with_data<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Self::Buffer> {
        Metal4Buffer::with_data(&self.device, data)
    }

    fn create_command_queue(&self) -> Result<Self::CommandQueue> {
        Metal4CommandQueue::new(&self.device)
    }

    fn create_compute_pipeline(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<Self::ComputePipeline> {
        // Metal 4 uses the same compute pipeline as Metal 3
        // The major differences are in tensor operations and ML encoding
        crate::metal3::Metal3ComputePipeline::from_source(
            self.device.as_metal3_device(),
            source,
            function_name,
        )
    }

    fn create_compute_pipeline_from_library(
        &self,
        library_data: &[u8],
        function_name: &str,
    ) -> Result<Self::ComputePipeline> {
        crate::metal3::Metal3ComputePipeline::from_library_data(
            self.device.as_metal3_device(),
            library_data,
            function_name,
        )
    }

    fn create_tensor(&self, desc: TensorDescriptor) -> Result<Self::Tensor> {
        Metal4Tensor::new(&self.device, desc)
    }

    fn synchronize(&self) -> Result<()> {
        // Metal 4 may have better sync primitives
        Ok(())
    }

    fn metal_version(&self) -> MetalVersion {
        MetalVersion::Metal4
    }
}

/// Metal 4 specific features.
impl Metal4Backend {
    /// Create a command allocator for per-frame memory management.
    ///
    /// This enables the triple-buffering pattern used by Metal 4.
    ///
    /// # Arguments
    ///
    /// * `max_frames_in_flight` - Maximum concurrent frames (typically 2-3)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let allocator = backend.create_command_allocator(3)?;
    ///
    /// // In render loop:
    /// allocator.reset(); // Start of frame
    /// let cmd = queue.create_command_buffer_with_allocator(&allocator)?;
    /// ```
    pub fn create_command_allocator(
        &self,
        max_frames_in_flight: usize,
    ) -> Result<Metal4CommandAllocator> {
        Metal4CommandAllocator::new(&self.device, max_frames_in_flight)
    }

    /// Create an argument table for faster buffer binding.
    ///
    /// MTL4ArgumentTable provides more efficient argument binding
    /// compared to repeated setBuffer calls.
    ///
    /// # Arguments
    ///
    /// * `desc` - Configuration for the argument table
    pub fn create_argument_table(
        &self,
        desc: ArgumentTableDescriptor,
    ) -> Result<Metal4ArgumentTable> {
        Metal4ArgumentTable::new(&self.device, desc)
    }

    /// Create an argument table with default settings and the given capacity.
    pub fn create_argument_table_with_capacity(
        &self,
        capacity: usize,
    ) -> Result<Metal4ArgumentTable> {
        Metal4ArgumentTable::with_capacity(&self.device, capacity)
    }

    /// Create a residency set for explicit memory management.
    ///
    /// MTLResidencySet allows explicit control over which resources
    /// are resident in GPU memory.
    pub fn create_residency_set(&self) -> Result<Metal4ResidencySet> {
        Metal4ResidencySet::new(&self.device)
    }

    /// Create a residency set builder for fluent construction.
    pub fn residency_set_builder(&self) -> ResidencySetBuilder {
        ResidencySetBuilder::new(&self.device)
    }

    /// Create an ML command encoder for efficient neural network inference.
    ///
    /// This uses the MTL4MachineLearningCommandEncoder when available.
    /// Currently returns an error as native Metal 4 ML encoding is not yet available.
    pub fn create_ml_encoder(&self) -> Result<Metal4MlEncoder> {
        if !is_available() {
            return Err(MetalError::creation_failed(
                "Metal4MlEncoder",
                "Metal 4 ML encoder requires macOS 26+ with native Metal 4 support",
            ));
        }
        Metal4MlEncoder::new(&self.device)
    }
}

/// Metal 4 ML command encoder.
///
/// Provides efficient neural network inference using MTL4MachineLearningCommandEncoder.
pub struct Metal4MlEncoder {
    // Placeholder - will wrap MTL4MachineLearningCommandEncoder when available
    _device: Arc<Metal4Device>,
}

impl Metal4MlEncoder {
    fn new(device: &Arc<Metal4Device>) -> Result<Self> {
        Ok(Self {
            _device: Arc::clone(device),
        })
    }

    /// Encode a neural network inference operation.
    pub fn encode_inference(
        &mut self,
        _weights: &Metal4Tensor,
        _input: &Metal4Tensor,
        _output: &mut Metal4Tensor,
    ) -> Result<()> {
        // TODO: Implement when Metal 4 APIs are available
        Err(MetalError::creation_failed(
            "encode_inference",
            "Metal 4 ML encoding not yet implemented",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal4_detection() {
        let features = Metal4Features::detect();
        println!("Metal 4 features: {}", features.summary());

        // On current systems (before macOS 26), Metal 4 should not be available
        // But our fallback implementations should still work
    }

    #[test]
    fn test_metal4_backend_creation() {
        // This should succeed with either native Metal 4 or fallbacks
        let result = Metal4Backend::new();
        if let Ok(backend) = result {
            if backend.is_native() {
                println!("Backend created with native Metal 4 support");
            } else {
                println!("Backend created successfully (using fallbacks)");
            }
        }
    }

    #[test]
    fn test_metal4_native_behavior() {
        // new_native() behavior depends on system capabilities
        let result = Metal4Backend::new_native();
        if is_available() {
            // On macOS 26+, native creation should succeed
            assert!(result.is_ok());
            println!("Native Metal 4 backend created successfully");
        } else {
            // On earlier systems, native creation should fail
            assert!(result.is_err());
            println!("Native Metal 4 not available (expected on pre-macOS 26)");
        }
    }

    #[test]
    fn test_metal_generation() {
        let gen = MetalGeneration::detect();
        println!("Detected Metal generation: {}", gen);
    }

    #[test]
    fn test_command_allocator_creation() {
        if let Ok(backend) = Metal4Backend::new() {
            let allocator = backend
                .create_command_allocator(3)
                .expect("Failed to create allocator");
            assert_eq!(allocator.max_frames_in_flight(), 3);
        }
    }

    #[test]
    fn test_argument_table_creation() {
        if let Ok(backend) = Metal4Backend::new() {
            let table = backend
                .create_argument_table_with_capacity(16)
                .expect("Failed to create argument table");
            assert_eq!(table.max_buffers(), 16);
        }
    }

    #[test]
    fn test_residency_set_creation() {
        if let Ok(backend) = Metal4Backend::new() {
            let set = backend
                .create_residency_set()
                .expect("Failed to create residency set");
            assert!(set.is_empty());
        }
    }

    #[test]
    fn test_barrier_creation() {
        let barrier = BarrierBuilder::new()
            .scope(crate::command::BarrierScope::Buffers)
            .build();

        assert_eq!(barrier.scope, crate::command::BarrierScope::Buffers);
        assert!(!barrier.has_resources());
    }
}
