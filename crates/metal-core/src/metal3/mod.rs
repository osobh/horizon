//! Metal 3 backend implementation.
//!
//! This module provides the stable Metal 3 backend for macOS 14+.
//! It uses `objc2-metal` for Rust bindings to the Metal API.

mod device;
mod buffer;
mod command;
mod compute;
mod tensor;

pub use device::Metal3Device;
pub use buffer::Metal3Buffer;
pub use command::{Metal3CommandQueue, Metal3CommandBuffer, Metal3ComputeEncoder};
pub use compute::Metal3ComputePipeline;
pub use tensor::Metal3Tensor;

use crate::backend::{MetalBackend, MetalDevice, MetalVersion};
use crate::buffer::MetalBuffer;
use crate::error::Result;
use crate::tensor::TensorDescriptor;

use std::sync::Arc;

use objc2_metal::MTLCreateSystemDefaultDevice;

/// Check if Metal 3 is available on this system.
pub fn is_available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// Metal 3 backend implementation.
///
/// This is the primary entry point for GPU operations using Metal 3.
/// It works on macOS 14+ and iOS 17+.
#[derive(Clone)]
pub struct Metal3Backend {
    device: Arc<Metal3Device>,
}

impl Metal3Backend {
    /// Create a new Metal 3 backend using the default system device.
    pub fn new() -> Result<Self> {
        let device = Metal3Device::system_default()?;
        Ok(Self {
            device: Arc::new(device),
        })
    }

    /// Create a backend from an existing device.
    pub fn from_device(device: Metal3Device) -> Self {
        Self {
            device: Arc::new(device),
        }
    }

    /// Get the raw Metal device.
    pub fn raw_device(&self) -> &Metal3Device {
        &self.device
    }
}

impl MetalBackend for Metal3Backend {
    type Buffer = Metal3Buffer;
    type CommandQueue = Metal3CommandQueue;
    type ComputePipeline = Metal3ComputePipeline;
    type Tensor = Metal3Tensor;

    fn device(&self) -> &dyn MetalDevice {
        self.device.as_ref()
    }

    fn create_buffer<T: bytemuck::Pod>(&self, count: usize) -> Result<Self::Buffer> {
        Metal3Buffer::new(&self.device, count * std::mem::size_of::<T>())
    }

    fn create_buffer_with_data<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Self::Buffer> {
        let mut buffer = self.create_buffer::<T>(data.len())?;
        buffer.copy_from_slice(data);
        Ok(buffer)
    }

    fn create_command_queue(&self) -> Result<Self::CommandQueue> {
        Metal3CommandQueue::new(&self.device)
    }

    fn create_compute_pipeline(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<Self::ComputePipeline> {
        Metal3ComputePipeline::from_source(&self.device, source, function_name)
    }

    fn create_compute_pipeline_from_library(
        &self,
        library_data: &[u8],
        function_name: &str,
    ) -> Result<Self::ComputePipeline> {
        Metal3ComputePipeline::from_library_data(&self.device, library_data, function_name)
    }

    fn create_tensor(&self, desc: TensorDescriptor) -> Result<Self::Tensor> {
        Metal3Tensor::new(&self.device, desc)
    }

    fn synchronize(&self) -> Result<()> {
        // Metal 3 doesn't have a global sync, we sync per command queue
        Ok(())
    }

    fn metal_version(&self) -> MetalVersion {
        MetalVersion::Metal3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal3_availability() {
        let available = is_available();
        println!("Metal 3 available: {}", available);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_create_backend() {
        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = Metal3Backend::new();
        assert!(backend.is_ok(), "Failed to create Metal 3 backend");

        let backend = backend.unwrap();
        println!("Device: {}", backend.device().name());
    }
}
