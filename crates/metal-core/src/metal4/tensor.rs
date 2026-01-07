//! Metal 4 native tensor support.
//!
//! Metal 4 introduces MTLTensor as a first-class type, replacing
//! the need for MPS matrices in many ML workloads.

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal4::{Metal4Buffer, Metal4Device};
use crate::tensor::{MetalTensor, TensorDType, TensorDescriptor};

/// Metal 4 native tensor.
///
/// When Metal 4 is available, this wraps an MTLTensor.
/// Until then, it provides a compatible API using regular buffers.
pub struct Metal4Tensor {
    /// The underlying buffer holding tensor data
    buffer: Metal4Buffer,
    /// Tensor descriptor (shape, dtype, layout)
    descriptor: TensorDescriptor,
}

impl Metal4Tensor {
    /// Create a new tensor with the given descriptor.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    /// * `desc` - Tensor descriptor specifying shape, dtype, layout
    ///
    /// # Errors
    ///
    /// Returns an error if tensor creation fails.
    pub fn new(device: &Metal4Device, desc: TensorDescriptor) -> Result<Self> {
        let size = desc.size_bytes();
        let buffer = Metal4Buffer::new(device, size)?;

        Ok(Self {
            buffer,
            descriptor: desc,
        })
    }

    /// Create a tensor from existing data.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal 4 device
    /// * `desc` - Tensor descriptor
    /// * `data` - Initial data (must match descriptor size)
    pub fn with_data<T: bytemuck::Pod>(
        device: &Metal4Device,
        desc: TensorDescriptor,
        data: &[T],
    ) -> Result<Self> {
        let expected_elements = desc.numel();
        if data.len() != expected_elements {
            return Err(MetalError::creation_failed(
                "Metal4Tensor",
                &format!(
                    "Data length {} does not match tensor shape {:?} (expected {})",
                    data.len(),
                    desc.shape,
                    expected_elements
                ),
            ));
        }

        let buffer = Metal4Buffer::with_data(device, data)?;

        Ok(Self {
            buffer,
            descriptor: desc,
        })
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &Metal4Buffer {
        &self.buffer
    }

    /// Get mutable access to the underlying buffer.
    pub fn buffer_mut(&mut self) -> &mut Metal4Buffer {
        &mut self.buffer
    }

    /// Get the tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Check if this is a native MTLTensor (Metal 4 only).
    ///
    /// **Current Status:** Always returns `false` (Metal 3 fallback mode).
    ///
    /// Native MTLTensor requires macOS 15+ with Metal 4 support.
    pub fn is_native_tensor(&self) -> bool {
        // FALLBACK: Always false until Metal 4 runtime detection is implemented.
        //
        // TODO(metal4): Add macOS version check:
        //   if #available(macOS 15, *) { /* check for MTLTensor support */ }
        //
        // When Metal 4 is available, this will return true if the tensor is backed
        // by a native MTLTensor rather than an MTLBuffer-based emulation.
        false
    }

    /// Get the stride for each dimension.
    pub fn strides(&self) -> Vec<usize> {
        self.descriptor.effective_strides()
    }

    /// Reshape the tensor (must have same total elements).
    ///
    /// This is a zero-copy operation that just changes the descriptor.
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let new_elements: usize = new_shape.iter().product();
        let old_elements = self.descriptor.numel();

        if new_elements != old_elements {
            return Err(MetalError::creation_failed(
                "Metal4Tensor::reshape",
                &format!(
                    "Cannot reshape {} elements to {} elements",
                    old_elements, new_elements
                ),
            ));
        }

        self.descriptor = TensorDescriptor::new(self.descriptor.dtype, new_shape);
        Ok(())
    }

    /// Transpose dimensions (for 2D tensors).
    ///
    /// This is a zero-copy operation that changes the layout.
    pub fn transpose(&mut self) -> Result<()> {
        if self.descriptor.shape.len() != 2 {
            return Err(MetalError::creation_failed(
                "Metal4Tensor::transpose",
                "Transpose only supported for 2D tensors",
            ));
        }

        let shape = self.descriptor.shape.clone();
        let new_shape = vec![shape[1], shape[0]];

        // Update strides for transposed layout
        let strides = vec![1, shape[1]];

        self.descriptor =
            TensorDescriptor::new(self.descriptor.dtype, new_shape).with_strides(strides);
        Ok(())
    }
}

impl MetalTensor for Metal4Tensor {
    fn shape(&self) -> &[usize] {
        &self.descriptor.shape
    }

    fn dtype(&self) -> TensorDType {
        self.descriptor.dtype
    }

    fn gpu_address(&self) -> u64 {
        self.buffer.gpu_address()
    }

    fn copy_from_slice<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<()> {
        let expected = self.descriptor.numel();
        if data.len() != expected {
            return Err(MetalError::creation_failed(
                "Metal4Tensor::copy_from_slice",
                &format!(
                    "Data length {} does not match tensor size {}",
                    data.len(),
                    expected
                ),
            ));
        }
        let contents = self.buffer.contents_mut::<T>();
        contents.copy_from_slice(data);
        Ok(())
    }

    fn copy_to_slice<T: bytemuck::Pod>(&self, data: &mut [T]) -> Result<()> {
        let expected = self.descriptor.numel();
        if data.len() < expected {
            return Err(MetalError::creation_failed(
                "Metal4Tensor::copy_to_slice",
                &format!(
                    "Destination length {} is smaller than tensor size {}",
                    data.len(),
                    expected
                ),
            ));
        }
        let contents = self.buffer.contents::<T>();
        data[..expected].copy_from_slice(contents);
        Ok(())
    }
}

// Safety: Tensors with shared storage can be accessed from any thread
unsafe impl Send for Metal4Tensor {}
unsafe impl Sync for Metal4Tensor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        if let Ok(device) = Metal4Device::system_default() {
            let desc = TensorDescriptor::matrix(TensorDType::Float32, 32, 64);
            let tensor = Metal4Tensor::new(&device, desc).expect("Failed to create tensor");

            assert_eq!(tensor.shape(), &[32, 64]);
            assert_eq!(tensor.dtype(), TensorDType::Float32);
            assert_eq!(tensor.numel(), 32 * 64);
            assert_eq!(tensor.size_bytes(), 32 * 64 * 4);
        }
    }

    #[test]
    fn test_tensor_with_data() {
        if let Ok(device) = Metal4Device::system_default() {
            let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
            let desc = TensorDescriptor::matrix(TensorDType::Float32, 10, 10);
            let tensor =
                Metal4Tensor::with_data(&device, desc, &data).expect("Failed to create tensor");

            assert_eq!(tensor.shape(), &[10, 10]);
            let mut output = vec![0.0f32; 100];
            tensor.copy_to_slice(&mut output).expect("Failed to copy");
            assert_eq!(output[0], 0.0);
            assert_eq!(output[99], 99.0);
        }
    }

    #[test]
    fn test_tensor_reshape() {
        if let Ok(device) = Metal4Device::system_default() {
            let desc = TensorDescriptor::matrix(TensorDType::Float32, 4, 8);
            let mut tensor = Metal4Tensor::new(&device, desc).expect("Failed to create tensor");

            tensor.reshape(vec![2, 16]).expect("Reshape failed");
            assert_eq!(tensor.shape(), &[2, 16]);

            tensor.reshape(vec![32]).expect("Reshape failed");
            assert_eq!(tensor.shape(), &[32]);

            // Invalid reshape should fail
            assert!(tensor.reshape(vec![64]).is_err());
        }
    }

    #[test]
    fn test_tensor_transpose() {
        if let Ok(device) = Metal4Device::system_default() {
            let desc = TensorDescriptor::matrix(TensorDType::Float32, 4, 8);
            let mut tensor = Metal4Tensor::new(&device, desc).expect("Failed to create tensor");

            tensor.transpose().expect("Transpose failed");
            assert_eq!(tensor.shape(), &[8, 4]);
        }
    }

    #[test]
    fn test_native_tensor_detection() {
        if let Ok(device) = Metal4Device::system_default() {
            let desc = TensorDescriptor::vector(TensorDType::Float32, 16);
            let tensor = Metal4Tensor::new(&device, desc).expect("Failed to create tensor");

            // Should be false until Metal 4 is available
            assert!(!tensor.is_native_tensor());
        }
    }
}
