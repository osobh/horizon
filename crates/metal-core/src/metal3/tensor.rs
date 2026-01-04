//! Metal 3 tensor implementation.
//!
//! Since Metal 3 doesn't have native MTLTensor, we implement
//! tensors using buffers with shape/stride metadata.

use crate::buffer::MetalBuffer;
use crate::error::{MetalError, Result};
use crate::metal3::{Metal3Buffer, Metal3Device};
use crate::tensor::{MetalTensor, TensorDType, TensorDescriptor};

/// Metal 3 tensor implementation.
///
/// Uses a buffer-backed approach since Metal 3 lacks native tensors.
pub struct Metal3Tensor {
    buffer: Metal3Buffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: TensorDType,
}

impl Metal3Tensor {
    /// Create a new tensor.
    pub fn new(device: &Metal3Device, desc: TensorDescriptor) -> Result<Self> {
        let size_bytes = desc.size_bytes();
        let buffer = Metal3Buffer::new(device, size_bytes)?;

        let strides = desc.effective_strides();

        Ok(Self {
            buffer,
            shape: desc.shape,
            strides,
            dtype: desc.dtype,
        })
    }

    /// Create a tensor initialized with data.
    pub fn with_data<T: bytemuck::Pod>(
        device: &Metal3Device,
        desc: TensorDescriptor,
        data: &[T],
    ) -> Result<Self> {
        if data.len() != desc.numel() {
            return Err(MetalError::InvalidTensorDescriptor(format!(
                "Data length {} doesn't match tensor size {}",
                data.len(),
                desc.numel()
            )));
        }

        let buffer = Metal3Buffer::with_data(device, data)?;
        let strides = desc.effective_strides();

        Ok(Self {
            buffer,
            shape: desc.shape,
            strides,
            dtype: desc.dtype,
        })
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &Metal3Buffer {
        &self.buffer
    }

    /// Get a mutable reference to the underlying buffer.
    pub fn buffer_mut(&mut self) -> &mut Metal3Buffer {
        &mut self.buffer
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Calculate the linear index for a multi-dimensional index.
    pub fn linear_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum()
    }

    /// Reshape the tensor (view only, no data copy).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<TensorView<'_>> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(MetalError::InvalidTensorDescriptor(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                self.numel(),
                new_shape,
                new_numel
            )));
        }

        // Compute new strides (row-major)
        let mut new_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        Ok(TensorView {
            tensor: self,
            shape: new_shape,
            strides: new_strides,
        })
    }
}

impl MetalTensor for Metal3Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> TensorDType {
        self.dtype
    }

    fn gpu_address(&self) -> u64 {
        self.buffer.gpu_address()
    }

    fn copy_from_slice<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<()> {
        let expected_len = self.numel();
        if data.len() != expected_len {
            return Err(MetalError::BufferSizeMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }

        self.buffer.copy_from_slice(data);
        Ok(())
    }

    fn copy_to_slice<T: bytemuck::Pod>(&self, data: &mut [T]) -> Result<()> {
        let expected_len = self.numel();
        if data.len() != expected_len {
            return Err(MetalError::BufferSizeMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }

        self.buffer.copy_to_slice(data);
        Ok(())
    }
}

impl std::fmt::Debug for Metal3Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal3Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("dtype", &self.dtype)
            .field("size_bytes", &self.size_bytes())
            .finish()
    }
}

/// A view into a tensor with different shape.
pub struct TensorView<'a> {
    tensor: &'a Metal3Tensor,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    /// Get the shape of this view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of this view.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &Metal3Tensor {
        self.tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_tensor_creation() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let desc = TensorDescriptor::matrix(TensorDType::Float32, 4, 4);
        let tensor = Metal3Tensor::new(&device, desc).unwrap();

        assert_eq!(tensor.shape(), &[4, 4]);
        assert_eq!(tensor.numel(), 16);
        assert_eq!(tensor.dtype(), TensorDType::Float32);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_tensor_with_data() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let desc = TensorDescriptor::vector(TensorDType::Float32, 8);
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();

        let tensor = Metal3Tensor::with_data(&device, desc, &data).unwrap();

        let mut output = vec![0.0f32; 8];
        tensor.copy_to_slice(&mut output).unwrap();

        assert_eq!(output, data);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_tensor_reshape() {
        use crate::metal3::is_available;

        if !is_available() {
            println!("Skipping test - Metal not available");
            return;
        }

        let device = Metal3Device::system_default().unwrap();
        let desc = TensorDescriptor::matrix(TensorDType::Float32, 4, 6);
        let tensor = Metal3Tensor::new(&device, desc).unwrap();

        let view = tensor.reshape(vec![2, 12]).unwrap();
        assert_eq!(view.shape(), &[2, 12]);

        let view = tensor.reshape(vec![24]).unwrap();
        assert_eq!(view.shape(), &[24]);
    }
}
