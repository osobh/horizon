//! Tensor abstraction for ML operations.
//!
//! Tensors are multi-dimensional arrays optimized for ML workloads.
//! In Metal 4, MTLTensor is a native type. In Metal 3, we use
//! MPS matrices and manual implementations.

use crate::error::Result;

/// Tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point (half precision).
    Float16,
    /// 16-bit brain floating point.
    BFloat16,
    /// 32-bit signed integer.
    Int32,
    /// 16-bit signed integer.
    Int16,
    /// 8-bit signed integer.
    Int8,
    /// 32-bit unsigned integer.
    UInt32,
    /// 16-bit unsigned integer.
    UInt16,
    /// 8-bit unsigned integer.
    UInt8,
    /// Boolean.
    Bool,
}

impl TensorDType {
    /// Get the size in bytes for this dtype.
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDType::Float32 | TensorDType::Int32 | TensorDType::UInt32 => 4,
            TensorDType::Float16
            | TensorDType::BFloat16
            | TensorDType::Int16
            | TensorDType::UInt16 => 2,
            TensorDType::Int8 | TensorDType::UInt8 | TensorDType::Bool => 1,
        }
    }

    /// Check if this is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            TensorDType::Float32 | TensorDType::Float16 | TensorDType::BFloat16
        )
    }

    /// Check if this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            TensorDType::Int32
                | TensorDType::Int16
                | TensorDType::Int8
                | TensorDType::UInt32
                | TensorDType::UInt16
                | TensorDType::UInt8
        )
    }
}

/// Tensor usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorUsage {
    /// Tensor will be used in compute shaders.
    pub compute: bool,
    /// Tensor will be used in render shaders.
    pub render: bool,
    /// Tensor will be used in ML operations.
    pub machine_learning: bool,
}

impl Default for TensorUsage {
    fn default() -> Self {
        Self {
            compute: true,
            render: false,
            machine_learning: true,
        }
    }
}

impl TensorUsage {
    /// Create usage for compute only.
    pub fn compute() -> Self {
        Self {
            compute: true,
            render: false,
            machine_learning: false,
        }
    }

    /// Create usage for ML only.
    pub fn machine_learning() -> Self {
        Self {
            compute: false,
            render: false,
            machine_learning: true,
        }
    }

    /// Create usage for both compute and ML.
    pub fn compute_and_ml() -> Self {
        Self {
            compute: true,
            render: false,
            machine_learning: true,
        }
    }
}

/// Descriptor for creating a tensor.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Data type.
    pub dtype: TensorDType,
    /// Shape (dimensions).
    pub shape: Vec<usize>,
    /// Optional strides (if not specified, computed from shape).
    pub strides: Option<Vec<usize>>,
    /// Usage flags.
    pub usage: TensorUsage,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor.
    pub fn new(dtype: TensorDType, shape: Vec<usize>) -> Self {
        Self {
            dtype,
            shape,
            strides: None,
            usage: TensorUsage::default(),
            label: None,
        }
    }

    /// Create a 1D tensor descriptor.
    pub fn vector(dtype: TensorDType, len: usize) -> Self {
        Self::new(dtype, vec![len])
    }

    /// Create a 2D tensor descriptor (matrix).
    pub fn matrix(dtype: TensorDType, rows: usize, cols: usize) -> Self {
        Self::new(dtype, vec![rows, cols])
    }

    /// Create a 3D tensor descriptor.
    pub fn tensor3d(dtype: TensorDType, d0: usize, d1: usize, d2: usize) -> Self {
        Self::new(dtype, vec![d0, d1, d2])
    }

    /// Create a 4D tensor descriptor (batch of 3D tensors).
    pub fn tensor4d(dtype: TensorDType, d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Self::new(dtype, vec![d0, d1, d2, d3])
    }

    /// Set custom strides.
    pub fn with_strides(mut self, strides: Vec<usize>) -> Self {
        self.strides = Some(strides);
        self
    }

    /// Set usage flags.
    pub fn with_usage(mut self, usage: TensorUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Set a label for debugging.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Compute default strides (row-major).
    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// Get the effective strides.
    pub fn effective_strides(&self) -> Vec<usize> {
        self.strides
            .clone()
            .unwrap_or_else(|| self.default_strides())
    }
}

/// Trait for Metal tensors.
///
/// Tensors are optimized for ML operations and can be used
/// with Metal Performance Shaders or Metal 4's native ML encoder.
pub trait MetalTensor: Send + Sync {
    /// Get the tensor shape.
    fn shape(&self) -> &[usize];

    /// Get the tensor data type.
    fn dtype(&self) -> TensorDType;

    /// Get the rank (number of dimensions).
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements.
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get the size in bytes.
    fn size_bytes(&self) -> usize {
        self.numel() * self.dtype().size_bytes()
    }

    /// Get the GPU address.
    fn gpu_address(&self) -> u64;

    /// Copy data from a slice into the tensor.
    fn copy_from_slice<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<()>;

    /// Copy data from the tensor into a slice.
    fn copy_to_slice<T: bytemuck::Pod>(&self, data: &mut [T]) -> Result<()>;
}

/// Neural network layer weights.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Weight tensor.
    pub weights: TensorDescriptor,
    /// Optional bias tensor.
    pub bias: Option<TensorDescriptor>,
}

impl LayerWeights {
    /// Create weights for a linear layer.
    pub fn linear(
        dtype: TensorDType,
        in_features: usize,
        out_features: usize,
        with_bias: bool,
    ) -> Self {
        Self {
            weights: TensorDescriptor::matrix(dtype, out_features, in_features),
            bias: if with_bias {
                Some(TensorDescriptor::vector(dtype, out_features))
            } else {
                None
            },
        }
    }

    /// Create weights for a 2D convolution layer.
    pub fn conv2d(
        dtype: TensorDType,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        with_bias: bool,
    ) -> Self {
        Self {
            weights: TensorDescriptor::tensor4d(
                dtype,
                out_channels,
                in_channels,
                kernel_size.0,
                kernel_size.1,
            ),
            bias: if with_bias {
                Some(TensorDescriptor::vector(dtype, out_channels))
            } else {
                None
            },
        }
    }
}
