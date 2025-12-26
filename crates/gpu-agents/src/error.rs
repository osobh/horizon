//! GPU agent error types

use std::fmt;

#[derive(Debug)]
pub enum GpuAgentError {
    CudaError(cudarc::driver::DriverError),
    AllocationError(String),
    KernelLaunchError(String),
    InvalidConfiguration(String),
}

impl fmt::Display for GpuAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaError(e) => write!(f, "CUDA error: {}", e),
            Self::AllocationError(msg) => write!(f, "GPU allocation error: {}", msg),
            Self::KernelLaunchError(msg) => write!(f, "Kernel launch error: {}", msg),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
        }
    }
}

impl std::error::Error for GpuAgentError {}

impl From<cudarc::driver::DriverError> for GpuAgentError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::CudaError(e)
    }
}
