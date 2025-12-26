//! CUDA integration for GPU container system
//!
//! This crate provides the foundation for CUDA operations including:
//! - CUDA toolkit detection and initialization
//! - Kernel loading and compilation (PTX/CUBIN)
//! - Memory management integration
//! - Stream management
//! - Error handling

#![warn(missing_docs)]

pub mod context;
pub mod detection;
pub mod error;
pub mod kernel;
pub mod memory;
pub mod stream;

pub use context::{Context, ContextFlags, ContextProperties};
pub use detection::{CudaToolkit, ToolkitInfo};
pub use error::{CudaError, CudaResult};
pub use kernel::{
    CompilationOptions, CompileOptions, Kernel, KernelArg, KernelAttributes, KernelMetadata,
    KernelModule, KernelSource, LaunchConfig, LaunchConfigBuilder, OccupancyInfo, ProfileInfo,
    SourceType,
};
pub use memory::{DeviceMemory, MemoryPool};
pub use stream::{Stream, StreamFlags};

/// Re-export common types
pub mod prelude {
    pub use crate::{
        CompilationOptions, CompileOptions, Context, ContextFlags, ContextProperties, CudaError,
        CudaResult, CudaToolkit, DeviceMemory, Kernel, KernelArg, KernelAttributes, KernelMetadata,
        KernelModule, KernelSource, LaunchConfig, LaunchConfigBuilder, MemoryPool, OccupancyInfo,
        ProfileInfo, SourceType, Stream, StreamFlags, ToolkitInfo,
    };
}

/// Initialize CUDA subsystem
pub fn init() -> CudaResult<()> {
    detection::initialize_cuda()
}

/// Get CUDA toolkit information
pub fn toolkit_info() -> CudaResult<ToolkitInfo> {
    detection::get_toolkit_info()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod edge_case_tests;

#[cfg(test)]
mod additional_tests;
