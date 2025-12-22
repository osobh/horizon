//! # StratoSwarm Metal Core
//!
//! Metal GPU backend abstraction layer for StratoSwarm.
//! Provides a unified API over Metal 3 (stable) and Metal 4 (feature-gated).
//!
//! ## Features
//!
//! - `metal3` (default): Stable Metal 3 backend for macOS 14+
//! - `metal4`: Metal 4 backend for macOS 26+ (when available)
//!
//! ## Architecture
//!
//! The crate provides trait-based abstractions that work across Metal versions:
//!
//! - [`MetalBackend`]: Main entry point for GPU operations
//! - [`MetalBuffer`]: GPU memory buffer with unified memory access
//! - [`MetalCommandQueue`]: Command submission and synchronization
//! - [`MetalComputePipeline`]: Compute shader execution
//!
//! ## Example
//!
//! ```ignore
//! use stratoswarm_metal_core::{Metal3Backend, MetalBackend};
//!
//! let backend = Metal3Backend::new()?;
//! let buffer = backend.create_buffer::<f32>(1024)?;
//!
//! // Unified memory - direct CPU access
//! {
//!     let data = buffer.contents_mut();
//!     for i in 0..1024 {
//!         data[i] = i as f32;
//!     }
//! }
//!
//! // GPU compute...
//! ```

#![cfg(target_os = "macos")]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod backend;
pub mod buffer;
pub mod command;
pub mod compute;
pub mod device;
pub mod error;
pub mod sync;
pub mod tensor;

#[cfg(feature = "metal3")]
pub mod metal3;

#[cfg(feature = "metal4")]
pub mod metal4;

// Re-export core traits
pub use backend::{MetalBackend, MetalDevice};
pub use buffer::MetalBuffer;
pub use command::{MetalCommandBuffer, MetalCommandQueue, MetalComputeEncoder};
pub use compute::MetalComputePipeline;
pub use error::{MetalError, Result};
pub use sync::MetalSync;
pub use tensor::{MetalTensor, TensorDescriptor, TensorDType};

// Re-export default backend
#[cfg(feature = "metal3")]
pub use metal3::Metal3Backend;

#[cfg(feature = "metal4")]
pub use metal4::Metal4Backend;

/// Check if Metal is available on this system.
pub fn is_metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Try to get the default Metal device
        metal3::is_available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Get the best available Metal backend.
#[cfg(feature = "metal3")]
pub fn create_default_backend() -> Result<Metal3Backend> {
    Metal3Backend::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        // This will only pass on macOS with Metal support
        #[cfg(target_os = "macos")]
        {
            let available = is_metal_available();
            println!("Metal available: {}", available);
            // Don't assert - CI may not have Metal
        }
    }
}
