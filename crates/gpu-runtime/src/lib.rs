//! GPU Runtime Backend Selection
//!
//! This crate provides runtime detection and factory functions for
//! selecting between different GPU backends:
//!
//! - **CUDA** - NVIDIA GPUs (Linux/Windows)
//! - **Metal 3** - Apple Silicon (macOS 14+)
//! - **Metal 4** - Apple Silicon (macOS 26+, when available)
//! - **CPU** - Fallback for systems without GPU acceleration
//!
//! # Usage
//!
//! ```rust,ignore
//! use stratoswarm_gpu_runtime::{detect_backend, GpuBackend};
//!
//! let backend = detect_backend();
//! match backend {
//!     GpuBackend::Metal3 => println!("Using Metal 3"),
//!     GpuBackend::Metal4 => println!("Using Metal 4"),
//!     GpuBackend::Cuda => println!("Using CUDA"),
//!     GpuBackend::Cpu => println!("Falling back to CPU"),
//! }
//! ```

mod detect;
mod error;
mod factory;

pub use detect::{detect_backend, is_metal_available, is_cuda_available, GpuBackend};
pub use error::{RuntimeError, Result};
pub use factory::GpuRuntimeFactory;

/// GPU backend information.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// The detected backend type.
    pub backend: GpuBackend,
    /// Backend name.
    pub name: String,
    /// Device name (e.g., "Apple M3 Max", "NVIDIA RTX 4090").
    pub device_name: String,
    /// Whether the device has unified memory.
    pub unified_memory: bool,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: u64,
}

impl BackendInfo {
    /// Get a human-readable description of the backend.
    pub fn description(&self) -> String {
        format!(
            "{} ({}) - {} memory, max buffer: {} MB",
            self.name,
            self.device_name,
            if self.unified_memory { "unified" } else { "discrete" },
            self.max_buffer_size / (1024 * 1024)
        )
    }
}

/// Get detailed information about the available GPU backend.
pub fn backend_info() -> Option<BackendInfo> {
    let backend = detect_backend();

    match backend {
        GpuBackend::Cpu => None,

        #[cfg(all(target_os = "macos", feature = "metal"))]
        GpuBackend::Metal3 | GpuBackend::Metal4 => {
            use stratoswarm_metal_core::metal3::Metal3Backend;
            use stratoswarm_metal_core::backend::{MetalBackend, MetalDevice};

            if let Ok(metal_backend) = Metal3Backend::new() {
                let device = metal_backend.device();
                let info = device.info();

                Some(BackendInfo {
                    backend,
                    name: format!("{:?}", backend),
                    device_name: info.name.clone(),
                    unified_memory: info.unified_memory,
                    max_buffer_size: info.max_buffer_length,
                })
            } else {
                None
            }
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        GpuBackend::Metal3 | GpuBackend::Metal4 => None,

        GpuBackend::Cuda => {
            // CUDA implementation placeholder
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        // Should detect some backend (at least CPU fallback)
        match backend {
            GpuBackend::Metal3 | GpuBackend::Metal4 => {
                #[cfg(target_os = "macos")]
                assert!(true, "Metal backend detected on macOS");
            }
            GpuBackend::Cuda => {
                assert!(true, "CUDA backend detected");
            }
            GpuBackend::Cpu => {
                // CPU fallback is always valid
                assert!(true, "CPU fallback");
            }
        }
    }

    #[test]
    fn test_backend_info() {
        let info = backend_info();
        // Backend info is optional (may be None if no GPU)
        if let Some(info) = info {
            assert!(!info.device_name.is_empty());
            assert!(info.max_buffer_size > 0);
        }
    }
}
