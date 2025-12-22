//! GPU runtime factory.
//!
//! Factory functions for creating backend-specific engines.

use crate::detect::GpuBackend;
use crate::error::{RuntimeError, Result};

/// Factory for creating GPU runtime components.
///
/// The factory automatically selects the best available backend
/// or allows manual backend selection.
pub struct GpuRuntimeFactory {
    /// The backend to use.
    backend: GpuBackend,
}

impl GpuRuntimeFactory {
    /// Create a factory using automatic backend detection.
    pub fn auto() -> Self {
        Self {
            backend: crate::detect_backend(),
        }
    }

    /// Create a factory with a specific backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the specified backend is not available.
    pub fn with_backend(backend: GpuBackend) -> Result<Self> {
        // Verify the backend is available
        match backend {
            GpuBackend::Metal3 => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    if !stratoswarm_metal_core::metal3::is_available() {
                        return Err(RuntimeError::backend_not_available("Metal 3"));
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err(RuntimeError::feature_not_enabled("metal"));
                }
            }
            GpuBackend::Metal4 => {
                #[cfg(all(target_os = "macos", feature = "metal4"))]
                {
                    if !stratoswarm_metal_core::metal4::is_available() {
                        return Err(RuntimeError::backend_not_available("Metal 4"));
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "metal4")))]
                {
                    return Err(RuntimeError::feature_not_enabled("metal4"));
                }
            }
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    if !crate::is_cuda_available() {
                        return Err(RuntimeError::backend_not_available("CUDA"));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(RuntimeError::feature_not_enabled("cuda"));
                }
            }
            GpuBackend::Cpu => {
                // CPU is always available
            }
        }

        Ok(Self { backend })
    }

    /// Get the selected backend.
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Check if GPU acceleration is available.
    pub fn has_gpu(&self) -> bool {
        self.backend.is_gpu()
    }

    /// Create a Metal 3 backend.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal 3 is not the selected backend.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn create_metal3_backend(&self) -> Result<stratoswarm_metal_core::metal3::Metal3Backend> {
        if self.backend != GpuBackend::Metal3 {
            return Err(RuntimeError::backend_not_available("Metal 3"));
        }

        stratoswarm_metal_core::metal3::Metal3Backend::new()
            .map_err(|e| RuntimeError::initialization_failed("Metal 3", e.to_string()))
    }

    /// Create a Metal 4 backend.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal 4 is not the selected backend.
    #[cfg(all(target_os = "macos", feature = "metal4"))]
    pub fn create_metal4_backend(&self) -> Result<stratoswarm_metal_core::metal4::Metal4Backend> {
        if self.backend != GpuBackend::Metal4 {
            return Err(RuntimeError::backend_not_available("Metal 4"));
        }

        stratoswarm_metal_core::metal4::Metal4Backend::new()
            .map_err(|e| RuntimeError::initialization_failed("Metal 4", e.to_string()))
    }
}

impl Default for GpuRuntimeFactory {
    fn default() -> Self {
        Self::auto()
    }
}

/// Configuration for GPU runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Preferred backend (None = auto-detect).
    pub preferred_backend: Option<GpuBackend>,
    /// Enable debug logging.
    pub debug: bool,
    /// Maximum buffer size hint (bytes).
    pub max_buffer_size: Option<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            debug: false,
            max_buffer_size: None,
        }
    }
}

impl RuntimeConfig {
    /// Create config with a preferred backend.
    pub fn with_backend(backend: GpuBackend) -> Self {
        Self {
            preferred_backend: Some(backend),
            ..Default::default()
        }
    }

    /// Enable debug mode.
    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Set maximum buffer size hint.
    pub fn with_max_buffer_size(mut self, size: u64) -> Self {
        self.max_buffer_size = Some(size);
        self
    }

    /// Create a factory from this config.
    pub fn create_factory(&self) -> Result<GpuRuntimeFactory> {
        if let Some(backend) = self.preferred_backend {
            GpuRuntimeFactory::with_backend(backend)
        } else {
            Ok(GpuRuntimeFactory::auto())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_auto() {
        let factory = GpuRuntimeFactory::auto();
        // Should detect some backend
        let backend = factory.backend();
        assert!(matches!(
            backend,
            GpuBackend::Metal3 | GpuBackend::Metal4 | GpuBackend::Cuda | GpuBackend::Cpu
        ));
    }

    #[test]
    fn test_factory_cpu_always_available() {
        let factory = GpuRuntimeFactory::with_backend(GpuBackend::Cpu);
        assert!(factory.is_ok());
        assert!(!factory.unwrap().has_gpu());
    }

    #[test]
    fn test_config_builder() {
        let config = RuntimeConfig::default()
            .with_debug()
            .with_max_buffer_size(1024 * 1024 * 1024);

        assert!(config.debug);
        assert_eq!(config.max_buffer_size, Some(1024 * 1024 * 1024));
    }

    #[test]
    fn test_config_with_backend() {
        let config = RuntimeConfig::with_backend(GpuBackend::Cpu);
        assert_eq!(config.preferred_backend, Some(GpuBackend::Cpu));

        let factory = config.create_factory();
        assert!(factory.is_ok());
    }
}
