//! GPU backend detection.
//!
//! Automatically detects the best available GPU backend for the current system.

/// Available GPU backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// NVIDIA CUDA (Linux/Windows).
    Cuda,
    /// Apple Metal 3 (macOS 14+).
    Metal3,
    /// Apple Metal 4 (macOS 26+).
    Metal4,
    /// CPU fallback (no GPU acceleration).
    Cpu,
}

impl GpuBackend {
    /// Get a human-readable name for the backend.
    pub fn name(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA",
            GpuBackend::Metal3 => "Metal 3",
            GpuBackend::Metal4 => "Metal 4",
            GpuBackend::Cpu => "CPU",
        }
    }

    /// Check if this is a GPU-accelerated backend.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, GpuBackend::Cpu)
    }

    /// Check if this is a Metal backend.
    pub fn is_metal(&self) -> bool {
        matches!(self, GpuBackend::Metal3 | GpuBackend::Metal4)
    }

    /// Check if this is the CUDA backend.
    pub fn is_cuda(&self) -> bool {
        matches!(self, GpuBackend::Cuda)
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Detect the best available GPU backend for the current system.
///
/// Detection order:
/// 1. Metal 4 (if macOS 26+ and feature enabled)
/// 2. Metal 3 (if macOS and feature enabled)
/// 3. CUDA (if feature enabled and CUDA available)
/// 4. CPU fallback
pub fn detect_backend() -> GpuBackend {
    // Check for Metal 4 first (requires macOS 26+)
    #[cfg(all(target_os = "macos", feature = "metal4"))]
    {
        if is_metal4_available() {
            return GpuBackend::Metal4;
        }
    }

    // Check for Metal 3 (requires macOS 14+)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if is_metal3_available() {
            return GpuBackend::Metal3;
        }
    }

    // Check for CUDA
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            return GpuBackend::Cuda;
        }
    }

    // Fallback to CPU
    GpuBackend::Cpu
}

/// Check if any Metal backend is available.
pub fn is_metal_available() -> bool {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        is_metal3_available() || is_metal4_available()
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        false
    }
}

/// Check if Metal 3 is available.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn is_metal3_available() -> bool {
    stratoswarm_metal_core::metal3::is_available()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn is_metal3_available() -> bool {
    false
}

/// Check if Metal 4 is available.
#[cfg(all(target_os = "macos", feature = "metal4"))]
fn is_metal4_available() -> bool {
    stratoswarm_metal_core::metal4::is_available()
}

#[cfg(not(all(target_os = "macos", feature = "metal4")))]
fn is_metal4_available() -> bool {
    false
}

/// Check if CUDA is available.
///
/// **Current Status:** Always returns `false` (detection not implemented).
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // TODO(cuda): Implement CUDA detection.
        //
        // Detection should check:
        // 1. CUDA driver is installed (libcuda.so / nvcuda.dll)
        // 2. At least one CUDA-capable GPU is present
        // 3. Driver version meets minimum requirements
        //
        // Use cudarc::driver::CudaDevice::count() when available.
        false
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get all available backends on this system.
pub fn available_backends() -> Vec<GpuBackend> {
    let mut backends = Vec::new();

    #[cfg(all(target_os = "macos", feature = "metal4"))]
    {
        if is_metal4_available() {
            backends.push(GpuBackend::Metal4);
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if is_metal3_available() {
            backends.push(GpuBackend::Metal3);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            backends.push(GpuBackend::Cuda);
        }
    }

    // CPU is always available
    backends.push(GpuBackend::Cpu);

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(GpuBackend::Cuda.name(), "CUDA");
        assert_eq!(GpuBackend::Metal3.name(), "Metal 3");
        assert_eq!(GpuBackend::Metal4.name(), "Metal 4");
        assert_eq!(GpuBackend::Cpu.name(), "CPU");
    }

    #[test]
    fn test_backend_is_gpu() {
        assert!(GpuBackend::Cuda.is_gpu());
        assert!(GpuBackend::Metal3.is_gpu());
        assert!(GpuBackend::Metal4.is_gpu());
        assert!(!GpuBackend::Cpu.is_gpu());
    }

    #[test]
    fn test_backend_is_metal() {
        assert!(!GpuBackend::Cuda.is_metal());
        assert!(GpuBackend::Metal3.is_metal());
        assert!(GpuBackend::Metal4.is_metal());
        assert!(!GpuBackend::Cpu.is_metal());
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        // CPU should always be available
        assert!(backends.contains(&GpuBackend::Cpu));
    }

    #[test]
    fn test_detect_backend_returns_valid() {
        let backend = detect_backend();
        // Should return a valid backend
        assert!(matches!(
            backend,
            GpuBackend::Cuda | GpuBackend::Metal3 | GpuBackend::Metal4 | GpuBackend::Cpu
        ));
    }
}
