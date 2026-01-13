//! CUDA toolkit detection and initialization

#[cfg(not(cuda_mock))]
use crate::error::CudaError;
use crate::error::CudaResult;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();
static mut INITIALIZED: bool = false;

/// CUDA toolkit version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaVersion {
    /// Major version number
    pub major: u32,
    /// Minor version number  
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
}

impl CudaVersion {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version meets minimum requirements
    pub fn meets_minimum(&self, min_major: u32, min_minor: u32) -> bool {
        self.major > min_major || (self.major == min_major && self.minor >= min_minor)
    }
}

impl std::fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Information about detected CUDA toolkit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolkitInfo {
    /// CUDA version
    pub version: CudaVersion,
    /// Installation path
    pub path: PathBuf,
    /// Number of available devices
    pub device_count: u32,
    /// Whether toolkit is in mock mode
    pub is_mock: bool,
    /// Driver version string
    pub driver_version: String,
}

/// CUDA toolkit detection and management
pub struct CudaToolkit {
    info: Option<ToolkitInfo>,
}

impl CudaToolkit {
    /// Create a new toolkit detector
    pub fn new() -> Self {
        Self { info: None }
    }

    /// Detect CUDA installation
    pub fn detect(&mut self) -> CudaResult<&ToolkitInfo> {
        if self.info.is_some() {
            return Ok(self.info.as_ref().unwrap());
        }

        #[cfg(cuda_mock)]
        {
            self.info = Some(Self::create_mock_info());
            Ok(self.info.as_ref().unwrap())
        }

        #[cfg(not(cuda_mock))]
        {
            self.detect_real_cuda()
        }
    }

    /// Create mock toolkit info for testing
    #[cfg(cuda_mock)]
    fn create_mock_info() -> ToolkitInfo {
        ToolkitInfo {
            version: CudaVersion::new(12, 0, 0),
            path: PathBuf::from("/mock/cuda"),
            device_count: 4,
            is_mock: true,
            driver_version: "535.154.05".to_string(),
        }
    }

    /// Detect real CUDA installation
    #[cfg(not(cuda_mock))]
    fn detect_real_cuda(&mut self) -> CudaResult<&ToolkitInfo> {
        use std::process::Command;

        // Try to run nvcc --version
        let output = Command::new("nvcc")
            .arg("--version")
            .output()
            .map_err(|_| CudaError::ToolkitNotFound)?;

        if !output.status.success() {
            return Err(CudaError::ToolkitNotFound);
        }

        // Parse version from output
        let version_str = String::from_utf8_lossy(&output.stdout);
        let version = Self::parse_nvcc_version(&version_str)?;

        // Get CUDA path from environment or nvcc location
        let cuda_path = Self::find_cuda_path()?;

        // Get device count (would use CUDA API in real implementation)
        let device_count = Self::get_device_count()?;

        // Get driver version
        let driver_version = Self::get_driver_version()?;

        self.info = Some(ToolkitInfo {
            version,
            path: cuda_path,
            device_count,
            is_mock: false,
            driver_version,
        });

        Ok(self.info.as_ref().unwrap())
    }

    #[cfg(not(cuda_mock))]
    fn parse_nvcc_version(output: &str) -> CudaResult<CudaVersion> {
        // Example: "Cuda compilation tools, release 12.0, V12.0.140"
        let version_line = output
            .lines()
            .find(|line| line.contains("release"))
            .ok_or_else(|| CudaError::InitializationFailed {
                message: "Could not parse nvcc version".to_string(),
            })?;

        let parts: Vec<&str> = version_line.split(',').collect();
        if parts.len() < 2 {
            return Err(CudaError::InitializationFailed {
                message: "Invalid nvcc version format".to_string(),
            });
        }

        let release_part = parts[1].trim();
        let version_str = release_part.strip_prefix("release ").ok_or_else(|| {
            CudaError::InitializationFailed {
                message: "Invalid release format".to_string(),
            }
        })?;

        let version_parts: Vec<&str> = version_str.split('.').collect();
        if version_parts.len() < 2 {
            return Err(CudaError::InitializationFailed {
                message: "Invalid version number format".to_string(),
            });
        }

        let major = version_parts[0]
            .parse()
            .map_err(|_| CudaError::InitializationFailed {
                message: "Invalid major version".to_string(),
            })?;

        let minor = version_parts[1]
            .parse()
            .map_err(|_| CudaError::InitializationFailed {
                message: "Invalid minor version".to_string(),
            })?;

        Ok(CudaVersion::new(major, minor, 0))
    }

    #[cfg(not(cuda_mock))]
    fn find_cuda_path() -> CudaResult<PathBuf> {
        // Check environment variables
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            return Ok(PathBuf::from(cuda_path));
        }

        if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
            return Ok(PathBuf::from(cuda_home));
        }

        // Check common paths
        let common_paths = ["/usr/local/cuda", "/opt/cuda"];

        for path in &common_paths {
            let cuda_path = PathBuf::from(path);
            if cuda_path.exists() {
                return Ok(cuda_path);
            }
        }

        Err(CudaError::ToolkitNotFound)
    }

    #[cfg(not(cuda_mock))]
    fn get_device_count() -> CudaResult<u32> {
        // In real implementation, would use cudaGetDeviceCount
        // For now, return 1 as placeholder
        Ok(1)
    }

    #[cfg(not(cuda_mock))]
    fn get_driver_version() -> CudaResult<String> {
        // In real implementation, would use cudaDriverGetVersion
        Ok("Unknown".to_string())
    }
}

/// Initialize CUDA subsystem
pub fn initialize_cuda() -> CudaResult<()> {
    let mut result = Ok(());

    INIT.call_once(|| {
        #[cfg(cuda_mock)]
        {
            // SAFETY: INITIALIZED is only written inside call_once, which guarantees
            // single-threaded execution. After call_once completes, INITIALIZED is
            // only read, never written again.
            unsafe {
                INITIALIZED = true;
            }
        }

        #[cfg(not(cuda_mock))]
        {
            // In real implementation, would call cudaInit
            let mut toolkit = CudaToolkit::new();
            match toolkit.detect() {
                Ok(info) => {
                    if !info.version.meets_minimum(11, 0) {
                        result = Err(CudaError::UnsupportedVersion {
                            version: info.version.to_string(),
                        });
                    } else {
                        // SAFETY: INITIALIZED is only written inside call_once, which
                        // guarantees single-threaded execution. No data races possible.
                        unsafe {
                            INITIALIZED = true;
                        }
                    }
                }
                Err(e) => result = Err(e),
            }
        }
    });

    result
}

/// Get CUDA toolkit information
pub fn get_toolkit_info() -> CudaResult<ToolkitInfo> {
    // SAFETY: INITIALIZED is only written inside call_once (single-threaded).
    // Reading it here may race with a concurrent initialize_cuda(), but the
    // worst case is we call initialize_cuda() redundantly (which is idempotent).
    if !unsafe { INITIALIZED } {
        initialize_cuda()?;
    }

    let mut toolkit = CudaToolkit::new();
    toolkit.detect().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_version() {
        let version = CudaVersion::new(12, 0, 1);
        assert_eq!(version.major, 12);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 1);
        assert_eq!(version.to_string(), "12.0.1");
    }

    #[test]
    fn test_version_meets_minimum() {
        let version = CudaVersion::new(12, 0, 0);
        assert!(version.meets_minimum(11, 0));
        assert!(version.meets_minimum(12, 0));
        assert!(!version.meets_minimum(12, 1));
        assert!(!version.meets_minimum(13, 0));
    }

    #[test]
    fn test_toolkit_detection() {
        let mut toolkit = CudaToolkit::new();

        #[cfg(cuda_mock)]
        {
            let info = toolkit.detect()?;
            assert_eq!(info.version.major, 12);
            assert_eq!(info.device_count, 4);
            assert!(info.is_mock);
            assert_eq!(info.path, PathBuf::from("/mock/cuda"));
        }
    }

    #[test]
    fn test_initialize_cuda() {
        #[cfg(cuda_mock)]
        {
            assert!(initialize_cuda().is_ok());
            assert!(unsafe { INITIALIZED });
        }
    }

    #[test]
    fn test_get_toolkit_info() {
        #[cfg(cuda_mock)]
        {
            let info = get_toolkit_info().unwrap();
            assert_eq!(info.version.major, 12);
            assert!(info.is_mock);
        }
    }

    #[test]
    #[cfg(not(cuda_mock))]
    fn test_parse_nvcc_version() {
        let output = r#"nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0"#;

        let version = CudaToolkit::parse_nvcc_version(output)?;
        assert_eq!(version.major, 12);
        assert_eq!(version.minor, 2);
    }
}
