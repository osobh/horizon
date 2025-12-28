//! Metal 4 runtime detection.
//!
//! Provides functions to detect Metal 4 availability and features at runtime.
//! This allows the library to use Metal 4 APIs when available while falling
//! back to Metal 3 on older systems.
//!
//! # Detection Strategy
//!
//! 1. **OS Version Check**: Metal 4 requires macOS 26+
//! 2. **Symbol Detection**: Check for MTL4* classes using Objective-C runtime
//! 3. **Device Queries**: Check device capabilities
//!
//! # Example
//!
//! ```ignore
//! if is_metal4_available() {
//!     // Use Metal 4 APIs
//! } else {
//!     // Fall back to Metal 3
//! }
//!
//! let features = Metal4Features::detect();
//! if features.has_native_tensors {
//!     // Use MTLTensor directly
//! }
//! ```

use std::sync::OnceLock;

/// Cached detection results.
static METAL4_FEATURES: OnceLock<Metal4Features> = OnceLock::new();

/// Detected Metal 4 feature set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metal4Features {
    /// macOS version (major, minor, patch)
    pub macos_version: (u32, u32, u32),
    /// Metal 4 APIs are available
    pub metal4_available: bool,
    /// Native MTLTensor is available
    pub has_native_tensors: bool,
    /// MTL4CommandAllocator is available
    pub has_command_allocator: bool,
    /// MTL4ArgumentTable is available
    pub has_argument_table: bool,
    /// MTLResidencySet is available
    pub has_residency_set: bool,
    /// MTL4MachineLearningCommandEncoder is available
    pub has_ml_encoder: bool,
    /// MTLTextureViewPool is available
    pub has_texture_view_pool: bool,
    /// Command buffer reuse is available
    pub has_command_buffer_reuse: bool,
    /// Batch command commit is available
    pub has_batch_commit: bool,
}

impl Default for Metal4Features {
    fn default() -> Self {
        Self {
            macos_version: (0, 0, 0),
            metal4_available: false,
            has_native_tensors: false,
            has_command_allocator: false,
            has_argument_table: false,
            has_residency_set: false,
            has_ml_encoder: false,
            has_texture_view_pool: false,
            has_command_buffer_reuse: false,
            has_batch_commit: false,
        }
    }
}

impl Metal4Features {
    /// Detect available features at runtime.
    ///
    /// This function is cached - calling it multiple times returns the same result.
    pub fn detect() -> &'static Self {
        METAL4_FEATURES.get_or_init(|| {
            let version = get_macos_version();
            let mut features = Self {
                macos_version: version,
                ..Default::default()
            };

            // Metal 4 requires macOS 26.0 or later
            if version.0 >= 26 {
                features.metal4_available = true;
                features.has_native_tensors = check_class_exists("MTLTensor");
                features.has_command_allocator = check_class_exists("MTL4CommandAllocator");
                features.has_argument_table = check_class_exists("MTL4ArgumentTable");
                features.has_residency_set = check_class_exists("MTLResidencySet");
                features.has_ml_encoder = check_class_exists("MTL4MachineLearningCommandEncoder");
                features.has_texture_view_pool = check_class_exists("MTLTextureViewPool");
                // These are methods, not classes - we'll check device capabilities instead
                features.has_command_buffer_reuse = features.has_command_allocator;
                features.has_batch_commit = features.has_command_allocator;
            }

            features
        })
    }

    /// Check if all Metal 4 features are available.
    pub fn has_all(&self) -> bool {
        self.metal4_available
            && self.has_native_tensors
            && self.has_command_allocator
            && self.has_argument_table
            && self.has_residency_set
    }

    /// Check if any Metal 4 features are available.
    pub fn has_any(&self) -> bool {
        self.metal4_available
            || self.has_native_tensors
            || self.has_command_allocator
    }

    /// Get a human-readable summary of available features.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.has_native_tensors {
            features.push("MTLTensor");
        }
        if self.has_command_allocator {
            features.push("MTL4CommandAllocator");
        }
        if self.has_argument_table {
            features.push("MTL4ArgumentTable");
        }
        if self.has_residency_set {
            features.push("MTLResidencySet");
        }
        if self.has_ml_encoder {
            features.push("MTL4MLEncoder");
        }
        if self.has_texture_view_pool {
            features.push("MTLTextureViewPool");
        }

        if features.is_empty() {
            format!(
                "Metal 4 not available (macOS {}.{}.{})",
                self.macos_version.0, self.macos_version.1, self.macos_version.2
            )
        } else {
            format!(
                "Metal 4 features on macOS {}.{}.{}: {}",
                self.macos_version.0,
                self.macos_version.1,
                self.macos_version.2,
                features.join(", ")
            )
        }
    }
}

/// Check if Metal 4 is available on this system.
///
/// This is a convenience function that checks the OS version.
pub fn is_metal4_available() -> bool {
    Metal4Features::detect().metal4_available
}

/// Check if native MTLTensor is available.
///
/// MTLTensor is a GPU-native tensor type introduced in Metal 4.
pub fn is_native_tensor_available() -> bool {
    Metal4Features::detect().has_native_tensors
}

/// Check if the MTL4CommandAllocator is available.
pub fn is_command_allocator_available() -> bool {
    Metal4Features::detect().has_command_allocator
}

/// Check if the MTL4ArgumentTable is available.
pub fn is_argument_table_available() -> bool {
    Metal4Features::detect().has_argument_table
}

/// Check if MTLResidencySet is available.
pub fn is_residency_set_available() -> bool {
    Metal4Features::detect().has_residency_set
}

/// Get the current macOS version.
fn get_macos_version() -> (u32, u32, u32) {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        // Use sw_vers to get the OS version
        if let Ok(output) = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
        {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                let parts: Vec<&str> = version_str.trim().split('.').collect();
                let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                return (major, minor, patch);
            }
        }

        // Fallback: Try using sysctl
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("kern.osproductversion")
            .output()
        {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                let parts: Vec<&str> = version_str.trim().split('.').collect();
                let major = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
                let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                return (major, minor, patch);
            }
        }

        (0, 0, 0)
    }

    #[cfg(not(target_os = "macos"))]
    {
        (0, 0, 0)
    }
}

/// Check if an Objective-C class exists at runtime.
fn check_class_exists(class_name: &str) -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;

        // Use objc_getClass to check if the class exists
        extern "C" {
            fn objc_getClass(name: *const std::ffi::c_char) -> *const std::ffi::c_void;
        }

        if let Ok(c_name) = CString::new(class_name) {
            let class = unsafe { objc_getClass(c_name.as_ptr()) };
            return !class.is_null();
        }

        false
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = class_name;
        false
    }
}

/// Metal API generation for compatibility tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalGeneration {
    /// Metal 1.x (iOS 8, macOS 10.11)
    Metal1,
    /// Metal 2.x (iOS 11, macOS 10.13)
    Metal2,
    /// Metal 3 (iOS 16, macOS 13)
    Metal3,
    /// Metal 4 (iOS 19, macOS 26)
    Metal4,
}

impl MetalGeneration {
    /// Detect the highest available Metal generation.
    pub fn detect() -> Self {
        let features = Metal4Features::detect();
        if features.metal4_available {
            MetalGeneration::Metal4
        } else if features.macos_version.0 >= 13 {
            MetalGeneration::Metal3
        } else if features.macos_version.0 >= 10 && features.macos_version.1 >= 13 {
            MetalGeneration::Metal2
        } else {
            MetalGeneration::Metal1
        }
    }

    /// Check if a specific generation is available.
    pub fn is_available(generation: MetalGeneration) -> bool {
        Self::detect() >= generation
    }

    /// Get the minimum macOS version for this generation.
    pub fn min_macos_version(&self) -> (u32, u32) {
        match self {
            MetalGeneration::Metal1 => (10, 11),
            MetalGeneration::Metal2 => (10, 13),
            MetalGeneration::Metal3 => (13, 0),
            MetalGeneration::Metal4 => (26, 0),
        }
    }
}

impl std::fmt::Display for MetalGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalGeneration::Metal1 => write!(f, "Metal 1"),
            MetalGeneration::Metal2 => write!(f, "Metal 2"),
            MetalGeneration::Metal3 => write!(f, "Metal 3"),
            MetalGeneration::Metal4 => write!(f, "Metal 4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        let features = Metal4Features::detect();
        // Just verify detection runs without panicking
        println!("Detected features: {}", features.summary());
    }

    #[test]
    fn test_macos_version() {
        let version = get_macos_version();
        // On macOS, we should get a valid version
        #[cfg(target_os = "macos")]
        {
            assert!(version.0 > 0, "Major version should be non-zero on macOS");
        }
        println!("macOS version: {}.{}.{}", version.0, version.1, version.2);
    }

    #[test]
    fn test_metal_generation_ordering() {
        assert!(MetalGeneration::Metal4 > MetalGeneration::Metal3);
        assert!(MetalGeneration::Metal3 > MetalGeneration::Metal2);
        assert!(MetalGeneration::Metal2 > MetalGeneration::Metal1);
    }

    #[test]
    fn test_metal_generation_detect() {
        let gen = MetalGeneration::detect();
        // On current systems, should be at least Metal 3
        #[cfg(target_os = "macos")]
        {
            let features = Metal4Features::detect();
            if features.macos_version.0 >= 13 {
                assert!(gen >= MetalGeneration::Metal3);
            }
        }
        println!("Detected Metal generation: {}", gen);
    }

    #[test]
    fn test_convenience_functions() {
        // These should all work without panicking
        let _ = is_metal4_available();
        let _ = is_native_tensor_available();
        let _ = is_command_allocator_available();
        let _ = is_argument_table_available();
        let _ = is_residency_set_available();
    }

    #[test]
    fn test_feature_summary() {
        let features = Metal4Features::detect();
        let summary = features.summary();
        assert!(!summary.is_empty());
        println!("{}", summary);
    }

    #[test]
    fn test_min_macos_version() {
        assert_eq!(MetalGeneration::Metal3.min_macos_version(), (13, 0));
        assert_eq!(MetalGeneration::Metal4.min_macos_version(), (26, 0));
    }

    #[test]
    fn test_class_check() {
        // NSObject should always exist
        #[cfg(target_os = "macos")]
        {
            assert!(check_class_exists("NSObject"));
            // A made-up class should not exist
            assert!(!check_class_exists("ThisClassDefinitelyDoesNotExist123"));
        }
    }

    #[test]
    fn test_default_features() {
        let features = Metal4Features::default();
        assert!(!features.metal4_available);
        assert!(!features.has_native_tensors);
        assert!(!features.has_any());
        assert!(!features.has_all());
    }
}
