//! Device-related utilities.
//!
//! This module provides utilities for working with Metal devices,
//! separate from the main backend trait.

use crate::backend::{DeviceInfo, GpuFamily};

/// Get the recommended threadgroup size for a given total thread count.
pub fn recommended_threadgroup_size(total_threads: u64, max_threads: u64) -> u64 {
    // Common threadgroup sizes that work well on Apple Silicon
    const SIZES: [u64; 5] = [1024, 512, 256, 128, 64];

    for &size in &SIZES {
        if size <= max_threads && total_threads >= size {
            return size;
        }
    }

    // Fall back to max threads or 32
    max_threads.min(32)
}

/// Get recommended 2D threadgroup dimensions.
pub fn recommended_threadgroup_size_2d(
    width: u64,
    height: u64,
    max_threads: u64,
) -> (u64, u64) {
    // Try common 2D sizes
    const SIZES: [(u64, u64); 4] = [(16, 16), (8, 8), (16, 8), (8, 16)];

    for &(w, h) in &SIZES {
        if w * h <= max_threads && width >= w && height >= h {
            return (w, h);
        }
    }

    // Fall back to 8x8 or smaller
    let side = (max_threads as f64).sqrt() as u64;
    (side.min(width), side.min(height))
}

/// Calculate the number of threadgroups needed.
pub fn calculate_threadgroups(total: u64, group_size: u64) -> u64 {
    (total + group_size - 1) / group_size
}

/// Calculate 2D threadgroup count.
pub fn calculate_threadgroups_2d(
    width: u64,
    height: u64,
    group_width: u64,
    group_height: u64,
) -> (u64, u64) {
    (
        (width + group_width - 1) / group_width,
        (height + group_height - 1) / group_height,
    )
}

/// Parse a GPU family from device information.
pub fn parse_gpu_family(name: &str) -> GpuFamily {
    let name_lower = name.to_lowercase();

    // Check for Apple Silicon
    if name_lower.contains("apple") {
        // Try to extract the generation number
        if let Some(pos) = name_lower.find('m') {
            let after_m = &name_lower[pos + 1..];
            if let Some(num) = after_m.chars().next().and_then(|c| c.to_digit(10)) {
                return GpuFamily::Apple(num as u8);
            }
        }
        return GpuFamily::Apple(1);
    }

    // Check for Mac family
    if name_lower.contains("mac") || name_lower.contains("amd") || name_lower.contains("intel") {
        return GpuFamily::Mac(2);
    }

    GpuFamily::Unknown
}

/// Device capabilities for feature queries.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device information.
    pub info: DeviceInfo,
    /// Supports ray tracing.
    pub ray_tracing: bool,
    /// Supports mesh shaders.
    pub mesh_shaders: bool,
    /// Supports 32-bit atomics in threadgroup memory.
    pub threadgroup_atomics_32: bool,
    /// Supports 64-bit atomics.
    pub atomics_64: bool,
    /// Supports SIMD-scoped operations.
    pub simd_scoped_operations: bool,
    /// Supports quad-scoped operations.
    pub quad_scoped_operations: bool,
    /// Maximum SIMD width.
    pub simd_width: u32,
}

impl DeviceCapabilities {
    /// Create capabilities for Apple Silicon.
    pub fn apple_silicon(generation: u8) -> Self {
        Self {
            info: DeviceInfo {
                name: format!("Apple M{}", generation),
                unified_memory: true,
                max_buffer_length: 1 << 30, // 1 GB default
                max_threads_per_threadgroup: 1024,
                max_threadgroup_memory_length: 32768,
                gpu_family: GpuFamily::Apple(generation),
            },
            ray_tracing: generation >= 3,
            mesh_shaders: generation >= 3,
            threadgroup_atomics_32: true,
            atomics_64: generation >= 2,
            simd_scoped_operations: true,
            quad_scoped_operations: true,
            simd_width: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_threadgroup_size() {
        assert_eq!(recommended_threadgroup_size(10000, 1024), 1024);
        assert_eq!(recommended_threadgroup_size(100, 1024), 64);
        assert_eq!(recommended_threadgroup_size(10000, 256), 256);
    }

    #[test]
    fn test_calculate_threadgroups() {
        assert_eq!(calculate_threadgroups(1000, 256), 4);
        assert_eq!(calculate_threadgroups(1024, 256), 4);
        assert_eq!(calculate_threadgroups(1025, 256), 5);
    }

    #[test]
    fn test_parse_gpu_family() {
        assert!(matches!(parse_gpu_family("Apple M3 Max"), GpuFamily::Apple(3)));
        assert!(matches!(parse_gpu_family("Apple M1"), GpuFamily::Apple(1)));
        assert!(matches!(parse_gpu_family("AMD Radeon Pro 5500M"), GpuFamily::Mac(_)));
    }
}
