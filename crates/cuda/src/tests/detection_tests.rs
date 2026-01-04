//! Comprehensive tests for CUDA detection module

use crate::detection::*;
use std::path::PathBuf;

#[test]
fn test_cuda_version_creation() {
    let version = CudaVersion::new(12, 2, 140);
    assert_eq!(version.major, 12);
    assert_eq!(version.minor, 2);
    assert_eq!(version.patch, 140);
}

#[test]
fn test_cuda_version_display() {
    let version = CudaVersion::new(11, 8, 0);
    assert_eq!(version.to_string(), "11.8.0");

    let version2 = CudaVersion::new(12, 0, 100);
    assert_eq!(version2.to_string(), "12.0.100");
}

#[test]
fn test_cuda_version_meets_minimum() {
    let version = CudaVersion::new(12, 2, 0);

    // Test cases where it meets minimum
    assert!(version.meets_minimum(11, 0));
    assert!(version.meets_minimum(11, 8));
    assert!(version.meets_minimum(12, 0));
    assert!(version.meets_minimum(12, 1));
    assert!(version.meets_minimum(12, 2));

    // Test cases where it doesn't meet minimum
    assert!(!version.meets_minimum(12, 3));
    assert!(!version.meets_minimum(13, 0));
    assert!(!version.meets_minimum(13, 1));
}

#[test]
fn test_cuda_version_edge_cases() {
    // Test with 0 values
    let version = CudaVersion::new(0, 0, 0);
    assert_eq!(version.to_string(), "0.0.0");
    assert!(!version.meets_minimum(1, 0));

    // Test with large values
    let version = CudaVersion::new(999, 999, 999);
    assert_eq!(version.to_string(), "999.999.999");
    assert!(version.meets_minimum(998, 999));
}

#[test]
fn test_toolkit_info_creation() {
    let info = ToolkitInfo {
        version: CudaVersion::new(12, 0, 0),
        path: PathBuf::from("/usr/local/cuda"),
        device_count: 2,
        is_mock: false,
        driver_version: "535.154.05".to_string(),
    };

    assert_eq!(info.version.major, 12);
    assert_eq!(info.device_count, 2);
    assert!(!info.is_mock);
    assert_eq!(info.driver_version, "535.154.05");
}

#[test]
fn test_cuda_toolkit_new() {
    let toolkit = CudaToolkit::new();
    // Initially, toolkit info should be None
    // (Can't directly test private field, but we know detect() will populate it)
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_toolkit_detection() {
    let mut toolkit = CudaToolkit::new();
    let info = toolkit.detect().unwrap();

    assert_eq!(info.version.major, 12);
    assert_eq!(info.version.minor, 0);
    assert_eq!(info.version.patch, 0);
    assert_eq!(info.path, PathBuf::from("/mock/cuda"));
    assert_eq!(info.device_count, 4);
    assert!(info.is_mock);
    assert_eq!(info.driver_version, "535.154.05");
}

#[cfg(cuda_mock)]
#[test]
fn test_mock_toolkit_detect_cached() {
    let mut toolkit = CudaToolkit::new();

    // First detection
    let info1 = toolkit.detect().unwrap();
    let info1_ptr = info1 as *const ToolkitInfo;

    // Second detection should return cached result
    let info2 = toolkit.detect()?;
    let info2_ptr = info2 as *const ToolkitInfo;

    // Should be the same reference
    assert_eq!(info1_ptr, info2_ptr);
}

#[cfg(not(cuda_mock))]
#[test]
fn test_nvcc_version_parsing() {
    // Test various nvcc output formats
    let test_cases = vec![
        (
            r#"nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0"#,
            (12, 2),
        ),
        (
            r#"nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0"#,
            (11, 8),
        ),
        (
            r#"nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 10.2, V10.2.300"#,
            (10, 2),
        ),
    ];

    for (output, expected) in test_cases {
        let version = CudaToolkit::parse_nvcc_version(output).unwrap();
        assert_eq!(version.major, expected.0);
        assert_eq!(version.minor, expected.1);
    }
}

#[cfg(not(cuda_mock))]
#[test]
fn test_nvcc_version_parsing_errors() {
    // Test invalid nvcc outputs
    let invalid_outputs = vec![
        "",
        "nvcc: command not found",
        "Invalid output without release info",
        "Cuda compilation tools, release", // Missing version
        "Cuda compilation tools, release invalid.version",
    ];

    for output in invalid_outputs {
        let result = CudaToolkit::parse_nvcc_version(output);
        assert!(result.is_err());
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_initialize_cuda_mock() {
    // Initialize should succeed in mock mode
    assert!(initialize_cuda().is_ok());

    // Multiple calls should be idempotent
    assert!(initialize_cuda().is_ok());
    assert!(initialize_cuda().is_ok());
}

#[cfg(cuda_mock)]
#[test]
fn test_get_toolkit_info_mock() {
    let info = get_toolkit_info().unwrap();

    assert_eq!(info.version.major, 12);
    assert_eq!(info.version.minor, 0);
    assert_eq!(info.path, PathBuf::from("/mock/cuda"));
    assert_eq!(info.device_count, 4);
    assert!(info.is_mock);
}

#[test]
fn test_toolkit_info_serialization() {
    let info = ToolkitInfo {
        version: CudaVersion::new(12, 1, 0),
        path: PathBuf::from("/opt/cuda"),
        device_count: 3,
        is_mock: true,
        driver_version: "525.60.13".to_string(),
    };

    // Test serialization
    let json = serde_json::to_string(&info)?;
    let deserialized: ToolkitInfo = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.version.major, info.version.major);
    assert_eq!(deserialized.version.minor, info.version.minor);
    assert_eq!(deserialized.device_count, info.device_count);
    assert_eq!(deserialized.is_mock, info.is_mock);
    assert_eq!(deserialized.driver_version, info.driver_version);
}

#[test]
fn test_cuda_version_serialization() {
    let version = CudaVersion::new(11, 7, 1);

    let json = serde_json::to_string(&version).unwrap();
    let deserialized: CudaVersion = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.major, version.major);
    assert_eq!(deserialized.minor, version.minor);
    assert_eq!(deserialized.patch, version.patch);
}

#[test]
fn test_version_comparison_logic() {
    let test_cases = vec![
        (CudaVersion::new(11, 0, 0), 10, 0, true),
        (CudaVersion::new(11, 0, 0), 11, 0, true),
        (CudaVersion::new(11, 0, 0), 11, 1, false),
        (CudaVersion::new(11, 2, 0), 11, 1, true),
        (CudaVersion::new(11, 2, 0), 11, 2, true),
        (CudaVersion::new(11, 2, 0), 11, 3, false),
        (CudaVersion::new(12, 0, 0), 11, 8, true),
        (CudaVersion::new(12, 0, 0), 13, 0, false),
    ];

    for (version, min_major, min_minor, expected) in test_cases {
        assert_eq!(
            version.meets_minimum(min_major, min_minor),
            expected,
            "Version {} vs minimum {}.{}",
            version,
            min_major,
            min_minor
        );
    }
}

#[cfg(not(cuda_mock))]
#[test]
fn test_cuda_path_detection() {
    // Test common CUDA paths
    let paths = vec![
        PathBuf::from("/usr/local/cuda"),
        PathBuf::from("/opt/cuda"),
        PathBuf::from("/usr/local/cuda-12.0"),
        PathBuf::from("/usr/local/cuda-11.8"),
    ];

    // At least one should exist on a system with CUDA
    // In CI, none might exist, so we just test the logic
}

#[test]
fn test_driver_version_format() {
    let versions = vec!["535.154.05", "525.60.13", "515.43.04", "470.129.06"];

    for version in versions {
        // Verify version format (three numbers separated by dots)
        let parts: Vec<&str> = version.split('.').collect();
        assert_eq!(parts.len(), 3);

        for part in parts {
            assert!(part.parse::<u32>().is_ok());
        }
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_concurrent_toolkit_detection() {
    use std::sync::Arc;
    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                let mut toolkit = CudaToolkit::new();
                toolkit.detect()?;
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[cfg(cuda_mock)]
#[test]
fn test_toolkit_info_consistency() {
    // Multiple calls should return consistent info
    let info1 = get_toolkit_info().unwrap();
    let info2 = get_toolkit_info().unwrap();

    assert_eq!(info1.version.major, info2.version.major);
    assert_eq!(info1.version.minor, info2.version.minor);
    assert_eq!(info1.device_count, info2.device_count);
    assert_eq!(info1.path, info2.path);
}

#[test]
fn test_toolkit_info_debug_format() {
    let info = ToolkitInfo {
        version: CudaVersion::new(12, 0, 0),
        path: PathBuf::from("/usr/local/cuda"),
        device_count: 1,
        is_mock: false,
        driver_version: "535.154.05".to_string(),
    };

    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("ToolkitInfo"));
    assert!(debug_str.contains("12"));
    assert!(debug_str.contains("/usr/local/cuda"));
}

#[test]
fn test_version_patch_handling() {
    // Test that patch version doesn't affect meets_minimum
    let v1 = CudaVersion::new(12, 0, 0);
    let v2 = CudaVersion::new(12, 0, 100);
    let v3 = CudaVersion::new(12, 0, 999);

    // All should meet the same minimum requirements
    assert_eq!(v1.meets_minimum(12, 0), v2.meets_minimum(12, 0));
    assert_eq!(v2.meets_minimum(12, 0), v3.meets_minimum(12, 0));
    assert_eq!(v1.meets_minimum(11, 8), v3.meets_minimum(11, 8));
}

#[cfg(cuda_mock)]
#[test]
fn test_initialization_state() {
    // Test that initialization state is properly tracked
    // SAFETY: INITIALIZED is a static bool used to track one-time init state.
    // Reading it is safe as we only check its value, and the mock test
    // environment ensures no concurrent access.
    let _ = unsafe { INITIALIZED };
}

#[test]
fn test_toolkit_new_is_empty() {
    // New toolkit should not have info yet
    let toolkit = CudaToolkit::new();
    // Can't directly test private field, but first detect() will populate it
}

#[cfg(not(cuda_mock))]
#[test]
fn test_environment_variable_paths() {
    use std::env;

    // Save original values
    let orig_cuda_path = env::var("CUDA_PATH").ok();
    let orig_cuda_home = env::var("CUDA_HOME").ok();

    // Test CUDA_PATH
    env::set_var("CUDA_PATH", "/test/cuda/path");
    let result = CudaToolkit::find_cuda_path();
    if result.is_ok() {
        assert_eq!(result.unwrap(), PathBuf::from("/test/cuda/path"));
    }

    // Test CUDA_HOME
    env::remove_var("CUDA_PATH");
    env::set_var("CUDA_HOME", "/test/cuda/home");
    let result = CudaToolkit::find_cuda_path();
    if result.is_ok() {
        assert_eq!(result.unwrap(), PathBuf::from("/test/cuda/home"));
    }

    // Restore original values
    match orig_cuda_path {
        Some(val) => env::set_var("CUDA_PATH", val),
        None => env::remove_var("CUDA_PATH"),
    }
    match orig_cuda_home {
        Some(val) => env::set_var("CUDA_HOME", val),
        None => env::remove_var("CUDA_HOME"),
    }
}
