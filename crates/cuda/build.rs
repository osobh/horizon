//! Build script for CUDA integration
//! Detects CUDA installation and sets up compilation environment

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Tell rustc about our custom cfg flags
    println!("cargo:rustc-check-cfg=cfg(cuda_mock)");
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");

    // Check if we're in mock mode
    if env::var("CARGO_FEATURE_MOCK").is_ok() {
        println!("cargo:rustc-cfg=cuda_mock");
        return;
    }

    // Try to find CUDA installation
    let cuda_path = find_cuda_root();

    if let Some(cuda_root) = cuda_path {
        println!("cargo:rustc-cfg=cuda_available");

        // Set up include paths
        let include_path = cuda_root.join("include");
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_PATH={}",
            include_path.display()
        );

        // Set up library paths
        let lib_path = if cfg!(target_os = "windows") {
            cuda_root.join("lib").join("x64")
        } else {
            cuda_root.join("lib64")
        };
        println!("cargo:rustc-link-search=native={}", lib_path.display());

        // Link CUDA libraries
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=nvrtc");

        // Export CUDA root for runtime use
        println!("cargo:rustc-env=CUDA_ROOT={}", cuda_root.display());
    } else {
        // CUDA not found, we'll use mock implementations
        println!("cargo:rustc-cfg=cuda_mock");
        println!("cargo:warning=CUDA not found, using mock implementations");
    }
}

fn find_cuda_root() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(cuda_path);
        if path.exists() {
            return Some(path);
        }
    }

    // Check CUDA_HOME
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let path = PathBuf::from(cuda_home);
        if path.exists() {
            return Some(path);
        }
    }

    // Check common installation paths
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
        ]
    } else {
        vec![
            "/usr/local/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
            "/opt/cuda",
        ]
    };

    for path in common_paths {
        let cuda_path = PathBuf::from(path);
        if cuda_path.exists() {
            return Some(cuda_path);
        }
    }

    // Try to find nvcc in PATH
    if let Ok(nvcc_path) = which::which("nvcc") {
        // Get parent directory of nvcc (usually .../bin/nvcc)
        if let Some(bin_dir) = nvcc_path.parent() {
            if let Some(cuda_root) = bin_dir.parent() {
                return Some(cuda_root.to_path_buf());
            }
        }
    }

    None
}
