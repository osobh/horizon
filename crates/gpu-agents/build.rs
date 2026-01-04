use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");
    println!("cargo:rerun-if-changed=src/memory/");
    println!("cargo:rerun-if-changed=CMakeLists.txt");

    // Check if CUDA is available
    if !check_cuda_available() {
        println!("cargo:warning=CUDA not found - gpu-agents will use stub implementations");
        println!(
            "cargo:warning=To enable CUDA support, install CUDA toolkit and ensure nvcc is in PATH"
        );
        return;
    }

    // Detect CUDA version and GPU
    let cuda_info = detect_cuda_environment();
    println!("cargo:warning=CUDA Version: {}", cuda_info.version);
    println!("cargo:warning=GPU: {}", cuda_info.gpu_name);
    println!(
        "cargo:warning=Compute Capability: sm_{}",
        cuda_info.compute_capability
    );

    // Set cfg flag for conditional compilation in Rust code
    println!("cargo:rustc-cfg=has_cuda");

    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_lib_path = PathBuf::from(&out_dir).join("cuda_build");

    // Create build directory
    std::fs::create_dir_all(&cuda_lib_path).expect("Failed to create CUDA build directory");

    // Get the source directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let source_dir = PathBuf::from(manifest_dir);

    // Set architecture based on detection
    let cuda_arch = if cuda_info.is_rtx5090 {
        "75;80;86;89;90;110" // Include RTX 5090 (sm_110)
    } else if cuda_info.compute_capability >= 89 {
        "75;80;86;89;90" // RTX 4090 and similar
    } else {
        "75;80;86" // Older GPUs
    };

    // Find CUDA installation for CMake
    let cuda_install_path = find_cuda_path();

    // Run CMake to configure with detected settings
    let cmake_config = Command::new("cmake")
        .current_dir(&cuda_lib_path)
        .arg(&source_dir) // Point to source directory with CMakeLists.txt
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={}", cuda_arch))
        .arg("-DCMAKE_CXX_COMPILER=g++")
        .arg(format!(
            "-DCMAKE_CUDA_COMPILER={}/bin/nvcc",
            cuda_install_path
        ))
        .arg(format!("-DCUDAToolkit_ROOT={}", cuda_install_path))
        .env("CXX", "g++")
        .env("CUDA_PATH", &cuda_install_path)
        .env("CUDACXX", format!("{}/bin/nvcc", cuda_install_path))
        .output()
        .expect("Failed to run CMake configuration");

    if !cmake_config.status.success() {
        panic!(
            "CMake configuration failed: {}",
            String::from_utf8_lossy(&cmake_config.stderr)
        );
    }

    // Build the CUDA library
    let cmake_build = Command::new("cmake")
        .current_dir(&cuda_lib_path)
        .args(["--build", ".", "--target", "gpu_agent_kernels"])
        .output()
        .expect("Failed to build CUDA library");

    if !cmake_build.status.success() {
        panic!(
            "CMake build failed: {}",
            String::from_utf8_lossy(&cmake_build.stderr)
        );
    }

    // Link the built library
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=static=gpu_agent_kernels");

    // Link CUDA runtime libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cuda"); // CUDA driver API for async operations

    // Link CUDA 13.0 specific libraries if available
    if cuda_info.version_major >= 13 {
        println!("cargo:rustc-link-lib=cudadevrt"); // Device runtime for RDC
        println!("cargo:rustc-link-lib=nvrtc"); // Runtime compilation
                                                // nvJitLink is optional and may not be available
        println!("cargo:rustc-link-lib=nvJitLink");
    }

    println!("cargo:rustc-link-search=native={cuda_install_path}/lib64");
    println!("cargo:rustc-link-search=native={cuda_install_path}/lib");
}

/// Check if CUDA toolkit is available on this system
fn check_cuda_available() -> bool {
    // Check if nvcc is available
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !nvcc_available {
        return false;
    }

    // Check if CUDA installation path exists
    find_cuda_path_option().is_some()
}

/// Find the CUDA installation path, returning None if not found
fn find_cuda_path_option() -> Option<String> {
    // Check environment variables first
    if let Ok(path) = env::var("CUDA_PATH") {
        if Path::new(&path).join("bin/nvcc").exists() {
            return Some(path);
        }
    }
    if let Ok(path) = env::var("CUDA_HOME") {
        if Path::new(&path).join("bin/nvcc").exists() {
            return Some(path);
        }
    }

    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-12.0",
        "/usr/local/cuda",
        "/opt/cuda",
    ];

    for path in common_paths {
        if Path::new(path).join("bin/nvcc").exists() {
            return Some(path.to_string());
        }
    }

    None
}

/// Find CUDA path, panics if not found (should only be called after check_cuda_available)
fn find_cuda_path() -> String {
    find_cuda_path_option().expect("CUDA path not found (should have been checked earlier)")
}

// CUDA environment detection
struct CudaInfo {
    version: String,
    version_major: u32,
    gpu_name: String,
    compute_capability: u32,
    is_rtx5090: bool,
}

fn detect_cuda_environment() -> CudaInfo {
    // Detect CUDA version
    let nvcc_output = Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let mut version_major = 12; // Default to CUDA 12
    let mut version = "12.0".to_string();

    // Parse CUDA version from nvcc output
    if let Some(line) = nvcc_output.lines().find(|l| l.contains("release")) {
        if let Some(ver_str) = line.split("release").nth(1) {
            if let Some(ver) = ver_str.trim().split(',').next() {
                version = ver.trim().to_string();
                let parts: Vec<&str> = ver.split('.').collect();
                if !parts.is_empty() {
                    version_major = parts[0].parse().unwrap_or(12);
                }
            }
        }
    }

    // Detect GPU using nvidia-smi
    let smi_output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let mut gpu_name = "Unknown GPU".to_string();
    let mut compute_capability = 86; // Default to sm_86
    let mut is_rtx5090 = false;

    if let Some(line) = smi_output.lines().next() {
        let parts: Vec<&str> = line.split(',').collect();
        if !parts.is_empty() {
            gpu_name = parts[0].trim().to_string();

            // Check for RTX 5090
            if gpu_name.contains("RTX 5090") || gpu_name.contains("RTX 50") {
                is_rtx5090 = true;
                compute_capability = 110; // sm_110 for Blackwell
            } else if parts.len() > 1 {
                // Parse compute capability (e.g., "8.9" -> 89)
                let cc_str = parts[1].trim();
                if let Some(dot_pos) = cc_str.find('.') {
                    let major = cc_str[..dot_pos].parse().unwrap_or(8);
                    let minor = cc_str[dot_pos + 1..].parse().unwrap_or(6);
                    compute_capability = major * 10 + minor;
                }
            }
        }
    }

    // Special handling for CUDA 13.0
    if version_major >= 13 {
        println!("cargo:warning=CUDA 13.0+ detected - enabling async memory features");
    }

    CudaInfo {
        version,
        version_major,
        gpu_name,
        compute_capability,
        is_rtx5090,
    }
}
