//! System Commands
//!
//! Commands for detecting local system information including GPUs.

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Detected GPU information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedGpu {
    pub id: u32,
    pub name: String,
    pub vendor: String,
    pub memory_gb: Option<f32>,
    pub driver_version: Option<String>,
    pub gpu_type: GpuType,
}

/// Type of GPU detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuType {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

/// System information summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub os_name: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f32,
    pub available_memory_gb: f32,
    pub gpus: Vec<DetectedGpu>,
}

/// Detect GPUs on the local system.
#[tauri::command]
pub async fn detect_gpus() -> Result<Vec<DetectedGpu>, String> {
    let mut gpus = Vec::new();

    // On macOS, check for Apple Silicon GPU
    #[cfg(target_os = "macos")]
    {
        // Check if we're on Apple Silicon
        if std::env::consts::ARCH == "aarch64" {
            // Apple Silicon has unified memory GPU
            let sys = System::new_all();
            let total_memory = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

            // Get chip name from system_profiler if available
            let chip_name = std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Apple Silicon".to_string());

            let gpu_name = if chip_name.contains("M1") {
                "Apple M1 GPU"
            } else if chip_name.contains("M2") {
                "Apple M2 GPU"
            } else if chip_name.contains("M3") {
                "Apple M3 GPU"
            } else if chip_name.contains("M4") {
                "Apple M4 GPU"
            } else {
                "Apple GPU"
            };

            gpus.push(DetectedGpu {
                id: 0,
                name: gpu_name.to_string(),
                vendor: "Apple".to_string(),
                // Unified memory - GPU can use all of it
                memory_gb: Some(total_memory),
                driver_version: None,
                gpu_type: GpuType::Apple,
            });
        }
    }

    // On Linux/Windows, try to detect NVIDIA GPUs via nvidia-smi
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 {
                        let id: u32 = parts[0].parse().unwrap_or(gpus.len() as u32);
                        let name = parts[1].to_string();
                        let memory_mb: f32 = parts[2].parse().unwrap_or(0.0);
                        let driver = parts[3].to_string();

                        gpus.push(DetectedGpu {
                            id,
                            name,
                            vendor: "NVIDIA".to_string(),
                            memory_gb: Some(memory_mb / 1024.0),
                            driver_version: Some(driver),
                            gpu_type: GpuType::Nvidia,
                        });
                    }
                }
            }
        }
    }

    // If no GPUs detected, return a message
    if gpus.is_empty() {
        tracing::info!("No GPUs detected on this system");
    } else {
        tracing::info!("Detected {} GPU(s)", gpus.len());
    }

    Ok(gpus)
}

/// Get local system information.
#[tauri::command]
pub async fn get_system_info() -> Result<SystemInfo, String> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let os_name = System::name().unwrap_or_else(|| "unknown".to_string());
    let os_version = System::os_version().unwrap_or_else(|| "unknown".to_string());

    // Get CPU info
    let cpu_model = sys.cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let cpu_cores = sys.cpus().len();

    // Get memory info
    let total_memory_gb = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let available_memory_gb = sys.available_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

    // Detect GPUs
    let gpus = detect_gpus().await?;

    Ok(SystemInfo {
        hostname,
        os_name,
        os_version,
        cpu_model,
        cpu_cores,
        total_memory_gb,
        available_memory_gb,
        gpus,
    })
}
