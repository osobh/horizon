//! Remote node detection
//!
//! Detects operating system, architecture, and hardware capabilities
//! of remote nodes via SSH.

use anyhow::{Context, Result};

use super::ssh::SshSession;
use hpc_inventory::{Architecture, GpuInfo, HardwareProfile, OsType};

/// Node detector for gathering system information via SSH
pub struct NodeDetector<'a> {
    session: &'a SshSession,
}

impl<'a> NodeDetector<'a> {
    /// Create a new node detector
    pub fn new(session: &'a SshSession) -> Self {
        Self { session }
    }

    /// Detect the operating system
    pub async fn detect_os(&self) -> Result<OsType> {
        // Try uname first (Linux/macOS/BSD)
        let output = self.session.exec("uname -s 2>/dev/null || echo UNKNOWN").await?;
        let os_str = output.stdout.trim().to_lowercase();

        match os_str.as_str() {
            "linux" => Ok(OsType::Linux),
            "darwin" => Ok(OsType::Darwin),
            _ => {
                // Check for Windows
                let win_check = self
                    .session
                    .exec("echo %OS% 2>nul || echo NOTWINDOWS")
                    .await?;

                if win_check.stdout.contains("Windows") {
                    Ok(OsType::Windows)
                } else {
                    Err(anyhow::anyhow!("Unknown operating system: {}", os_str))
                }
            }
        }
    }

    /// Detect the CPU architecture
    pub async fn detect_arch(&self) -> Result<Architecture> {
        let output = self.session.exec("uname -m 2>/dev/null || echo unknown").await?;
        let arch_str = output.stdout.trim().to_lowercase();

        Ok(match arch_str.as_str() {
            "x86_64" | "amd64" => Architecture::Amd64,
            "aarch64" | "arm64" => Architecture::Arm64,
            _ => Architecture::Unknown,
        })
    }

    /// Detect both OS and architecture
    pub async fn detect_platform(&self) -> Result<(OsType, Architecture)> {
        let os = self.detect_os().await?;
        let arch = self.detect_arch().await?;
        Ok((os, arch))
    }

    /// Detect full hardware profile
    pub async fn detect_hardware(&self) -> Result<HardwareProfile> {
        let os = self.detect_os().await.ok();

        let cpu_model = self.detect_cpu_model(os.as_ref()).await.unwrap_or_default();
        let cpu_cores = self.detect_cpu_cores(os.as_ref()).await.unwrap_or(0);
        let memory_gb = self.detect_memory(os.as_ref()).await.unwrap_or(0.0);
        let storage_gb = self.detect_storage().await.unwrap_or(0.0);
        let gpus = self.detect_gpus().await.unwrap_or_default();

        Ok(HardwareProfile {
            cpu_model,
            cpu_cores,
            memory_gb,
            storage_gb,
            gpus,
        })
    }

    /// Detect CPU model name
    async fn detect_cpu_model(&self, os: Option<&OsType>) -> Result<String> {
        let cmd = match os {
            Some(OsType::Linux) => {
                "grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs"
            }
            Some(OsType::Darwin) => "sysctl -n machdep.cpu.brand_string 2>/dev/null",
            _ => "echo unknown",
        };

        let output = self.session.exec(cmd).await?;
        let model = output.stdout.trim();

        if model.is_empty() || model == "unknown" {
            Ok("Unknown CPU".to_string())
        } else {
            Ok(model.to_string())
        }
    }

    /// Detect number of CPU cores
    async fn detect_cpu_cores(&self, os: Option<&OsType>) -> Result<u32> {
        let cmd = match os {
            Some(OsType::Linux) => "nproc 2>/dev/null || grep -c processor /proc/cpuinfo",
            Some(OsType::Darwin) => "sysctl -n hw.ncpu 2>/dev/null",
            _ => "echo 0",
        };

        let output = self.session.exec(cmd).await?;
        output
            .stdout
            .trim()
            .parse()
            .context("Failed to parse CPU core count")
    }

    /// Detect total memory in GB
    async fn detect_memory(&self, os: Option<&OsType>) -> Result<f32> {
        let cmd = match os {
            Some(OsType::Linux) => {
                "grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}'"
            }
            Some(OsType::Darwin) => "sysctl -n hw.memsize 2>/dev/null",
            _ => "echo 0",
        };

        let output = self.session.exec(cmd).await?;
        let value: u64 = output
            .stdout
            .trim()
            .parse()
            .context("Failed to parse memory")?;

        // Linux returns KB, Darwin returns bytes
        let gb = match os {
            Some(OsType::Linux) => value as f32 / 1024.0 / 1024.0,
            Some(OsType::Darwin) => value as f32 / 1024.0 / 1024.0 / 1024.0,
            _ => 0.0,
        };

        Ok(gb)
    }

    /// Detect total storage in GB
    async fn detect_storage(&self) -> Result<f32> {
        // Get root filesystem size
        let output = self
            .session
            .exec("df -BG / 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G'")
            .await?;

        output
            .stdout
            .trim()
            .parse()
            .context("Failed to parse storage size")
    }

    /// Detect GPUs
    async fn detect_gpus(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try NVIDIA first
        if self.session.command_exists("nvidia-smi").await.unwrap_or(false) {
            if let Ok(nvidia_gpus) = self.detect_nvidia_gpus().await {
                gpus.extend(nvidia_gpus);
            }
        }

        // Try AMD ROCm
        if self.session.command_exists("rocm-smi").await.unwrap_or(false) {
            if let Ok(amd_gpus) = self.detect_amd_gpus().await {
                gpus.extend(amd_gpus);
            }
        }

        // Check for Apple Silicon (macOS)
        let os = self.detect_os().await.ok();
        if matches!(os, Some(OsType::Darwin)) {
            let arch = self.detect_arch().await.ok();
            if matches!(arch, Some(Architecture::Arm64)) {
                // Apple Silicon has integrated GPU
                gpus.push(GpuInfo {
                    index: 0,
                    name: "Apple Silicon GPU".to_string(),
                    memory_mb: 0, // Unified memory, varies
                    vendor: "apple".to_string(),
                });
            }
        }

        Ok(gpus)
    }

    /// Detect NVIDIA GPUs using nvidia-smi
    async fn detect_nvidia_gpus(&self) -> Result<Vec<GpuInfo>> {
        let output = self
            .session
            .exec(
                "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null",
            )
            .await?;

        if !output.success() {
            return Ok(Vec::new());
        }

        let mut gpus = Vec::new();
        for line in output.stdout.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                let index: u32 = parts[0].parse().unwrap_or(0);
                let name = parts[1].to_string();
                let memory_mb: u64 = parts[2].parse().unwrap_or(0);

                gpus.push(GpuInfo {
                    index,
                    name,
                    memory_mb,
                    vendor: "nvidia".to_string(),
                });
            }
        }

        Ok(gpus)
    }

    /// Detect AMD GPUs using rocm-smi
    async fn detect_amd_gpus(&self) -> Result<Vec<GpuInfo>> {
        let output = self
            .session
            .exec("rocm-smi --showproductname --showmeminfo vram --csv 2>/dev/null")
            .await?;

        if !output.success() {
            return Ok(Vec::new());
        }

        // ROCm-SMI CSV parsing is complex, simplified version
        let mut gpus = Vec::new();
        let mut index = 0;

        for line in output.stdout.lines().skip(1) {
            // Skip header
            if line.contains("GPU") || line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if !parts.is_empty() {
                gpus.push(GpuInfo {
                    index,
                    name: parts.first().unwrap_or(&"AMD GPU").to_string(),
                    memory_mb: 0, // Would need separate parsing
                    vendor: "amd".to_string(),
                });
                index += 1;
            }
        }

        Ok(gpus)
    }

    /// Check if Docker is available
    pub async fn has_docker(&self) -> Result<bool> {
        let output = self.session.exec("docker info >/dev/null 2>&1 && echo yes").await?;
        Ok(output.stdout.trim() == "yes")
    }

    /// Check if Docker Compose is available
    pub async fn has_docker_compose(&self) -> Result<bool> {
        // Check for both v1 and v2
        let output = self
            .session
            .exec("(docker compose version >/dev/null 2>&1 || docker-compose version >/dev/null 2>&1) && echo yes")
            .await?;
        Ok(output.stdout.trim() == "yes")
    }

    /// Check if an HPC agent is already installed
    pub async fn has_agent(&self) -> Result<bool> {
        let output = self
            .session
            .exec("command -v swarmlet >/dev/null 2>&1 && echo yes")
            .await?;
        Ok(output.stdout.trim() == "yes")
    }

    /// Get agent version if installed
    pub async fn agent_version(&self) -> Result<Option<String>> {
        let output = self.session.exec("swarmlet --version 2>/dev/null").await?;

        if output.success() && !output.stdout.is_empty() {
            Ok(Some(output.stdout.trim().to_string()))
        } else {
            Ok(None)
        }
    }

    /// Check system requirements for HPC deployment
    pub async fn check_requirements(&self) -> Result<RequirementCheck> {
        let os = self.detect_os().await.ok();
        let arch = self.detect_arch().await.ok();
        let has_docker = self.has_docker().await.unwrap_or(false);
        let has_agent = self.has_agent().await.unwrap_or(false);

        // Check for curl or wget
        let has_curl = self.session.command_exists("curl").await.unwrap_or(false);
        let has_wget = self.session.command_exists("wget").await.unwrap_or(false);

        // Check for systemd (Linux)
        let has_systemd = if matches!(os, Some(OsType::Linux)) {
            self.session
                .command_exists("systemctl")
                .await
                .unwrap_or(false)
        } else {
            false
        };

        // Get memory
        let memory_gb = self.detect_memory(os.as_ref()).await.unwrap_or(0.0);

        let mut issues = Vec::new();

        // Check minimum requirements
        if os.is_none() {
            issues.push("Could not detect operating system".to_string());
        }

        if arch.is_none() || matches!(arch, Some(Architecture::Unknown)) {
            issues.push("Unsupported CPU architecture".to_string());
        }

        if !has_curl && !has_wget {
            issues.push("Neither curl nor wget available for downloads".to_string());
        }

        if memory_gb < 1.0 {
            issues.push(format!(
                "Low memory: {:.1} GB (minimum 1 GB recommended)",
                memory_gb
            ));
        }

        Ok(RequirementCheck {
            os,
            arch,
            has_docker,
            has_docker_compose: self.has_docker_compose().await.unwrap_or(false),
            has_agent,
            has_curl,
            has_wget,
            has_systemd,
            memory_gb,
            issues,
        })
    }
}

/// Result of requirement checking
#[derive(Debug, Clone)]
pub struct RequirementCheck {
    /// Detected OS
    pub os: Option<OsType>,
    /// Detected architecture
    pub arch: Option<Architecture>,
    /// Docker available
    pub has_docker: bool,
    /// Docker Compose available
    pub has_docker_compose: bool,
    /// HPC agent already installed
    pub has_agent: bool,
    /// Curl available
    pub has_curl: bool,
    /// Wget available
    pub has_wget: bool,
    /// Systemd available (Linux)
    pub has_systemd: bool,
    /// Available memory in GB
    pub memory_gb: f32,
    /// List of issues found
    pub issues: Vec<String>,
}

impl RequirementCheck {
    /// Check if all requirements are met
    pub fn is_ready(&self) -> bool {
        self.issues.is_empty()
            && self.os.is_some()
            && self.arch.is_some()
            && !matches!(self.arch, Some(Architecture::Unknown))
    }

    /// Get platform string (e.g., "linux/amd64")
    pub fn platform_str(&self) -> String {
        match (&self.os, &self.arch) {
            (Some(os), Some(arch)) => format!("{}/{}", os, arch),
            _ => "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requirement_check_ready() {
        let check = RequirementCheck {
            os: Some(OsType::Linux),
            arch: Some(Architecture::Amd64),
            has_docker: true,
            has_docker_compose: true,
            has_agent: false,
            has_curl: true,
            has_wget: false,
            has_systemd: true,
            memory_gb: 16.0,
            issues: vec![],
        };

        assert!(check.is_ready());
        assert_eq!(check.platform_str(), "linux/amd64");
    }

    #[test]
    fn test_requirement_check_not_ready() {
        let check = RequirementCheck {
            os: None,
            arch: Some(Architecture::Unknown),
            has_docker: false,
            has_docker_compose: false,
            has_agent: false,
            has_curl: false,
            has_wget: false,
            has_systemd: false,
            memory_gb: 0.5,
            issues: vec!["Could not detect OS".to_string()],
        };

        assert!(!check.is_ready());
    }
}
