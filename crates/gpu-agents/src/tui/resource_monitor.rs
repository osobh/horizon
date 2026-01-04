//! Resource monitoring for GPU/CPU/Memory usage

use anyhow::Result;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::benchmarks::progress_monitor::ResourceSnapshot;

/// Resource monitor for tracking system metrics
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Whether resource monitoring is enabled
    enabled: bool,
    /// Latest resource snapshot
    latest_snapshot: Option<ResourceSnapshot>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(enabled: bool) -> Result<Self> {
        Ok(Self {
            enabled,
            latest_snapshot: None,
        })
    }

    /// Get the latest resource snapshot
    pub async fn get_latest_snapshot(&mut self) -> Result<Option<ResourceSnapshot>> {
        if !self.enabled {
            return Ok(None);
        }

        let snapshot = self.collect_resource_snapshot().await?;
        self.latest_snapshot = Some(snapshot.clone());
        Ok(Some(snapshot))
    }

    /// Collect current resource usage
    async fn collect_resource_snapshot(&self) -> Result<ResourceSnapshot> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Get CPU and memory info
        let (cpu_usage, memory_used, memory_total) = self.get_system_stats()?;

        // Get GPU info
        let (gpu_usage, gpu_memory_used, gpu_memory_total, gpu_temp) = self.get_gpu_stats()?;

        Ok(ResourceSnapshot {
            timestamp,
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_used,
            memory_total_mb: memory_total,
            gpu_usage_percent: gpu_usage,
            gpu_memory_used_mb: gpu_memory_used,
            gpu_memory_total_mb: gpu_memory_total,
            gpu_temperature_c: gpu_temp,
            disk_io_read_mb_s: 0.0,  // TODO: Implement disk I/O monitoring
            disk_io_write_mb_s: 0.0, // TODO: Implement disk I/O monitoring
        })
    }

    /// Get system CPU and memory statistics
    fn get_system_stats(&self) -> Result<(f64, f64, f64)> {
        #[cfg(target_os = "linux")]
        {
            // Parse /proc/meminfo for memory stats
            let meminfo = std::fs::read_to_string("/proc/meminfo")?;
            let mut total_kb = 0;
            let mut available_kb = 0;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    total_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                } else if line.starts_with("MemAvailable:") {
                    available_kb = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                }
            }

            let memory_total_mb = total_kb as f64 / 1024.0;
            let memory_used_mb = (total_kb - available_kb) as f64 / 1024.0;

            // Get CPU usage (simplified - would need proper calculation over time intervals)
            let cpu_usage = 0.0; // TODO: Implement proper CPU usage calculation

            Ok((cpu_usage, memory_used_mb, memory_total_mb))
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            Ok((0.0, 0.0, 1.0))
        }
    }

    /// Get GPU statistics using nvidia-smi
    fn get_gpu_stats(&self) -> Result<(f64, f64, f64, f64)> {
        let output = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = stdout.trim().split(',').collect();

                if parts.len() >= 4 {
                    let gpu_usage = parts[0].trim().parse::<f64>().unwrap_or(0.0);
                    let gpu_memory_used = parts[1].trim().parse::<f64>().unwrap_or(0.0);
                    let gpu_memory_total = parts[2].trim().parse::<f64>().unwrap_or(1.0);
                    let gpu_temperature = parts[3].trim().parse::<f64>().unwrap_or(0.0);

                    Ok((
                        gpu_usage,
                        gpu_memory_used,
                        gpu_memory_total,
                        gpu_temperature,
                    ))
                } else {
                    // Fallback values
                    Ok((0.0, 0.0, 1.0, 0.0))
                }
            }
            Err(_) => {
                // nvidia-smi not available
                Ok((0.0, 0.0, 1.0, 0.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_monitor_new() {
        // RED PHASE: Test will fail until ResourceMonitor is properly implemented
        let monitor = ResourceMonitor::new(true);
        assert!(monitor.is_ok());

        let monitor = monitor?;
        assert!(monitor.enabled);
        assert!(monitor.latest_snapshot.is_none());
    }

    #[test]
    fn test_resource_monitor_disabled() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = ResourceMonitor::new(false)?;
        assert!(!monitor.enabled);
    }

    #[tokio::test]
    async fn test_get_latest_snapshot_disabled() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ResourceMonitor::new(false)?;
        let snapshot = monitor.get_latest_snapshot().await?;
        assert!(snapshot.is_none());
    }

    #[tokio::test]
    async fn test_get_latest_snapshot_enabled() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ResourceMonitor::new(true)?;
        let snapshot = monitor.get_latest_snapshot().await?;
        assert!(snapshot.is_some());

        let snapshot = snapshot?;
        assert!(snapshot.timestamp > 0);
        assert!(snapshot.memory_total_mb >= 0.0);
        assert!(snapshot.gpu_memory_total_mb >= 0.0);
    }

    #[test]
    fn test_get_system_stats() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = ResourceMonitor::new(true)?;
        let stats = monitor.get_system_stats();

        // Should not fail, even if values are not accurate
        assert!(stats.is_ok());

        let (cpu, mem_used, mem_total) = stats?;
        assert!(cpu >= 0.0);
        assert!(mem_used >= 0.0);
        assert!(mem_total >= 0.0);
    }

    #[test]
    fn test_get_gpu_stats() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = ResourceMonitor::new(true)?;
        let stats = monitor.get_gpu_stats();

        // Should not fail, even if nvidia-smi is not available
        assert!(stats.is_ok());

        let (gpu_usage, gpu_mem_used, gpu_mem_total, gpu_temp) = stats?;
        assert!(gpu_usage >= 0.0);
        assert!(gpu_mem_used >= 0.0);
        assert!(gpu_mem_total >= 0.0);
        assert!(gpu_temp >= 0.0);
    }
}
