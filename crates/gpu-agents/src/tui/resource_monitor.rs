//! Resource monitoring for GPU/CPU/Memory usage

use anyhow::Result;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::benchmarks::progress_monitor::ResourceSnapshot;

/// CPU statistics from /proc/stat
#[derive(Debug, Clone, Default)]
struct CpuStats {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
}

impl CpuStats {
    fn total(&self) -> u64 {
        self.user + self.nice + self.system + self.idle + self.iowait + self.irq + self.softirq
    }

    fn busy(&self) -> u64 {
        self.user + self.nice + self.system + self.irq + self.softirq
    }
}

/// Disk I/O statistics from /proc/diskstats
#[derive(Debug, Clone)]
struct DiskStats {
    read_sectors: u64,
    write_sectors: u64,
    timestamp: Instant,
}

impl Default for DiskStats {
    fn default() -> Self {
        Self {
            read_sectors: 0,
            write_sectors: 0,
            timestamp: Instant::now(),
        }
    }
}

/// Resource monitor for tracking system metrics
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Whether resource monitoring is enabled
    enabled: bool,
    /// Latest resource snapshot
    latest_snapshot: Option<ResourceSnapshot>,
    /// Previous CPU stats for calculating usage
    prev_cpu_stats: Option<CpuStats>,
    /// Previous disk stats for calculating I/O rates
    prev_disk_stats: Option<DiskStats>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(enabled: bool) -> Result<Self> {
        Ok(Self {
            enabled,
            latest_snapshot: None,
            prev_cpu_stats: None,
            prev_disk_stats: None,
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
    async fn collect_resource_snapshot(&mut self) -> Result<ResourceSnapshot> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Get CPU and memory info
        let (cpu_usage, memory_used, memory_total) = self.get_system_stats()?;

        // Get GPU info
        let (gpu_usage, gpu_memory_used, gpu_memory_total, gpu_temp) = self.get_gpu_stats()?;

        // Get disk I/O rates
        let (disk_read_mb_s, disk_write_mb_s) = self.get_disk_io_stats()?;

        Ok(ResourceSnapshot {
            timestamp,
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_used,
            memory_total_mb: memory_total,
            gpu_usage_percent: gpu_usage,
            gpu_memory_used_mb: gpu_memory_used,
            gpu_memory_total_mb: gpu_memory_total,
            gpu_temperature_c: gpu_temp,
            disk_io_read_mb_s: disk_read_mb_s,
            disk_io_write_mb_s: disk_write_mb_s,
        })
    }

    /// Get system CPU and memory statistics
    fn get_system_stats(&mut self) -> Result<(f64, f64, f64)> {
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

            // Calculate CPU usage from /proc/stat
            let cpu_usage = self.calculate_cpu_usage().unwrap_or(0.0);

            Ok((cpu_usage, memory_used_mb, memory_total_mb))
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            Ok((0.0, 0.0, 1.0))
        }
    }

    /// Calculate CPU usage by comparing current and previous /proc/stat readings
    #[cfg(target_os = "linux")]
    fn calculate_cpu_usage(&mut self) -> Result<f64> {
        let stat = std::fs::read_to_string("/proc/stat")?;

        // Parse the first line (aggregate CPU stats)
        let cpu_line = stat.lines().next().ok_or_else(|| {
            anyhow::anyhow!("Failed to read CPU stats from /proc/stat")
        })?;

        let parts: Vec<&str> = cpu_line.split_whitespace().collect();
        if parts.len() < 8 || parts[0] != "cpu" {
            return Ok(0.0);
        }

        let current = CpuStats {
            user: parts[1].parse().unwrap_or(0),
            nice: parts[2].parse().unwrap_or(0),
            system: parts[3].parse().unwrap_or(0),
            idle: parts[4].parse().unwrap_or(0),
            iowait: parts[5].parse().unwrap_or(0),
            irq: parts[6].parse().unwrap_or(0),
            softirq: parts[7].parse().unwrap_or(0),
        };

        // Calculate usage based on delta from previous reading
        let usage = if let Some(ref prev) = self.prev_cpu_stats {
            let total_delta = current.total().saturating_sub(prev.total());
            let busy_delta = current.busy().saturating_sub(prev.busy());

            if total_delta > 0 {
                (busy_delta as f64 / total_delta as f64) * 100.0
            } else {
                0.0
            }
        } else {
            // First reading - can't calculate delta yet
            0.0
        };

        // Store current stats for next calculation
        self.prev_cpu_stats = Some(current);

        Ok(usage)
    }

    #[cfg(not(target_os = "linux"))]
    fn calculate_cpu_usage(&mut self) -> Result<f64> {
        Ok(0.0)
    }

    /// Get disk I/O statistics from /proc/diskstats
    fn get_disk_io_stats(&mut self) -> Result<(f64, f64)> {
        #[cfg(target_os = "linux")]
        {
            let diskstats = std::fs::read_to_string("/proc/diskstats")?;

            let mut total_read_sectors: u64 = 0;
            let mut total_write_sectors: u64 = 0;

            // Sum up all disk devices (skip partitions - look for sd*, nvme*n*, etc.)
            for line in diskstats.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 14 {
                    continue;
                }

                let device_name = parts[2];

                // Only count whole disks, not partitions
                // sda, sdb, nvme0n1, etc. - not sda1, nvme0n1p1
                let is_whole_disk = (device_name.starts_with("sd") && device_name.len() == 3)
                    || (device_name.starts_with("nvme") && device_name.contains("n") && !device_name.contains("p"))
                    || (device_name.starts_with("vd") && device_name.len() == 3);

                if is_whole_disk {
                    // Field 6 = sectors read, Field 10 = sectors written
                    let read_sectors: u64 = parts[5].parse().unwrap_or(0);
                    let write_sectors: u64 = parts[9].parse().unwrap_or(0);

                    total_read_sectors += read_sectors;
                    total_write_sectors += write_sectors;
                }
            }

            let now = Instant::now();
            let current = DiskStats {
                read_sectors: total_read_sectors,
                write_sectors: total_write_sectors,
                timestamp: now,
            };

            // Calculate rates based on delta from previous reading
            let (read_mb_s, write_mb_s) = if let Some(ref prev) = self.prev_disk_stats {
                let elapsed = now.duration_since(prev.timestamp).as_secs_f64();

                if elapsed > 0.0 {
                    let read_sectors_delta = current.read_sectors.saturating_sub(prev.read_sectors);
                    let write_sectors_delta = current.write_sectors.saturating_sub(prev.write_sectors);

                    // Sectors are typically 512 bytes
                    let sector_size_mb = 512.0 / (1024.0 * 1024.0);

                    let read_mb_s = (read_sectors_delta as f64 * sector_size_mb) / elapsed;
                    let write_mb_s = (write_sectors_delta as f64 * sector_size_mb) / elapsed;

                    (read_mb_s, write_mb_s)
                } else {
                    (0.0, 0.0)
                }
            } else {
                // First reading - can't calculate rate yet
                (0.0, 0.0)
            };

            // Store current stats for next calculation
            self.prev_disk_stats = Some(current);

            Ok((read_mb_s, write_mb_s))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Ok((0.0, 0.0))
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
    fn test_resource_monitor_new() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = ResourceMonitor::new(true);
        assert!(monitor.is_ok());

        let monitor = monitor?;
        assert!(monitor.enabled);
        assert!(monitor.latest_snapshot.is_none());
        Ok(())
    }

    #[test]
    fn test_resource_monitor_disabled() -> Result<(), Box<dyn std::error::Error>> {
        let monitor = ResourceMonitor::new(false)?;
        assert!(!monitor.enabled);
        Ok(())
    }

    #[tokio::test]
    async fn test_get_latest_snapshot_disabled() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ResourceMonitor::new(false)?;
        let snapshot = monitor.get_latest_snapshot().await?;
        assert!(snapshot.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_get_latest_snapshot_enabled() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ResourceMonitor::new(true)?;
        let snapshot = monitor.get_latest_snapshot().await?;
        assert!(snapshot.is_some());

        let snapshot = snapshot.unwrap();
        assert!(snapshot.timestamp > 0);
        assert!(snapshot.memory_total_mb >= 0.0);
        assert!(snapshot.gpu_memory_total_mb >= 0.0);
        Ok(())
    }

    #[test]
    fn test_get_system_stats() -> Result<(), Box<dyn std::error::Error>> {
        let mut monitor = ResourceMonitor::new(true)?;
        let stats = monitor.get_system_stats();

        // Should not fail, even if values are not accurate
        assert!(stats.is_ok());

        let (cpu, mem_used, mem_total) = stats?;
        assert!(cpu >= 0.0);
        assert!(mem_used >= 0.0);
        assert!(mem_total >= 0.0);
        Ok(())
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
