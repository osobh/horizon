//! Resource Monitor for GPU Utilization
//!
//! Monitors GPU resources including memory, compute, and power consumption.

use anyhow::{Context, Result};
use cudarc::driver::CudaDevice;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Resource types to monitor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// GPU compute utilization
    Compute,
    /// GPU memory usage
    Memory,
    /// Memory bandwidth
    Bandwidth,
    /// Power consumption
    Power,
    /// Temperature
    Temperature,
    /// Clock speeds
    Clocks,
}

/// Resource measurement
#[derive(Debug, Clone)]
pub struct ResourceMeasurement {
    pub resource_type: ResourceType,
    pub value: f32,
    pub unit: String,
    pub timestamp: Instant,
}

/// Resource limits and thresholds
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage (bytes)
    pub max_memory: u64,
    /// Memory usage warning threshold (0.0 - 1.0)
    pub memory_warning_threshold: f32,
    /// Maximum temperature (Celsius)
    pub max_temperature: f32,
    /// Temperature warning threshold
    pub temperature_warning: f32,
    /// Maximum power consumption (Watts)
    pub max_power: f32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 32 * 1024 * 1024 * 1024, // 32GB for RTX 5090
            memory_warning_threshold: 0.9,       // 90% warning
            max_temperature: 83.0,               // 83°C max
            temperature_warning: 75.0,           // 75°C warning
            max_power: 450.0,                    // 450W TDP
        }
    }
}

/// Resource alert
#[derive(Debug, Clone)]
pub struct ResourceAlert {
    pub resource_type: ResourceType,
    pub severity: AlertSeverity,
    pub message: String,
    pub current_value: f32,
    pub threshold: f32,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Resource monitor
pub struct ResourceMonitor {
    device: Arc<CudaDevice>,
    limits: ResourceLimits,
    measurements: Arc<DashMap<ResourceType, VecDeque<ResourceMeasurement>>>,
    alerts: Arc<RwLock<VecDeque<ResourceAlert>>>,
    is_monitoring: Arc<AtomicBool>,
    measurement_interval: Duration,
}

impl ResourceMonitor {
    /// Create new resource monitor
    pub fn new(device: Arc<CudaDevice>, limits: ResourceLimits) -> Self {
        let measurements = DashMap::new();
        for resource_type in &[
            ResourceType::Compute,
            ResourceType::Memory,
            ResourceType::Bandwidth,
            ResourceType::Power,
            ResourceType::Temperature,
            ResourceType::Clocks,
        ] {
            measurements.insert(*resource_type, VecDeque::with_capacity(1000));
        }

        Self {
            device,
            limits,
            measurements: Arc::new(measurements),
            alerts: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            is_monitoring: Arc::new(AtomicBool::new(false)),
            measurement_interval: Duration::from_millis(100),
        }
    }

    /// Start monitoring resources
    pub async fn start_monitoring(&self) -> Result<()> {
        if self.is_monitoring.swap(true, Ordering::Relaxed) {
            return Ok(()); // Already monitoring
        }

        let device = self.device.clone();
        let measurements = self.measurements.clone();
        let alerts = self.alerts.clone();
        let is_monitoring = self.is_monitoring.clone();
        let interval = self.measurement_interval;
        let limits = self.limits.clone();

        tokio::spawn(async move {
            while is_monitoring.load(Ordering::Relaxed) {
                // Measure resources
                let current_measurements = Self::measure_resources(&device).await;

                // Store measurements
                for measurement in current_measurements {
                    if let Some(mut queue) = measurements.get_mut(&measurement.resource_type) {
                        if queue.len() >= 1000 {
                            queue.pop_front();
                        }
                        queue.push_back(measurement.clone());
                    }

                    // Check for alerts
                    if let Some(alert) = Self::check_alert(&measurement, &limits) {
                        let mut alert_queue = alerts.write().await;
                        if alert_queue.len() >= 100 {
                            alert_queue.pop_front();
                        }
                        alert_queue.push_back(alert);
                    }
                }

                tokio::time::sleep(interval).await;
            }
        });

        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.is_monitoring.store(false, Ordering::Relaxed);
    }

    /// Measure current resources (simulated)
    async fn measure_resources(device: &Arc<CudaDevice>) -> Vec<ResourceMeasurement> {
        let mut measurements = Vec::new();
        let now = Instant::now();

        // Simulate GPU measurements
        // In real implementation, use NVML or CUPTI

        // Compute utilization
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Compute,
            value: 75.0 + (rand::random::<f32>() * 20.0 - 10.0),
            unit: "%".to_string(),
            timestamp: now,
        });

        // Memory usage (simulated as 60-80% of 32GB)
        let memory_gb = 19.2 + (rand::random::<f32>() * 6.4);
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Memory,
            value: memory_gb,
            unit: "GB".to_string(),
            timestamp: now,
        });

        // Memory bandwidth (simulated as 600-800 GB/s)
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Bandwidth,
            value: 600.0 + (rand::random::<f32>() * 200.0),
            unit: "GB/s".to_string(),
            timestamp: now,
        });

        // Power consumption (simulated as 300-400W)
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Power,
            value: 300.0 + (rand::random::<f32>() * 100.0),
            unit: "W".to_string(),
            timestamp: now,
        });

        // Temperature (simulated as 65-75°C)
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Temperature,
            value: 65.0 + (rand::random::<f32>() * 10.0),
            unit: "°C".to_string(),
            timestamp: now,
        });

        // Clock speed (simulated as 2.3-2.5 GHz)
        measurements.push(ResourceMeasurement {
            resource_type: ResourceType::Clocks,
            value: 2300.0 + (rand::random::<f32>() * 200.0),
            unit: "MHz".to_string(),
            timestamp: now,
        });

        measurements
    }

    /// Check if measurement triggers an alert
    fn check_alert(
        measurement: &ResourceMeasurement,
        limits: &ResourceLimits,
    ) -> Option<ResourceAlert> {
        match measurement.resource_type {
            ResourceType::Memory => {
                let memory_bytes = measurement.value * 1024.0 * 1024.0 * 1024.0;
                let usage_ratio = memory_bytes / limits.max_memory as f32;

                if usage_ratio > limits.memory_warning_threshold {
                    Some(ResourceAlert {
                        resource_type: ResourceType::Memory,
                        severity: if usage_ratio > 0.95 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("Memory usage at {:.1}%", usage_ratio * 100.0),
                        current_value: measurement.value,
                        threshold: (limits.max_memory as f32 * limits.memory_warning_threshold)
                            / (1024.0 * 1024.0 * 1024.0),
                        timestamp: measurement.timestamp,
                    })
                } else {
                    None
                }
            }
            ResourceType::Temperature => {
                if measurement.value > limits.temperature_warning {
                    Some(ResourceAlert {
                        resource_type: ResourceType::Temperature,
                        severity: if measurement.value > limits.max_temperature {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("Temperature at {:.1}°C", measurement.value),
                        current_value: measurement.value,
                        threshold: limits.temperature_warning,
                        timestamp: measurement.timestamp,
                    })
                } else {
                    None
                }
            }
            ResourceType::Power => {
                if measurement.value > limits.max_power * 0.9 {
                    Some(ResourceAlert {
                        resource_type: ResourceType::Power,
                        severity: if measurement.value > limits.max_power {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("Power consumption at {:.1}W", measurement.value),
                        current_value: measurement.value,
                        threshold: limits.max_power * 0.9,
                        timestamp: measurement.timestamp,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get latest measurement for resource type
    pub async fn get_latest_measurement(
        &self,
        resource_type: ResourceType,
    ) -> Option<ResourceMeasurement> {
        self.measurements
            .get(&resource_type)
            .and_then(|queue| queue.back().cloned())
    }

    /// Get measurement history for resource type
    pub async fn get_measurement_history(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> Vec<ResourceMeasurement> {
        let cutoff = Instant::now() - duration;

        self.measurements
            .get(&resource_type)
            .map(|queue| {
                queue
                    .iter()
                    .filter(|m| m.timestamp > cutoff)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get recent alerts
    pub async fn get_recent_alerts(&self, count: usize) -> Vec<ResourceAlert> {
        let alerts = self.alerts.read().await;
        alerts.iter().rev().take(count).cloned().collect()
    }

    /// Calculate resource statistics
    pub async fn calculate_statistics(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> ResourceStatistics {
        let history = self.get_measurement_history(resource_type, duration).await;

        if history.is_empty() {
            return ResourceStatistics::default();
        }

        let values: Vec<f32> = history.iter().map(|m| m.value).collect();
        let sum: f32 = values.iter().sum();
        let count = values.len() as f32;
        let average = sum / count;

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let variance = values.iter().map(|v| (v - average).powi(2)).sum::<f32>() / count;
        let std_dev = variance.sqrt();

        ResourceStatistics {
            resource_type,
            average,
            min,
            max,
            std_dev,
            sample_count: values.len(),
            duration,
        }
    }

    /// Generate resource report
    pub async fn generate_report(&self) -> String {
        let mut report = String::from("GPU Resource Monitor Report:\n\n");

        // Current measurements
        report.push_str("Current Resources:\n");
        for resource_type in &[
            ResourceType::Compute,
            ResourceType::Memory,
            ResourceType::Bandwidth,
            ResourceType::Power,
            ResourceType::Temperature,
            ResourceType::Clocks,
        ] {
            if let Some(measurement) = self.get_latest_measurement(*resource_type).await {
                report.push_str(&format!(
                    "  {:?}: {:.1} {}\n",
                    resource_type, measurement.value, measurement.unit
                ));
            }
        }

        // Statistics (last 5 minutes)
        report.push_str("\nStatistics (5 min):\n");
        let duration = Duration::from_secs(300);
        for resource_type in &[
            ResourceType::Compute,
            ResourceType::Memory,
            ResourceType::Temperature,
        ] {
            let stats = self.calculate_statistics(*resource_type, duration).await;
            if stats.sample_count > 0 {
                report.push_str(&format!(
                    "  {:?}: avg={:.1}, min={:.1}, max={:.1}, std={:.1}\n",
                    resource_type, stats.average, stats.min, stats.max, stats.std_dev
                ));
            }
        }

        // Recent alerts
        let alerts = self.get_recent_alerts(5).await;
        if !alerts.is_empty() {
            report.push_str("\nRecent Alerts:\n");
            for alert in alerts {
                report.push_str(&format!(
                    "  [{:?}] {:?}: {}\n",
                    alert.severity, alert.resource_type, alert.message
                ));
            }
        }

        report
    }
}

/// Resource statistics
#[derive(Debug, Clone)]
pub struct ResourceStatistics {
    pub resource_type: ResourceType,
    pub average: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
    pub sample_count: usize,
    pub duration: Duration,
}

impl Default for ResourceStatistics {
    fn default() -> Self {
        Self {
            resource_type: ResourceType::Compute,
            average: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
            sample_count: 0,
            duration: Duration::ZERO,
        }
    }
}
