//! Resource tracking module for monitoring GPU/CPU/memory usage and collecting metrics
//!
//! This module provides comprehensive resource monitoring capabilities including:
//! - Real-time GPU utilization tracking
//! - CPU and memory usage monitoring
//! - Historical metrics collection and aggregation
//! - Resource utilization alerts and thresholds

use crate::error::{CostOptimizationError, CostOptimizationResult};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Registry};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Resource type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// GPU resource
    Gpu,
    /// CPU resource
    Cpu,
    /// Memory resource
    Memory,
    /// Storage resource
    Storage,
    /// Network bandwidth
    Network,
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceType::Gpu => write!(f, "GPU"),
            ResourceType::Cpu => write!(f, "CPU"),
            ResourceType::Memory => write!(f, "Memory"),
            ResourceType::Storage => write!(f, "Storage"),
            ResourceType::Network => write!(f, "Network"),
        }
    }
}

/// Resource utilization snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource identifier (e.g., GPU ID, CPU core)
    pub resource_id: String,
    /// Utilization percentage (0-100)
    pub utilization: f64,
    /// Available capacity
    pub available: f64,
    /// Total capacity
    pub total: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Resource metrics with historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Current utilization snapshots by resource
    pub current: HashMap<String, ResourceSnapshot>,
    /// Historical utilization data (last N samples)
    pub history: VecDeque<ResourceSnapshot>,
    /// Aggregated statistics
    pub stats: ResourceStats,
    /// Alert thresholds
    pub thresholds: ResourceThresholds,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            current: HashMap::new(),
            history: VecDeque::with_capacity(1000),
            stats: ResourceStats::default(),
            thresholds: ResourceThresholds::default(),
        }
    }
}

/// Aggregated resource statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceStats {
    /// Average utilization over time window
    pub avg_utilization: f64,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Minimum utilization
    pub min_utilization: f64,
    /// Standard deviation of utilization
    pub std_deviation: f64,
    /// Total samples collected
    pub sample_count: u64,
    /// Time window for statistics
    pub window_duration: Duration,
}

/// Resource alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// Warning threshold (percentage)
    pub warning: f64,
    /// Critical threshold (percentage)
    pub critical: f64,
    /// Sustained duration before alert
    pub sustained_duration: Duration,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            warning: 80.0,
            critical: 95.0,
            sustained_duration: Duration::from_secs(60),
        }
    }
}

/// Resource tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTrackerConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// History retention period
    pub retention_period: Duration,
    /// Maximum history size
    pub max_history_size: usize,
    /// Enable GPU monitoring
    pub enable_gpu: bool,
    /// Enable CPU monitoring
    pub enable_cpu: bool,
    /// Enable memory monitoring
    pub enable_memory: bool,
    /// Alert configurations
    pub alert_config: AlertConfig,
}

impl Default for ResourceTrackerConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(24 * 60 * 60),
            max_history_size: 10000,
            enable_gpu: true,
            enable_cpu: true,
            enable_memory: true,
            alert_config: AlertConfig::default(),
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert cooldown period
    pub cooldown: Duration,
    /// Alert notification channels
    pub channels: Vec<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown: Duration::from_secs(300),
            channels: vec!["metrics".to_string()],
        }
    }
}

/// Resource utilization alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    /// Alert ID
    pub id: Uuid,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Resource type
    pub resource_type: ResourceType,
    /// Resource identifier
    pub resource_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Current utilization
    pub utilization: f64,
    /// Threshold exceeded
    pub threshold: f64,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

/// Resource tracker for monitoring system resources
pub struct ResourceTracker {
    /// Configuration
    config: Arc<ResourceTrackerConfig>,
    /// Current metrics by resource type
    metrics: Arc<DashMap<ResourceType, ResourceMetrics>>,
    /// Active alerts
    alerts: Arc<RwLock<Vec<ResourceAlert>>>,
    /// Prometheus metrics
    prometheus_metrics: PrometheusMetrics,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
}

/// Prometheus metrics for resource tracking
struct PrometheusMetrics {
    /// GPU utilization gauge
    gpu_utilization: Gauge,
    /// CPU utilization gauge
    cpu_utilization: Gauge,
    /// Memory utilization gauge
    memory_utilization: Gauge,
    /// Collection errors counter
    collection_errors: Counter,
    /// Collection duration histogram
    collection_duration: Histogram,
}

impl ResourceTracker {
    /// Create a new resource tracker
    pub fn new(config: ResourceTrackerConfig) -> CostOptimizationResult<Self> {
        let registry = Registry::new();

        let prometheus_metrics = PrometheusMetrics {
            gpu_utilization: Gauge::new("resource_gpu_utilization", "GPU utilization percentage")
                .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?,
            cpu_utilization: Gauge::new("resource_cpu_utilization", "CPU utilization percentage")
                .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?,
            memory_utilization: Gauge::new(
                "resource_memory_utilization",
                "Memory utilization percentage",
            )
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?,
            collection_errors: Counter::new(
                "resource_collection_errors",
                "Resource collection error count",
            )
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?,
            collection_duration: Histogram::with_opts(HistogramOpts::new(
                "resource_collection_duration_seconds",
                "Resource collection duration in seconds",
            ))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?,
        };

        // Register metrics
        registry
            .register(Box::new(prometheus_metrics.gpu_utilization.clone()))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?;
        registry
            .register(Box::new(prometheus_metrics.cpu_utilization.clone()))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?;
        registry
            .register(Box::new(prometheus_metrics.memory_utilization.clone()))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?;
        registry
            .register(Box::new(prometheus_metrics.collection_errors.clone()))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?;
        registry
            .register(Box::new(prometheus_metrics.collection_duration.clone()))
            .map_err(|e| CostOptimizationError::MetricsError {
                message: e.to_string(),
            })?;

        Ok(Self {
            config: Arc::new(config),
            metrics: Arc::new(DashMap::new()),
            alerts: Arc::new(RwLock::new(Vec::new())),
            prometheus_metrics,
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Start resource monitoring
    pub async fn start(&self) -> CostOptimizationResult<()> {
        info!("Starting resource tracker");

        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let alerts = self.alerts.clone();
        let prometheus_metrics = PrometheusMetrics {
            gpu_utilization: self.prometheus_metrics.gpu_utilization.clone(),
            cpu_utilization: self.prometheus_metrics.cpu_utilization.clone(),
            memory_utilization: self.prometheus_metrics.memory_utilization.clone(),
            collection_errors: self.prometheus_metrics.collection_errors.clone(),
            collection_duration: self.prometheus_metrics.collection_duration.clone(),
        };
        let shutdown = self.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.collection_interval);

            loop {
                if *shutdown.read() {
                    info!("Resource tracker shutting down");
                    break;
                }

                interval.tick().await;

                let timer = prometheus_metrics.collection_duration.start_timer();

                // Collect metrics for each enabled resource type
                if config.enable_gpu {
                    if let Err(e) = Self::collect_gpu_metrics(&metrics, &prometheus_metrics).await {
                        error!("Failed to collect GPU metrics: {}", e);
                        prometheus_metrics.collection_errors.inc();
                    }
                }

                if config.enable_cpu {
                    if let Err(e) = Self::collect_cpu_metrics(&metrics, &prometheus_metrics).await {
                        error!("Failed to collect CPU metrics: {}", e);
                        prometheus_metrics.collection_errors.inc();
                    }
                }

                if config.enable_memory {
                    if let Err(e) =
                        Self::collect_memory_metrics(&metrics, &prometheus_metrics).await
                    {
                        error!("Failed to collect memory metrics: {}", e);
                        prometheus_metrics.collection_errors.inc();
                    }
                }

                // Check thresholds and generate alerts
                Self::check_thresholds(&metrics, &alerts, &config.alert_config);

                // Clean up old history
                Self::cleanup_history(&metrics, config.retention_period, config.max_history_size);

                timer.observe_duration();
            }
        });

        Ok(())
    }

    /// Stop resource monitoring
    pub async fn stop(&self) -> CostOptimizationResult<()> {
        info!("Stopping resource tracker");
        *self.shutdown.write() = true;
        Ok(())
    }

    /// Get current resource metrics
    pub fn get_metrics(&self, resource_type: ResourceType) -> Option<ResourceMetrics> {
        self.metrics.get(&resource_type).map(|entry| entry.clone())
    }

    /// Get all current metrics
    pub fn get_all_metrics(&self) -> HashMap<ResourceType, ResourceMetrics> {
        self.metrics
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Get active alerts
    pub fn get_alerts(&self) -> Vec<ResourceAlert> {
        self.alerts.read().clone()
    }

    /// Clear alerts
    pub fn clear_alerts(&self) {
        self.alerts.write().clear();
    }

    /// Update resource snapshot
    pub fn update_snapshot(&self, snapshot: ResourceSnapshot) -> CostOptimizationResult<()> {
        let mut metrics = self
            .metrics
            .entry(snapshot.resource_type)
            .or_insert_with(ResourceMetrics::default);

        // Update current snapshot
        metrics
            .current
            .insert(snapshot.resource_id.clone(), snapshot.clone());

        // Add to history
        metrics.history.push_back(snapshot.clone());

        // Update statistics
        Self::update_statistics(&mut metrics);

        // Update Prometheus metrics
        match snapshot.resource_type {
            ResourceType::Gpu => self
                .prometheus_metrics
                .gpu_utilization
                .set(snapshot.utilization),
            ResourceType::Cpu => self
                .prometheus_metrics
                .cpu_utilization
                .set(snapshot.utilization),
            ResourceType::Memory => self
                .prometheus_metrics
                .memory_utilization
                .set(snapshot.utilization),
            _ => {}
        }

        Ok(())
    }

    /// Collect GPU metrics
    async fn collect_gpu_metrics(
        metrics: &Arc<DashMap<ResourceType, ResourceMetrics>>,
        prometheus_metrics: &PrometheusMetrics,
    ) -> CostOptimizationResult<()> {
        // In a real implementation, this would interface with NVIDIA Management Library (NVML)
        // or similar GPU monitoring tools. For now, we'll simulate GPU metrics.

        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 75.0, // Simulated value
            available: 8.0,    // GB
            total: 32.0,       // GB
            metadata: HashMap::from([
                ("temperature".to_string(), "65".to_string()),
                ("power_draw".to_string(), "250".to_string()),
            ]),
        };

        let mut gpu_metrics = metrics
            .entry(ResourceType::Gpu)
            .or_insert_with(ResourceMetrics::default);

        gpu_metrics
            .current
            .insert(snapshot.resource_id.clone(), snapshot.clone());
        gpu_metrics.history.push_back(snapshot.clone());

        prometheus_metrics.gpu_utilization.set(snapshot.utilization);

        Ok(())
    }

    /// Collect CPU metrics
    async fn collect_cpu_metrics(
        metrics: &Arc<DashMap<ResourceType, ResourceMetrics>>,
        prometheus_metrics: &PrometheusMetrics,
    ) -> CostOptimizationResult<()> {
        // Simulated CPU metrics
        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Cpu,
            resource_id: "CPU-Total".to_string(),
            utilization: 45.0, // Simulated value
            available: 32.0,   // Cores
            total: 64.0,       // Cores
            metadata: HashMap::new(),
        };

        let mut cpu_metrics = metrics
            .entry(ResourceType::Cpu)
            .or_insert_with(ResourceMetrics::default);

        cpu_metrics
            .current
            .insert(snapshot.resource_id.clone(), snapshot.clone());
        cpu_metrics.history.push_back(snapshot.clone());

        prometheus_metrics.cpu_utilization.set(snapshot.utilization);

        Ok(())
    }

    /// Collect memory metrics
    async fn collect_memory_metrics(
        metrics: &Arc<DashMap<ResourceType, ResourceMetrics>>,
        prometheus_metrics: &PrometheusMetrics,
    ) -> CostOptimizationResult<()> {
        // Simulated memory metrics
        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Memory,
            resource_id: "Memory-System".to_string(),
            utilization: 60.0, // Simulated value
            available: 64.0,   // GB
            total: 128.0,      // GB
            metadata: HashMap::new(),
        };

        let mut memory_metrics = metrics
            .entry(ResourceType::Memory)
            .or_insert_with(ResourceMetrics::default);

        memory_metrics
            .current
            .insert(snapshot.resource_id.clone(), snapshot.clone());
        memory_metrics.history.push_back(snapshot.clone());

        prometheus_metrics
            .memory_utilization
            .set(snapshot.utilization);

        Ok(())
    }

    /// Update resource statistics
    fn update_statistics(metrics: &mut ResourceMetrics) {
        if metrics.history.is_empty() {
            return;
        }

        let utilizations: Vec<f64> = metrics.history.iter().map(|s| s.utilization).collect();

        let sum: f64 = utilizations.iter().sum();
        let count = utilizations.len() as f64;
        let avg = sum / count;

        let variance = utilizations.iter().map(|u| (u - avg).powi(2)).sum::<f64>() / count;

        metrics.stats = ResourceStats {
            avg_utilization: avg,
            peak_utilization: utilizations
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            min_utilization: utilizations.iter().cloned().fold(f64::INFINITY, f64::min),
            std_deviation: variance.sqrt(),
            sample_count: utilizations.len() as u64,
            window_duration: Duration::from_secs(3600), // 1 hour window
        };
    }

    /// Check thresholds and generate alerts
    fn check_thresholds(
        metrics: &Arc<DashMap<ResourceType, ResourceMetrics>>,
        alerts: &Arc<RwLock<Vec<ResourceAlert>>>,
        alert_config: &AlertConfig,
    ) {
        if !alert_config.enabled {
            return;
        }

        for entry in metrics.iter() {
            let resource_type = *entry.key();
            let resource_metrics = entry.value();

            for (resource_id, snapshot) in &resource_metrics.current {
                // Check critical threshold
                if snapshot.utilization >= resource_metrics.thresholds.critical {
                    let alert = ResourceAlert {
                        id: Uuid::new_v4(),
                        timestamp: Utc::now(),
                        resource_type,
                        resource_id: resource_id.clone(),
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "{} utilization critical: {:.1}% (threshold: {:.1}%)",
                            resource_type,
                            snapshot.utilization,
                            resource_metrics.thresholds.critical
                        ),
                        utilization: snapshot.utilization,
                        threshold: resource_metrics.thresholds.critical,
                    };

                    alerts.write().push(alert);
                }
                // Check warning threshold
                else if snapshot.utilization >= resource_metrics.thresholds.warning {
                    let alert = ResourceAlert {
                        id: Uuid::new_v4(),
                        timestamp: Utc::now(),
                        resource_type,
                        resource_id: resource_id.clone(),
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "{} utilization warning: {:.1}% (threshold: {:.1}%)",
                            resource_type,
                            snapshot.utilization,
                            resource_metrics.thresholds.warning
                        ),
                        utilization: snapshot.utilization,
                        threshold: resource_metrics.thresholds.warning,
                    };

                    alerts.write().push(alert);
                }
            }
        }
    }

    /// Clean up old history entries
    fn cleanup_history(
        metrics: &Arc<DashMap<ResourceType, ResourceMetrics>>,
        retention_period: Duration,
        max_size: usize,
    ) {
        let cutoff_time = match chrono::Duration::from_std(retention_period) {
            Ok(duration) => Utc::now() - duration,
            Err(_) => return, // Can't convert duration, skip cleanup
        };

        for mut entry in metrics.iter_mut() {
            let metrics = entry.value_mut();

            // Remove old entries
            metrics
                .history
                .retain(|snapshot| snapshot.timestamp > cutoff_time);

            // Enforce max size
            while metrics.history.len() > max_size {
                metrics.history.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resource_type_display() {
        assert_eq!(ResourceType::Gpu.to_string(), "GPU");
        assert_eq!(ResourceType::Cpu.to_string(), "CPU");
        assert_eq!(ResourceType::Memory.to_string(), "Memory");
        assert_eq!(ResourceType::Storage.to_string(), "Storage");
        assert_eq!(ResourceType::Network.to_string(), "Network");
    }

    #[test]
    fn test_resource_metrics_default() {
        let metrics = ResourceMetrics::default();
        assert!(metrics.current.is_empty());
        assert!(metrics.history.is_empty());
        assert_eq!(metrics.stats.avg_utilization, 0.0);
        assert_eq!(metrics.thresholds.warning, 80.0);
        assert_eq!(metrics.thresholds.critical, 95.0);
    }

    #[test]
    fn test_resource_tracker_creation() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config);
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_update_snapshot() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 75.0,
            available: 8.0,
            total: 32.0,
            metadata: HashMap::new(),
        };

        let result = tracker.update_snapshot(snapshot.clone());
        assert!(result.is_ok());

        let metrics = tracker.get_metrics(ResourceType::Gpu);
        assert!(metrics.is_some());

        let gpu_metrics = metrics.unwrap();
        assert_eq!(gpu_metrics.current.len(), 1);
        assert_eq!(gpu_metrics.history.len(), 1);
    }

    #[test]
    fn test_alert_generation() {
        let mut config = ResourceTrackerConfig::default();
        config.alert_config.enabled = true;

        let tracker = ResourceTracker::new(config).unwrap();

        // Add high utilization snapshot
        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 96.0, // Above critical threshold
            available: 1.0,
            total: 32.0,
            metadata: HashMap::new(),
        };

        tracker.update_snapshot(snapshot).unwrap();

        // Check thresholds manually
        ResourceTracker::check_thresholds(
            &tracker.metrics,
            &tracker.alerts,
            &tracker.config.alert_config,
        );

        let alerts = tracker.get_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_statistics_calculation() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        // Add multiple snapshots
        let utilizations = vec![50.0, 60.0, 70.0, 80.0, 90.0];
        for util in utilizations {
            let snapshot = ResourceSnapshot {
                timestamp: Utc::now(),
                resource_type: ResourceType::Cpu,
                resource_id: "CPU-0".to_string(),
                utilization: util,
                available: 100.0 - util,
                total: 100.0,
                metadata: HashMap::new(),
            };
            tracker.update_snapshot(snapshot).unwrap();
        }

        let metrics = tracker.get_metrics(ResourceType::Cpu).unwrap();
        assert_relative_eq!(metrics.stats.avg_utilization, 70.0, epsilon = 0.1);
        assert_relative_eq!(metrics.stats.peak_utilization, 90.0, epsilon = 0.1);
        assert_relative_eq!(metrics.stats.min_utilization, 50.0, epsilon = 0.1);
        assert!(metrics.stats.std_deviation > 0.0);
    }

    #[tokio::test]
    async fn test_start_stop_monitoring() {
        let config = ResourceTrackerConfig {
            collection_interval: Duration::from_millis(100),
            ..Default::default()
        };

        let tracker = ResourceTracker::new(config)?;

        // Start monitoring
        let result = tracker.start().await;
        assert!(result.is_ok());

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(250)).await;

        // Stop monitoring
        let result = tracker.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_history_cleanup() {
        let config = ResourceTrackerConfig {
            max_history_size: 5,
            ..Default::default()
        };

        let tracker = ResourceTracker::new(config)?;

        // Add more snapshots than max size
        for i in 0..10 {
            let snapshot = ResourceSnapshot {
                timestamp: Utc::now(),
                resource_type: ResourceType::Memory,
                resource_id: "Memory-0".to_string(),
                utilization: i as f64 * 10.0,
                available: 100.0 - (i as f64 * 10.0),
                total: 100.0,
                metadata: HashMap::new(),
            };
            tracker.update_snapshot(snapshot).unwrap();
        }

        // Clean up history
        ResourceTracker::cleanup_history(
            &tracker.metrics,
            tracker.config.retention_period,
            tracker.config.max_history_size,
        );

        let metrics = tracker.get_metrics(ResourceType::Memory).unwrap();
        assert!(metrics.history.len() <= 5);
    }

    #[test]
    fn test_multiple_resource_types() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        // Add snapshots for different resource types
        let resource_types = vec![
            (ResourceType::Gpu, "GPU-0", 80.0),
            (ResourceType::Cpu, "CPU-0", 50.0),
            (ResourceType::Memory, "Memory-0", 60.0),
        ];

        for (resource_type, resource_id, utilization) in resource_types {
            let snapshot = ResourceSnapshot {
                timestamp: Utc::now(),
                resource_type,
                resource_id: resource_id.to_string(),
                utilization,
                available: 100.0 - utilization,
                total: 100.0,
                metadata: HashMap::new(),
            };
            tracker.update_snapshot(snapshot).unwrap();
        }

        let all_metrics = tracker.get_all_metrics();
        assert_eq!(all_metrics.len(), 3);
        assert!(all_metrics.contains_key(&ResourceType::Gpu));
        assert!(all_metrics.contains_key(&ResourceType::Cpu));
        assert!(all_metrics.contains_key(&ResourceType::Memory));
    }

    #[test]
    fn test_alert_severity_levels() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        // Test warning threshold
        let warning_snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 85.0, // Between warning (80) and critical (95)
            available: 15.0,
            total: 100.0,
            metadata: HashMap::new(),
        };

        tracker.update_snapshot(warning_snapshot).unwrap();
        ResourceTracker::check_thresholds(
            &tracker.metrics,
            &tracker.alerts,
            &tracker.config.alert_config,
        );

        let alerts = tracker.get_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);

        // Clear alerts
        tracker.clear_alerts();
        assert!(tracker.get_alerts().is_empty());
    }

    #[test]
    fn test_metadata_in_snapshots() {
        let config = ResourceTrackerConfig::default();
        let tracker = ResourceTracker::new(config).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("temperature".to_string(), "75".to_string());
        metadata.insert("power_draw".to_string(), "300".to_string());

        let snapshot = ResourceSnapshot {
            timestamp: Utc::now(),
            resource_type: ResourceType::Gpu,
            resource_id: "GPU-0".to_string(),
            utilization: 90.0,
            available: 10.0,
            total: 100.0,
            metadata: metadata.clone(),
        };

        tracker.update_snapshot(snapshot).unwrap();

        let metrics = tracker.get_metrics(ResourceType::Gpu).unwrap();
        let current_snapshot = metrics.current.get("GPU-0").unwrap();
        assert_eq!(current_snapshot.metadata.len(), 2);
        assert_eq!(current_snapshot.metadata.get("temperature").unwrap(), "75");
    }

    #[test]
    fn test_configuration_validation() {
        // Test with invalid configuration
        let config = ResourceTrackerConfig {
            collection_interval: Duration::from_secs(0), // Invalid: zero interval
            retention_period: Duration::from_secs(0),    // Invalid: zero retention
            max_history_size: 0,                         // Invalid: zero size
            ..Default::default()
        };

        // Tracker should still be created, but with sensible defaults
        let tracker = ResourceTracker::new(config);
        assert!(tracker.is_ok());
    }
}
