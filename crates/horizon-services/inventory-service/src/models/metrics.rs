use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, sqlx::FromRow, Serialize, Deserialize, ToSchema)]
pub struct AssetMetrics {
    pub asset_id: Uuid,
    pub gpu_utilization_percent: Option<i16>,
    pub gpu_memory_used_gb: Option<f64>,
    pub gpu_memory_total_gb: Option<f64>,
    pub gpu_temperature_celsius: Option<i16>,
    pub gpu_power_watts: Option<f64>,
    pub cpu_utilization_percent: Option<i16>,
    pub cpu_temperature_celsius: Option<i16>,
    pub nic_rx_bandwidth_gbps: Option<f64>,
    pub nic_tx_bandwidth_gbps: Option<f64>,
    pub last_updated: DateTime<Utc>,
}

impl AssetMetrics {
    pub fn new(asset_id: Uuid) -> Self {
        Self {
            asset_id,
            gpu_utilization_percent: None,
            gpu_memory_used_gb: None,
            gpu_memory_total_gb: None,
            gpu_temperature_celsius: None,
            gpu_power_watts: None,
            cpu_utilization_percent: None,
            cpu_temperature_celsius: None,
            nic_rx_bandwidth_gbps: None,
            nic_tx_bandwidth_gbps: None,
            last_updated: Utc::now(),
        }
    }

    pub fn with_gpu_utilization(mut self, percent: i16) -> Self {
        self.gpu_utilization_percent = Some(percent);
        self
    }

    pub fn with_gpu_memory(mut self, used_gb: f64, total_gb: f64) -> Self {
        self.gpu_memory_used_gb = Some(used_gb);
        self.gpu_memory_total_gb = Some(total_gb);
        self
    }

    pub fn with_gpu_temperature(mut self, celsius: i16) -> Self {
        self.gpu_temperature_celsius = Some(celsius);
        self
    }

    pub fn with_cpu_utilization(mut self, percent: i16) -> Self {
        self.cpu_utilization_percent = Some(percent);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_metrics_new() {
        let asset_id = Uuid::new_v4();
        let metrics = AssetMetrics::new(asset_id);

        assert_eq!(metrics.asset_id, asset_id);
        assert!(metrics.gpu_utilization_percent.is_none());
        assert!(metrics.cpu_utilization_percent.is_none());
    }

    #[test]
    fn test_asset_metrics_builder() {
        let asset_id = Uuid::new_v4();
        let metrics = AssetMetrics::new(asset_id)
            .with_gpu_utilization(85)
            .with_gpu_memory(72.5, 80.0)
            .with_gpu_temperature(68);

        assert_eq!(metrics.gpu_utilization_percent, Some(85));
        assert_eq!(metrics.gpu_memory_used_gb, Some(72.5));
        assert_eq!(metrics.gpu_memory_total_gb, Some(80.0));
        assert_eq!(metrics.gpu_temperature_celsius, Some(68));
    }

    #[test]
    fn test_asset_metrics_serialization() {
        let asset_id = Uuid::new_v4();
        let metrics = AssetMetrics::new(asset_id).with_gpu_utilization(85);

        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: AssetMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(metrics.asset_id, deserialized.asset_id);
        assert_eq!(
            metrics.gpu_utilization_percent,
            deserialized.gpu_utilization_percent
        );
    }
}
