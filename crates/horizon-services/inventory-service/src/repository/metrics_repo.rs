use sqlx::PgPool;
use uuid::Uuid;

use crate::error::{HpcError, Result};
use crate::models::AssetMetrics;

pub struct MetricsRepository {
    pool: PgPool,
}

impl MetricsRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    #[tracing::instrument(skip(self))]
    pub async fn get_by_asset(&self, asset_id: Uuid) -> Result<AssetMetrics> {
        let metrics = sqlx::query_as!(
            AssetMetrics,
            r#"
            SELECT
                asset_id,
                gpu_utilization_percent,
                gpu_memory_used_gb as "gpu_memory_used_gb: f64",
                gpu_memory_total_gb as "gpu_memory_total_gb: f64",
                gpu_temperature_celsius,
                gpu_power_watts as "gpu_power_watts: f64",
                cpu_utilization_percent,
                cpu_temperature_celsius,
                nic_rx_bandwidth_gbps as "nic_rx_bandwidth_gbps: f64",
                nic_tx_bandwidth_gbps as "nic_tx_bandwidth_gbps: f64",
                last_updated
            FROM asset_metrics
            WHERE asset_id = $1
            "#,
            asset_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            HpcError::not_found(
                "metrics",
                format!("Metrics for asset {} not found", asset_id),
            )
        })?;

        Ok(metrics)
    }

    #[tracing::instrument(skip(self))]
    pub async fn upsert(&self, metrics: &AssetMetrics) -> Result<AssetMetrics> {
        let updated = sqlx::query_as!(
            AssetMetrics,
            r#"
            INSERT INTO asset_metrics (
                asset_id, gpu_utilization_percent, gpu_memory_used_gb,
                gpu_memory_total_gb, gpu_temperature_celsius, gpu_power_watts,
                cpu_utilization_percent, cpu_temperature_celsius,
                nic_rx_bandwidth_gbps, nic_tx_bandwidth_gbps, last_updated
            )
            VALUES ($1, $2, ($3::double precision)::numeric, ($4::double precision)::numeric, $5, ($6::double precision)::numeric, $7, $8, ($9::double precision)::numeric, ($10::double precision)::numeric, $11)
            ON CONFLICT (asset_id)
            DO UPDATE SET
                gpu_utilization_percent = EXCLUDED.gpu_utilization_percent,
                gpu_memory_used_gb = EXCLUDED.gpu_memory_used_gb,
                gpu_memory_total_gb = EXCLUDED.gpu_memory_total_gb,
                gpu_temperature_celsius = EXCLUDED.gpu_temperature_celsius,
                gpu_power_watts = EXCLUDED.gpu_power_watts,
                cpu_utilization_percent = EXCLUDED.cpu_utilization_percent,
                cpu_temperature_celsius = EXCLUDED.cpu_temperature_celsius,
                nic_rx_bandwidth_gbps = EXCLUDED.nic_rx_bandwidth_gbps,
                nic_tx_bandwidth_gbps = EXCLUDED.nic_tx_bandwidth_gbps,
                last_updated = EXCLUDED.last_updated
            RETURNING
                asset_id,
                gpu_utilization_percent,
                gpu_memory_used_gb as "gpu_memory_used_gb: f64",
                gpu_memory_total_gb as "gpu_memory_total_gb: f64",
                gpu_temperature_celsius,
                gpu_power_watts as "gpu_power_watts: f64",
                cpu_utilization_percent,
                cpu_temperature_celsius,
                nic_rx_bandwidth_gbps as "nic_rx_bandwidth_gbps: f64",
                nic_tx_bandwidth_gbps as "nic_tx_bandwidth_gbps: f64",
                last_updated
            "#,
            metrics.asset_id,
            metrics.gpu_utilization_percent,
            metrics.gpu_memory_used_gb,
            metrics.gpu_memory_total_gb,
            metrics.gpu_temperature_celsius,
            metrics.gpu_power_watts,
            metrics.cpu_utilization_percent,
            metrics.cpu_temperature_celsius,
            metrics.nic_rx_bandwidth_gbps,
            metrics.nic_tx_bandwidth_gbps,
            metrics.last_updated
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_repository_creation() {
        let pool = PgPool::connect_lazy("postgres://localhost/test").unwrap();
        let repo = MetricsRepository::new(pool);
        assert!(std::mem::size_of_val(&repo) > 0);
    }
}
