use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub scheduler: SchedulerConfig,
    pub checkpoint: CheckpointConfig,
    pub inventory: InventoryConfig,
    pub server: ServerConfig,
    pub pricing: PricingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub scheduling_interval_ms: u64,
    pub max_queue_size: usize,
    pub enable_preemption: bool,
    pub enable_backfill: bool,
    pub preemption_grace_period_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub storage_path: String,
    pub s3_bucket: Option<String>,
    pub max_checkpoint_size_gb: u64,
    pub retention_days: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryConfig {
    pub base_url: String,
    pub timeout_secs: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

/// Pricing configuration for cost estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// GPU hourly rates by type (e.g., "H100" -> 4.0)
    pub gpu_hourly_rates: std::collections::HashMap<String, f64>,
    /// CPU cost per core per hour
    pub cpu_per_core_hour: f64,
    /// Memory cost per GB per hour
    pub memory_per_gb_hour: f64,
    /// Storage cost per GB per hour
    pub storage_per_gb_hour: f64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        let mut gpu_rates = std::collections::HashMap::new();
        gpu_rates.insert("H100".to_string(), 4.0);
        gpu_rates.insert("A100".to_string(), 2.5);
        gpu_rates.insert("V100".to_string(), 1.5);
        gpu_rates.insert("RTX4090".to_string(), 1.0);

        Self {
            gpu_hourly_rates: gpu_rates,
            cpu_per_core_hour: 0.05,
            memory_per_gb_hour: 0.01,
            storage_per_gb_hour: 0.0001,
        }
    }
}

impl Config {
    pub fn from_env() -> crate::Result<Self> {
        Ok(Self {
            database: DatabaseConfig {
                url: std::env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_dev".to_string()),
                max_connections: std::env::var("DB_MAX_CONNECTIONS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10),
                min_connections: std::env::var("DB_MIN_CONNECTIONS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2),
                connect_timeout_secs: 30,
            },
            scheduler: SchedulerConfig {
                scheduling_interval_ms: 100,
                max_queue_size: 10000,
                enable_preemption: true,
                enable_backfill: true,
                preemption_grace_period_secs: 60,
            },
            checkpoint: CheckpointConfig {
                storage_path: std::env::var("CHECKPOINT_PATH")
                    .unwrap_or_else(|_| "/tmp/checkpoints".to_string()),
                s3_bucket: std::env::var("CHECKPOINT_S3_BUCKET").ok(),
                max_checkpoint_size_gb: 100,
                retention_days: 7,
            },
            inventory: InventoryConfig {
                base_url: std::env::var("INVENTORY_SERVICE_URL")
                    .unwrap_or_else(|_| "http://localhost:8081".to_string()),
                timeout_secs: 30,
                retry_attempts: 3,
            },
            server: ServerConfig {
                host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8082),
            },
            pricing: PricingConfig::default(),
        })
    }

    pub fn scheduling_interval(&self) -> Duration {
        Duration::from_millis(self.scheduler.scheduling_interval_ms)
    }

    pub fn preemption_grace_period(&self) -> Duration {
        Duration::from_secs(self.scheduler.preemption_grace_period_secs)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env().expect("Failed to load default config")
    }
}
