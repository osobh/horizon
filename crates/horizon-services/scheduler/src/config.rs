use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub scheduler: SchedulerConfig,
    pub checkpoint: CheckpointConfig,
    pub inventory: InventoryConfig,
    pub server: ServerConfig,
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
