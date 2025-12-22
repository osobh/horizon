use crate::config::DatabaseConfig;
use crate::error::{HpcError, Result};
use sqlx::postgres::{PgPoolOptions, PgPool};

#[derive(Debug, Clone)]
pub struct DbPool {
    pool: PgPool,
}

impl DbPool {
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .connect(&config.url)
            .await
            .map_err(|e| HpcError::config(format!("Failed to connect to database: {}", e)))?;

        Ok(Self { pool })
    }

    pub fn inner(&self) -> &PgPool {
        &self.pool
    }
}
