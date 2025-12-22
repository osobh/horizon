use sqlx::postgres::{PgPool, PgPoolOptions};

use crate::{config::DatabaseConfig, error::Result};

#[derive(Clone)]
pub struct DbPool {
    pool: PgPool,
}

impl DbPool {
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&config.url)
            .await?;

        Ok(Self { pool })
    }

    pub fn inner(&self) -> &PgPool {
        &self.pool
    }
}
