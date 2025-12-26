use crate::config::DatabaseConfig;
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;

pub async fn create_pool(config: &DatabaseConfig) -> crate::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(Duration::from_secs(config.connect_timeout_secs))
        .connect(&config.url)
        .await?;

    Ok(pool)
}
