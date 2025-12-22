use crate::config::Config;
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::time::Duration;

pub async fn create_pool(config: &Config) -> crate::error::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .min_connections(config.database.min_connections)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&config.database.url)
        .await?;

    Ok(pool)
}

pub async fn run_migrations(pool: &PgPool) -> crate::error::Result<()> {
    sqlx::migrate!("./migrations")
        .run(pool)
        .await
        .map_err(|e| crate::error::HpcError::internal(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation_requires_database() {
        let config = Config::default();
        assert!(config.database.url.contains("postgres"));
    }
}
