use crate::config::DatabaseConfig;
use crate::error::{HpcError, Result};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::time::Duration;

pub async fn create_pool(config: &DatabaseConfig) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&config.url)
        .await
        .map_err(|e| HpcError::config(format!("Failed to connect to database: {}", e)))?;

    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_pool_invalid_url() {
        let config = DatabaseConfig {
            url: "invalid://url".to_string(),
            max_connections: 5,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(create_pool(&config));
        assert!(result.is_err());
    }
}
