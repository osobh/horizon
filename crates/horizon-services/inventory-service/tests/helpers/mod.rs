use axum::Router;
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::time::Duration;
use uuid::Uuid;

use inventory_service::api::create_routes;

pub async fn setup_test_db() -> PgPool {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/inventory_test".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(3))
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database");

    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    clean_database(&pool).await;

    pool
}

pub async fn clean_database(pool: &PgPool) {
    sqlx::query("TRUNCATE asset_reservations, asset_metrics, asset_history, assets CASCADE")
        .execute(pool)
        .await
        .expect("Failed to clean database");
}

pub fn create_test_app(pool: PgPool) -> Router {
    create_routes(pool)
}

#[allow(dead_code)]
pub fn random_uuid() -> Uuid {
    Uuid::new_v4()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires PostgreSQL instance
    async fn test_setup_test_db() {
        let pool = setup_test_db().await;
        assert!(sqlx::query("SELECT 1").execute(&pool).await.is_ok());
    }
}
