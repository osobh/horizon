use scheduler::{config::Config, scheduler::Scheduler};
use sqlx::PgPool;
use std::sync::Arc;

/// Test fixture that sets up the application with a real database
pub struct TestApp {
    pub pool: PgPool,
    pub scheduler: Arc<Scheduler>,
}

impl TestApp {
    pub async fn new() -> Self {
        // Get database URL from environment or use default
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/scheduler_test".to_string());

        // Create connection pool
        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("Failed to run migrations");

        // Clean up existing test data
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&pool)
            .await
            .expect("Failed to clean up test data");

        // Create config
        let config = Config::from_env().expect("Failed to load config");

        // Create scheduler
        let scheduler = Scheduler::new(config.clone(), pool.clone())
            .await
            .expect("Failed to create scheduler");

        Self {
            pool,
            scheduler: Arc::new(scheduler),
        }
    }

    pub async fn cleanup(&self) {
        // Clean up test data after each test
        sqlx::query("TRUNCATE jobs, user_usage, job_events CASCADE")
            .execute(&self.pool)
            .await
            .expect("Failed to clean up test data");
    }
}
