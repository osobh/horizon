use anyhow::Context;
use initiative_tracker::{
    api::{create_router, handlers::AppState},
    config::Config,
    db::{create_pool, InitiativeRepository},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tracing_config = hpc_tracing::TracingConfig {
        service_name: "initiative-tracker".to_string(),
        log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        otlp_endpoint: None,
    };
    let _guard = hpc_tracing::init(tracing_config)?;

    let config = Config::from_env().context("Failed to load configuration")?;

    info!("Starting initiative-tracker service");
    info!("Service address: {}", config.http_addr);

    info!("Connecting to database...");
    let pool = create_pool(&config.database)
        .await
        .context("Failed to create database pool")?;

    info!("Running migrations...");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .context("Failed to run migrations")?;

    let repository = InitiativeRepository::new(pool);
    let state = Arc::new(AppState::new(repository));
    let router = create_router(state);

    info!("Starting HTTP server on {}", config.http_addr);
    let listener = TcpListener::bind(config.http_addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
