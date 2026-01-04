use std::sync::Arc;

use cost_reporter::{
    api::{create_router, AppState},
    config::ReporterConfig,
    db::create_pool,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load configuration
    let config = ReporterConfig::default();
    tracing::info!(
        "Starting cost-reporter service on {}:{}",
        config.host,
        config.port
    );

    // Create database pool
    let pool = create_pool(&config.database_url).await?;
    tracing::info!("Database pool created");

    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await?;
    tracing::info!("Database migrations complete");

    // Create app state
    let state = Arc::new(AppState::new(config.clone(), pool));

    // Create router
    let app = create_router(state);

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
