use anyhow::Result;
use executive_intelligence::api::{create_router, handlers::AppState};
use sqlx::postgres::PgPoolOptions;
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .compact()
        .init();

    info!("Starting Executive Intelligence Service");

    // Get database URL from environment
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://horizon:horizon@localhost:5432/horizon_intelligence".to_string());

    info!("Connecting to database: {}", database_url.split('@').last().unwrap_or("unknown"));

    // Create database connection pool
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    info!("Database connection pool established");

    // Run database migrations
    info!("Running database migrations...");
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await?;
    info!("Database migrations completed");

    // Create app state with database pool
    let state = Arc::new(AppState::new(pool));
    let router = create_router(state);

    // Bind to port 8092
    let addr = SocketAddr::from(([0, 0, 0, 0], 8092));
    info!("Executive Intelligence Service listening on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
